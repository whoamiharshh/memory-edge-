"""Qdrant Server storage for the fleet (the ONLY cloud module that imports qdrant_client).

Per tenant, two collections (tenant isolation by construction, not only by filter):
  events_<tenant>   every accepted ShareEvent; point id = event_id; vector `vib` (fingerprint)
  cases_<tenant>    one point per case group (component, fault_class); vectors `vib` (centroid of its events'
                    fingerprints) and `note` (server-side bge embedding of the case's redacted summary text).
  mirror_<tenant>   the device-facing copy of cases_<tenant>, laid out exactly like a device's mirror shard (vib,
                    note, and a `note_bm25` IDF sparse vector computed with Edge's own BM25) in ONE shard and ONE
                    segment, so devices can pull it as Qdrant shard snapshots (full, then partial).
Idempotent ingest: insert_only on the deterministic event id (a replay is reported as `duplicate`).
Case writes: compare-and-set on the payload field `version` (conditional update), retried by the caller.
`seq` is a per-tenant monotonic counter stamped on every case write; devices pull cases with seq > cursor
(scroll fallback) or compare it with the mirror head before asking for a partial snapshot.

Works against a real server (url="http://127.0.0.1:6333") or in-process (":memory:") for tests. Snapshots need a
real server; in-process stores report supports_snapshots = False and devices fall back to the scroll pull.
"""
from __future__ import annotations

import re
import queue
import threading
import time
import zlib
from typing import Any, Iterator

import httpx
from qdrant_client import QdrantClient, models as m

from edge.fingerprint import DIM as FP_DIM
from edge.store_edge import BM25_AVG_LEN, bm25_sparse

NOTE_DIM = 384
_SAFE = re.compile(r"^[a-z0-9_-]{1,40}$")
GZIP_LEVEL = 6
CHUNK = 1 << 20


class SnapshotError(RuntimeError):
    """Qdrant Server refused or failed a snapshot request."""


QDRANT_TIMEOUT_S = 20      # every Qdrant call fails after this instead of hanging (then the device retries)


def _cname(kind: str, tenant: str) -> str:
    if not _SAFE.match(tenant):
        raise ValueError("invalid tenant id")
    return f"{kind}_{tenant}"


class _PooledClient:
    """Looks like one QdrantClient; every method call borrows a client from a small pool for the length of that call.
    A few warm connections instead of one per short-lived server thread (bench/fleet_scale.py profile: 13 % of the
    cloud's time was spent opening new connections), and never one client object used by two threads at once (a shared
    client broke under 20 concurrent devices, WinError 10038)."""

    def __init__(self, url: str, size: int = 8):
        self._url, self._size = url, size
        self._free: queue.LifoQueue = queue.LifoQueue()
        self._made = 0
        self._lock = threading.Lock()

    def _borrow(self) -> QdrantClient:
        try:
            return self._free.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._made < self._size:
                    self._made += 1
                    # keep-alive ON: qdrant-client turns it off for localhost by default; measured here (28 Sep 2026,
                    # 300 retrieves) p50 4.6 -> 2.9 ms and p95 23.9 -> 4.2 ms with it (every call was a new TCP connection)
                    return QdrantClient(url=self._url, timeout=QDRANT_TIMEOUT_S, limits=httpx.Limits(
                        max_connections=4, max_keepalive_connections=4, keepalive_expiry=30))
            return self._free.get()

    def __getattr__(self, name):
        def call(*a, **kw):
            c = self._borrow()
            try:
                return getattr(c, name)(*a, **kw)
            finally:
                self._free.put(c)
        return call


class _GroupWriter:
    """Group commit for event inserts. Every push hands its events to ONE writer thread, which takes everything waiting
    (up to MAX_POINTS), does one id lookup and one insert_only upsert per tenant, and answers each push.
    Why (bench/fleet_scale.py): per-event-id lock stripes held across Qdrant calls overlapped between hundreds of
    concurrent pushes and serialised them (~90 ms each, ~100 events/s); and each Qdrant write has a ~9 ms fixed cost
    (measured: 1 point/write ~100 points/s, 100 points/write ~4,500/s). One writer also keeps check-then-insert exact:
    the first copy of an event id is "accepted", every later one "duplicate"."""

    MAX_POINTS = 1000
    GATHER_S = 0.003           # after the first request arrives, wait this long for others to join the batch

    def __init__(self, store: "CloudStore"):
        self.store = store
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="event-writer", daemon=True).start()

    def submit(self, tenant: str, items: list[tuple[str, dict, list[float]]]) -> dict[str, str]:
        done = threading.Event()
        box: dict = {}
        self._q.put((tenant, items, done, box))
        if not done.wait(QDRANT_TIMEOUT_S * 3):
            raise TimeoutError("event writer did not answer in time (the device will retry)")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _run(self) -> None:
        while True:
            batch = [self._q.get()]
            deadline = time.monotonic() + self.GATHER_S
            n = len(batch[0][1])
            while n < self.MAX_POINTS:
                try:
                    batch.append(self._q.get(timeout=max(0.0, deadline - time.monotonic())))
                    n += len(batch[-1][1])
                except queue.Empty:
                    break
            by_tenant: dict[str, list] = {}
            for job in batch:
                by_tenant.setdefault(job[0], []).append(job)
            for tenant, jobs in by_tenant.items():
                try:
                    results = self.store._insert_batch(tenant, [it for j in jobs for it in j[1]])
                    for _, items, done, box in jobs:
                        box["result"] = {eid: results[eid] for eid, _, _ in items}
                        done.set()
                except Exception as e:                   # every waiting push gets the error -> devices retry
                    for _, _, done, box in jobs:
                        box["error"] = e
                        done.set()


class CloudStore:
    def __init__(self, url: str | None = None, location: str | None = None):
        # One Qdrant client PER THREAD for a server: FastAPI runs requests on a thread pool, and a shared client broke
        # under 20 concurrent devices on Windows ("WinError 10038 not a socket", bench/scale_fleet.py). In-memory
        # Qdrant (tests) lives inside one client object, so that one stays shared.
        self._local = threading.local()
        self._shared = _PooledClient(url.rstrip("/")) if url else QdrantClient(location=location or ":memory:")
        self.backend = url or location or ":memory:"
        self.url = url.rstrip("/") if url else None
        self.supports_snapshots = self.url is not None
        self._lock = threading.RLock()                      # schema creation and sequence numbers only
        self._stripes = [threading.Lock() for _ in range(64)]
        self._writer: _GroupWriter | None = None
        self._snap_cache: dict[str, tuple[int, bytes]] = {}    # tenant -> (mirror seq, gzip full snapshot)
        self.mirror_version: dict[str, int] = {}               # bumped on every mirror write (response caches)
        self.events_version: dict[str, int] = {}               # bumped on every event insert / field change
        self._snap_locks: dict[str, threading.Lock] = {}
        self._ready: set[str] = set()
        self._seq: dict[str, int] = {}

    @property
    def client(self) -> QdrantClient:
        return self._shared

    @property
    def _http(self) -> httpx.Client | None:
        if self.url is None:
            return None
        h = getattr(self._local, "http", None)
        if h is None:
            h = self._local.http = httpx.Client(timeout=httpx.Timeout(120, connect=10))
        return h

    # ---- schema -----------------------------------------------------------------------------------------
    def ensure_tenant(self, tenant: str) -> None:
        if tenant in self._ready:
            return
        with self._lock:
            ev, cs = _cname("events", tenant), _cname("cases", tenant)
            if not self.client.collection_exists(ev):
                self.client.create_collection(ev, vectors_config={"vib": m.VectorParams(size=FP_DIM, distance=m.Distance.EUCLID)})
                for k in ("case_id", "status", "site_id", "action_code", "outcome"):
                    self.client.create_payload_index(ev, k, m.PayloadSchemaType.KEYWORD)
            if not self.client.collection_exists(cs):
                self.client.create_collection(cs, vectors_config={
                    "vib": m.VectorParams(size=FP_DIM, distance=m.Distance.EUCLID),
                    "note": m.VectorParams(size=NOTE_DIM, distance=m.Distance.COSINE)})
                for k in ("component", "fault_class", "status"):
                    self.client.create_payload_index(cs, k, m.PayloadSchemaType.KEYWORD)
                self.client.create_payload_index(cs, "seq", m.PayloadSchemaType.INTEGER)
            mr = _cname("mirror", tenant)
            if not self.client.collection_exists(mr):
                # one shard + one segment + a 1 MB WAL: a shard snapshot then carries one segment of pre-allocated
                # pages instead of one per CPU (measured in spike/spike_snapshot.py: 580 MB -> 148 MB raw).
                self.client.create_collection(
                    mr, shard_number=1,
                    optimizers_config=m.OptimizersConfigDiff(default_segment_number=1),
                    wal_config=m.WalConfigDiff(wal_capacity_mb=1, wal_segments_ahead=0),
                    vectors_config={"vib": m.VectorParams(size=FP_DIM, distance=m.Distance.EUCLID),
                                    "note": m.VectorParams(size=NOTE_DIM, distance=m.Distance.COSINE)},
                    sparse_vectors_config={"note_bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)})
                for k in ("type", "component", "fault_class", "status"):
                    self.client.create_payload_index(mr, k, m.PayloadSchemaType.KEYWORD)
                self.client.create_payload_index(mr, "seq", m.PayloadSchemaType.INTEGER)
            self._seq[tenant] = self._max_seq(tenant)
            self._backfill_mirror(tenant)
            self._ready.add(tenant)

    def _max_seq(self, tenant: str, kind: str = "cases") -> int:
        pts, _ = self.client.scroll(_cname(kind, tenant), limit=1, with_payload=["seq"],
                                    order_by=m.OrderBy(key="seq", direction=m.Direction.DESC))
        return int(pts[0].payload["seq"]) if pts else 0

    # ---- device-facing mirror collection ----------------------------------------------------------------
    def _mirror_point(self, case_id: str, payload: dict, vib: list[float], note: list[float]) -> m.PointStruct:
        idx, val = bm25_sparse(payload.get("text", ""), BM25_AVG_LEN)
        return m.PointStruct(id=case_id, payload=payload, vector={
            "vib": vib, "note": note, "note_bm25": m.SparseVector(indices=idx, values=val)})

    def _mirror_put(self, tenant: str, case_id: str, payload: dict, vib: list[float], note: list[float]) -> None:
        """Copy one case write into the mirror. Conditional on version, so a slower writer of an older version
        can never overwrite a newer one (a missing point is simply inserted)."""
        cond = m.Filter(must=[m.FieldCondition(key="version", range=m.Range(lt=payload["version"]))])
        self.client.upsert(_cname("mirror", tenant), [self._mirror_point(case_id, payload, vib, note)],
                           update_filter=cond, wait=True)
        self.mirror_version[tenant] = self.mirror_version.get(tenant, 0) + 1

    def _backfill_mirror(self, tenant: str) -> None:
        """Bring the mirror level with cases_<tenant> (e.g. a mirror collection added to an existing tenant)."""
        if self._max_seq(tenant, "mirror") >= self._seq[tenant]:
            return
        off = None
        while True:
            pts, off = self.client.scroll(_cname("cases", tenant), limit=128, offset=off, with_payload=True,
                                          with_vectors=True)
            for p in pts:
                self._mirror_put(tenant, str(p.id), dict(p.payload), list(p.vector["vib"]), list(p.vector["note"]))
            if off is None:
                return

    def mirror_head(self, tenant: str, since: int | None = None) -> dict:
        """Newest mirror seq, case count, snapshot support; with `since`, also how many cases changed after it
        (the device uses that to choose a scroll delta or a snapshot)."""
        self.ensure_tenant(tenant)
        mr = _cname("mirror", tenant)
        out = {"seq": self._max_seq(tenant, "mirror"), "cases": int(self.client.count(mr, exact=True).count),
               "snapshots": self.supports_snapshots}
        if since is not None:
            out["changed"] = int(self.client.count(mr, exact=True, count_filter=m.Filter(
                must=[m.FieldCondition(key="seq", range=m.Range(gt=since))])).count)
        return out

    def open_mirror_snapshot(self, tenant: str, manifest: dict | None = None) -> tuple[int, Iterator[bytes] | None]:
        """Ask Qdrant Server for a snapshot of the tenant's mirror shard: full (manifest None) or partial (the
        device's snapshot_manifest). Returns (304, None) when the device is already current, else (200, a gzip
        stream). Qdrant's shard snapshot is mostly pre-allocated zero pages, so gzip shrinks it ~1000x (spike)."""
        if not self.supports_snapshots:
            raise SnapshotError("snapshots need a Qdrant Server (this store is in-process)")
        self.ensure_tenant(tenant)
        if manifest is None:
            # a FULL snapshot is the same for every device at the same mirror version: build it once, serve it to all
            # (bench/scale_fleet.py: 20 devices each triggering their own snapshot took ~78 s per device)
            seq = self._max_seq(tenant, "mirror")
            with self._snap_locks.setdefault(tenant, threading.Lock()):
                hit = self._snap_cache.get(tenant)
                if hit is None or hit[0] != seq:
                    status, stream = self._open_snapshot_stream(tenant, None)
                    hit = (seq, b"".join(stream))
                    self._snap_cache[tenant] = hit
            data = hit[1]
            return 200, iter([data[i:i + CHUNK] for i in range(0, len(data), CHUNK)])
        return self._open_snapshot_stream(tenant, manifest)

    def _open_snapshot_stream(self, tenant: str, manifest: dict | None) -> tuple[int, Iterator[bytes] | None]:
        base = f"{self.url}/collections/{_cname('mirror', tenant)}/shards/0/snapshot"
        req = (self._http.build_request("POST", base + "/partial/create", json=manifest) if manifest is not None
               else self._http.build_request("GET", base))
        r = self._http.send(req, stream=True)
        if r.status_code == 304:
            r.close()
            return 304, None
        if r.status_code != 200:
            detail = r.read()[:300].decode("utf-8", "replace")
            r.close()
            raise SnapshotError(f"Qdrant snapshot HTTP {r.status_code}: {detail}")

        def gz() -> Iterator[bytes]:
            z = zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, 31)       # wbits 31 = gzip container
            try:
                for chunk in r.iter_bytes(CHUNK):
                    out = z.compress(chunk)
                    if out:
                        yield out
                yield z.flush()
            finally:
                r.close()
        return 200, gz()

    def next_seq(self, tenant: str) -> int:
        with self._lock:
            self._seq[tenant] = max(self._seq.get(tenant, 0) + 1, int(time.time() * 1000))
            return self._seq[tenant]

    # ---- events -----------------------------------------------------------------------------------------
    def insert_event(self, tenant: str, event_id: str, payload: dict, vib: list[float]) -> str:
        """'accepted' or 'duplicate'. insert_only: an existing event is never modified by a replay."""
        self.ensure_tenant(tenant)
        c = _cname("events", tenant)
        # a lock per event-id STRIPE, not one global lock: the check-then-insert stays atomic for the same event, while
        # one slow Qdrant call can no longer stall every device's push (bench/scale_fleet.py found that freeze)
        with self._stripes[hash(event_id) % len(self._stripes)]:
            if self.client.retrieve(c, [event_id], with_payload=False):
                return "duplicate"
            self.client.upsert(c, [m.PointStruct(id=event_id, vector={"vib": vib}, payload=payload)],
                               update_mode=m.UpdateMode.INSERT_ONLY, wait=True)
            return "accepted"

    def insert_events(self, tenant: str, items: list[tuple[str, dict, list[float]]]) -> dict[str, str]:
        """accepted / duplicate per event id. Against a Qdrant Server the events go through the group-commit writer
        (_GroupWriter); in-process Qdrant (tests) writes directly."""
        if not items:
            return {}
        self.ensure_tenant(tenant)
        if self.url is not None:
            if self._writer is None:
                with self._lock:
                    if self._writer is None:
                        self._writer = _GroupWriter(self)
            return self._writer.submit(tenant, items)
        return self._insert_batch(tenant, items)

    def _insert_batch(self, tenant: str, items: list[tuple[str, dict, list[float]]]) -> dict[str, str]:
        """Batch form of insert_event: ONE lookup of the ids and ONE insert_only upsert for the new ones (a push of 50
        events was 100 Qdrant calls; bench/scale_fleet.py). The stripe locks of all ids are taken in a fixed order,
        so concurrent batches cannot deadlock and the check-then-insert stays atomic per event."""
        if not items:
            return {}
        self.ensure_tenant(tenant)
        c = _cname("events", tenant)
        locks = sorted({hash(eid) % len(self._stripes) for eid, _, _ in items})
        for k in locks:
            self._stripes[k].acquire()
        try:
            ids_ = list(dict.fromkeys(eid for eid, _, _ in items))
            have = {str(r.id) for r in self.client.retrieve(c, ids_, with_payload=False)}
            out, new = {}, []
            for eid, payload, vib in items:
                if eid in have or eid in out:
                    out.setdefault(eid, "duplicate")
                    continue
                out[eid] = "accepted"
                new.append(m.PointStruct(id=eid, vector={"vib": vib}, payload=payload))
            if new:
                self.client.upsert(c, new, update_mode=m.UpdateMode.INSERT_ONLY, wait=True)
                self.events_version[tenant] = self.events_version.get(tenant, 0) + 1
            return out
        finally:
            for k in reversed(locks):
                self._stripes[k].release()

    def get_event(self, tenant: str, event_id: str) -> dict | None:
        self.ensure_tenant(tenant)
        r = self.client.retrieve(_cname("events", tenant), [event_id], with_payload=True)
        return r[0].payload if r else None

    def set_event_fields(self, tenant: str, event_id: str, fields: dict) -> None:
        self.client.set_payload(_cname("events", tenant), payload=fields, points=[event_id], wait=True)
        self.events_version[tenant] = self.events_version.get(tenant, 0) + 1

    def events(self, tenant: str, case_id: str | None = None, with_vectors: bool = False) -> list[dict]:
        self.ensure_tenant(tenant)
        flt = m.Filter(must=[m.FieldCondition(key="case_id", match=m.MatchValue(value=case_id))]) if case_id else None
        out, off = [], None
        while True:
            pts, off = self.client.scroll(_cname("events", tenant), scroll_filter=flt, limit=256, offset=off,
                                          with_payload=True, with_vectors=with_vectors)
            for p in pts:
                d = dict(p.payload)
                if with_vectors:
                    d["fingerprint"] = list(p.vector["vib"])
                out.append(d)
            if off is None:
                return out

    # ---- cases ------------------------------------------------------------------------------------------
    def get_case(self, tenant: str, case_id: str, with_vectors: bool = False) -> dict | None:
        self.ensure_tenant(tenant)
        r = self.client.retrieve(_cname("cases", tenant), [case_id], with_payload=True, with_vectors=with_vectors)
        if not r:
            return None
        d = dict(r[0].payload)
        if with_vectors:
            d["_vectors"] = {k: list(v) for k, v in r[0].vector.items()}
        return d

    def put_case(self, tenant: str, case_id: str, payload: dict, vib: list[float], note: list[float],
                 expected_version: int | None) -> bool:
        """CAS write. expected_version None = create (insert_only). Returns True if our write is the stored one."""
        self.ensure_tenant(tenant)
        c = _cname("cases", tenant)
        new_version = (expected_version or 0) + 1
        body = payload | {"version": new_version, "seq": self.next_seq(tenant)}
        pt = m.PointStruct(id=case_id, vector={"vib": vib, "note": note}, payload=body)
        if expected_version is None:
            self.client.upsert(c, [pt], update_mode=m.UpdateMode.INSERT_ONLY, wait=True)
        else:
            cond = m.Filter(must=[m.FieldCondition(key="version", match=m.MatchValue(value=expected_version))])
            self.client.upsert(c, [pt], update_filter=cond, wait=True)
        cur = self.get_case(tenant, case_id)       # a rejected condition is silent: re-read to know
        won = bool(cur) and cur.get("version") == new_version and cur.get("seq") == body["seq"]
        if won:
            self._mirror_put(tenant, case_id, body, vib, note)
        return won

    def cases(self, tenant: str, since: int = 0, limit: int = 200, with_vectors: bool = False) -> list[dict]:
        self.ensure_tenant(tenant)
        pts, _ = self.client.scroll(
            _cname("cases", tenant), limit=limit, with_payload=True, with_vectors=with_vectors,
            scroll_filter=m.Filter(must=[m.FieldCondition(key="seq", range=m.Range(gt=since))]),
            order_by=m.OrderBy(key="seq", direction=m.Direction.ASC))
        out = []
        for p in pts:
            d = dict(p.payload)
            if with_vectors:
                d["_vectors"] = {k: list(v) for k, v in p.vector.items()}
            out.append(d)
        return out

    def search_cases(self, tenant: str, note: list[float], limit: int = 10, flt: dict | None = None) -> list[dict]:
        self.ensure_tenant(tenant)
        cond = [m.FieldCondition(key=k, match=m.MatchValue(value=v)) for k, v in (flt or {}).items()]
        res = self.client.query_points(_cname("cases", tenant), query=note, using="note", limit=limit,
                                       query_filter=m.Filter(must=cond) if cond else None, with_payload=True)
        return [dict(p.payload) | {"_score": p.score} for p in res.points]
