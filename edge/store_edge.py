"""Adapter over Qdrant Edge. The ONLY module in this repo that imports qdrant_edge (the Edge API is beta and
drifts; qdrant-edge-py is pinned to 0.8.0, and everything version-specific lives here).

Durability rule (kill test K4, spike/spike_durability.py): Edge writes do not survive a hard kill unless
flush() is called. Every write method here flushes before it returns, so "returned" means "durable".

Layout on disk:  <root>/shard/            the Edge shard
                 <root>/store_meta.json   dims, fp_version, text model, BM25 avg_len (checked on reopen)

Snapshots (the fleet mirror; Qdrant's documented dual-shard pattern, spike/spike_snapshot.py): a store can be
built from a Qdrant Server shard snapshot (from_snapshot) and brought up to date with a partial snapshot
(snapshot_manifest -> server -> apply_snapshot). The server computes its BM25 sparse vectors with bm25_sparse()
below, i.e. with Edge's own tokenizer, so a device's BM25 queries match the mirrored documents.

Vectors on a point (all optional per point, e.g. baseline points carry only `vib`):
  vib        z-scored DSP fingerprint, Euclid (score = distance, lower is closer)
  note       dense text embedding, Cosine (computed by the caller; this module does not embed text)
  note_bm25  sparse BM25 from Edge's built-in tokenizer, IDF modifier (computed here from `bm25_text`)

Filters are plain dicts so callers never touch qdrant_edge types:
  {"machine_id": "m1"}              match value
  {"type": ["episode", "fleet"]}    match any
  {"version": {"gte": 2}}           numeric range
  {"!status": "retracted"}          leading "!" = must_not
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from qdrant_edge import (Bm25, Bm25Config, CountRequest, Distance, EdgeConfig, EdgeShard, EdgeSparseVectorParams,
                         EdgeVectorParams, FacetRequest, FieldCondition, Filter, Fusion, MatchAny, MatchValue,
                         Modifier, PayloadSchemaType, Point, PointVectors, Prefetch, Query, QueryRequest, RangeFloat,
                         ScrollRequest, SparseVector, UpdateMode, UpdateOperation)

from edge.fingerprint import DIM as FP_DIM, FP_VERSION

VIB, NOTE, NOTE_BM25 = "vib", "note", "note_bm25"
NOTE_DIM = 384                                   # BAAI/bge-small-en-v1.5
TEXT_MODEL = "BAAI/bge-small-en-v1.5"
BM25_AVG_LEN = 6.86                              # MEASURED mean tokens per problem text on the maintenance logbook
                                                 # (bench/results/text_retrieval.json); Qdrant's default is 256
RRF_K = 60
KEYWORD_INDEXES = ("type", "machine_id", "device_id", "site_id", "component", "fault_class",
                   "action_code", "outcome", "share_state", "status")
META_FILE = "store_meta.json"


TRANSIENT_IO = ("os error 5)", "os error 32)", "os error 33)")   # access denied / sharing / lock violation
FLUSH_RETRIES = 6


def _flush(shard: EdgeShard) -> None:
    """flush() with a short retry on TRANSIENT Windows file errors. Seen live once (28 Sep 2026): 'Failed to flush
    segment state ... Access is denied (os error 5)' while another program (typically the virus scanner) briefly
    held a file Edge was replacing. The data is already in the shard; flush only persists it, so a retry is safe.
    Anything else, or a lock that outlasts ~3 s, is raised (the journal then re-applies the op on the next boot)."""
    for i in range(FLUSH_RETRIES):
        try:
            shard.flush()
            return
        except Exception as e:
            if i == FLUSH_RETRIES - 1 or not any(t in str(e) for t in TRANSIENT_IO):
                raise
            time.sleep(0.05 * 2 ** i)


class StoreConfigError(RuntimeError):
    """The shard on disk was built with different dimensions / fingerprint version than the caller expects."""


@dataclass
class StorePoint:
    id: str
    payload: dict[str, Any]
    vib: Sequence[float] | None = None
    note: Sequence[float] | None = None
    bm25_text: str | None = None


@dataclass
class Record:
    id: str
    payload: dict[str, Any]
    vectors: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    id: str
    score: float
    payload: dict[str, Any]
    legs: dict[str, int | None] = field(default_factory=dict)   # per-leg 1-based rank (search(explain=True))


def canonical_id(pid: str | uuid.UUID) -> str:
    """Edge ids are UUIDs (or u64; we only use UUIDs). Raises ValueError on anything else."""
    return str(pid if isinstance(pid, uuid.UUID) else uuid.UUID(str(pid)))


def build_filter(spec: Mapping[str, Any] | None) -> Filter | None:
    if not spec:
        return None
    must, must_not = [], []
    for key, val in spec.items():
        target = must_not if key.startswith("!") else must
        key = key.lstrip("!")
        if isinstance(val, Mapping):
            target.append(FieldCondition(key, range=RangeFloat(**val)))
        elif isinstance(val, (list, tuple, set, frozenset)):
            target.append(FieldCondition(key, match=MatchAny(list(val))))
        else:
            target.append(FieldCondition(key, match=MatchValue(val)))
    return Filter(must=must or None, must_not=must_not or None)


def _check_vector(name: str, v: Sequence[float] | None, dim: int) -> list[float] | None:
    if v is None:
        return None
    out = [float(x) for x in v]
    if len(out) != dim:
        raise ValueError(f"{name} vector has {len(out)} dims, store expects {dim}")
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{name} vector contains NaN/inf")
    return out


_BM25_CACHE: dict[float, Bm25] = {}


def bm25_sparse(text: str, avg_len: float = BM25_AVG_LEN) -> tuple[list[int], list[float]]:
    """Edge's built-in BM25 document vector as plain lists (indices, values). Used by the cloud to fill the
    `note_bm25` sparse vector of the server-side mirror collection, so both sides share one tokenizer."""
    bm = _BM25_CACHE.get(float(avg_len))
    if bm is None:
        bm = _BM25_CACHE[float(avg_len)] = Bm25(Bm25Config(avg_len=float(avg_len)))
    s = bm.embed_document(text)
    return [int(i) for i in s.indices], [float(v) for v in s.values]


def _vectors_out(v: Any) -> dict[str, Any]:
    if not v:
        return {}
    return {k: ({"indices": list(x.indices), "values": list(x.values)} if isinstance(x, SparseVector) else list(x))
            for k, x in v.items()}


class EdgeStore:
    """One device's local memory shard. Thread-safe within a process (all shard access is serialised)."""

    def __init__(self, root: str | os.PathLike, *, vib_dim: int = FP_DIM, note_dim: int = NOTE_DIM,
                 fp_version: str = FP_VERSION, text_model: str = TEXT_MODEL, bm25_avg_len: float = BM25_AVG_LEN,
                 allow_text_model_change: bool = False):
        """allow_text_model_change: a store built with another text model opens anyway, with
        `pending_text_model` set; the caller must then run migrate_text_model() before writing or querying text
        vectors (docs/RESEARCH.md H.3: new named vector, re-embed, then switch). Other mismatches always raise."""
        self.root = pathlib.Path(root)
        self._lock = threading.RLock()
        self.pending_text_model: dict | None = None
        wanted = {"vib_dim": vib_dim, "note_dim": note_dim, "fp_version": fp_version, "text_model": text_model}
        meta_path = self.root / META_FILE
        shard_path = self.root / "shard"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            diff = {k: (meta.get(k), v) for k, v in wanted.items() if meta.get(k) != v}
            text_only = set(diff) <= {"text_model", "note_dim"}
            if diff and allow_text_model_change and text_only:
                self.pending_text_model = {"from": meta.get("text_model"), "to": text_model, "dim": note_dim}
            elif diff:
                raise StoreConfigError(f"store at {self.root} was built with different settings (disk, code): {diff}")
            self.meta = meta
            self._shard = EdgeShard.load(str(shard_path))
        else:
            shard_path.mkdir(parents=True, exist_ok=True)
            self._shard = EdgeShard.create(str(shard_path), EdgeConfig(
                vectors={VIB: EdgeVectorParams(size=vib_dim, distance=Distance.Euclid),
                         NOTE: EdgeVectorParams(size=note_dim, distance=Distance.Cosine)},
                sparse_vectors={NOTE_BM25: EdgeSparseVectorParams(modifier=Modifier.Idf)}))
            for key in KEYWORD_INDEXES:
                self._shard.update(UpdateOperation.create_field_index(key, PayloadSchemaType.Keyword))
            _flush(self._shard)
            self.meta = {**wanted, "bm25_avg_len": bm25_avg_len, "schema_version": 1}
            tmp = meta_path.with_suffix(".tmp")        # meta written last + atomically: its presence = shard ready
            tmp.write_text(json.dumps(self.meta, indent=2))
            tmp.replace(meta_path)
        self._bm25 = Bm25(Bm25Config(avg_len=float(self.meta["bm25_avg_len"])))
        self.note_name = self.meta.get("note_vector", NOTE)     # physical name of the active dense text vector

    def _write_meta(self) -> None:
        meta_path = self.root / META_FILE
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta, indent=2))
        tmp.replace(meta_path)

    def migrate_text_model(self, embed_documents: Callable[[list[str]], list[list[float]]],
                           text_of: Callable[[dict[str, Any]], str | None], batch: int = 64) -> dict:
        """Switch the dense text vector to a new model (H.3): create a NEW named vector, re-embed every point whose
        payload yields text (text_of), switch the meta to the new vector (written last: a crash before that simply
        re-runs the migration), then drop the old vector. The BM25 vector is model-free and is not touched."""
        p = self.pending_text_model
        if not p:
            return {"migrated": 0}
        new = "note_" + hashlib.sha1(p["to"].encode()).hexdigest()[:10]
        with self._lock:
            try:
                self._shard.update(UpdateOperation.create_dense_vector(new, p["dim"], Distance.Cosine))
            except Exception as e:                     # re-run after a crash: the vector is already there
                if "exist" not in str(e).lower():
                    raise
            _flush(self._shard)
        todo = [(r.id, t) for r in self.scroll() if (t := text_of(r.payload))]
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            vecs = embed_documents([t for _, t in chunk])
            pvs = [PointVectors(canonical_id(pid), {new: _check_vector(new, v, p["dim"])}) for (pid, _), v in zip(chunk, vecs)]
            with self._lock:
                self._shard.update(UpdateOperation.update_vectors(pvs))
                _flush(self._shard)
        old = self.note_name
        with self._lock:
            self.meta = {**self.meta, "text_model": p["to"], "note_dim": p["dim"], "note_vector": new,
                         "previous_text_models": [*self.meta.get("previous_text_models", []), p["from"]]}
            self._write_meta()
            self.note_name = new
            try:
                self._shard.update(UpdateOperation.delete_vector_name(old))
                _flush(self._shard)
            except Exception:                          # a leftover old vector wastes space but is never queried
                pass
        self.pending_text_model = None
        return {"migrated": len(todo), "from": p["from"], "to": p["to"], "vector": new}

    # ---- lifecycle ------------------------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._shard is not None:
                _flush(self._shard)
                self._shard.close()
                self._shard = None

    def __enter__(self) -> "EdgeStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def optimize(self) -> None:
        """Edge has no background optimizer; call this periodically (e.g. after bulk loads)."""
        with self._lock:
            self._shard.optimize()
            _flush(self._shard)

    # ---- snapshots (fleet mirror) ---------------------------------------------------------------------
    @classmethod
    def from_snapshot(cls, snapshot_path: str | os.PathLike, root: str | os.PathLike, **settings) -> "EdgeStore":
        """Create a NEW store at `root` from a Qdrant Server shard snapshot, then open and probe it. `settings` are
        the constructor's keyword settings (dims, fp_version, text_model, bm25_avg_len); the snapshot's collection
        must have been created with the same vectors (vib Euclid, note Cosine, note_bm25 sparse IDF)."""
        root = pathlib.Path(root)
        if root.exists():
            raise FileExistsError(f"{root} already exists; restore into a fresh directory")
        defaults = {"vib_dim": FP_DIM, "note_dim": NOTE_DIM, "fp_version": FP_VERSION, "text_model": TEXT_MODEL,
                    "bm25_avg_len": BM25_AVG_LEN}
        meta = defaults | settings
        root.mkdir(parents=True)
        EdgeShard.unpack_snapshot(str(snapshot_path), str(root / "shard"))
        meta_path = root / META_FILE
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**meta, "schema_version": 1, "source": "snapshot"}, indent=2))
        tmp.replace(meta_path)
        store = cls(root, **meta)
        try:
            store.probe()
        except Exception:
            store.close()
            raise
        return store

    def probe(self) -> None:
        """Run one query per named vector. Raises if the shard lacks a vector or its dims differ from the meta
        (Edge exposes no config getter in 0.8.0, so this is how a restored snapshot is checked)."""
        unit = [0.0] * self.meta["note_dim"]
        unit[0] = 1.0
        with self._lock:
            for q in (Query.Nearest([0.0] * self.meta["vib_dim"], using=VIB), Query.Nearest(unit, using=self.note_name),
                      Query.Nearest(self._bm25.embed_query("probe"), using=NOTE_BM25)):
                self._shard.query(QueryRequest(limit=1, query=q))

    def snapshot_manifest(self) -> dict:
        """The shard's segment manifest; the server answers it with a partial snapshot of what changed."""
        with self._lock:
            return self._shard.snapshot_manifest()

    def apply_snapshot(self, snapshot_path: str | os.PathLike) -> None:
        """Apply a (partial) server snapshot in place. Reads are blocked meanwhile (the store lock), as Qdrant's
        docs require writes to pause during a restore. Raises on failure; the caller then rebuilds the mirror."""
        tmp_dir = self.root / "snapshot_tmp"
        tmp_dir.mkdir(exist_ok=True)
        with self._lock:
            try:
                self._shard.update_from_snapshot(str(snapshot_path), str(tmp_dir))
                _flush(self._shard)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            self.probe()

    # ---- writes (all durable on return) ---------------------------------------------------------------
    def _to_point(self, p: StorePoint) -> Point:
        vec: dict[str, Any] = {}
        if (v := _check_vector(VIB, p.vib, self.meta["vib_dim"])) is not None:
            vec[VIB] = v
        if (v := _check_vector(NOTE, p.note, self.meta["note_dim"])) is not None:
            vec[self.note_name] = v
        if p.bm25_text:
            vec[NOTE_BM25] = self._bm25.embed_document(p.bm25_text)
        if not vec:
            raise ValueError(f"point {p.id} has no vectors")
        return Point(canonical_id(p.id), vec, dict(p.payload))

    def upsert(self, points: Sequence[StorePoint], *, insert_only: bool = False) -> list[str]:
        """Write points; returns the ids actually written. With insert_only, ids that already exist are left
        untouched and are not returned (idempotent replay). Validation errors raise before anything is written."""
        if not points:
            return []
        built = [self._to_point(p) for p in points]
        with self._lock:
            ids = [canonical_id(p.id) for p in built]          # Point.id comes back as uuid.UUID
            if insert_only:
                existing = {canonical_id(r.id) for r in self._shard.retrieve(ids, False, False)}
                written = [i for i in ids if i not in existing]
                self._shard.update(UpdateOperation.upsert_points(built, update_mode=UpdateMode.InsertOnly))
            else:
                written = ids
                self._shard.update(UpdateOperation.upsert_points(built))
            _flush(self._shard)
        return written

    def cas_update(self, point: StorePoint, expected_version: int) -> bool:
        """Compare-and-set on the payload field `version`: replace the point only if its stored version equals
        expected_version; the stored version becomes expected_version + 1. Returns False (nothing written) if the
        point is missing or its version moved on. Edge rejects a stale condition silently (spike finding), so the
        current version is read under the lock first; the Edge condition is kept as a second guard."""
        built = self._to_point(StorePoint(point.id, {**point.payload, "version": expected_version + 1},
                                          point.vib, point.note, point.bm25_text))
        pid = canonical_id(point.id)
        with self._lock:
            cur = self._shard.retrieve([pid], True, False)
            if not cur or cur[0].payload.get("version") != expected_version:
                return False
            cond = Filter(must=[FieldCondition("version", range=RangeFloat(gte=expected_version, lte=expected_version))])
            self._shard.update(UpdateOperation.upsert_points([built], condition=cond))
            _flush(self._shard)
            return self._shard.retrieve([pid], True, False)[0].payload.get("version") == expected_version + 1

    def modify(self, pid: str, fn: Callable[[dict[str, Any]], Mapping[str, Any]]) -> dict[str, Any] | None:
        """Atomic (in-process) read-modify-write of payload fields: fn(current_payload) -> fields to set.
        Returns the new payload, or None if the point does not exist."""
        pid = canonical_id(pid)
        with self._lock:
            cur = self._shard.retrieve([pid], True, False)
            if not cur:
                return None
            changes = dict(fn(dict(cur[0].payload)))
            if changes:
                self._shard.update(UpdateOperation.set_payload([pid], changes))
                _flush(self._shard)
            return {**cur[0].payload, **changes}

    def set_payload_where(self, filter: Mapping[str, Any], fields: Mapping[str, Any]) -> None:
        """Merge `fields` into the payload of every point matching `filter` (durable on return)."""
        flt = build_filter(filter)
        if flt is None:
            raise ValueError("refusing to set payload on every point: give a filter")
        with self._lock:
            self._shard.update(UpdateOperation.set_payload_by_filter(flt, dict(fields)))
            _flush(self._shard)

    def delete_where(self, filter: Mapping[str, Any]) -> None:
        """Delete every point matching `filter` (durable on return). Refuses an empty filter."""
        flt = build_filter(filter)
        if flt is None:
            raise ValueError("refusing to delete every point: give a filter")
        with self._lock:
            self._shard.update(UpdateOperation.delete_points_by_filter(flt))
            _flush(self._shard)

    # ---- reads ----------------------------------------------------------------------------------------
    def retrieve(self, ids: Sequence[str], *, with_vectors: bool = False) -> list[Record]:
        """Records in the order requested; missing ids are skipped."""
        want = [canonical_id(i) for i in ids]
        with self._lock:
            got = {canonical_id(r.id): r for r in self._shard.retrieve(want, True, with_vectors)}
        return [Record(i, dict(got[i].payload), _vectors_out(got[i].vector) if with_vectors else {})
                for i in want if i in got]

    def get(self, pid: str, *, with_vectors: bool = False) -> Record | None:
        r = self.retrieve([pid], with_vectors=with_vectors)
        return r[0] if r else None

    def nearest(self, vib: Sequence[float], *, filter: Mapping[str, Any] | None = None, limit: int = 1) -> list[Hit]:
        """Nearest points by fingerprint. score = Euclidean distance (ascending)."""
        q = Query.Nearest(_check_vector(VIB, vib, self.meta["vib_dim"]), using=VIB)
        with self._lock:
            res = self._shard.query(QueryRequest(limit=limit, query=q, filter=build_filter(filter), with_payload=True))
        return [Hit(canonical_id(r.id), float(r.score), dict(r.payload)) for r in res]

    def _legs(self, vib, note, text) -> dict[str, Any]:
        legs: dict[str, Any] = {}
        if vib is not None:
            legs[VIB] = Query.Nearest(_check_vector(VIB, vib, self.meta["vib_dim"]), using=VIB)
        if note is not None:
            legs[NOTE] = Query.Nearest(_check_vector(NOTE, note, self.meta["note_dim"]), using=self.note_name)
        if text:
            legs[NOTE_BM25] = Query.Nearest(self._bm25.embed_query(text), using=NOTE_BM25)
        return legs

    def search(self, *, vib: Sequence[float] | None = None, note: Sequence[float] | None = None,
               text: str | None = None, filter: Mapping[str, Any] | None = None, limit: int = 10,
               prefetch_limit: int = 50, weights: Mapping[str, float] | None = None,
               explain: bool = False) -> list[Hit]:
        """Hybrid search in ONE Edge request: a prefetch per supplied leg, fused with reciprocal rank fusion.
        score = RRF score (rank-based; not comparable across queries, so never threshold it).
        explain=True additionally runs each leg alone and reports each hit's rank in that leg."""
        legs = self._legs(vib, note, text)
        if not legs:
            raise ValueError("search needs at least one of vib, note, text")
        flt = build_filter(filter)
        w = [float(weights.get(k, 1.0)) for k in legs] if weights else None
        req = QueryRequest(limit=limit, with_payload=True, filter=flt, query=Fusion.Rrf(RRF_K, w),
                           prefetches=[Prefetch(limit=prefetch_limit, query=q, filter=flt) for q in legs.values()])
        with self._lock:
            res = self._shard.query(req)
            ranks: dict[str, dict[str, int]] = {}
            if explain:
                for name, q in legs.items():
                    alone = self._shard.query(QueryRequest(limit=prefetch_limit, query=q, filter=flt))
                    ranks[name] = {canonical_id(r.id): i + 1 for i, r in enumerate(alone)}
        hits = [Hit(canonical_id(r.id), float(r.score), dict(r.payload)) for r in res]
        if explain:
            for h in hits:
                h.legs = {name: ranks[name].get(h.id) for name in legs}
        return hits

    def count(self, filter: Mapping[str, Any] | None = None) -> int:
        with self._lock:
            return int(self._shard.count(CountRequest(filter=build_filter(filter))))

    def facet(self, key: str, *, filter: Mapping[str, Any] | None = None, limit: int = 50) -> dict[Any, int]:
        """Value -> exact count for a keyword-indexed payload field."""
        with self._lock:
            res = self._shard.facet(FacetRequest(key=key, limit=limit, exact=True, filter=build_filter(filter)))
        return {h.value: int(h.count) for h in res.hits}

    def scroll(self, *, filter: Mapping[str, Any] | None = None, batch: int = 256,
               with_vectors: bool = False) -> Iterator[Record]:
        """Iterate all matching records (pages of `batch`). Each page is read under the lock; iteration is not
        a snapshot, so concurrent writes may or may not be seen."""
        offset = None
        flt = build_filter(filter)
        while True:
            with self._lock:
                recs, offset = self._shard.scroll(ScrollRequest(offset=offset, limit=batch, filter=flt,
                                                                with_payload=True, with_vector=with_vectors))
            for r in recs:
                yield Record(canonical_id(r.id), dict(r.payload), _vectors_out(r.vector) if with_vectors else {})
            if offset is None:
                return

    def info(self) -> dict[str, Any]:
        with self._lock:
            i = self._shard.info()
        return {"points": int(i.points_count), "segments": int(i.segments_count), **self.meta}
