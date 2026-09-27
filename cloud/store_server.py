"""Qdrant Server storage for the fleet (the ONLY cloud module that imports qdrant_client).

Per tenant, two collections (tenant isolation by construction, not only by filter):
  events_<tenant>   every accepted ShareEvent; point id = event_id; vector `vib` (fingerprint)
  cases_<tenant>    one point per case group (component, fault_class); vectors `vib` (centroid of its events'
                    fingerprints) and `note` (server-side bge embedding of the case's redacted summary text).
Idempotent ingest: insert_only on the deterministic event id (a replay is reported as `duplicate`).
Case writes: compare-and-set on the payload field `version` (conditional update), retried by the caller.
`seq` is a per-tenant monotonic counter stamped on every case write; devices pull cases with seq > cursor.

Works against a real server (url="http://127.0.0.1:6333") or in-process (":memory:") for tests.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any

from qdrant_client import QdrantClient, models as m

from edge.fingerprint import DIM as FP_DIM

NOTE_DIM = 384
_SAFE = re.compile(r"^[a-z0-9_-]{1,40}$")


def _cname(kind: str, tenant: str) -> str:
    if not _SAFE.match(tenant):
        raise ValueError("invalid tenant id")
    return f"{kind}_{tenant}"


class CloudStore:
    def __init__(self, url: str | None = None, location: str | None = None):
        self.client = QdrantClient(url=url, timeout=10) if url else QdrantClient(location=location or ":memory:")
        self.backend = url or location or ":memory:"
        self._lock = threading.RLock()
        self._ready: set[str] = set()
        self._seq: dict[str, int] = {}

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
            self._seq[tenant] = self._max_seq(tenant)
            self._ready.add(tenant)

    def _max_seq(self, tenant: str) -> int:
        pts, _ = self.client.scroll(_cname("cases", tenant), limit=1, with_payload=["seq"],
                                    order_by=m.OrderBy(key="seq", direction=m.Direction.DESC))
        return int(pts[0].payload["seq"]) if pts else 0

    def next_seq(self, tenant: str) -> int:
        with self._lock:
            self._seq[tenant] = max(self._seq.get(tenant, 0) + 1, int(time.time() * 1000))
            return self._seq[tenant]

    # ---- events -----------------------------------------------------------------------------------------
    def insert_event(self, tenant: str, event_id: str, payload: dict, vib: list[float]) -> str:
        """'accepted' or 'duplicate'. insert_only: an existing event is never modified by a replay."""
        self.ensure_tenant(tenant)
        c = _cname("events", tenant)
        with self._lock:
            if self.client.retrieve(c, [event_id], with_payload=False):
                return "duplicate"
            self.client.upsert(c, [m.PointStruct(id=event_id, vector={"vib": vib}, payload=payload)],
                               update_mode=m.UpdateMode.INSERT_ONLY, wait=True)
            return "accepted"

    def get_event(self, tenant: str, event_id: str) -> dict | None:
        self.ensure_tenant(tenant)
        r = self.client.retrieve(_cname("events", tenant), [event_id], with_payload=True)
        return r[0].payload if r else None

    def set_event_fields(self, tenant: str, event_id: str, fields: dict) -> None:
        self.client.set_payload(_cname("events", tenant), payload=fields, points=[event_id], wait=True)

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
        return bool(cur) and cur.get("version") == new_version and cur.get("seq") == body["seq"]

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
