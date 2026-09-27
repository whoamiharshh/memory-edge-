"""Adapter over Qdrant Edge. The ONLY module in this repo that imports qdrant_edge (the Edge API is beta and
drifts; qdrant-edge-py is pinned to 0.8.0, and everything version-specific lives here).

Durability rule (kill test K4, spike/spike_durability.py): Edge writes do not survive a hard kill unless
flush() is called. Every write method here flushes before it returns, so "returned" means "durable".

Layout on disk:  <root>/shard/            the Edge shard
                 <root>/store_meta.json   dims, fp_version, text model, BM25 avg_len (checked on reopen)

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

import json
import math
import os
import pathlib
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from qdrant_edge import (Bm25, Bm25Config, CountRequest, Distance, EdgeConfig, EdgeShard, EdgeSparseVectorParams,
                         EdgeVectorParams, FacetRequest, FieldCondition, Filter, Fusion, MatchAny, MatchValue,
                         Modifier, PayloadSchemaType, Point, Prefetch, Query, QueryRequest, RangeFloat,
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


def _vectors_out(v: Any) -> dict[str, Any]:
    if not v:
        return {}
    return {k: ({"indices": list(x.indices), "values": list(x.values)} if isinstance(x, SparseVector) else list(x))
            for k, x in v.items()}


class EdgeStore:
    """One device's local memory shard. Thread-safe within a process (all shard access is serialised)."""

    def __init__(self, root: str | os.PathLike, *, vib_dim: int = FP_DIM, note_dim: int = NOTE_DIM,
                 fp_version: str = FP_VERSION, text_model: str = TEXT_MODEL, bm25_avg_len: float = BM25_AVG_LEN):
        self.root = pathlib.Path(root)
        self._lock = threading.RLock()
        wanted = {"vib_dim": vib_dim, "note_dim": note_dim, "fp_version": fp_version, "text_model": text_model}
        meta_path = self.root / META_FILE
        shard_path = self.root / "shard"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            diff = {k: (meta.get(k), v) for k, v in wanted.items() if meta.get(k) != v}
            if diff:
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
            self._shard.flush()
            self.meta = {**wanted, "bm25_avg_len": bm25_avg_len, "schema_version": 1}
            tmp = meta_path.with_suffix(".tmp")        # meta written last + atomically: its presence = shard ready
            tmp.write_text(json.dumps(self.meta, indent=2))
            tmp.replace(meta_path)
        self._bm25 = Bm25(Bm25Config(avg_len=float(self.meta["bm25_avg_len"])))

    # ---- lifecycle ------------------------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._shard is not None:
                self._shard.flush()
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
            self._shard.flush()

    # ---- writes (all durable on return) ---------------------------------------------------------------
    def _to_point(self, p: StorePoint) -> Point:
        vec: dict[str, Any] = {}
        if (v := _check_vector(VIB, p.vib, self.meta["vib_dim"])) is not None:
            vec[VIB] = v
        if (v := _check_vector(NOTE, p.note, self.meta["note_dim"])) is not None:
            vec[NOTE] = v
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
            self._shard.flush()
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
            self._shard.flush()
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
                self._shard.flush()
            return {**cur[0].payload, **changes}

    def set_payload_where(self, filter: Mapping[str, Any], fields: Mapping[str, Any]) -> None:
        """Merge `fields` into the payload of every point matching `filter` (durable on return)."""
        flt = build_filter(filter)
        if flt is None:
            raise ValueError("refusing to set payload on every point: give a filter")
        with self._lock:
            self._shard.update(UpdateOperation.set_payload_by_filter(flt, dict(fields)))
            self._shard.flush()

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
            legs[NOTE] = Query.Nearest(_check_vector(NOTE, note, self.meta["note_dim"]), using=NOTE)
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
