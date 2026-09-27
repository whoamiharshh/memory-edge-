"""Spike K5: Qdrant's dual-shard sync on OUR versions (Qdrant Server 1.19.1 binary + qdrant-edge-py 0.8.0).

Checks, against a throwaway server on ports 6433/6434:
  1. a server collection with named dense vectors + an IDF sparse vector whose values come from Edge's own BM25
  2. full shard snapshot  GET  /collections/{c}/shards/0/snapshot  -> EdgeShard.unpack_snapshot -> EdgeShard.load
  3. hybrid query (dense + BM25 prefetch, RRF) on the unpacked shard
  4. more server writes -> snapshot_manifest() -> POST /collections/{c}/shards/0/snapshot/partial/create
     -> update_from_snapshot() -> new + updated points visible, sizes reported
Run: .venv\\Scripts\\python.exe spike\\spike_snapshot.py
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import httpx
from qdrant_client import QdrantClient, models as m
from qdrant_edge import (Bm25, Bm25Config, EdgeShard, Fusion, Prefetch, Query, QueryRequest)

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:6433"
C = "spike_mirror"


def start_server(tmp: pathlib.Path) -> subprocess.Popen:
    env = os.environ | {"QDRANT__STORAGE__STORAGE_PATH": str(tmp / "storage"),
                        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(tmp / "snapshots"),
                        "QDRANT__SERVICE__HTTP_PORT": "6433", "QDRANT__SERVICE__GRPC_PORT": "6434",
                        "QDRANT__TELEMETRY_DISABLED": "true"}
    p = subprocess.Popen([str(ROOT / "qdrant_server" / "qdrant.exe")], cwd=tmp, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            if httpx.get(f"{URL}/readyz", timeout=1).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    p.kill()
    raise SystemExit("server did not start")


def sparse(bm25: Bm25, text: str) -> m.SparseVector:
    s = bm25.embed_document(text)
    return m.SparseVector(indices=list(s.indices), values=list(s.values))


def point(bm25, i: int, text: str, seq: int) -> m.PointStruct:
    vib = [float((i * 7 + k) % 5) for k in range(27)]
    note = [0.0] * 384
    note[i % 384] = 1.0
    return m.PointStruct(id=f"00000000-0000-0000-0000-{i:012d}", payload={"i": i, "text": text, "seq": seq,
                                                                          "component": "bearing"},
                         vector={"vib": vib, "note": note, "note_bm25": sparse(bm25, text)})


def download(r: httpx.Response, path: pathlib.Path) -> int:
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_bytes():
            f.write(chunk)
    return path.stat().st_size


def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="spike_snap_"))
    srv = start_server(tmp)
    out = {}
    try:
        cl = QdrantClient(url=URL)
        compact = os.environ.get("SPIKE_DEFAULT_CONFIG") != "1"
        extra = dict(optimizers_config=m.OptimizersConfigDiff(default_segment_number=1),
                     wal_config=m.WalConfigDiff(wal_capacity_mb=1, wal_segments_ahead=0)) if compact else {}
        out["compact_config"] = compact
        cl.create_collection(C, shard_number=1, **extra,
                             vectors_config={"vib": m.VectorParams(size=27, distance=m.Distance.EUCLID),
                                             "note": m.VectorParams(size=384, distance=m.Distance.COSINE)},
                             sparse_vectors_config={"note_bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)})
        cl.create_payload_index(C, "component", m.PayloadSchemaType.KEYWORD)
        bm25 = Bm25(Bm25Config(avg_len=6.86))
        cl.upsert(C, [point(bm25, i, f"bearing inner race fault action replace bearing case {i}", i) for i in range(20)],
                  wait=True)

        # 2. full snapshot -> unpack -> load
        edge_dir = tmp / "edge" / "mirror"
        with httpx.stream("GET", f"{URL}/collections/{C}/shards/0/snapshot", timeout=60) as r:
            out["full_snapshot_bytes"] = download(r, tmp / "full.snapshot")
        EdgeShard.unpack_snapshot(str(tmp / "full.snapshot"), str(edge_dir))
        shard = EdgeShard.load(str(edge_dir))
        out["after_full_points"] = shard.info().points_count

        # 3. hybrid on the unpacked shard
        q = QueryRequest(limit=3, with_payload=True, query=Fusion.Rrf(60, None), prefetches=[
            Prefetch(limit=10, query=Query.Nearest([0.0] * 26 + [1.0], using="vib")),
            Prefetch(limit=10, query=Query.Nearest(bm25.embed_query("case 7"), using="note_bm25"))])
        out["hybrid_top_after_full"] = [h.payload["i"] for h in shard.query(q)]

        # 4. partial update
        cl.upsert(C, [point(bm25, i, f"outer race fault lubrication failed case {i}", 100 + i) for i in range(20, 25)]
                  + [point(bm25, 3, "UPDATED case 3 retracted evidence", 200)], wait=True)
        manifest = shard.snapshot_manifest()
        out["manifest_type"] = type(manifest).__name__
        mbytes = json.dumps(manifest).encode()
        out["manifest_json_bytes"] = len(mbytes)
        with httpx.stream("POST", f"{URL}/collections/{C}/shards/0/snapshot/partial/create", content=mbytes,
                          headers={"content-type": "application/json"}, timeout=60) as r:
            out["partial_status"] = r.status_code
            out["partial_snapshot_bytes"] = download(r, tmp / "partial.snapshot")
        shard.update_from_snapshot(str(tmp / "partial.snapshot"))
        out["after_partial_points"] = shard.info().points_count
        got3 = shard.retrieve(["00000000-0000-0000-0000-000000000003"], True, False)
        out["updated_point_3_text"] = got3[0].payload["text"] if got3 else None
        out["bm25_hit_for_lubrication"] = [h.payload["i"] for h in shard.query(QueryRequest(
            limit=3, with_payload=True, query=Query.Nearest(bm25.embed_query("lubrication"), using="note_bm25")))]

        # 5. a partial with NO server change
        manifest = shard.snapshot_manifest()
        with httpx.stream("POST", f"{URL}/collections/{C}/shards/0/snapshot/partial/create",
                          content=json.dumps(manifest).encode(), headers={"content-type": "application/json"},
                          timeout=60) as r:
            out["noop_partial_status"] = r.status_code        # 304 Not Modified = mirror already current
            if r.status_code != 304:
                out["noop_partial_bytes"] = download(r, tmp / "partial2.snapshot")
                shard.update_from_snapshot(str(tmp / "partial2.snapshot"))
        out["after_noop_points"] = shard.info().points_count
        shard.close()
        reloaded = EdgeShard.load(str(edge_dir))
        out["after_reload_points"] = reloaded.info().points_count
        reloaded.close()
        out["manifest_sample"] = json.dumps(manifest)[:400]
    finally:
        srv.kill()
        srv.wait()
        if os.environ.get("SPIKE_KEEP") == "1":
            out["kept_tmp"] = str(tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
