"""Day-1 spike: prove every Qdrant Edge behaviour our design depends on (kill tests K1, K4).

Run: .venv\\Scripts\\python.exe spike\\spike_edge.py
Each check prints PASS/FAIL with evidence. No mocks: this talks to the real embedded engine.
"""
import os, shutil, subprocess, sys, tempfile, time, uuid, random
from qdrant_edge import (
    Bm25, Bm25Config, Distance, EdgeConfig, EdgeShard, EdgeSparseVectorParams, EdgeVectorParams,
    FacetRequest, FieldCondition, Filter, Fusion, MatchValue, Modifier, PayloadSchemaType, Point,
    Prefetch, Query, QueryRequest, CountRequest, UpdateMode, UpdateOperation, RangeFloat,
)

RESULTS = []
def check(name, ok, evidence=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {evidence}")

def config():
    return EdgeConfig(
        vectors={
            "vib": EdgeVectorParams(size=8, distance=Distance.Euclid),
            "note": EdgeVectorParams(size=4, distance=Distance.Cosine),
        },
        sparse_vectors={"note_bm25": EdgeSparseVectorParams(modifier=Modifier.Idf)},
    )

def rand(n): return [random.random() for _ in range(n)]

def main():
    random.seed(7)
    root = tempfile.mkdtemp(prefix="edge_spike_")
    path = os.path.join(root, "shard")
    os.makedirs(path)
    shard = EdgeShard.create(path, config())
    bm25 = Bm25(Bm25Config(avg_len=6.0))
    notes = ["bearing replaced vibration normal", "lubrication added fault persists",
             "misalignment corrected coupling", "inner race spall bearing replaced", "fan blade cleaned"]
    pts = []
    for i, t in enumerate(notes):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"devA:{i}"))
        pts.append(Point(pid, {"vib": rand(8), "note": rand(4), "note_bm25": bm25.embed_document(t)},
                         {"i": i, "text": t, "version": 1, "component": "bearing" if "bearing" in t else "other"}))
    shard.update(UpdateOperation.upsert_points(pts))
    shard.update(UpdateOperation.create_field_index("component", PayloadSchemaType.Keyword))
    check("upsert named dense+sparse", shard.count(CountRequest()) == 5, f"count={shard.count(CountRequest())}")

    # dense query per named vector
    r = shard.query(QueryRequest(limit=3, query=Query.Nearest(rand(8), using="vib"), with_payload=True))
    check("dense query using=vib (Euclid)", len(r) == 3, [round(p.score, 3) for p in r])

    # sparse BM25 query
    r = shard.query(QueryRequest(limit=2, query=Query.Nearest(bm25.embed_query("bearing replaced"), using="note_bm25"), with_payload=True))
    check("BM25 sparse query", r and "bearing" in r[0].payload["text"], [p.payload["text"] for p in r])

    # hybrid: prefetch + RRF in one request (K1)
    try:
        r = shard.query(QueryRequest(limit=3, with_payload=True, prefetches=[
            Prefetch(limit=5, query=Query.Nearest(rand(8), using="vib")),
            Prefetch(limit=5, query=Query.Nearest(rand(4), using="note")),
            Prefetch(limit=5, query=Query.Nearest(bm25.embed_query("bearing"), using="note_bm25")),
        ], query=Fusion.Rrf(60)))
        check("hybrid prefetch + Fusion.Rrf (K1)", len(r) == 3, [(p.payload["i"], round(p.score, 4)) for p in r])
    except Exception as e:
        check("hybrid prefetch + Fusion.Rrf (K1)", False, repr(e))
    try:
        r = shard.query(QueryRequest(limit=3, prefetches=[
            Prefetch(limit=5, query=Query.Nearest(rand(8), using="vib")),
            Prefetch(limit=5, query=Query.Nearest(bm25.embed_query("bearing"), using="note_bm25"))],
            query=Fusion.Rrf(60, [2.0, 1.0])))
        check("weighted RRF", len(r) == 3, "weights accepted")
    except Exception as e:
        check("weighted RRF", False, repr(e))

    # filter
    f = Filter(must=[FieldCondition("component", match=MatchValue("bearing"))])
    r = shard.query(QueryRequest(limit=5, query=Query.Nearest(rand(8), using="vib"), filter=f, with_payload=True))
    check("payload filter", all(p.payload["component"] == "bearing" for p in r) and len(r) == 2, len(r))

    # CAS: conditional upsert only if version==1
    target = pts[0].id
    newp = Point(target, {"vib": rand(8), "note": rand(4), "note_bm25": bm25.embed_document("edited")},
                 {"i": 0, "text": "edited", "version": 2, "component": "bearing"})
    vcond = lambda v: Filter(must=[FieldCondition("version", range=RangeFloat(gte=v, lte=v))])
    shard.update(UpdateOperation.upsert_points([newp], condition=vcond(1)))
    v_after_first = shard.retrieve([target], True, False)[0].payload["version"]
    stale = Point(target, {"vib": rand(8), "note": rand(4), "note_bm25": bm25.embed_document("stale")},
                  {"i": 0, "text": "stale-writer", "version": 2, "component": "bearing"})
    shard.update(UpdateOperation.upsert_points([stale], condition=vcond(1)))  # base version 1 is now wrong
    rec = shard.retrieve([target], True, False)[0].payload
    check("CAS: matching condition applies", v_after_first == 2, v_after_first)
    check("CAS: stale condition rejected (no overwrite)", rec["text"] == "edited", rec["text"])

    # InsertOnly idempotency
    dup = Point(pts[1].id, {"vib": rand(8), "note": rand(4), "note_bm25": bm25.embed_document("x")},
                {"i": 1, "text": "SHOULD NOT APPEAR", "version": 1, "component": "other"})
    shard.update(UpdateOperation.upsert_points([dup], update_mode=UpdateMode.InsertOnly))
    t1 = shard.retrieve([pts[1].id], True, False)[0].payload["text"]
    check("InsertOnly leaves existing point untouched", t1 == notes[1], t1)

    fr = shard.facet(FacetRequest(key="component", limit=10))
    check("facet", {h.value: h.count for h in fr.hits}.get("bearing") == 2, [(h.value, h.count) for h in fr.hits])
    t0 = time.perf_counter(); shard.optimize(); check("optimize()", True, f"{(time.perf_counter()-t0)*1000:.1f} ms")
    man = shard.snapshot_manifest()
    check("snapshot_manifest()", man is not None, str(man)[:120])
    shard.close()
    s2 = EdgeShard.load(path)
    check("close + load persists", s2.count(CountRequest()) == 5, s2.count(CountRequest()))
    s2.close()
    durability(root)
    shutil.rmtree(root, ignore_errors=True)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\nSUMMARY: {len(RESULTS)-len(failed)}/{len(RESULTS)} passed. Failed: {failed}")
    sys.exit(1 if failed else 0)

CHILD = r'''
import sys, uuid, random, time
from qdrant_edge import *
path = sys.argv[1]
cfg = EdgeConfig(vectors={"vib": EdgeVectorParams(size=8, distance=Distance.Euclid)})
s = EdgeShard.create(path, cfg)
for i in range(200):
    s.update(UpdateOperation.upsert_points([Point(str(uuid.uuid5(uuid.NAMESPACE_URL, str(i))), {"vib": [random.random() for _ in range(8)]}, {"i": i})]))
    print("ACK", i + 1, flush=True)
time.sleep(60)  # parent hard-kills us here: no close(), no flush()
'''

def durability(root):
    """K4: are acknowledged writes visible after a hard kill (TerminateProcess) with no close()/flush()?"""
    path = os.path.join(root, "crash_shard"); os.makedirs(path)
    p = subprocess.Popen([sys.executable, "-c", CHILD, path], stdout=subprocess.PIPE, text=True)
    acked = 0
    for line in p.stdout:
        if line.startswith("ACK"):
            acked = int(line.split()[1])
            if acked == 200:
                break
    p.kill(); p.wait()
    try:
        s = EdgeShard.load(path)
        n = s.count(CountRequest()); s.close()
        check("K4 durability after hard kill (no close/flush)", n == acked, f"acked={acked} recovered={n}")
    except Exception as e:
        check("K4 durability after hard kill (no close/flush)", False, f"acked={acked} load error {e!r}")

if __name__ == "__main__":
    main()
