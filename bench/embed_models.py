"""Which text model? RESEARCH.md I.1 picked bge-small with all-MiniLM-L6-v2 as fallback and said "final pick by
benchmark"; this is that benchmark (it had been assumed, not measured).

Same protocol as bench/retrieval_text.py (Annotated Maintenance Logbook, 500 queries, seed 7, relevant = same annotated
(problem, part), identical texts excluded), run once per candidate model, all through Qdrant Edge:
dense only, and the hybrid the device actually uses (dense + Edge BM25, RRF). Also: model files on disk, embed
latency per note (p50/p95, CPU), and time to embed all 6,169 records.
Candidates (all 384-d, all in FastEmbed): BAAI/bge-small-en-v1.5 (MIT), sentence-transformers/all-MiniLM-L6-v2
(Apache-2.0), snowflake/snowflake-arctic-embed-xs (Apache-2.0). Licences from the FastEmbed model table.
The first run downloads the two new models into models_cache/ (one time, network); afterwards it is offline.
Run: .venv\\Scripts\\python.exe -m bench.embed_models
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import re
import shutil
import tempfile
import time

import numpy as np

from bench.retrieval_text import DEPTH, N_QUERIES, SEED, load, metrics
from edge.store_edge import EdgeStore, StorePoint
from shared import ids
from shared.embed import CACHE, FastEmbedder

OUT = pathlib.Path(__file__).resolve().parent / "results"
MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2", "snowflake/snowflake-arctic-embed-xs"]


def model_mb(name: str) -> float:
    key = name.split("/")[-1].lower()
    dirs = [d for d in CACHE.iterdir() if d.is_dir() and key in d.name.lower()]
    return round(sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file()) / 2 ** 20, 1)


def run_model(name, recs, queries, groups, by_i, avg_len) -> dict:
    emb = FastEmbedder(name, 384, offline=False)
    t = time.perf_counter()
    vecs = emb.embed_documents([r["problem"] for r in recs])
    embed_all_s = time.perf_counter() - t
    lat = []
    for r in random.Random(3).sample(recs, 100):
        t = time.perf_counter()
        emb.embed_query(r["problem"])
        lat.append((time.perf_counter() - t) * 1000)
    root = tempfile.mkdtemp(prefix="embbench_")
    res = {"dense": [], "hybrid_rrf": []}
    try:
        store = EdgeStore(root, bm25_avg_len=avg_len, text_model=name)
        pts = [StorePoint(ids.make_id("log", r["i"]), {"i": r["i"]}, note=v, bm25_text=r["problem"])
               for r, v in zip(recs, vecs)]
        for s in range(0, len(pts), 500):
            store.upsert(pts[s:s + 500])
        store.optimize()
        for q in queries:
            relevant = {j for j in groups[(q["tag"], q["part"])] if by_i[j]["norm"] != q["norm"]}
            qv = emb.embed_query(q["problem"])
            for m, kw in (("dense", {"note": qv}), ("hybrid_rrf", {"note": qv, "text": q["problem"]})):
                hits = store.search(limit=DEPTH + 40, prefetch_limit=100, **kw)
                ranked = [h.payload["i"] for h in hits if by_i[h.payload["i"]]["norm"] != q["norm"]]
                res[m].append(metrics(ranked, relevant))
        store.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    agg = lambda v, k: round(float(np.mean([x[k] for x in v])), 3)
    return {"model": name, "model_files_mb": model_mb(name),
            "embed_ms_p50": round(float(np.percentile(lat, 50)), 2), "embed_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "embed_all_records_s": round(embed_all_s, 1),
            **{m: {"P@3": agg(v, 0), "R@10": agg(v, 1), "MRR@10": agg(v, 2)} for m, v in res.items()}}


def main() -> dict:
    recs = load()
    avg_len = float(np.mean([len(re.findall(r"\w+", r["problem"])) for r in recs]))
    groups = collections.defaultdict(set)
    for r in recs:
        if r["tag"] and r["part"]:
            groups[(r["tag"], r["part"])].add(r["i"])
    by_i = {r["i"]: r for r in recs}
    cands = [r for r in recs if r["tag"] and r["part"] and
             any(by_i[j]["norm"] != r["norm"] for j in groups[(r["tag"], r["part"])])]
    queries = random.Random(SEED).sample(cands, N_QUERIES)
    rows = []
    for name in MODELS:
        try:
            rows.append(run_model(name, recs, queries, groups, by_i, avg_len))
        except Exception as e:                      # a model that cannot be loaded is reported, not hidden
            rows.append({"model": name, "error": f"{type(e).__name__}: {e}"[:300]})
        print(rows[-1], flush=True)
    out = {"method": __doc__.strip().splitlines()[0], "n_records": len(recs), "n_queries": N_QUERIES, "seed": SEED,
           "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "embed_models.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
