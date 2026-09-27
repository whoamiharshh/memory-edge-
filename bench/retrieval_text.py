"""Text retrieval benchmark on REAL maintenance text: dense (bge-small) vs BM25 (Qdrant Edge built-in) vs
RRF hybrid (one Qdrant Edge prefetch+fusion query) - the same store code the device uses.

Data: Annotated Maintenance Logbook (Zenodo 17903357, CC BY 4.0; aviation, a proxy domain for our motors).
Index: every record's PROBLEM text. Query: the PROBLEM text of a sampled record.
Relevant: other records with the same (TAGEDPROBLEM, PART) annotation, both non-empty.
Leakage guard: records whose normalised PROBLEM text is identical to the query are removed from the ranking
and from the relevant set (3,512 distinct problems among 6,169 records - duplicates would make it trivial).
Metrics: P@3, R@10, MRR@10 over N_QUERIES queries with >= 1 relevant record (fixed seed).
Run: .venv\\Scripts\\python.exe -m bench.retrieval_text
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import pathlib
import random
import re
import shutil
import tempfile
import time

import numpy as np

from edge.store_edge import EdgeStore, StorePoint
from shared import ids
from shared.embed import BgeEmbedder
from shared.redact import normalise

OUT = pathlib.Path(__file__).resolve().parent / "results"
N_QUERIES = 500
SEED = 7
DEPTH = 10


def load() -> list[dict]:
    f = glob.glob(str(pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "logbook" / "*.csv"))[0]
    rows = list(csv.reader(open(f, encoding="utf-8", errors="replace")))[1:]
    return [{"i": k, "problem": r[1].strip(), "norm": normalise(r[1]), "tag": r[5].strip().upper(),
             "part": r[4].strip().upper()} for k, r in enumerate(rows) if r[1].strip()]


def metrics(ranked: list[int], relevant: set[int]) -> tuple[float, float, float]:
    p3 = sum(r in relevant for r in ranked[:3]) / 3
    r10 = sum(r in relevant for r in ranked[:DEPTH]) / len(relevant)
    mrr = next((1 / (k + 1) for k, r in enumerate(ranked[:DEPTH]) if r in relevant), 0.0)
    return p3, r10, mrr


def main() -> dict:
    recs = load()
    toks = [len(re.findall(r"\w+", r["problem"])) for r in recs]
    avg_len = float(np.mean(toks))
    groups = collections.defaultdict(set)
    for r in recs:
        if r["tag"] and r["part"]:
            groups[(r["tag"], r["part"])].add(r["i"])
    by_i = {r["i"]: r for r in recs}
    cands = [r for r in recs if r["tag"] and r["part"] and
             any(by_i[j]["norm"] != r["norm"] for j in groups[(r["tag"], r["part"])])]
    queries = random.Random(SEED).sample(cands, N_QUERIES)

    emb = BgeEmbedder()
    t = time.perf_counter()
    vecs = emb.embed_documents([r["problem"] for r in recs])
    embed_s = time.perf_counter() - t
    root = tempfile.mkdtemp(prefix="text_bench_")
    try:
        store = EdgeStore(root, bm25_avg_len=avg_len)
        pts = [StorePoint(ids.make_id("log", r["i"]), {"i": r["i"]}, note=v, bm25_text=r["problem"])
               for r, v in zip(recs, vecs)]
        for s in range(0, len(pts), 500):
            store.upsert(pts[s:s + 500])
        store.optimize()
        res = {m: [] for m in ("dense", "bm25", "hybrid_rrf")}
        lat = {m: [] for m in res}
        for q in queries:
            relevant = {j for j in groups[(q["tag"], q["part"])] if by_i[j]["norm"] != q["norm"]}
            qv = emb.embed_query(q["problem"])
            for m, kw in (("dense", {"note": qv}), ("bm25", {"text": q["problem"]}),
                          ("hybrid_rrf", {"note": qv, "text": q["problem"]})):
                t = time.perf_counter()
                hits = store.search(limit=DEPTH + 40, prefetch_limit=100, **kw)
                lat[m].append((time.perf_counter() - t) * 1000)
                ranked = [h.payload["i"] for h in hits if by_i[h.payload["i"]]["norm"] != q["norm"]]
                res[m].append(metrics(ranked, relevant))
        store.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    out = {"method": __doc__.strip().splitlines()[0], "dataset": "Annotated Maintenance Logbook (Zenodo 17903357, CC BY 4.0)",
           "n_records": len(recs), "n_queries": N_QUERIES, "seed": SEED, "bm25_avg_len_measured": round(avg_len, 2),
           "embed_model": emb.name, "embed_all_records_s": round(embed_s, 1),
           "results": {m: {"P@3": round(float(np.mean([x[0] for x in v])), 3), "R@10": round(float(np.mean([x[1] for x in v])), 3),
                           "MRR@10": round(float(np.mean([x[2] for x in v])), 3),
                           "query_ms_p50": round(float(np.percentile(lat[m], 50)), 3),
                           "query_ms_p95": round(float(np.percentile(lat[m], 95)), 3)} for m, v in res.items()}}
    OUT.mkdir(exist_ok=True)
    (OUT / "text_retrieval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
