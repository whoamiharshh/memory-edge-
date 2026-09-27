"""K2 kill test: does vibration-fingerprint retrieval transfer to bearings it has never seen?

Retrieval runs through a real Qdrant Edge shard (the same engine the product uses).
Metrics per split and normalisation:
  P@3   : fraction of the top-3 neighbours whose fault class matches the query's class
  kNN-5 : majority class of the top-5 neighbours == query class (accuracy)
Chance level for P@3 is reported as the class prior of the index.

Run: .venv\\Scripts\\python.exe -m bench.retrieval_vib
"""
from __future__ import annotations

import collections
import json
import pathlib
import shutil
import tempfile
import time

import numpy as np
from qdrant_edge import (CountRequest, Distance, EdgeConfig, EdgeShard, EdgeVectorParams, Point, Query,
                         QueryRequest, UpdateOperation)

from data.splits import bearing_split, build_dataset, leaky_split
from edge.fingerprint import PHYSICS_SLICE

OUT = pathlib.Path(__file__).resolve().parent / "results"


def evaluate(X: np.ndarray, y: np.ndarray, idx: np.ndarray, qry: np.ndarray, distance) -> dict:
    root = tempfile.mkdtemp(prefix="k2_")
    try:
        shard = EdgeShard.create(root, EdgeConfig(vectors={"vib": EdgeVectorParams(size=X.shape[1], distance=distance)}))
        pts = [Point(int(i), {"vib": X[i].tolist()}, {"cls": str(y[i])}) for i in idx]
        for s in range(0, len(pts), 512):
            shard.update(UpdateOperation.upsert_points(pts[s:s + 512]))
        shard.optimize()
        p3, knn, lat = [], [], []
        per_class = collections.defaultdict(list)
        for q in qry:
            t = time.perf_counter()
            res = shard.query(QueryRequest(limit=5, query=Query.Nearest(X[q].tolist(), using="vib"), with_payload=True))
            lat.append((time.perf_counter() - t) * 1000)
            labels = [r.payload["cls"] for r in res]
            p = sum(l == y[q] for l in labels[:3]) / 3
            p3.append(p)
            maj = collections.Counter(labels).most_common(1)[0][0]
            knn.append(maj == y[q])
            per_class[str(y[q])].append(maj == y[q])
        prior = collections.Counter(y[idx])
        chance = sum((prior[c] / len(idx)) * (np.sum(y[qry] == c) / len(qry)) for c in prior)
        shard.close()
        return {"P@3": round(float(np.mean(p3)), 3), "kNN5_acc": round(float(np.mean(knn)), 3),
                "chance_P@3": round(float(chance), 3),
                "per_class_kNN5": {c: round(float(np.mean(v)), 3) for c, v in sorted(per_class.items())},
                "n_index": int(len(idx)), "n_query": int(len(qry)),
                "query_ms_p50": round(float(np.percentile(lat, 50)), 3),
                "query_ms_p95": round(float(np.percentile(lat, 95)), 3)}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def relative_orders(X: np.ndarray) -> np.ndarray:
    """Defect-order pattern: bpfo/bpfi/bsf energies minus their row mean -> 'which defect dominates',
    independent of how severe (loud) the fault is."""
    o = X[:, PHYSICS_SLICE][:, 1:]
    return o - o.mean(axis=1, keepdims=True)


def dominant_defect_rule(X: np.ndarray, y: np.ndarray, qry: np.ndarray) -> dict:
    """No index at all: label = argmax defect order. Only meaningful for fault windows."""
    names = np.array(["outer_race", "inner_race", "ball"])
    fault_q = qry[y[qry] != "normal"]
    pred = names[np.argmax(X[fault_q][:, PHYSICS_SLICE][:, 1:], axis=1)]
    per = {str(c): round(float(np.mean(pred[y[fault_q] == c] == c)), 3) for c in names}
    return {"fault_acc": round(float(np.mean(pred == y[fault_q])), 3), "per_class": per, "n": int(len(fault_q))}


def standardise(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu, sd = X[ref].mean(axis=0), X[ref].std(axis=0)
    return (X - mu) / np.where(sd < 1e-9, 1e-9, sd)


def main() -> dict:
    d = build_dataset()
    y = d["cls"]
    results = {}
    for split_name, (idx, qry) in {"bearing_level (honest)": bearing_split(d), "random_windows (leaky)": leaky_split(d)}.items():
        healthy_ref = idx[y[idx] == "normal"]                    # baseline fit only on indexed healthy data
        Xp = d["X"][:, PHYSICS_SLICE]
        Xs = d["X"][:, :PHYSICS_SLICE.start]
        rel = relative_orders(d["X"])
        variants = {
            "physics-relative (pattern) / Euclid": (rel, Distance.Euclid),
            "physics-relative + healthy-z stats / Euclid": (np.hstack([rel * 3.0, standardise(Xs, healthy_ref) * 0.2]), Distance.Euclid),
            "v1-stats-only healthy-z / Euclid": (standardise(Xs, healthy_ref), Distance.Euclid),
            "full(v2) healthy-z / Euclid": (standardise(d["X"], healthy_ref), Distance.Euclid),
            "full(v2) index-z / Euclid": (standardise(d["X"], idx), Distance.Euclid),
            "physics-only index-z / Euclid": (standardise(Xp, idx), Distance.Euclid),
            "physics-only index-z / Cosine": (standardise(Xp, idx), Distance.Cosine),
        }
        for vname, (X, dist) in variants.items():
            results[f"{split_name} | {vname}"] = evaluate(X, y, idx, qry, dist)
        results[f"{split_name} | RULE dominant-defect (no retrieval)"] = dominant_defect_rule(d["X"], y, qry)
    OUT.mkdir(exist_ok=True)
    (OUT / "k2_vib_retrieval.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if "P@3" not in v:
            print(f"{k:60s} fault_acc={v['fault_acc']:.3f} per_class={v['per_class']} n={v['n']}")
            continue
        print(f"{k:60s} P@3={v['P@3']:.3f} kNN5={v['kNN5_acc']:.3f} chance={v['chance_P@3']:.3f} "
              f"p50={v['query_ms_p50']}ms per_class={v['per_class_kNN5']}")
    return results


if __name__ == "__main__":
    main()
