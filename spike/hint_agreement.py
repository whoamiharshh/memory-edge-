"""Spike: precision of the fault-type hint when the physics rule and the fleet model AGREE (abstain otherwise).
Unseen-bearing protocol as in spike/order_features.py. Per recording (all its 1 s segments): physics vote (shipped
envelope rule, kurtogram band) and fleet-model vote; both must name the same class for the hint to be shown.
Run: .venv\Scripts\python.exe -m spike.hint_agreement
"""
from __future__ import annotations

import collections

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edge import physics as P
from spike.order_features import SEG_S, group, order_feats
from spike.physics_v2 import cwru, hust, uottawa


def rows(source):
    out = []
    for d, name, x, fs, rpm, truth, geo, _ in source:
        if truth == "healthy":
            continue
        n = int(SEG_S * fs)
        segs = [x[i:i + n] for i in range(0, min(len(x) - n + 1, 10 * n), n)]
        feats = [order_feats(s, fs, rpm / 60.0, geo) for s in segs]
        keys = ("bpfo", "bpfi", "bsf", "ftf") if d.startswith("uottawa") else ("bpfo", "bpfi", "bsf")
        pv = collections.Counter()
        for s in segs:
            sc = P.bearing_defect_scores(s, fs, rpm / 60.0, geo, band=P.kurtogram_band(s, fs))
            pv[P.DEFECT_TO_CLASS[max(keys, key=lambda k: sc[k])]] += 1
        out.append((d, name, truth, group(d, name), np.array(feats), pv.most_common(1)[0][0]))
    return out


def run(label, R, model):
    groups = sorted({r[3] for r in R})
    stats = collections.Counter()
    for gr in groups:
        tr = [r for r in R if r[3] != gr]
        te = [r for r in R if r[3] == gr]
        X = np.concatenate([r[4] for r in tr]); y = np.concatenate([[r[2]] * len(r[4]) for r in tr])
        clf = model()
        clf.fit(X, y)
        for r in te:
            mv = collections.Counter(clf.predict(r[4])).most_common(1)[0][0]
            stats["n"] += 1
            stats["model_ok"] += mv == r[2]
            stats["phys_ok"] += r[5] == r[2]
            if mv == r[5]:
                stats["agree"] += 1
                stats["agree_ok"] += mv == r[2]
    n = stats["n"]
    print(f"{label:22s} recordings {n} | physics {stats['phys_ok'] / n:.2f} | fleet model {stats['model_ok'] / n:.2f} | "
          f"both agree on {stats['agree']}/{n} ({stats['agree'] / n:.0%}) -> correct {stats['agree_ok']}/{stats['agree']} "
          f"({stats['agree_ok'] / max(1, stats['agree']):.0%})", flush=True)


if __name__ == "__main__":
    LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    HGB = lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)
    data = {"uottawa-acc": rows(r for r in uottawa() if r[0] == "uottawa-acc"),
            "uottawa-mic": rows(r for r in uottawa() if r[0] == "uottawa-mic"),
            "cwru": rows(cwru()), "hust": rows(hust())}
    for name, R in data.items():
        run(name + " LR", R, LR)
        run(name + " HGB", R, HGB)
