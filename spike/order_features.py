"""Spike: can a FLEET of identical machines learn fault types that the textbook rule misses?

Order-domain envelope features (dimensionless, so they transfer across speeds): for each defect frequency (BPFO,
BPFI, 2xBSF, FTF) and harmonics 1-3, log(peak / median envelope level); shaft-rate sidebands around BPFI (inner-race
signature); envelope energy at shaft harmonics 1x-5x; plus kurtosis/crest of the raw segment.
Evaluation is always on bearings the model has NEVER seen (no leakage):
  UOttawa  leave-one-bearing-out (20 folds; each bearing = its developing + faulty recordings)
  CWRU     leave-one-fault-size-out (7/14/21 mil = three physically different bearings per class)
  HUST     leave-one-bearing-type-out (6204..6208)
Run: .venv\\Scripts\\python.exe -m spike.order_features
"""
from __future__ import annotations

import collections
import math

import numpy as np
import scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edge import physics as P
from spike.physics_v2 import cwru, hust, uottawa

SEG_S = 1.0


def order_feats(x, fs, shaft, geo):
    band = P.kurtogram_band(x, fs)
    f, p = P.envelope_spectrum(x, fs, band)
    ref = float(np.median(p[(f > 5) & (f < 1000)])) + 1e-30
    lg = lambda v: math.log10(v / ref + 1e-12)
    out = []
    for name in ("bpfo", "bpfi", "bsf", "ftf"):
        o = geo.orders()[name]
        for h in (1, 2, 3):
            out.append(lg(P.peak_near(f, p, h * o * shaft, 0.015)))
    bpfi = geo.orders()["bpfi"] * shaft
    out.append(lg(P.peak_near(f, p, bpfi - shaft, 0.015) + P.peak_near(f, p, bpfi + shaft, 0.015)))
    for k in range(1, 6):
        out.append(lg(P.peak_near(f, p, k * shaft, 0.015)))
    y = x - x.mean()
    out += [math.log10(st.kurtosis(y, fisher=False) + 1e-9), math.log10(np.abs(y).max() / (y.std() + 1e-12))]
    return out


def dataset(source, keep_healthy=False):
    X, y, g, ds = [], [], [], []
    for d, name, x, fs, rpm, truth, geo, _ in source:
        if truth == "healthy" and not keep_healthy:
            continue
        n = int(SEG_S * fs)
        for i in range(0, min(len(x) - n + 1, 10 * n), n):
            X.append(order_feats(x[i:i + n], fs, rpm / 60.0, geo))
            y.append(truth)
            g.append(group(d, name))
            ds.append(d)
    return np.array(X), np.array(y), np.array(g), np.array(ds)


def group(d, name):
    if d.startswith("uottawa"):
        return name.split("_")[1]                      # bearing number
    if d == "hust":
        return name[1]                                 # bearing type digit
    from data.fetch_data import CWRU
    return str(CWRU[int(name)][1])                     # fault size = physical bearing


def lobo(X, y, g, names):
    """Leave-one-group-out, vote per recording is approximated by vote per group-class."""
    correct, total = collections.Counter(), collections.Counter()
    seg_ok = 0
    for gr in sorted(set(g)):
        tr, te = g != gr, g == gr
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        seg_ok += int((pred == y[te]).sum())
        for cls in set(y[te]):
            m = y[te] == cls
            vote = collections.Counter(pred[m]).most_common(1)[0][0]
            correct[cls] += vote == cls
            total[cls] += 1
    return seg_ok / len(y), {c: f"{correct[c]}/{total[c]}" for c in total}, sum(correct.values()) / sum(total.values())


if __name__ == "__main__":
    for label, src in (("uottawa-acc", lambda: (r for r in uottawa() if r[0] == "uottawa-acc")),
                       ("uottawa-mic", lambda: (r for r in uottawa() if r[0] == "uottawa-mic")),
                       ("cwru", cwru), ("hust", hust)):
        X, y, g, ds = dataset(src())
        seg, per, rec = lobo(X, y, g, None)
        print(f"{label:12s} unseen-bearing: segment acc {seg:.3f} | per bearing-class vote {rec:.3f} {per}", flush=True)
