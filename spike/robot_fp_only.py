"""Spike: robot detector on the device's own stored data only (the 27-number fp-ft1 fingerprint, z-scored against the
normal traces of the training folds), LR balanced, p > 0.5; 10 x 5-fold CV, real UCI data.
Run: .venv\Scripts\python.exe -m spike.robot_fp_only
"""
import collections

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from bench.robot_failures import NORMAL
from data.fetch_uci_robot import load
from edge.fingerprint import Baseline
from edge.profiles import ForceTorque

FT = ForceTorque()
for lp in ("lp1", "lp2", "lp3", "lp4", "lp5"):
    inst = load(lp)
    X = np.stack([FT.features(np.asarray(r)) for _, r in inst]); lab = np.array([l for l, _ in inst]); y = np.array([l not in NORMAL for l in lab])
    strat = lab if min(collections.Counter(lab).values()) >= 5 else y
    fa, det = [], []
    for seed in range(10):
        al = np.zeros(len(y), bool)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
            bl = Baseline.fit(X[tr][~y[tr]], FT.fp_version, FT.min_std)
            m = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(np.clip(bl.z(X[tr]), -50, 50), y[tr])
            al[te] = m.predict_proba(np.clip(bl.z(X[te]), -50, 50))[:, 1] > 0.5
        fa.append(al[~y].mean()); det.append(al[y].mean())
    print(lp, f"FA {np.mean(fa):.3f} (range {min(fa):.2f}-{max(fa):.2f})  detection {np.mean(det):.3f}")
