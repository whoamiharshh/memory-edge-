"""Spike: robot detector threshold from TRAINING-set scores (overfit) vs from OUT-OF-FOLD scores inside the training
folds; plain p > 0.5 for reference. 10 x stratified 5-fold CV (seeds 0-9) per learning problem, real UCI data.
Run: .venv\Scripts\python.exe -m spike.robot_threshold
"""
import collections

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bench.robot_failures import NORMAL
from bench.robot_model import feats
from data.fetch_uci_robot import load

mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))
for lp in ("lp1", "lp2", "lp3", "lp4", "lp5"):
    inst = load(lp)
    X = np.stack([feats(r) for _, r in inst]); lab = np.array([l for l, _ in inst]); y = np.array([l not in NORMAL for l in lab])
    strat = lab if min(collections.Counter(lab).values()) >= 5 else y
    res = collections.defaultdict(lambda: [[], []])
    for seed in range(10):
        al = {k: np.zeros(len(y), bool) for k in ("train_scores", "oof_scores", "p05")}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
            m = mk().fit(X[tr], y[tr])
            pt = m.predict_proba(X[te])[:, 1]
            thr_train = float(np.quantile(m.predict_proba(X[tr])[:, 1][~y[tr]], 0.95)) + 1e-9
            inner = StratifiedKFold(4, shuffle=True, random_state=seed)
            oof = cross_val_predict(mk(), X[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
            thr_oof = float(np.quantile(oof[~y[tr]], 0.95)) + 1e-9
            al["train_scores"][te] = pt > thr_train
            al["oof_scores"][te] = pt > thr_oof
            al["p05"][te] = pt > 0.5
        for k, a in al.items():
            res[k][0].append(a[~y].mean()); res[k][1].append(a[y].mean())
    print(lp, f"normals {int((~y).sum())}, failures {int(y.sum())}", {k: f"FA {np.mean(v[0]):.3f} (range {min(v[0]):.2f}-{max(v[0]):.2f}) det {np.mean(v[1]):.3f}" for k, v in res.items()})
