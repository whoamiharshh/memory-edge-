"""Robots: a failure detector TRAINED ON REAL labelled data (UCI Robot Execution Failures, CC BY 4.0), compared with
the unsupervised gate (bench/robot_failures.py), especially on subtle failures.

SHIPPED METHOD ("device_learned_detector", edge/local_detector.py): exactly what a device stores - the fp-ft1
fingerprint z-scored against the healthy traces of the training folds - class-balanced logistic regression, alarm when
p > 0.5 (no threshold tuning), 10 x stratified 5-fold CV (seeds 0-9), mean and range over the repetitions.
EARLIER METHOD (kept for comparison, "previous_*"): fingerprint + temporal-shape features, threshold set so that 5 %
of the TRAINING normals alarm - overfit scores made that threshold too low (9-15 % false alarms).
Honest limit: 47-164 traces per problem; results have wide uncertainty; the classes differ between problems, so a
model is trained per problem (a robot cell would train on its own labelled history).
Run: .venv\\Scripts\\python.exe -m bench.robot_model
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bench.robot_failures import NORMAL
from data.fetch_uci_robot import load
from edge.profiles import ForceTorque

OUT = pathlib.Path(__file__).resolve().parent / "results"
FT = ForceTorque()


def feats(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    d = np.diff(x, axis=0)
    sl = lambda v: np.sign(v) * np.log1p(np.abs(v))
    return np.concatenate([FT.features(x), sl(x[-1] - x[0]), sl(np.abs(d).max(axis=0))])


def run(lp: str, make) -> dict:
    inst = load(lp)
    X = np.stack([feats(r) for _, r in inst])
    lab = np.array([l for l, _ in inst])
    y = np.array([l not in NORMAL for l in lab])
    alerts = np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab if min(collections.Counter(lab).values()) >= 5 else y):
        m = make().fit(X[tr], y[tr])
        p_tr = m.predict_proba(X[tr])[:, 1]
        thr = float(np.quantile(p_tr[~y[tr]], 0.95)) + 1e-9          # <= 5 % of TRAINING normals alarm
        alerts[te] = m.predict_proba(X[te])[:, 1] > thr
    per = {c: round(float(alerts[lab == c].mean()), 3) for c in sorted(set(lab)) if c not in NORMAL}
    return {"false_alarm_rate": round(float(alerts[~y].mean()), 3), "detection_rate": round(float(alerts[y].mean()), 3),
            "per_failure_type": per, "traces": int(len(y))}


def device_detector(lp: str) -> dict:
    from edge.fingerprint import Baseline
    from edge.local_detector import probability, train
    inst = load(lp)
    X = np.stack([FT.features(np.asarray(r)) for _, r in inst])
    lab = np.array([l for l, _ in inst])
    y = np.array([l not in NORMAL for l in lab])
    strat = lab if min(collections.Counter(lab).values()) >= 5 else y
    fa, det, per = [], [], collections.defaultdict(list)
    for seed in range(10):
        al = np.zeros(len(y), bool)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
            bl = Baseline.fit(X[tr][~y[tr]], FT.fp_version, FT.min_std)
            m = train(bl.z(X[tr][~y[tr]]), bl.z(X[tr][y[tr]]))
            al[te] = [probability(m, z) > 0.5 for z in bl.z(X[te])]
        fa.append(al[~y].mean()); det.append(al[y].mean())
        for c in set(lab[y]):
            per[c].append(al[lab == c].mean())
    return {"false_alarm_rate": round(float(np.mean(fa)), 3), "false_alarm_range": [round(min(fa), 3), round(max(fa), 3)],
            "detection_rate": round(float(np.mean(det)), 3), "detection_range": [round(min(det), 3), round(max(det), 3)],
            "per_failure_type": {c: round(float(np.mean(v)), 3) for c, v in sorted(per.items())},
            "normal_traces": int((~y).sum()), "failure_traces": int(y.sum())}


def main() -> dict:
    makers = {"logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")),
              "gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0,
                                                                          class_weight="balanced")}
    out = {"method": __doc__.strip().splitlines()[0], "dataset": "UCI Robot Execution Failures, CC BY 4.0 (real)",
           "results": {lp: {"device_learned_detector": device_detector(lp)}
                       | {f"previous_{name}": run(lp, mk) for name, mk in makers.items()}
                       for lp in ("lp1", "lp2", "lp3", "lp4", "lp5")}}
    try:
        gate = json.loads((OUT / "robot_failures.json").read_text())["problems"]
        out["unsupervised_gate_for_comparison"] = {lp: {k: v for k, v in r.items() if k != "held_out_normals_per_seed"}
                                                   for lp, r in gate.items()}
    except (OSError, KeyError):
        pass
    OUT.mkdir(exist_ok=True)
    (OUT / "robot_model.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["results"], indent=1))
    return out


if __name__ == "__main__":
    main()
