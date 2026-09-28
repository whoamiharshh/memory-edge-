"""A detector each device learns from ITS OWN history: healthy baseline fingerprints (label 0) vs fingerprints of
episodes the technician CONFIRMED as faults (label 1). Logistic regression (class-balanced), exported as plain numbers
and applied with numpy; its accuracy on this machine is measured by 5-fold cross-validation at training time and shown.

Why: the unsupervised gate misses subtle robot failures (bench/robot_failures.py: 36-55 % on two tasks). Trained per
robot task on real labelled traces, this detector found 96-99 % of failures with 0-10 % false alarms
(bench/robot_model.py). It only ever ADDS alarms to the gate (a window it scores > 0.5 opens or grows an episode),
and it is enabled per profile (force-torque by default).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

MIN_NORMAL, MIN_FAULT = 10, 5
CLIP = 50.0


def train(normal_z: np.ndarray, fault_z: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    if len(normal_z) < MIN_NORMAL or len(fault_z) < MIN_FAULT:
        raise ValueError(f"needs >= {MIN_NORMAL} healthy and >= {MIN_FAULT} confirmed-fault fingerprints; have "
                         f"{len(normal_z)} and {len(fault_z)}")
    X = np.clip(np.vstack([normal_z, fault_z]), -CLIP, CLIP)
    y = np.r_[np.zeros(len(normal_z)), np.ones(len(fault_z))]
    mk = lambda: LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")
    alarm = np.zeros(len(y), bool)
    k = int(min(5, min(len(normal_z), len(fault_z))))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        alarm[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] > 0.5
    m = mk().fit(X, y)
    return {"coef": np.round(m.coef_[0], 6).tolist(), "intercept": round(float(m.intercept_[0]), 6),
            "trained_on": {"healthy": int(len(normal_z)), "confirmed_fault": int(len(fault_z))},
            "cross_validated": {"false_alarm_rate": round(float(alarm[y == 0].mean()), 3),
                                "detection_rate": round(float(alarm[y == 1].mean()), 3), "folds": k},
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}


def probability(model: dict, z: np.ndarray) -> float:
    s = float(np.clip(np.asarray(z, dtype=float), -CLIP, CLIP) @ np.asarray(model["coef"]) + model["intercept"])
    return 1.0 / (1.0 + np.exp(-s))
