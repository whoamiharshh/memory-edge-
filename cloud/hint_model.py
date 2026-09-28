"""Train the fleet-learned fault hint from technician-confirmed evidence (edge/fleet_hint.py applies it).

Input: live (not retracted or quarantined) events that carry order features and a technician-confirmed bearing fault class.
Model: centred features -> multinomial logistic regression (scikit-learn), exported as plain JSON numbers.
Honesty: its accuracy is measured leave-one-DEVICE-out (every case is predicted by a model that never saw that
device) and shipped with the model; the UI shows that number, not the training accuracy.
"""
from __future__ import annotations

import collections
import datetime as dt
from typing import Any

import numpy as np

from edge.physics import ORDER_FEATURE_NAMES, ORDER_FEATURES_VERSION

CLASSES = ("inner_race", "outer_race", "ball", "cage")
MIN_PER_CLASS = 2
MIN_DEVICES = 2


def _fit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    # centred, NOT standardised: the features are all log10 ratios on one scale already, and standardising lets pure
    # noise columns weigh as much as a real defect line when the fleet has few cases (bench/fault_hint.py: centred
    # >= standardised on all four real datasets; tests/unit/test_fleet_hint.py shows the small-sample failure)
    mean, scale = X.mean(axis=0), np.ones(X.shape[1])
    clf = LogisticRegression(max_iter=3000, C=0.5).fit((X - mean) / scale, y)
    return clf, mean, scale


def train(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns {"model": {...} | None, "why": str}. `events` are cloud event payloads (any case)."""
    rows = [e for e in events if e.get("status", "active") == "active" and e.get("order_features")
            and e.get("of_version") == ORDER_FEATURES_VERSION and e.get("fault_class") in CLASSES
            and e.get("fault_class_source") == "technician"]
    # one physical repair counts once (same collapse rule as the tallies)
    seen, uniq = set(), []
    for e in rows:
        k = (e.get("device_id"), e.get("content_hash") or e.get("event_id"))
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    per = collections.Counter(e["fault_class"] for e in uniq)
    classes = sorted(c for c, n in per.items() if n >= MIN_PER_CLASS)
    uniq = [e for e in uniq if e["fault_class"] in classes]
    devices = sorted({e["device_id"] for e in uniq})
    if len(classes) < 2 or len(devices) < MIN_DEVICES:
        return {"model": None, "why": f"needs confirmed cases of >= 2 bearing fault classes (>= {MIN_PER_CLASS} "
                                      f"each) from >= {MIN_DEVICES} devices; have {dict(per)} from {len(devices)} device(s)"}
    X = np.asarray([e["order_features"] for e in uniq], dtype=float)
    y = np.asarray([e["fault_class"] for e in uniq])
    g = np.asarray([e["device_id"] for e in uniq])
    ok = n = 0
    for dev in devices:                                  # leave-one-device-out accuracy
        tr, te = g != dev, g == dev
        if len(set(y[tr])) < 2:
            continue
        clf, m, s = _fit(X[tr], y[tr])
        ok += int((clf.predict((X[te] - m) / s) == y[te]).sum())
        n += int(te.sum())
    clf, mean, scale = _fit(X, y)
    coef = clf.coef_ if len(clf.classes_) > 2 else np.vstack([-clf.coef_[0] / 2, clf.coef_[0] / 2])
    icpt = clf.intercept_ if len(clf.classes_) > 2 else np.array([-clf.intercept_[0] / 2, clf.intercept_[0] / 2])
    model = {"of_version": ORDER_FEATURES_VERSION, "feature_names": ORDER_FEATURE_NAMES,
             "classes": [str(c) for c in clf.classes_], "mean": mean.round(6).tolist(), "scale": scale.round(6).tolist(),
             "coef": np.round(coef, 6).tolist(), "intercept": np.round(icpt, 6).tolist(),
             "trained_on": {"cases": len(uniq), "devices": len(devices), "per_class": dict(collections.Counter(y.tolist()))},
             "unseen_device_accuracy": round(ok / n, 3) if n else None, "unseen_device_cases": n,
             "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    return {"model": model, "why": "trained"}
