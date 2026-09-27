"""Vehicle early-warning risk hint for the `telemetry` profile.

The model (knowledge/vehicle_risk_model.json) is a logistic regression TRAINED ON REAL DATA ONLY: the SCANIA
Component X validation split (Scania CV AB, CC BY 4.0), tested on its test split (bench/vehicle_scania.py): ROC-AUC
0.75; alerting the riskiest ~10 % of trucks caught 32 % of those repaired within 48 time steps (precision 9.4 % vs a
2.8 % base rate). Stored as plain coefficients (no pickle: loading it cannot run code). It is a HINT shown with those
numbers; it never decides anything.
Features (identical to bench.vehicle_scania.truck_features): the last increment's fingerprint (27), its z-deviation
from the truck's own early baseline (27, clipped to +-20), the distance to that baseline, log(1+age), log(1+readouts),
and whether a baseline existed.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

MODEL = pathlib.Path(__file__).resolve().parents[1] / "knowledge" / "vehicle_risk_model.json"
MEASURED = "ROC-AUC 0.75 on 5,045 held-out real trucks; top-10 % alerts caught 32 % of repairs within 48 steps"
_M: dict | None = None


def model() -> dict | None:
    global _M
    if _M is None and MODEL.exists():
        _M = json.loads(MODEL.read_text())
    return _M


def features(last_raw, baseline, age: float, n_readouts: int) -> np.ndarray:
    last = np.asarray(last_raw, dtype=np.float64)
    if baseline is not None:
        z = np.clip(baseline.z(last), -20, 20)
        dist, has = float(np.sqrt((z ** 2).sum())), 1.0
    else:
        z, dist, has = np.zeros(len(last)), 0.0, 0.0
    return np.concatenate([last, z, [dist, math.log1p(age), math.log1p(n_readouts), has]])


def risk(last_raw, baseline, age: float, n_readouts: int) -> dict | None:
    m = model()
    if m is None:
        return None
    x = (features(last_raw, baseline, age, n_readouts) - np.asarray(m["mean"])) / np.asarray(m["scale"])
    p = 1.0 / (1.0 + math.exp(-(float(np.dot(np.asarray(m["coef"]), x)) + m["intercept"])))
    return {"probability": round(p, 4), "alert": p >= m["threshold"], "threshold": round(m["threshold"], 4),
            "target": m["target"], "measured": MEASURED,
            "trained_on": "real data: SCANIA Component X validation split (CC BY 4.0)"}
