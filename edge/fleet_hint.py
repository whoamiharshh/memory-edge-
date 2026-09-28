"""Fleet-learned fault hint, applied on the device (numpy only, offline).

The cloud trains a small multinomial logistic regression on the order features (edge/physics.order_features) of
TECHNICIAN-CONFIRMED cases from the whole fleet (cloud/hint_model.py) and measures it on devices it did not train on.
Devices pull the coefficients as plain JSON with the mirror (no pickle, nothing executable) and combine it with the
physics rule:

  both name the same class       -> "confident"   (measured on unseen bearings: HUST 100 %, UOttawa accel 92 %,
                                                    CWRU 87 % of such cases correct; bench/fault_hint.py)
  they disagree / no fleet model -> "uncertain"   the UI says to inspect; the technician's confirmation decides

The hint never shares, closes or decides anything: the fleet groups evidence by the CONFIRMED class only.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from edge.physics import ORDER_FEATURE_NAMES, ORDER_FEATURES_VERSION

MIN_PROB = 0.5          # a fleet prediction below this probability is reported but never counts as agreement


def valid(model: dict | None) -> bool:
    try:
        k, d = len(model["classes"]), len(ORDER_FEATURE_NAMES)
        return (model.get("of_version") == ORDER_FEATURES_VERSION and k >= 2
                and np.asarray(model["coef"], dtype=float).shape == (k, d)
                and len(model["intercept"]) == k and len(model["mean"]) == d and len(model["scale"]) == d
                and all(math.isfinite(float(v)) for v in np.ravel(model["coef"])))
    except (KeyError, TypeError, ValueError):
        return False


def predict(model: dict, feats: list[float]) -> dict[str, Any]:
    x = (np.asarray(feats, dtype=float) - np.asarray(model["mean"], dtype=float)) / np.asarray(model["scale"], dtype=float)
    z = np.asarray(model["coef"], dtype=float) @ x + np.asarray(model["intercept"], dtype=float)
    p = np.exp(z - z.max())
    p /= p.sum()
    i = int(np.argmax(p))
    return {"fault_class": model["classes"][i], "probability": round(float(p[i]), 3),
            "probabilities": {c: round(float(v), 3) for c, v in zip(model["classes"], p)}}


def combine(physics_hint: dict | None, feats: list[float] | None, model: dict | None) -> dict[str, Any]:
    """The hint shown to the technician: physics rule + fleet model + whether they agree."""
    ph = (physics_hint or {}).get("fault_class", "unknown")
    out: dict[str, Any] = {"physics": ph, "fleet": None, "confidence": "uncertain",
                           "fault_class": ph, "why": "physics rule only (no fleet model yet: it needs confirmed "
                                                     "cases of at least two fault classes)"}
    if feats is None:
        out["why"] = "physics rule only (no order features for this signal: needs >= 0.5 s at a known shaft speed)"
        return out
    if not valid(model):
        return out
    fp = predict(model, feats)
    tr = model.get("trained_on", {})
    cv = model.get("unseen_device_accuracy")
    out["fleet"] = fp | {"cases": tr.get("cases"), "devices": tr.get("devices"), "unseen_device_accuracy": cv}
    if ph != "unknown" and fp["fault_class"] == ph and fp["probability"] >= MIN_PROB:
        out.update(confidence="confident", fault_class=ph,
                   why=f"physics rule AND the fleet model (learned from {tr.get('cases')} confirmed cases on "
                       f"{tr.get('devices')} devices) both say {ph.replace('_', ' ')}")
    else:
        out.update(confidence="uncertain", fault_class=ph if ph != "unknown" else fp["fault_class"],
                   why=f"physics rule says {ph.replace('_', ' ')}, fleet model says "
                       f"{fp['fault_class'].replace('_', ' ')} (p={fp['probability']}): inspect before deciding")
    return out
