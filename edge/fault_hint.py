"""Physics hint for the bearing fault class: which defect frequency dominates the envelope spectrum
(BPFO -> outer race, BPFI -> inner race, BSF -> ball). No retrieval, no training.

It is a HINT shown with its measured accuracy (bench/results/k2_vib_retrieval.json, bearing-level split,
unseen bearings). The technician confirms the class; fleet grouping uses the confirmed class (K2 decision).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from edge.fingerprint import DEFECT_ORDERS, PHYSICS_SLICE

K2_RESULTS = pathlib.Path(__file__).resolve().parents[1] / "bench" / "results" / "k2_vib_retrieval.json"
RULE_KEY = "bearing_level (honest) | RULE dominant-defect (no retrieval)"
ORDER_TO_CLASS = {"bpfo": "outer_race", "bpfi": "inner_race", "bsf": "ball"}


def _measured() -> dict:
    try:
        r = json.loads(K2_RESULTS.read_text())[RULE_KEY]
        return {"overall": r["fault_acc"], "per_class": r["per_class"], "n": r["n"],
                "source": "bench/results/k2_vib_retrieval.json (bearing-level split)"}
    except (OSError, KeyError, ValueError):
        return {"overall": None, "per_class": {}, "n": 0, "source": "benchmark not run"}


MEASURED = _measured()


def suggest(raw_features: np.ndarray) -> dict:
    """raw_features: un-normalised fingerprint of one window (or the mean of several)."""
    orders = np.asarray(raw_features)[PHYSICS_SLICE]
    names = list(DEFECT_ORDERS)
    if not np.any(orders):
        return {"fault_class": "unknown", "why": "no shaft speed available", "measured_accuracy": None}
    defects = names[1:]
    i = int(np.argmax(orders[1:]))
    cls = ORDER_TO_CLASS[defects[i]]
    return {"fault_class": cls,
            "why": f"{defects[i].upper()} envelope energy dominates ({orders[1:][i]:.2f} log-ratio)",
            "measured_accuracy": MEASURED["per_class"].get(cls),
            "measured_overall": MEASURED["overall"], "measured_source": MEASURED["source"]}
