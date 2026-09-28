"""Fleet-learned fault hint: trained in the cloud from confirmed cases, applied on the device as plain numbers.
The feature vectors here are SYNTHETIC (test fixtures only; the shipped model is trained on real fleet evidence and
measured on real data in bench/fault_hint.py)."""
import json

import numpy as np

from cloud.hint_model import train
from edge import fleet_hint
from edge import physics as P

D = len(P.ORDER_FEATURE_NAMES)


def _events(n_dev=4, per=3, seed=0):
    rng = np.random.default_rng(seed)
    evs = []
    for d in range(n_dev):
        for cls, col in (("inner_race", 3), ("outer_race", 0)):
            for k in range(per):
                f = rng.normal(0, 0.2, D)
                f[col] += 2.0
                evs.append({"event_id": f"{d}-{cls}-{k}", "device_id": f"dev{d}", "fault_class": cls,
                            "fault_class_source": "technician", "order_features": f.tolist(), "of_version": "of-v1",
                            "status": "active"})
    return evs


def test_order_features_are_20_finite_numbers():
    fs, shaft = 12000.0, 29.95
    t = np.arange(int(fs)) / fs
    x = np.sin(2 * np.pi * 3500 * t) * (1 + (np.sin(2 * np.pi * 5.4152 * shaft * t) > 0.95)) + 0.05 * np.random.default_rng(0).normal(size=len(t))
    f = P.order_features(x, fs, shaft, P.BEARINGS["SKF6205-CWRU"])
    assert len(f) == D == 20 and all(np.isfinite(f))
    assert f[3] > f[0]                               # BPFI h1 above BPFO h1 for an inner-race modulation


def test_training_needs_two_classes_and_two_devices():
    assert train(_events(n_dev=1))["model"] is None
    one_class = [e for e in _events() if e["fault_class"] == "inner_race"]
    assert "2 bearing fault classes" in train(one_class)["why"]


def test_model_is_plain_json_measured_on_unseen_devices_and_predicts():
    out = train(_events())
    m = out["model"]
    assert m is not None and fleet_hint.valid(json.loads(json.dumps(m)))
    assert m["unseen_device_accuracy"] == 1.0 and m["unseen_device_cases"] == 24
    f = np.zeros(D); f[3] = 2.0
    assert fleet_hint.predict(m, f.tolist())["fault_class"] == "inner_race"


def test_retracted_and_unconfirmed_evidence_is_not_learned():
    evs = _events()
    for e in evs[:6]:
        e["status"] = "retracted"
    for e in evs[6:9]:
        e["fault_class_source"] = "physics_hint"
    assert train(evs)["model"]["trained_on"]["cases"] == len(evs) - 9


def test_combine_is_confident_only_when_physics_and_fleet_agree():
    m = train(_events())["model"]
    f = np.zeros(D); f[3] = 2.0
    agree = fleet_hint.combine({"fault_class": "inner_race"}, f.tolist(), m)
    assert agree["confidence"] == "confident" and agree["fault_class"] == "inner_race"
    disagree = fleet_hint.combine({"fault_class": "ball"}, f.tolist(), m)
    assert disagree["confidence"] == "uncertain" and "inspect" in disagree["why"]
    assert fleet_hint.combine({"fault_class": "ball"}, f.tolist(), None)["confidence"] == "uncertain"
    assert fleet_hint.combine({"fault_class": "ball"}, None, m)["fleet"] is None


def test_tampered_model_is_ignored():
    m = train(_events())["model"]
    assert not fleet_hint.valid(m | {"coef": [[1.0]]})
    assert not fleet_hint.valid(m | {"of_version": "of-v9"})
    assert not fleet_hint.valid({"classes": ["a"]})
