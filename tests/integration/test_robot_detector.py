"""The robot's own learned detector, end to end on REAL UCI robot traces (lp2: collisions the unsupervised gate
misses about half the time). The technician confirms the failures the gate did catch; the device learns from them and
then also catches failures the gate alone misses."""
import numpy as np
import pytest

from data.fetch_uci_robot import OUT as UCI_RAW, load
from edge.local_detector import MIN_FAULT

pytestmark = pytest.mark.skipif(not UCI_RAW.exists(), reason="UCI robot data not downloaded (data/fetch_uci_robot.py)")


def test_learned_detector_adds_the_failures_the_gate_misses(make_device):
    inst = load("lp2")
    rng = np.random.default_rng(3)
    normals = [np.asarray(r) for l, r in inst if l == "normal"]
    fails = [(l, np.asarray(r)) for l, r in inst if l != "normal"]
    rng.shuffle(normals)
    order = rng.permutation(len(fails))
    fails = [fails[i] for i in order]
    d, _ = make_device("robot1", "cell1", profile="force-torque")
    d.fit_baseline(np.stack([d.profile.features(x) for x in normals[:14]]))
    # phase 1: failures happen; the technician confirms those the gate flagged and TEACHES those it missed (the
    # operator saw the collision)
    taught = 0
    for label, x in fails[:14]:
        r = d.ingest_window(d.profile.features(x))
        if r["state"] != "normal":
            d.set_fault_class(r["episode_id"], "collision")
        else:
            assert d.teach_fault(d.profile.features(x), "collision")["action"] == "KEEP_LOCAL"
            taught += 1
        d.ingest_window(d.profile.features(normals[0]))          # healthy motion between failures
    assert taught > 0                                             # the gate alone missed some
    m = d.train_local_detector()
    assert m["trained_on"]["confirmed_fault"] >= MIN_FAULT and 0 <= m["cross_validated"]["false_alarm_rate"] <= 1
    # phase 2: new traces
    gate_only = learned = 0
    for label, x in fails[14:]:
        r = d.ingest_window(d.profile.features(x))
        learned += r["state"] != "normal"
        gate_only += r["state"] != "normal" and "learned_detector" not in r
        d.ingest_window(d.profile.features(normals[1]))
    assert learned > gate_only                                   # the learned detector added real detections
    assert learned >= 0.8 * len(fails[14:])
    healthy = [d.ingest_window(d.profile.features(x))["state"] for x in normals[14:]]
    assert sum(s != "normal" for s in healthy) <= 2              # few false alarms on unseen healthy motion
    assert d.stats()["local_detector"]["trained_on"]["confirmed_fault"] == m["trained_on"]["confirmed_fault"]


def test_detector_refuses_to_train_without_enough_confirmed_faults(make_device):
    d, _ = make_device("robot2", "cell2", profile="force-torque")
    inst = load("lp2")
    d.fit_baseline(np.stack([d.profile.features(np.asarray(r)) for l, r in inst if l == "normal"][:12]))
    with pytest.raises(ValueError, match="confirmed-fault"):
        d.train_local_detector()
