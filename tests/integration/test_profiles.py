"""One memory engine, many kinds of edge device (edge/profiles.py). Each test runs the REAL Device (Qdrant Edge, gate,
verifier, policy) with a live-sensor style input, through the full loop: baseline -> fault -> episode + physics hint
-> action -> fix verified by the signal -> SHARE."""
import numpy as np
import pytest

from edge import physics as P
from edge.device import Device, DeviceConfig
from shared.embed import HashEmbedder

RNG = np.random.default_rng(0)


def device(tmp_path, profile, **params):
    return Device(DeviceConfig(device_id="d1", site_id="s1", machine_id="m1", root=tmp_path / profile,
                               profile=profile, profile_params=params), HashEmbedder())


def close_the_loop(dev, fault_class, action, healthy_signal):
    """Technician confirms + acts; the healthy signal then verifies the fix; returns the episode."""
    ep = dev.episodes()[0]["episode_id"]
    dev.set_fault_class(ep, fault_class)
    dev.record_action(ep, action)
    healthy_signal()
    dev.confirm_outcome(ep, "worked")
    return dev.episode(ep)


# ---- phone on a desk fan: coin taped to a blade = imbalance ---------------------------------------------------
FS_PHONE, SHAFT = 60.0, 17.0          # 60 Hz DeviceMotion, fan at 1020 rpm


def phone(seconds, imbalance=0.0, misalign=0.0):
    t = np.arange(int(FS_PHONE * seconds)) / FS_PHONE
    base = 0.05 * np.sin(2 * np.pi * SHAFT * t)                        # a healthy fan still has a little 1x
    y = base + imbalance * np.sin(2 * np.pi * SHAFT * t + 0.3) + misalign * np.sin(2 * np.pi * 2 * SHAFT * t)
    noise = lambda: 0.03 * RNG.normal(size=len(t))
    return np.stack([0.2 + y + noise(), 0.1 + 0.4 * y + noise(), 9.81 + noise()], axis=1)   # m/s^2 incl. gravity


def test_phone_accelerometer_fan_imbalance_full_loop(tmp_path):
    dev = device(tmp_path, "lowrate-accel", shaft_hz=SHAFT)
    try:
        dev.start_baseline_capture(20)
        out = dev.ingest_signal(phone(95), FS_PHONE)                   # 95 s of known-good running
        assert any(r["state"] == "baseline" for r in out) and dev.gate is not None
        assert all(r["state"] == "normal" for r in dev.ingest_signal(phone(30), FS_PHONE))
        states = [r["state"] for r in dev.ingest_signal(phone(40, imbalance=0.6), FS_PHONE)]
        assert states.count("new") == 1 and len(dev.episodes()) == 1
        ep = dev.episodes()[0]
        assert ep["fault_hint"]["fault_class"] == "imbalance"
        assert ep["physics"]["rotating"]["fault_class"] == "imbalance" and ep["physics"]["shaft_hz"] == pytest.approx(SHAFT)
        e = close_the_loop(dev, "imbalance", "rebalance", lambda: dev.ingest_signal(phone(100), FS_PHONE))
        assert e["verify"]["verdict"] == "symptom_resolved" and e["decision"]["action"] == "SHARE"
        assert e["decision"]["event_id"] and dev.outbox.counts() == {"queued": 1}
    finally:
        dev.close()


def test_phone_rule_is_honest_about_nyquist(tmp_path):
    """At 60 Hz the phone sees up to 30 Hz. A fan at 17 Hz has its 2x at 34 Hz: misalignment is NOT assessable and
    the hint must say so instead of calling it imbalance by silence. A slow machine (9 Hz) keeps 2x = 18 Hz visible."""
    from edge import profiles
    p = profiles.make("lowrate-accel", shaft_hz=SHAFT)
    h = p.hint(p.features(phone(8, misalign=0.6), FS_PHONE))
    assert "NOT assessable" in h["why"] and h["not_assessable"] == ["2x", "3x"]
    slow = profiles.make("lowrate-accel", shaft_hz=9.0)
    t = np.arange(int(FS_PHONE * 8)) / FS_PHONE
    x = 0.5 * np.sin(2 * np.pi * 9.0 * t) + 0.45 * np.sin(2 * np.pi * 18.0 * t) + 0.02 * RNG.normal(size=len(t))
    assert slow.hint(slow.features(x, FS_PHONE))["fault_class"] == "misalignment"


# ---- robot wrist force/torque: a collision ---------------------------------------------------------------------
def ft(n, collision=False):
    x = np.tile([2.0, -1.0, 12.0, 0.1, 0.05, 0.02], (n, 1)) + 0.3 * RNG.normal(size=(n, 6))
    if collision:
        x[n // 2:, 0] += 60.0                                         # sudden lateral force
        x[n // 2:, 4] += 4.0
    return x


def test_robot_force_torque_collision_opens_one_episode(tmp_path):
    dev = device(tmp_path, "force-torque")
    try:
        dev.start_baseline_capture(40)
        dev.ingest_signal(ft(15 * 45), 1.0)
        assert dev.gate is not None
        normal = [r["state"] for r in dev.ingest_signal(ft(15 * 10), 1.0)]
        assert normal.count("normal") >= len(normal) - 1                  # at most one borderline window
        bad = [r["state"] for r in dev.ingest_signal(ft(15, collision=True), 1.0)]
        assert "new" in bad
        assert dev.episodes()[0]["component"] == "robot_gripper"
    finally:
        dev.close()


# ---- kiosk error codes: a printer jam that the fix makes go away ----------------------------------------------
def buckets(n, jam=False):
    return [{"codes": ({"PRN-JAM": 3, "PRN-RETRY": 5} if jam else {"HEARTBEAT": 1}), "severity": 3 if jam else 0}
            for _ in range(n)]


def test_kiosk_event_codes_fix_verified_when_the_codes_stay_away(tmp_path):
    dev = device(tmp_path, "events")
    try:
        dev.start_baseline_capture(12)
        for b in buckets(12):
            dev.ingest_signal(b, 1.0)
        assert dev.gate is not None
        states = [dev.ingest_signal(b, 1.0)[0]["state"] for b in buckets(3, jam=True)]
        assert states == ["new", "merge", "merge"]
        e = close_the_loop(dev, "jam", "clear_jam", lambda: [dev.ingest_signal(b, 1.0) for b in buckets(20)])
        assert e["verify"]["verdict"] == "symptom_resolved" and e["decision"]["action"] == "SHARE"
        assert e["component"] == "kiosk"
    finally:
        dev.close()


# ---- rotating machine, any bearing geometry: inner-race defect on a 6206 ----------------------------------------
def bearing_signal(seconds, fs, shaft, geometry, defect=None):
    t = np.arange(int(fs * seconds)) / fs
    x = 0.02 * RNG.normal(size=len(t)) + 0.01 * np.sin(2 * np.pi * shaft * t)
    if defect:
        fd = geometry.orders()[defect] * shaft
        impacts = (np.sin(2 * np.pi * fd * t) > 0.97).astype(float)
        x = x + 0.4 * impacts * np.sin(2 * np.pi * 3800 * t)           # defect impacts ring a resonance
    return x


def test_rotating_hf_other_bearing_geometry_inner_race(tmp_path):
    geo, fs, shaft = P.BEARINGS["6206"], 51200.0, 29.0
    dev = device(tmp_path, "rotating-hf", bearing="6206", shaft_hz=shaft)
    try:
        dev.start_baseline_capture(20)
        dev.ingest_signal(bearing_signal(4.0, fs, shaft, geo), fs)
        assert dev.gate is not None
        states = [r["state"] for r in dev.ingest_signal(bearing_signal(1.5, fs, shaft, geo, "bpfi"), fs)]
        assert states[0] == "new"
        ep = dev.episodes()[0]
        assert ep["fault_hint"]["fault_class"] == "inner_race"
        assert ep["physics"]["bearing"]["fault_class"] == "inner_race"
        assert ep["physics"]["defect_frequencies_hz"]["bpfi"] == pytest.approx(geo.orders()["bpfi"] * shaft, rel=1e-3)
    finally:
        dev.close()


def test_profiles_share_one_fingerprint_slot():
    from edge import profiles
    for name, x, fs in (("rotating-hf", RNG.normal(size=8192), 12800.0), ("lowrate-accel", RNG.normal(size=(256, 3)), 60.0),
                        ("force-torque", RNG.normal(size=(15, 6)), 1.0), ("events", {"codes": {"E1": 2}}, 1.0),
                        ("bearing-12k", RNG.normal(size=4096), 12000.0)):
        p = profiles.make(name)
        f = p.features(x if name == "events" else (p.windows(x, fs)[0] if name not in ("events",) else x),
                       p.analysis_fs or fs)
        assert f.shape == (profiles.DIM,) and np.all(np.isfinite(f)), name
