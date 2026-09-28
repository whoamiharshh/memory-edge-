"""Phone-microphone path: the acoustic profile and POST /api/ingest/audio. The audio here is SYNTHETIC (a test
fixture: noise + motor tone, and the same plus periodic impacts at the outer-race defect rate); the real-data
measurement is bench/acoustic_uottawa.py on 20 naturally worn bearings."""
import base64

import numpy as np
from fastapi.testclient import TestClient

from edge import physics as P
from edge import profiles
from edge.api import create_app
from edge.sync_worker import SyncWorker

FS, SHAFT = 48000, 30.0
GEO = P.BEARINGS["6203-UO"]
H = {"X-Operator-Token": "op-1234567"}


def sound(seconds=1.0, fault=False, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * seconds)) / FS
    x = 0.02 * rng.normal(size=len(t)) + 0.05 * np.sin(2 * np.pi * SHAFT * t) + 0.03 * np.sin(2 * np.pi * 120 * t)
    if fault:
        fd = GEO.orders()["bpfo"] * SHAFT
        ring = np.sin(2 * np.pi * 5000 * t) * (np.sin(2 * np.pi * fd * t) > 0.97)
        x = x + 0.25 * ring
    return x


def b64(x):
    return base64.b64encode((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()).decode()


def test_acoustic_profile_shape_and_no_iso_zone():
    p = profiles.make("acoustic", bearing="6203-UO")
    ws = p.windows(sound(2.0), FS)
    assert p.analysis_fs == 16000 and len(ws) >= 2 and len(ws[0]) == 8192
    f = p.features(ws[0], p.analysis_fs, SHAFT * 60)
    assert f.shape == (27,) and np.all(np.isfinite(f))
    assert p.diagnose(sound(), FS, SHAFT * 60)["severity"]["zone"] is None


def test_audio_route_captures_detects_and_hints(make_device):
    d, _ = make_device("devA", "site1", profile="acoustic", profile_params={"bearing": "6203-UO"}, component="motor")
    c = TestClient(create_app(d, SyncWorker(d, None, None), "op-1234567"))
    assert c.post("/api/baseline/capture", json={"windows": 20}, headers=H).status_code == 200
    for k in range(10):
        r = c.post("/api/ingest/audio", json={"pcm16_b64": b64(sound(seed=k)), "fs": FS, "rpm": SHAFT * 60}, headers=H)
        assert r.status_code == 200, r.text
    assert d.gate is not None
    healthy = [c.post("/api/ingest/audio", json={"pcm16_b64": b64(sound(seed=100 + k)), "fs": FS, "rpm": SHAFT * 60},
                      headers=H).json() for k in range(3)]
    assert all(h["states"] == {"normal": h["windows"]} for h in healthy)
    r = c.post("/api/ingest/audio", json={"pcm16_b64": b64(sound(fault=True, seed=7)), "fs": FS, "rpm": SHAFT * 60},
               headers=H).json()
    assert r["last"]["state"] in ("new", "merge")
    ep = d.episodes()[0]
    assert ep["fault_hint"]["fault_class"] == "outer_race"
    assert len(ep["order_features"]) == 20 and ep["fleet_hint"]["physics"] == "outer_race"


def test_audio_route_refuses_bad_input(make_device):
    d, _ = make_device("devA", "site1", profile="acoustic", profile_params={"bearing": "6203-UO"})
    c = TestClient(create_app(d, SyncWorker(d, None, None), "op-1234567"))
    assert c.post("/api/ingest/audio", json={"pcm16_b64": "***not base64***", "fs": FS}, headers=H).status_code == 422
    assert c.post("/api/ingest/audio", json={"pcm16_b64": b64(np.zeros(FS)), "fs": FS}, headers=H).status_code == 422
    assert c.post("/api/ingest/audio", json={"pcm16_b64": b64(sound(0.1)), "fs": FS}, headers=H).status_code == 422
    assert c.post("/api/ingest/audio", json={"pcm16_b64": b64(sound()), "fs": FS}).status_code == 401


def test_audio_route_needs_the_acoustic_profile(make_device):
    d, _ = make_device("devA", "site1")                          # bearing-12k
    c = TestClient(create_app(d, SyncWorker(d, None, None), "op-1234567"))
    r = c.post("/api/ingest/audio", json={"pcm16_b64": b64(sound()), "fs": FS}, headers=H)
    assert r.status_code == 409 and "--profile acoustic" in r.json()["detail"]


def test_sensor_page_offers_the_microphone_with_processing_off():
    import pathlib
    ui = pathlib.Path(__file__).resolve().parents[2] / "edge" / "ui"
    js = (ui / "sensor.js").read_text(encoding="utf-8")
    assert "echoCancellation: false, noiseSuppression: false, autoGainControl: false" in js
    assert "/api/ingest/audio" in js and ".innerHTML" not in js
    assert "/static/mic-worklet.js" in (ui / "sw.js").read_text(encoding="utf-8")
