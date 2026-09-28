"""The external-sensor bridge: line parsing, chunking and the request it sends (hardware itself not tested)."""
import numpy as np
from fastapi.testclient import TestClient

from edge.api import create_app
from edge.sync_worker import SyncWorker
from tools import sensor_bridge as sb


def test_parser_accepts_common_formats_and_skips_junk():
    assert sb.parse_line("0.01, -0.02, 9.81\n", 3) == [0.01, -0.02, 9.81]
    assert sb.parse_line("1\t2\t3", 3) == [1.0, 2.0, 3.0]
    assert sb.parse_line("ax,ay,az", 3) is None and sb.parse_line("", 3) is None and sb.parse_line("1,2", 3) is None
    assert sb.parse_line("1e9,0,0", 3) is None


def test_chunks_and_body_shape():
    lines = ["ax,ay,az"] + [f"{i},{-i},{i * 2}" for i in range(10)]
    cs = list(sb.chunks(lines, 3, 4))
    assert [len(c) for c in cs] == [4, 4] and cs[1][0] == [4.0, -4.0, 8.0]
    assert sb.body(cs[0], 1600.0, 1500.0, "x")["axes"][0] == [0.0, -0.0, 0.0]
    assert "samples" in sb.body([[1.0], [2.0]], 100.0, None, "x")


def test_bridge_payload_is_accepted_by_a_device(make_device):
    d, _ = make_device("devA", "site1", profile="lowrate-accel", component="fan")
    c = TestClient(create_app(d, SyncWorker(d, None, None), "op-1234567"))
    h = {"X-Operator-Token": "op-1234567"}
    c.post("/api/baseline/capture", json={"windows": 10}, headers=h)
    fs, t = 400.0, np.arange(4000) / 400.0
    rows = [f"{0.1 * np.sin(2 * np.pi * 15 * x):.5f},{0.02 * np.cos(2 * np.pi * 15 * x):.5f},9.81" for x in t]
    for ch in sb.chunks(rows, 3, 1600):
        r = c.post("/api/ingest/signal", json=sb.body(ch, fs, 900.0, "bridge:test"), headers=h)
        assert r.status_code == 200, r.text
    assert d.capture_state() is None or d.capture_state()["have"] > 0
