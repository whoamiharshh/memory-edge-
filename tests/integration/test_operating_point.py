"""Automatic operating-point check: episodes at a speed/load the healthy baseline never saw are flagged, with a physics
suggestion; 'Not a fault: normal operation' teaches that operating point; taught points are never flagged."""
import numpy as np

from edge.device import Device, DeviceConfig
from edge.replay import Recordings
from shared.embed import HashEmbedder
from tests.conftest import needs_cwru

pytestmark = needs_cwru


def dev_with_baseline(tmp_path):
    d = Device(DeviceConfig(device_id="d", site_id="s", machine_id="m", root=tmp_path / "d"), HashEmbedder())
    base = Recordings.baseline()
    d.fit_baseline(base, [{"speed_hz": 29.95}] * (len(base) // 2) + [{"speed_hz": 29.5}] * (len(base) - len(base) // 2))
    return d


def test_taught_ranges_survive_a_restart(tmp_path):
    d = dev_with_baseline(tmp_path)
    assert d.taught == {"speed_hz": [29.5, 29.95]}
    d.close()
    d2 = Device(DeviceConfig(device_id="d", site_id="s", machine_id="m", root=tmp_path / "d"), HashEmbedder())
    try:
        assert d2.taught == {"speed_hz": [29.5, 29.95]}
    finally:
        d2.close()


def test_fault_at_a_taught_speed_is_not_flagged(tmp_path):
    d = dev_with_baseline(tmp_path)
    try:
        for w in Recordings.windows(105)[:5]:
            d.ingest_window(w, op={"speed_hz": 29.9})
        assert "untaught_operating_point" not in d.episodes()[0]
    finally:
        d.close()


def test_untaught_speed_is_flagged_with_a_physics_suggestion_and_can_be_taught(tmp_path):
    d = dev_with_baseline(tmp_path)
    try:
        for w in Recordings.windows(105)[:12]:                     # inner-race fault, at a speed never taught
            d.ingest_window(w, op={"speed_hz": 27.0})
        u = d.episodes()[0]["untaught_operating_point"]
        assert u["speed_hz"]["value"] == 27.0 and u["speed_hz"]["taught"] == [29.5, 29.95]
        assert len(u["sig_scores"]) == 10                          # median over the first 10 windows
        assert u["suggestion"].startswith("a fault signature is present")   # CWRU 105 has a clear BPFI signature
        eid = d.episodes()[0]["episode_id"]
        d.mark_normal(eid)                                         # (for the test: the technician teaches 27 Hz)
        assert d.taught["speed_hz"] == [27.0, 29.95]
        assert d.untaught({"speed_hz": 27.0}) == {} and d.untaught({"speed_hz": 20.0})
    finally:
        d.close()


def test_no_operating_data_means_no_claim(tmp_path):
    d = Device(DeviceConfig(device_id="d", site_id="s", machine_id="m", root=tmp_path / "d"), HashEmbedder())
    try:
        d.fit_baseline(Recordings.baseline())
        for w in Recordings.windows(105)[:3]:
            d.ingest_window(w, op={"speed_hz": 5.0})
        assert d.taught == {} and "untaught_operating_point" not in d.episodes()[0]
    finally:
        d.close()
