"""Regressions from the 28 Sep live run: a transient Windows file lock during Edge flush(), and a replay thread that
died without clearing `playing` (the scripted demo then waited forever)."""
import numpy as np
import pytest

from edge import replay as R
from edge.store_edge import _flush


class FakeShard:
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def flush(self):
        self.calls += 1
        if self.errors:
            raise Exception(self.errors.pop(0))


def test_flush_retries_transient_windows_lock():
    s = FakeShard(["Service runtime error: IO Error: Access is denied. (os error 5)"] * 2)
    _flush(s)
    assert s.calls == 3


def test_flush_raises_real_errors_immediately():
    s = FakeShard(["Service runtime error: disk full"])
    with pytest.raises(Exception, match="disk full"):
        _flush(s)
    assert s.calls == 1


def test_flush_gives_up_on_a_lock_that_does_not_clear():
    s = FakeShard(["(os error 32)"] * 50)
    with pytest.raises(Exception):
        _flush(s)
    assert s.calls == 6


def test_replay_that_crashes_stops_playing_and_reports(monkeypatch):
    monkeypatch.setattr(R.Recordings, "windows", classmethod(lambda cls, fid: np.zeros((5, 3))))
    n = {"i": 0}

    def ingest(x, src):
        n["i"] += 1
        if n["i"] == 3:
            raise RuntimeError("shard flush failed")
        return {"state": "normal"}

    r = R.ReplayRunner(ingest)
    r.play(99, interval=0.0)
    r.wait(5)
    assert r.state["playing"] is False and r.state["done"] == 2
    assert "shard flush failed" in r.state["error"]
