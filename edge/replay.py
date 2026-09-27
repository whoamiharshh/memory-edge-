"""Replay real CWRU recordings through a device as if they were a live sensor stream.

Windows come from data/splits.build_dataset() - the SAME fingerprint code (edge/fingerprint.py) run over the
real .mat files, cached so the demo does not redo the DSP (live fingerprinting is ~ms per window and is
exercised by tests/unit/test_fingerprint.py and bench/latency.py).
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from data.fetch_data import CWRU

HEALTHY_BASELINE_FILES = (97, 98)      # normal, loads 0 and 1: used to fit the machine's baseline
HEALTHY_REPLAY_FILES = (99, 100)       # normal, loads 2 and 3: post-repair "healthy" stream (not in the baseline)


class Recordings:
    _data = None
    _lock = threading.Lock()

    @classmethod
    def data(cls) -> dict:
        with cls._lock:
            if cls._data is None:
                from data.splits import build_dataset
                cls._data = build_dataset()
            return cls._data

    @classmethod
    def windows(cls, fid: int) -> np.ndarray:
        d = cls.data()
        w = d["X"][d["fid"] == fid]
        if not len(w):
            raise KeyError(f"no windows for CWRU file {fid}")
        return w

    @classmethod
    def baseline(cls, fids=HEALTHY_BASELINE_FILES) -> np.ndarray:
        return np.concatenate([cls.windows(f) for f in fids])

    _raw: dict[int, tuple[np.ndarray, float]] = {}

    @classmethod
    def raw_window(cls, fid: int, i: int) -> tuple[np.ndarray, float, float] | None:
        """The i-th raw 4096-sample window of a CWRU file at 12 kHz (+ fs, rpm), for the physics diagnosis. None if
        the raw .mat file is not on this machine (the cached fingerprints still replay)."""
        from data.splits import NOMINAL_RPM, RAW
        from edge.fingerprint import FS, load_cwru, windows
        with cls._lock:
            if fid not in cls._raw:
                path = RAW / "cwru" / f"{fid}.mat"
                if not path.exists():
                    return None
                x, rpm = load_cwru(path)
                if not np.isfinite(rpm) or rpm <= 0:
                    rpm = NOMINAL_RPM[CWRU[fid][2]]
                cls._raw[fid] = (windows(x), rpm)
        ws, rpm = cls._raw[fid]
        return (ws[i], float(FS), float(rpm)) if 0 <= i < len(ws) else None

    @staticmethod
    def catalogue() -> list[dict]:
        return [{"fid": f, "fault_class": c, "size_mil": s, "load_hp": l,
                 "role": "baseline" if f in HEALTHY_BASELINE_FILES else ("healthy replay" if c == "normal" else "fault")}
                for f, (c, s, l) in sorted(CWRU.items())]


class ReplayRunner:
    """Background feeder: plays one file's windows into ingest(), `interval` seconds apart."""

    def __init__(self, ingest: Callable[[np.ndarray, str], dict],
                 on_abnormal: Callable[[dict, int, int], None] | None = None):
        """on_abnormal(result, fid, window_index): called for abnormal windows (e.g. to attach the physics
        diagnosis computed from the raw signal of that window)."""
        self.ingest = ingest
        self.on_abnormal = on_abnormal
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state = {"playing": False, "fid": None, "done": 0, "total": 0, "last": None}

    def play(self, fid: int, n: int | None = None, interval: float = 0.15, start: int = 0) -> dict:
        self.stop()
        w = Recordings.windows(fid)[start:]
        w = w[:n] if n else w
        self._stop.clear()
        self.state = {"playing": True, "fid": fid, "done": 0, "total": len(w), "last": None, "error": None}

        def run():
            try:
                for k, x in enumerate(w):
                    if self._stop.is_set():
                        break
                    self.state["last"] = self.ingest(x, f"cwru:{fid}")
                    if self.on_abnormal and self.state["last"].get("state") in ("new", "merge"):
                        self.on_abnormal(self.state["last"], fid, start + k)
                    self.state["done"] += 1
                    if interval:
                        time.sleep(interval)
            except Exception as e:          # visible in /api/stats; never leave "playing" stuck on True
                self.state["error"] = f"{type(e).__name__}: {e}"[:300]
                raise
            finally:
                self.state["playing"] = False

        self._thread = threading.Thread(target=run, name="replay", daemon=True)
        self._thread.start()
        return dict(self.state)

    def wait(self, timeout: float = 120) -> None:
        if self._thread:
            self._thread.join(timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.state["playing"] = False
