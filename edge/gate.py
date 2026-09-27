"""Novelty gate: one nearest-neighbour query per signal window, answering "have I seen this state before?"

  near this machine's HEALTHY baseline (distance <= tau_normal)      -> "normal"   (nothing is written)
  near an exemplar of an ACTIVE episode (distance <= tau_merge)      -> "merge"    (occurrences++)
  otherwise                                                          -> "new"      (a new episode opens)
A "new" state that is close to an exemplar of a CLOSED episode is reported as a recurrence of that episode
(same-machine recurrence memory: K2 measured ~0.997 same-bearing retrieval precision).

Thresholds are prototype parameters calibrated from this machine's own healthy data (calibrate()):
  tau_normal = NORMAL_FACTOR x q99 of healthy-to-healthy nearest-neighbour distances (split-half)
  tau_merge  = MERGE_FACTOR x tau_normal
Measured on CWRU (bench/gate_verifier.py): unseen-load healthy windows stay under ~9.3 z-units of the
baseline, while every fault window is >= ~29.8, so the margin is wide on this dataset.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from edge.store_edge import EdgeStore

NORMAL_FACTOR = 2.0
MERGE_FACTOR = 3.0
RECURRENCE_FACTOR = 1.0     # "seen before" if within tau_merge of a closed episode's exemplar


@dataclass(frozen=True)
class GateConfig:
    tau_normal: float
    tau_merge: float
    calib_q99: float
    n_calib: int

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(healthy_z: np.ndarray, seed: int = 0) -> GateConfig:
    """Split-half calibration: distance of each held-out healthy window to its nearest neighbour in the
    other half. Needs >= 10 windows."""
    if len(healthy_z) < 10:
        raise ValueError("need at least 10 healthy windows to calibrate the gate")
    idx = np.random.default_rng(seed).permutation(len(healthy_z))
    a, b = healthy_z[idx[: len(idx) // 2]], healthy_z[idx[len(idx) // 2:]]
    d = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(axis=1)
    q99 = float(np.percentile(d, 99))
    tau_n = NORMAL_FACTOR * q99
    return GateConfig(tau_normal=tau_n, tau_merge=MERGE_FACTOR * tau_n, calib_q99=q99, n_calib=len(healthy_z))


@dataclass
class GateResult:
    state: str                       # normal | merge | new
    d_baseline: float
    d_episode: float | None
    episode_id: str | None           # merge target
    recurrence_of: str | None        # closed episode this new state resembles
    latency_ms: float


class NoveltyGate:
    def __init__(self, store: EdgeStore, machine_id: str, cfg: GateConfig):
        self.store, self.machine_id, self.cfg = store, machine_id, cfg

    def distance_to_baseline(self, z: np.ndarray) -> float:
        hits = self.store.nearest(z, filter={"type": "baseline", "machine_id": self.machine_id}, limit=1)
        if not hits:
            raise RuntimeError("no baseline stored for this machine; fit the baseline first")
        return hits[0].score

    def classify(self, z: np.ndarray) -> GateResult:
        t = time.perf_counter()
        d_base = self.distance_to_baseline(z)
        if d_base <= self.cfg.tau_normal:
            return GateResult("normal", d_base, None, None, None, (time.perf_counter() - t) * 1000)
        hits = self.store.nearest(z, filter={"type": "exemplar", "machine_id": self.machine_id}, limit=5)
        active = [h for h in hits if h.payload.get("episode_active")]
        if active and active[0].score <= self.cfg.tau_merge:
            h = active[0]
            return GateResult("merge", d_base, h.score, h.payload["episode_id"], None,
                              (time.perf_counter() - t) * 1000)
        closed = [h for h in hits if not h.payload.get("episode_active")]
        rec = closed[0].payload["episode_id"] if closed and closed[0].score <= self.cfg.tau_merge * RECURRENCE_FACTOR else None
        d_ep = hits[0].score if hits else None
        return GateResult("new", d_base, d_ep, None, rec, (time.perf_counter() - t) * 1000)
