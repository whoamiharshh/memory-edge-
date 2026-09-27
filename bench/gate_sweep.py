"""Novelty-gate threshold sweep (docs/RESEARCH.md Part O: "tau swept, curve published").

The gate has two prototype parameters (edge/gate.py): tau_normal = NORMAL_FACTOR x q99 of healthy nearest-neighbour
distances, and tau_merge = MERGE_FACTOR x tau_normal. Shipped values: 2.0 and 3.0. This sweeps each one with the
other held at its shipped value, on real CWRU fingerprints through the real Device + Qdrant Edge.

  normal-factor sweep (distance to baseline only, so one pass per window is enough):
      false-alarm rate  unseen-load healthy files 99+100: windows NOT classified normal
      detection rate    first 20 windows of all 36 fault recordings: windows classified abnormal
  merge-factor sweep (needs the full gate + episodes, fresh device per run):
      episodes / recording   first 20 windows of each of the 36 fault recordings (ideal 1)
      separated pairs        fault X -> healthy -> DIFFERENT fault Y, 12 pairs (ideal: 2 episodes each)
Honest limit: one healthy bearing exists in CWRU, so the "unseen" healthy data is other loads of the same bearing.
Run: .venv\\Scripts\\python.exe -m bench.gate_sweep
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import numpy as np

from bench.gate_verifier import distinct_faults_after_healthy_gap
from data.fetch_data import CWRU
from edge import gate as G
from edge.device import Device, DeviceConfig
from edge.replay import HEALTHY_REPLAY_FILES, Recordings
from shared.embed import HashEmbedder

OUT = pathlib.Path(__file__).resolve().parent / "results"
NORMAL_FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
MERGE_FACTORS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
FAULTS = sorted(f for f, v in CWRU.items() if v[0] != "normal")


def _device(base) -> tuple[Device, pathlib.Path]:
    root = pathlib.Path(tempfile.mkdtemp(prefix="sweep_"))
    dev = Device(DeviceConfig("bench", "s", "m", root), HashEmbedder())
    dev.fit_baseline(base)
    return dev, root


def normal_sweep(base) -> dict:
    dev, root = _device(base)
    try:
        q99 = dev.gate.cfg.calib_q99
        d = lambda w: dev.gate.distance_to_baseline(dev.baseline.z(w))
        healthy = [d(w) for f in HEALTHY_REPLAY_FILES for w in Recordings.windows(f)]
        fault = [d(w) for f in FAULTS for w in Recordings.windows(f)[:20]]
    finally:
        dev.close()
        shutil.rmtree(root, ignore_errors=True)
    rows = []
    for k in NORMAL_FACTORS:
        tau = k * q99
        rows.append({"normal_factor": k, "tau_normal": round(tau, 3),
                     "false_alarm_rate": round(float(np.mean(np.array(healthy) > tau)), 4),
                     "detection_rate": round(float(np.mean(np.array(fault) > tau)), 4)})
    return {"calib_q99": round(q99, 3), "healthy_windows": len(healthy), "fault_windows": len(fault),
            "healthy_distance_max": round(max(healthy), 3), "fault_distance_min": round(min(fault), 3), "rows": rows}


def merge_sweep(base) -> list[dict]:
    """Per recording, one fresh device runs three probes:
      A  first 20 windows                                   -> episodes opened (ideal 1)
      B  healthy gap (10), then the SAME fault again (w 20-34) while its episode is open -> still 1 episode
         (an intermittent fault must not be split into duplicates)
      C  action + healthy verification closes it, then the fault returns (w 40-42) -> flagged recurrence_of"""
    rows = []
    orig = G.MERGE_FACTOR
    healthy = Recordings.windows(HEALTHY_REPLAY_FILES[0])
    for mf in MERGE_FACTORS:
        G.MERGE_FACTOR = mf                               # read by calibrate() inside Device.fit_baseline
        try:
            eps, same_ok, rec_ok, rec_n = [], 0, 0, 0
            for fid in FAULTS:
                dev, root = _device(base)
                try:
                    W = Recordings.windows(fid)
                    for w in W[:20]:
                        dev.ingest_window(w)
                    eps.append(len(dev.episodes()))
                    for w in healthy[:10]:
                        dev.ingest_window(w)
                    for w in W[20:35]:
                        dev.ingest_window(w)
                    same_ok += len(dev.episodes()) == 1
                    first = dev.episodes()[-1]["episode_id"]
                    dev.record_action(first, "replace_bearing")
                    for w in healthy[10:10 + 20]:
                        dev.ingest_window(w)
                    dev.confirm_outcome(first, "worked")
                    if len(W) > 42 and dev.episode(first)["status"] == "closed":
                        rec_n += 1
                        out = [dev.ingest_window(w) for w in W[40:43]]
                        rec_ok += any(o.get("recurrence_of") == first for o in out)
                finally:
                    dev.close()
                    shutil.rmtree(root, ignore_errors=True)
            sep = distinct_faults_after_healthy_gap(base)
        finally:
            G.MERGE_FACTOR = orig
        rows.append({"merge_factor": mf, "episodes_per_recording_mean": round(float(np.mean(eps)), 3),
                     "recordings_with_exactly_1_episode": f"{sum(e == 1 for e in eps)}/{len(eps)}",
                     "same_fault_after_gap_stays_1_episode": f"{same_ok}/{len(FAULTS)}",
                     "recurrence_after_close_detected": f"{rec_ok}/{rec_n}",
                     "distinct_faults_separated": sep["separated"]})
        print(rows[-1], flush=True)
    return rows


def main() -> dict:
    base = Recordings.baseline()
    out = {"method": __doc__.strip().splitlines()[0], "shipped": {"normal_factor": G.NORMAL_FACTOR,
                                                                  "merge_factor": G.MERGE_FACTOR}}
    out["normal_factor_sweep"] = normal_sweep(base)
    for r in out["normal_factor_sweep"]["rows"]:
        print(r, flush=True)
    out["merge_factor_sweep"] = merge_sweep(base)
    OUT.mkdir(exist_ok=True)
    (OUT / "gate_sweep.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
