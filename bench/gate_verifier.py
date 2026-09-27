"""K3 kill test + novelty-gate benchmark, through the real Device (Qdrant Edge shard, real CWRU fingerprints).

Setup per fault recording (36 files): fresh device, baseline fitted on healthy files 97+98 (loads 0,1).
  Gate false alarms : unseen healthy files 99+100 (loads 2,3) -> fraction NOT classified "normal"
  Gate detection    : fault windows classified abnormal; episodes opened per recording (ideal = 1)
  K3 "fixed"        : first 20 fault windows -> action -> healthy replay (99/100 alternating) -> verdict?
  K3 "still faulty" : first 20 fault windows -> action -> the SAME fault continues -> verdict?
  false promotion   : a still-faulty replay verified as "symptom_resolved" (the dangerous error)
Run: .venv\\Scripts\\python.exe -m bench.gate_verifier
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import time

import numpy as np

from data.fetch_data import CWRU
from edge.device import Device, DeviceConfig
from edge.replay import HEALTHY_REPLAY_FILES, Recordings
from edge.verifier import REQUIRED_WINDOWS
from shared.embed import HashEmbedder

OUT = pathlib.Path(__file__).resolve().parent / "results"
FAULT_WINDOWS = 20


def distinct_faults_after_healthy_gap(base: np.ndarray) -> dict:
    """Fault X (15 windows) -> healthy (10) -> fault Y of a DIFFERENT class, same size & load (15).
    The gate should open a second episode (not merge Y into X). 12 pairs = every (size, load)."""
    order = {"inner_race": "ball", "ball": "outer_race", "outer_race": "inner_race"}
    by = {(c, s, l): f for f, (c, s, l) in CWRU.items()}
    ok, pairs = 0, []
    for (c, s, l), fx in sorted(by.items()):
        if c != "inner_race" and not (c == "ball" and s == 14) and not (c == "outer_race" and s == 21):
            continue                                   # 12 pairs: one per (size, load), rotating the class
        fy = by[(order[c], s, l)]
        root = pathlib.Path(tempfile.mkdtemp(prefix="sep_"))
        dev = Device(DeviceConfig("bench", "s", "m", root), HashEmbedder())
        try:
            dev.fit_baseline(base)
            for w in Recordings.windows(fx)[:15]:
                dev.ingest_window(w)
            for w in Recordings.windows(HEALTHY_REPLAY_FILES[0])[:10]:
                dev.ingest_window(w)
            for w in Recordings.windows(fy)[:15]:
                dev.ingest_window(w)
            n_ep = len(dev.episodes())
            ok += n_ep == 2
            pairs.append({"first": fx, "second": fy, "episodes": n_ep})
        finally:
            dev.close()
            shutil.rmtree(root, ignore_errors=True)
    return {"separated": f"{ok}/{len(pairs)}", "pairs": pairs}


def run() -> dict:
    base = Recordings.baseline()
    healthy_unseen = np.concatenate([Recordings.windows(f) for f in HEALTHY_REPLAY_FILES])
    rows = []
    false_alarm = None
    for k, (fid, (cls, size, load)) in enumerate(sorted((f, v) for f, v in CWRU.items() if v[0] != "normal")):
        root = pathlib.Path(tempfile.mkdtemp(prefix="k3_"))
        dev = Device(DeviceConfig("bench", "s", "m", root), HashEmbedder())
        try:
            g = dev.fit_baseline(base)
            if false_alarm is None:
                states = [dev.ingest_window(w)["state"] for w in healthy_unseen]
                false_alarm = {"windows": len(states), "not_normal": sum(s != "normal" for s in states),
                               "rate": round(sum(s != "normal" for s in states) / len(states), 4)}
            W = Recordings.windows(fid)
            # --- fixed scenario ---
            t = time.perf_counter()
            st = [dev.ingest_window(w)["state"] for w in W[:FAULT_WINDOWS]]
            ep1 = dev.episodes()[0]["episode_id"]
            dev.record_action(ep1, "replace_bearing")
            healthy = Recordings.windows(HEALTHY_REPLAY_FILES[k % 2])
            n_to_verdict = None
            for i, w in enumerate(healthy[:REQUIRED_WINDOWS * 2]):
                dev.ingest_window(w)
                if dev.episode(ep1)["verify"]["verdict"] != "verifying":
                    n_to_verdict = i + 1
                    break
            fixed_verdict = dev.episode(ep1)["verify"]["verdict"]
            dev.confirm_outcome(ep1, "worked")
            # --- still-faulty scenario: the fault returns, a lubrication attempt is made, the fault continues ---
            rest = W[FAULT_WINDOWS:]
            for w in rest[:5]:
                dev.ingest_window(w)
            eps = [e for e in dev.episodes() if e["episode_id"] != ep1]
            ep2 = max(eps, key=lambda e: e["occurrences"])["episode_id"]
            dev.record_action(ep2, "lubricate")
            post = rest[5:5 + REQUIRED_WINDOWS]
            for w in post:
                dev.ingest_window(w)
            still_verdict = dev.episode(ep2)["verify"]["verdict"]
            rows.append({"fid": fid, "class": cls, "size": size, "load": load,
                         "detected": sum(s != "normal" for s in st) / len(st),
                         "episodes_opened_first_20": sum(s == "new" for s in st),
                         "recurrence_detected": dev.episode(ep2).get("recurrence_of") == ep1,
                         "fixed_verdict": fixed_verdict, "windows_to_verdict": n_to_verdict,
                         "still_faulty_verdict": still_verdict,
                         "still_faulty_windows_available": int(len(post)),
                         "seconds": round(time.perf_counter() - t, 2)})
            print(f"{fid:4d} {cls:10s} {size:2d}mil load{load}: detect {rows[-1]['detected']:.2f} "
                  f"eps {rows[-1]['episodes_opened_first_20']} | fixed -> {fixed_verdict} ({n_to_verdict}) | "
                  f"still faulty -> {still_verdict}")
        finally:
            dev.close()
            shutil.rmtree(root, ignore_errors=True)
    separation = distinct_faults_after_healthy_gap(base)
    n = len(rows)
    evaluable = [r for r in rows if r["still_faulty_windows_available"] >= REQUIRED_WINDOWS]
    summary = {
        "method": __doc__.strip().splitlines()[0],
        "required_windows": REQUIRED_WINDOWS, "tau": {k: round(v, 3) for k, v in g.items() if k.startswith("tau")},
        "gate_false_alarm_unseen_healthy": false_alarm,
        "gate_detection_rate_fault_windows": round(float(np.mean([r["detected"] for r in rows])), 4),
        "episodes_per_recording_mean": round(float(np.mean([r["episodes_opened_first_20"] for r in rows])), 3),
        "recurrence_detected_rate": round(float(np.mean([r["recurrence_detected"] for r in rows])), 3),
        "K3_fixed_verified_resolved": f"{sum(r['fixed_verdict'] == 'symptom_resolved' for r in rows)}/{n}",
        "K3_still_faulty_verified_persists": f"{sum(r['still_faulty_verdict'] == 'symptom_persists' for r in evaluable)}/{len(evaluable)}",
        "K3_false_promotions": sum(r["still_faulty_verdict"] == "symptom_resolved" for r in rows),
        "windows_to_verdict_fixed_median": float(np.median([r["windows_to_verdict"] for r in rows if r["windows_to_verdict"]])),
        "distinct_faults_separated_by_healthy_gap": separation,
        "per_recording": rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "k3_gate_verifier.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_recording"}, indent=2))
    return summary


if __name__ == "__main__":
    run()
