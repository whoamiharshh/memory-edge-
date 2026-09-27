"""Robots (PS3): does the same memory engine flag robot execution failures from wrist force/torque?

Data: UCI Robot Execution Failures (CC BY 4.0, DOI 10.24432/C5M89N), 5 learning problems (lp1 approach to grasp,
lp2 part transfer, lp3 part position after transfer, lp4 approach to ungrasp, lp5 motion with part). Each instance
is 15 samples x 6 channels (Fx Fy Fz Tx Ty Tz) = one window for the `force-torque` profile.
Per problem and seed: the NORMAL instances ("normal"/"ok") are shuffled; 60 % fit the device's healthy baseline
(the shipped gate calibration, no tuning), the other 40 % measure false alarms; every failure instance measures
detection (gate state != normal), per failure type. 5 seeds, averaged (the data set is small).
Caveat: each instance starts right after the robot's own controller detected a failure, so failures are
comparatively obvious; this shows the engine transfers to robot data, not that it beats the robot's controller.
Run: .venv\\Scripts\\python.exe -m bench.robot_failures
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import shutil
import tempfile

import numpy as np

from data.fetch_uci_robot import load
from edge.device import Device, DeviceConfig
from shared.embed import HashEmbedder

OUT = pathlib.Path(__file__).resolve().parent / "results"
NORMAL = {"normal", "ok"}
SEEDS = 5


TUNE, TEST = ("lp1", "lp2"), ("lp3", "lp4", "lp5")
FACTORS = (0.75, 1.0, 1.25, 1.5, 2.0)
MAX_FALSE_ALARM = 0.05       # selection rule fixed before looking at the test tasks


def run_problem(lp: str, seed: int, normal_factor: float | None = None) -> dict:
    inst = load(lp)
    normals = [np.asarray(r) for lbl, r in inst if lbl in NORMAL]
    fails = [(lbl, np.asarray(r)) for lbl, r in inst if lbl not in NORMAL]
    random.Random(seed).shuffle(normals)
    k = max(10, int(0.6 * len(normals)))
    root = pathlib.Path(tempfile.mkdtemp(prefix="robot_"))
    dev = Device(DeviceConfig("r", "s", "m", root, profile="force-torque"), HashEmbedder())
    dev.profile.normal_factor = normal_factor
    try:
        dev.fit_baseline(np.stack([dev.profile.features(x) for x in normals[:k]]))
        state = lambda x: dev.ingest_window(dev.profile.features(x))["state"]
        held = [state(x) for x in normals[k:]]
        per = collections.defaultdict(list)
        for lbl, x in fails:
            per[lbl].append(state(x) != "normal")
            state(normals[0])                      # a normal window between failures ends the abnormal run
    finally:
        dev.close()
        shutil.rmtree(root, ignore_errors=True)
    return {"held_normals": len(held), "false_alarms": sum(s != "normal" for s in held),
            "per_class": {c: (sum(v), len(v)) for c, v in per.items()}}


def summarise(lps, factor) -> dict:
    fa = n = hit = tot = 0
    for lp in lps:
        for s in range(SEEDS):
            r = run_problem(lp, s, factor)
            fa, n = fa + r["false_alarms"], n + r["held_normals"]
            hit += sum(v[0] for v in r["per_class"].values())
            tot += sum(v[1] for v in r["per_class"].values())
    return {"false_alarm_rate": round(fa / n, 3), "detection_rate": round(hit / tot, 3)}


def tune() -> dict:
    """Pick the robot profile's healthy-radius factor on TUNE tasks only; report TEST tasks at default and chosen."""
    table = {f: summarise(TUNE, f) for f in FACTORS}
    ok = {f: r for f, r in table.items() if r["false_alarm_rate"] <= MAX_FALSE_ALARM}
    chosen = max(ok, key=lambda f: (ok[f]["detection_rate"], f)) if ok else 2.0
    return {"rule": f"on {TUNE}: highest detection with false alarms <= {MAX_FALSE_ALARM:.0%}",
            "tune_table": {str(f): r for f, r in table.items()}, "chosen_factor": chosen,
            "test_default_factor_2.0": summarise(TEST, None), f"test_chosen_factor_{chosen}": summarise(TEST, chosen),
            "test_per_task_chosen": {lp: summarise((lp,), chosen) for lp in TEST}}


def main() -> dict:
    out = {"method": __doc__.strip().splitlines()[0], "dataset": "UCI Robot Execution Failures, CC BY 4.0",
           "seeds": SEEDS, "problems": {}}
    out["tuning"] = tune()
    print(json.dumps(out["tuning"], indent=1), flush=True)
    for lp in ("lp1", "lp2", "lp3", "lp4", "lp5"):
        runs = [run_problem(lp, s) for s in range(SEEDS)]
        fa_n = sum(r["held_normals"] for r in runs)
        fa = sum(r["false_alarms"] for r in runs)
        classes = sorted({c for r in runs for c in r["per_class"]})
        det = {c: round(sum(r["per_class"][c][0] for r in runs) / sum(r["per_class"][c][1] for r in runs), 3) for c in classes}
        tot = sum(r["per_class"][c][1] for r in runs for c in classes)
        hit = sum(r["per_class"][c][0] for r in runs for c in classes)
        out["problems"][lp] = {"false_alarm_rate": round(fa / fa_n, 3), "held_out_normals_per_seed": runs[0]["held_normals"],
                               "detection_rate": round(hit / tot, 3), "detection_per_failure_type": det}
        print(lp, out["problems"][lp], flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "robot_failures.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
