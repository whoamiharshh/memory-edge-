"""Automatic operating-point check (answers "a healthy machine at a new load looks faulty"): does the device tell
"new operating point" apart from "fault", before any technician input?

HUST held-out data (as bench/hust_holdout.py), profile rotating-hf, shipped thresholds. The operating point is the
shaft speed from each recording (speed drops as load rises: ~24.9 Hz at 0 W to ~23 Hz at 400 W). Per bearing type:
  baseline      healthy 0 W + 200 W, with their speeds as the taught range
  healthy 400 W episodes that open: flagged "untaught operating point"? suggestion "probably normal"?
  faults 0/200 W (taught speeds): wrongly flagged "untaught"?
  faults 400 W: flagged untaught AND suggestion "fault signature present"?
The suggestion comes from physics: the median defect-envelope score of the episode's first windows against a
threshold chosen on CWRU (spike/signature_threshold.py), so this measurement on HUST is held out.
Run: .venv\\Scripts\\python.exe -m bench.operating_point
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import numpy as np

from bench.hust_holdout import CLASS, RAW, feats
from edge import profiles
from edge.device import Device, DeviceConfig
from shared.embed import HashEmbedder

OUT = pathlib.Path(__file__).resolve().parent / "results"


def device(prof, base, ops):
    root = pathlib.Path(tempfile.mkdtemp(prefix="oppt_"))
    dev = Device(DeviceConfig("b", "s", "m", root, profile="rotating-hf", profile_params=dict(prof.params)), HashEmbedder())
    dev.fit_baseline(base, ops)
    return dev, root


def episodes_of(dev, W, speed):
    for w in W:
        dev.ingest_window(w, op={"speed_hz": speed})
    return dev.episodes()


def main() -> dict:
    tally = {"healthy_400_episodes": 0, "healthy_400_flagged_untaught": 0, "healthy_400_suggested_normal": 0,
             "fault_taught_speed_recordings": 0, "fault_taught_speed_wrongly_untaught": 0,
             "fault_400_recordings": 0, "fault_400_flagged_untaught": 0, "fault_400_suggested_fault": 0}
    speeds = {}
    for b in "45678":
        prof = profiles.make("rotating-hf", bearing=f"620{b}")
        (F0, s0), (F2, s2), (F4, s4) = (feats(f"N{b}0{k}", prof) for k in "024")
        speeds[f"620{b}"] = {"0W": s0, "200W": s2, "400W": s4}
        base, ops = np.concatenate([F0, F2]), [{"speed_hz": s0}] * len(F0) + [{"speed_hz": s2}] * len(F2)
        dev, root = device(prof, base, ops)
        try:
            for e in episodes_of(dev, F4, s4):
                tally["healthy_400_episodes"] += 1
                u = e.get("untaught_operating_point")
                tally["healthy_400_flagged_untaught"] += bool(u)
                tally["healthy_400_suggested_normal"] += bool(u) and u["suggestion"].startswith("probably")
        finally:
            dev.close()
            shutil.rmtree(root, ignore_errors=True)
        for kind in CLASS:
            for load in "024":
                name = f"{kind}{b}0{load}"
                if not (RAW / f"{name}.mat").exists():
                    continue
                W, speed = feats(name, prof)
                dev, root = device(prof, base, ops)
                try:
                    eps = episodes_of(dev, W[:20], speed)
                finally:
                    dev.close()
                    shutil.rmtree(root, ignore_errors=True)
                if not eps:
                    continue
                u = eps[-1].get("untaught_operating_point")
                if load in "02":
                    tally["fault_taught_speed_recordings"] += 1
                    tally["fault_taught_speed_wrongly_untaught"] += bool(u)
                else:
                    tally["fault_400_recordings"] += 1
                    tally["fault_400_flagged_untaught"] += bool(u)
                    tally["fault_400_suggested_fault"] += bool(u) and u["suggestion"].startswith("a fault signature")
        print(f"620{b}", tally, flush=True)
    out = {"method": __doc__.strip().splitlines()[0], "shaft_speeds_hz": speeds, "tally": tally}
    OUT.mkdir(exist_ok=True)
    (OUT / "operating_point.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
