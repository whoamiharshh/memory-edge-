"""Phone-microphone answer to the "60 Hz phone" weakness, measured on REAL microphone recordings.

Data: University of Ottawa UODS-VAFDC (CC BY 4.0, DOI 10.17632/y2px5tg92h.1): 20 bearings whose faults DEVELOPED
NATURALLY (seals removed, degreased), each recorded healthy (stage 0), developing (1) and faulty (2); a microphone
(PCB 130F20, 2 cm from the bearing) and an accelerometer on the same run, 42 kHz, 10 s. Never used to tune anything
before this benchmark; thresholds as shipped (NORMAL_FACTOR 2.0, MERGE_FACTOR 1.5).

Per bearing = one machine with its own device (profile `acoustic` on the microphone, `rotating-hf` on the
accelerometer, bearing NSK 6203 from the paper):
  baseline        first half of its healthy recording
  false alarms    second half of the healthy recording (never seen)
  detection       share of abnormal windows in the first 15 windows of the developing and of the faulty recording
  fix verified    15 faulty windows -> action -> the rest of its healthy recording, then another bearing's healthy
                  recording ("a new bearing") -> symptom_resolved?   (N = 15: recordings are only 10 s long)
  still faulty    the fault continues after the action -> symptom_persists?
Also at a PHONE rate: the microphone decimated to 16 kHz happens inside the profile (analysis rate), i.e. exactly
what a phone's 44.1/48 kHz audio becomes.
Run: .venv\\Scripts\\python.exe -m bench.acoustic_uottawa
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import numpy as np
import scipy.io as sio

from edge import profiles
from edge.device import Device, DeviceConfig
from shared.embed import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "uottawa"
OUT = ROOT / "bench" / "results"
FS = 42000.0
N = 15
KIND = {"I": "inner_race", "O": "outer_race", "B": "ball", "C": "cage"}


def bearings() -> dict[int, str]:
    out = {}
    for f in RAW.glob("*_1.mat"):
        k, b, _ = f.stem.split("_")
        out[int(b)] = k
    return dict(sorted(out.items()))


def load(name: str, col: int) -> tuple[np.ndarray, float]:
    m = sio.loadmat(str(RAW / f"{name}.mat"))
    d = m[name]
    return d[:, col].astype(np.float64), float(d[0, 2])


_feat_cache: dict = {}


def feats(prof, name: str, col: int) -> np.ndarray:
    key = (prof.name, name, col)
    if key not in _feat_cache:
        x, rpm = load(name, col)
        _feat_cache[key] = np.stack([prof.features(w, prof.analysis_fs, rpm) for w in prof.windows(x, FS)])
    return _feat_cache[key]


def device(profile: str, base: np.ndarray):
    root = pathlib.Path(tempfile.mkdtemp(prefix="uo_"))
    d = Device(DeviceConfig("bench", "s", "m", root, profile=profile, profile_params={"bearing": "6203-UO"}),
               HashEmbedder())
    d.fit_baseline(base)
    return d, root


def verdict_after(dev, fault: np.ndarray, after: np.ndarray) -> str:
    for w in fault[:N]:
        dev.ingest_window(w)
    eps = [e for e in dev.episodes() if not e.get("dismissed")]
    if not eps:
        return "no_episode"
    ep = max(eps, key=lambda e: e["occurrences"])
    dev.record_action(ep["episode_id"], "replace_bearing", required_windows=N)
    v = "verifying"
    for w in after:
        dev.ingest_window(w)
        v = dev.episode(ep["episode_id"])["verify"]["verdict"]
        if v != "verifying":
            break
    return v


def run(profile: str, col: int) -> dict:
    prof = profiles.make(profile, bearing="6203-UO")
    B = bearings()
    rows = []
    for b, kind in B.items():
        H = feats(prof, f"H_{b}_0", col)
        half = len(H) // 2
        base, test = H[:half], H[half:]
        D1, D2 = feats(prof, f"{kind}_{b}_1", col), feats(prof, f"{kind}_{b}_2", col)
        other = next(o for o in B if o != b)
        Hn = feats(prof, f"H_{other}_0", col)
        row = {"bearing": b, "fault": KIND[kind]}
        dev, root = device(profile, base)
        try:
            row["false_alarms"] = sum(dev.ingest_window(w)["state"] != "normal" for w in test)
            row["healthy_windows"] = len(test)
        finally:
            dev.close(); shutil.rmtree(root, ignore_errors=True)
        for stage, W in (("developing", D1), ("faulty", D2)):
            dev, root = device(profile, base)
            try:
                row[f"detected_{stage}"] = round(sum(dev.ingest_window(w)["state"] != "normal" for w in W[:N]) / N, 3)
            finally:
                dev.close(); shutil.rmtree(root, ignore_errors=True)
        for label, after in (("fixed_same_bearing_healthy", np.concatenate([test, test])),
                             ("fixed_new_bearing", Hn), ("still_faulty", D2[N:])):
            dev, root = device(profile, base)
            try:
                row[label] = verdict_after(dev, D2, after)
            finally:
                dev.close(); shutil.rmtree(root, ignore_errors=True)
        rows.append(row)
        print(profile, row, flush=True)
    n = len(rows)
    return {
        "profile": profile, "channel": "microphone" if col == 1 else "accelerometer", "bearings": n,
        "false_alarms": {"windows": sum(r["healthy_windows"] for r in rows), "alarms": sum(r["false_alarms"] for r in rows)},
        "detection_developing": round(float(np.mean([r["detected_developing"] for r in rows])), 3),
        "detection_faulty": round(float(np.mean([r["detected_faulty"] for r in rows])), 3),
        "fix_verified_same_bearing_healthy": f"{sum(r['fixed_same_bearing_healthy'] == 'symptom_resolved' for r in rows)}/{n}",
        "fix_verified_new_bearing": f"{sum(r['fixed_new_bearing'] == 'symptom_resolved' for r in rows)}/{n}",
        "still_faulty_caught": f"{sum(r['still_faulty'] == 'symptom_persists' for r in rows)}/{n}",
        "rows": rows,
    }


def main() -> dict:
    out = {"method": __doc__.strip().splitlines()[0], "dataset": "UODS-VAFDC v1, CC BY 4.0, DOI 10.17632/y2px5tg92h.1",
           "protocol": __doc__.split("Per bearing")[1].split("Run:")[0].strip(), "N_verify": N,
           "microphone": run("acoustic", 1), "accelerometer": run("rotating-hf", 0)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acoustic_uottawa.json").write_text(json.dumps(out, indent=1))
    for k in ("microphone", "accelerometer"):
        r = out[k]
        print(k, {x: r[x] for x in r if x != "rows"})
    return out


if __name__ == "__main__":
    main()
