"""Naming imbalance and misalignment on ONE real machine (MaFaulDa, UFRJ): the check the University of Ottawa motor set
could not give (there every fault was a different motor).

Data: MaFaulDa - a SpectraQuest Machinery Fault Simulator recorded normal (49 speeds, 737-3686 rpm), with rotor
imbalance (6-35 g) and with horizontal (0.5-2.0 mm) and vertical (0.51-1.90 mm) shaft misalignment; 50 kHz, 5 s;
channels: tachometer, underhang accelerometer (axial, radial, tangential), overhang accelerometer (axial, radial,
tangential), microphone. Shaft speed = the file name (Hz; from the tachometer).
Protocol (nothing tuned here; shipped profiles and thresholds): the NORMAL recordings are split by speed - every other
speed is taught as healthy, the rest are held out. For each recording, its baseline = the taught normal recordings at
the two nearest speeds (a device watches its machine at its operating speed; vibration grows strongly with speed).
  detection    share of windows the gate flags (fault recordings), false alarms on the held-out normal speeds
  naming       majority hint over the flagged windows vs the truth: imbalance -> "imbalance", horizontal / vertical
               misalignment -> "misalignment" (relative order rule, shipped); the absolute textbook rule for contrast
Channels: overhang radial (next to the rotor disk), underhang radial, axial (underhang), microphone.
Run: .venv\\Scripts\\python.exe -m bench.mafaulda_rules [max_files_per_severity]
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

from edge import profiles
from edge.fingerprint import Baseline
from edge.gate import _nn, calibrate

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mafaulda"
OUT = ROOT / "bench" / "results"
FS = 50000.0
CHANNELS = {"overhang_radial": (5, "rotating-hf"), "underhang_radial": (2, "rotating-hf"),
            "underhang_axial": (1, "rotating-hf"), "microphone": (7, "acoustic")}
# The simulator's bearings, from the MaFaulDa page (checked 28 Sep 2026): 8 balls, ball diameter 0.7145 cm, cage
# diameter 2.8519 cm; the page's factors BPFO 2.9980 / BPFI 5.0020 / FTF 0.3750 follow exactly from these numbers.
GEOMETRY = {"n_elements": 8, "ball_d": 0.7145, "pitch_d": 2.8519, "name": "MaFaulDa simulator bearing"}
TRUTH = {"imbalance": "imbalance", "horizontal-misalignment": "misalignment", "vertical-misalignment": "misalignment"}


def speed(p: pathlib.Path) -> float:
    return float(p.stem)


_cache: dict = {}
FEAT_DIR = RAW / "_features_geo"        # features with the simulator's own bearing geometry


def _feat_file(p: pathlib.Path) -> pathlib.Path:
    return FEAT_DIR / (p.relative_to(RAW).as_posix().replace("/", "__") + ".npz")


def compute(p: pathlib.Path) -> str:
    """Read the CSV ONCE and fingerprint every channel of CHANNELS; cached on disk (runs in worker processes)."""
    out = _feat_file(p)
    if out.exists():
        return str(out)
    import pandas as pd
    data = pd.read_csv(p, header=None, dtype=np.float64, engine="c").to_numpy()
    hz = speed(p)
    arrays = {}
    for c, prof_name in CHANNELS.values():
        prof = profiles.make(prof_name, shaft_hz=hz, geometry=GEOMETRY)
        arrays[f"{c}_{prof_name}"] = np.stack([prof.features(w, prof.analysis_fs, hz * 60)
                                              for w in prof.windows(data[:, c], FS)])
    FEAT_DIR.mkdir(exist_ok=True)
    np.savez(out, **arrays)
    return str(out)


def precompute(files: list[pathlib.Path], workers: int = 8) -> None:
    import concurrent.futures as cf
    todo = [f for f in files if not _feat_file(f).exists()]
    with cf.ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(compute, todo, chunksize=4), 1):
            if k % 50 == 0:
                print(f"[features] {k}/{len(todo)}", flush=True)


def feats(p: pathlib.Path, col: int, profile: str) -> np.ndarray:
    key = (str(p), col, profile)
    if key not in _cache:
        z = np.load(compute(p))
        for k in z.files:
            c, prof_name = k.split("_", 1)
            _cache[(str(p), int(c), prof_name)] = z[k]
    return _cache[key]


def recordings(max_per: int | None) -> dict[str, list[tuple[str, pathlib.Path]]]:
    out = {}
    for kind in TRUTH:
        d = RAW / kind
        if not d.exists():
            continue
        rows = []
        for sev in sorted(p for p in d.iterdir() if p.is_dir()):
            files = sorted(sev.glob("*.csv"), key=speed)
            if max_per:
                idx = np.linspace(0, len(files) - 1, min(max_per, len(files))).round().astype(int)
                files = [files[i] for i in sorted(set(idx))]
            rows += [(sev.name, f) for f in files]
        out[kind] = rows
    return out


def main(max_per: int | None = None) -> dict:
    normals = sorted((RAW / "normal").glob("*.csv"), key=speed)
    taught, held = normals[0::2], normals[1::2]
    recs = recordings(max_per)
    precompute(normals + [f for rows in recs.values() for _, f in rows])
    res = {"method": __doc__.strip().splitlines()[0], "protocol": __doc__.split("Protocol")[1].split("Run:")[0].strip(),
           "files": {k: len(v) for k, v in recs.items()} | {"normal_taught": len(taught), "normal_held_out": len(held)},
           "channels": {}}
    for ch, (col, profile) in CHANNELS.items():
        prof = profiles.make(profile, geometry=GEOMETRY)

        def judge(f: pathlib.Path) -> tuple[float, str, str]:
            near = sorted(taught, key=lambda n: abs(speed(n) - speed(f)))[:2]
            base = np.concatenate([feats(n, col, profile) for n in near])
            bl = Baseline.fit(base, prof.fp_version, prof.min_std)
            zb = bl.z(base)
            tau = calibrate(zb).tau_normal
            W = feats(f, col, profile)
            z = bl.z(W)
            flagged = _nn(z, zb) > tau
            pr = profiles.make(profile, shaft_hz=speed(f), geometry=GEOMETRY)
            if not flagged.any():
                return 0.0, "unknown", "unknown"
            rel = collections.Counter(pr.hint(w, zz)["fault_class"] for w, zz, fl in zip(W, z, flagged) if fl)
            ab = collections.Counter(pr.hint(w)["fault_class"] for w, fl in zip(W, flagged) if fl)
            return float(flagged.mean()), rel.most_common(1)[0][0], ab.most_common(1)[0][0]

        fa = [judge(f)[0] for f in held]
        out = {"false_alarm_window_share_on_held_out_normal": round(float(np.mean(fa)), 3), "per_fault": {}}
        for kind, rows in recs.items():
            per_sev = collections.defaultdict(lambda: {"n": 0, "detected": [], "rel_ok": 0, "rel_wrong": 0, "abs_ok": 0,
                                                       "rel_names": collections.Counter()})
            for sev, f in rows:
                share, rel, ab = judge(f)
                s = per_sev[sev]
                s["n"] += 1
                s["detected"].append(share)
                s["rel_ok"] += rel == TRUTH[kind]
                s["rel_wrong"] += rel not in (TRUTH[kind], "unknown")
                s["abs_ok"] += ab == TRUTH[kind]
                s["rel_names"][rel] += 1
            n = sum(s["n"] for s in per_sev.values())
            out["per_fault"][kind] = {
                "recordings": n,
                "detected_window_share": round(float(np.mean(sum((s["detected"] for s in per_sev.values()), []))), 3),
                "named_correctly_relative_rule": f"{sum(s['rel_ok'] for s in per_sev.values())}/{n}",
                "named_WRONG_relative_rule": f"{sum(s['rel_wrong'] for s in per_sev.values())}/{n}",
                "named_correctly_absolute_rule": f"{sum(s['abs_ok'] for s in per_sev.values())}/{n}",
                "relative_rule_names": dict(sum((s["rel_names"] for s in per_sev.values()), collections.Counter())),
                "by_severity": {sev: {"n": s["n"], "detected": round(float(np.mean(s["detected"])), 3),
                                      "named_ok": f"{s['rel_ok']}/{s['n']}",
                                      "named_wrong": f"{s['rel_wrong']}/{s['n']}"} for sev, s in sorted(per_sev.items())}}
        res["channels"][ch] = out
        print(ch, json.dumps({k: (v if k != "per_fault" else {kk: {x: vv[x] for x in vv if x != "by_severity"}
                                                               for kk, vv in v.items()}) for k, v in out.items()}),
              flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "mafaulda_rules.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
