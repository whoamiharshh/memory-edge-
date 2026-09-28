"""Imbalance / misalignment hints on REAL motors: the absolute textbook order rules vs the RELATIVE rule shipped now
("which shaft order grew versus this machine's healthy state").

Data: University of Ottawa electric-motor set UOEMD-VAFCVS (CC BY 4.0, DOI 10.17632/msxs4vj48g.2), drive-fed Marathon
motors at constant 15/30/45/60 Hz, unloaded and loaded (8 conditions), accelerometer + microphone, 42 kHz.
CAVEAT (stated in every result): each fault is ONE physical motor, so the healthy baseline comes from a different
motor than the faulty one; motor-to-motor differences are mixed into the change. Nothing is trained here.
Per condition: baseline = first 5 s of the healthy motor H-H; tested = the last 5 s of H-H (should say "no shaft
fault"), rotor unbalance R-U (should say imbalance) and rotor misalignment R-M (should say misalignment).
Only windows the device's gate flags as NOT normal get a hint (as in the product: a healthy window opens no episode);
a recording whose windows are all normal counts as "no fault" (unknown). Majority hint over the flagged windows;
shaft speed from the healthy recording's 1x peak.
Run: .venv\\Scripts\\python.exe -m bench.motor_rules
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np
import scipy.io as sio

from edge import physics as P
from edge import profiles
from edge.fingerprint import Baseline
from edge.gate import _nn, calibrate

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "uottawa_motor"
OUT = ROOT / "bench" / "results"
FS = 42000.0
WANT = {"H_H": "unknown", "R_U": "imbalance", "R_M": "misalignment"}


def load(name: str, col: int) -> np.ndarray:
    return sio.loadmat(str(RAW / f"{name}.mat"))["data"][:, col].astype(np.float64)


def vote(prof, W, z, zb, tau, relative=True):
    flagged = _nn(z, zb) > tau
    if not flagged.any():
        return "unknown"
    return collections.Counter(prof.hint(w, zz if relative else None)["fault_class"]
                               for w, zz, f in zip(W, z, flagged) if f).most_common(1)[0][0]


def run(profile: str, col: int) -> dict:
    rows = []
    for sp in (1, 2, 3, 4):
        for ld in (0, 1):
            h = load(f"H_H_{sp}_{ld}", col)
            f, a = P.spectrum(load(f"H_H_{sp}_{ld}", 0)[:84000], FS)
            shaft = P.estimate_shaft_hz(f, a, sp * 15 * 0.9, sp * 15 * 1.05) or sp * 15.0
            prof = profiles.make(profile, shaft_hz=shaft)
            feats = lambda x: np.stack([prof.features(w, prof.analysis_fs, shaft * 60) for w in prof.windows(x, FS)])
            half = len(h) // 2
            base = feats(h[:half])
            bl = Baseline.fit(base, prof.fp_version, prof.min_std)
            zb = bl.z(base)
            tau = calibrate(zb).tau_normal
            for motor, x in (("H_H", h[half:]), ("R_U", load(f"R_U_{sp}_{ld}", col)[half:]),
                             ("R_M", load(f"R_M_{sp}_{ld}", col)[half:])):
                W = feats(x)
                rows.append({"speed_hz": sp * 15, "loaded": bool(ld), "motor": motor, "want": WANT[motor],
                             "relative": vote(prof, W, bl.z(W), zb, tau),
                             "absolute": vote(prof, W, bl.z(W), zb, tau, relative=False)})
    out = {}
    for rule in ("relative", "absolute"):
        per = {m: f"{sum(r[rule] == r['want'] for r in rows if r['motor'] == m)}/8" for m in WANT}
        out[rule] = per | {"all": f"{sum(r[rule] == r['want'] for r in rows)}/{len(rows)}"}
    print(profile, out, flush=True)
    return {"summary": out, "rows": rows}


def main():
    res = {"method": __doc__.strip().splitlines()[0], "protocol": __doc__.split("Data:")[1].split("Run:")[0].strip(),
           "accelerometer": run("rotating-hf", 0), "microphone": run("acoustic", 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "motor_rules.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
