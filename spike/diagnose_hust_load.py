"""Why does healthy HUST data at an unseen load look abnormal? Distance of healthy windows to the baseline, in units
of tau_normal (the gate's healthy radius), for several baseline choices, per bearing type:
  A  baseline 0+200 W        -> 400 W   (extrapolate the load: the benchmark's setting)
  B  baseline 0+400 W        -> 200 W   (interpolate the load)
  C  baseline all loads, first half of each recording -> second half of the SAME recordings (session variance only)
  D  like A but features divided by overall RMS level (amplitude-normalised shape; does load mostly change level?)
Run: .venv\\Scripts\\python.exe spike\\diagnose_hust_load.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.hust_holdout import feats                    # noqa: E402
from edge import profiles                               # noqa: E402
from edge.fingerprint import Baseline                   # noqa: E402
from edge.gate import calibrate                         # noqa: E402

LEVEL = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23]   # log-amplitude features (rotating-hf layout)


def ratio(base, test, drop_level=False):
    if drop_level:
        keep = [i for i in range(base.shape[1]) if i not in LEVEL]
        base, test = base[:, keep], test[:, keep]
    b = Baseline.fit(base)
    zb, zt = b.z(base), b.z(test)
    tau = calibrate(zb).tau_normal
    d = np.sqrt(((zt[:, None, :] - zb[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d / tau)), float(np.mean(d > tau))


for bno in "45678":
    p = profiles.make("rotating-hf", bearing=f"620{bno}")
    F = {load: feats(f"N{bno}0{load}", p)[0] for load in "024"}
    half = lambda x, h: x[: len(x) // 2] if h == 0 else x[len(x) // 2:]
    rows = {
        "A extrapolate 400W": ratio(np.concatenate([F["0"], F["2"]]), F["4"]),
        "B interpolate 200W": ratio(np.concatenate([F["0"], F["4"]]), F["2"]),
        "C same sessions, 2nd half": ratio(np.concatenate([half(F[k], 0) for k in F]), np.concatenate([half(F[k], 1) for k in F])),
        "D A without level features": ratio(np.concatenate([F["0"], F["2"]]), F["4"], drop_level=True),
    }
    print(f"620{bno}: " + " | ".join(f"{k}: median d/tau {m:.1f}, alarm {a:.0%}" for k, (m, a) in rows.items()))
