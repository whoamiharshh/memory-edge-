"""Do temporal force/torque features catch subtle robot failures better? Same protocol as bench/robot_failures.py
(60 % of normals = baseline, 40 % false alarms, all failures detection, 5 seeds, shipped gate), comparing the current
features (fp-ft1: per-channel mean/std/max + resultants) with a temporal variant (per channel: mean, first-to-last
change, largest step, and resultant force/torque shape). Chosen on lp1+lp2, then read on lp3-lp5.
Run: .venv\\Scripts\\python.exe spike\\robot_features.py
"""
import math
import pathlib
import random
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.robot_failures import NORMAL, SEEDS           # noqa: E402
from data.fetch_uci_robot import load                     # noqa: E402
from edge.device import Device, DeviceConfig              # noqa: E402
from edge.profiles import ForceTorque                     # noqa: E402
from shared.embed import HashEmbedder                     # noqa: E402

sl = lambda v: math.copysign(math.log1p(abs(v)), v)


def temporal(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 6)
    out = []
    for c in range(6):
        d = np.diff(x[:, c])
        out += [sl(x[:, c].mean()), sl(x[-1, c] - x[0, c]), sl(np.abs(d).max() if len(d) else 0.0)]
    f, t = np.linalg.norm(x[:, :3], axis=1), np.linalg.norm(x[:, 3:], axis=1)
    k = np.arange(len(f)) - (len(f) - 1) / 2
    slope = lambda y: float((k * (y - y.mean())).sum() / ((k ** 2).sum() or 1))
    out += [sl(f.mean()), sl(f.max()), sl(slope(f)), sl(t.mean()), sl(t.max()), sl(slope(t)),
            sl(np.abs(np.diff(f)).sum()), sl(np.abs(np.diff(t)).sum()), sl(f.argmax() / len(f))]
    return np.asarray(out)


def run(lps, feat):
    fa = n = hit = tot = 0
    for lp in lps:
        inst = load(lp)
        normals = [np.asarray(r) for lbl, r in inst if lbl in NORMAL]
        fails = [np.asarray(r) for lbl, r in inst if lbl not in NORMAL]
        for s in range(SEEDS):
            ns = normals[:]
            random.Random(s).shuffle(ns)
            k = max(10, int(0.6 * len(ns)))
            root = pathlib.Path(tempfile.mkdtemp(prefix="rf_"))
            dev = Device(DeviceConfig("r", "s", "m", root, profile="force-torque"), HashEmbedder())
            try:
                dev.fit_baseline(np.stack([feat(x) for x in ns[:k]]))
                st = lambda x: dev.ingest_window(feat(x))["state"]
                held = [st(x) for x in ns[k:]]
                fa, n = fa + sum(h != "normal" for h in held), n + len(held)
                for x in fails:
                    hit += st(x) != "normal"
                    tot += 1
                    st(ns[0])
            finally:
                dev.close()
                shutil.rmtree(root, ignore_errors=True)
    return {"false_alarm_rate": round(fa / n, 3), "detection_rate": round(hit / tot, 3)}


current = ForceTorque().features
for name, feat in (("current fp-ft1", current), ("temporal", temporal)):
    print(name, "| tune lp1+lp2:", run(("lp1", "lp2"), feat), "| test lp3-lp5:", run(("lp3", "lp4", "lp5"), feat), flush=True)
