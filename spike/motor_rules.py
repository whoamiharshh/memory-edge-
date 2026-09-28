"""Spike: the order-spectrum rules (imbalance / misalignment / looseness) on the University of Ottawa motor set
(real motors; one motor per fault, so this can only TEST fixed rules, never train anything).
Run: .venv\\Scripts\\python.exe -m spike.motor_rules
"""
import collections
import statistics

import numpy as np
import scipy.io as sio

from edge import physics as P

RAW = "data/raw/uottawa_motor"
res = collections.defaultdict(collections.Counter)
rows = []
for fault in ["H_H", "R_U", "R_M", "B_R", "S_W", "V_U", "K_A", "F_B"]:
    for sp in (1, 2, 3, 4):
        for ld in (0, 1):
            name = f"{fault}_{sp}_{ld}"
            d = sio.loadmat(f"{RAW}/{name}.mat")["data"]
            for ch, col in (("acc", 0), ("mic", 1)):
                x = d[:, col].astype(float)
                f, a = P.spectrum(x[:84000], 42000.0)
                shaft = P.estimate_shaft_hz(f, a, sp * 15 * 0.9, sp * 15 * 1.05) or sp * 15
                o = P.order_amplitudes(f, a, shaft)
                res[(fault, ch)][P.rotating_rules(f, a, shaft)["fault_class"]] += 1
                rows.append((fault, sp, ld, ch, shaft, o.get("1x", 0.0), o.get("2x", 0.0) / max(o.get("1x", 1e-12), 1e-12)))
for k, v in sorted(res.items()):
    print(k, dict(v))
for fault in ["H_H", "R_U", "R_M"]:
    for ch in ("acc", "mic"):
        a1 = [r[5] for r in rows if r[0] == fault and r[3] == ch]
        ratio = [r[6] for r in rows if r[0] == fault and r[3] == ch]
        print(fault, ch, "median 1x", round(statistics.median(a1), 5), "median 2x/1x", round(statistics.median(ratio), 3))
# paired by speed/load: is the unbalanced motor's 1x above the healthy motor's? misaligned motor's 2x/1x?
for ch in ("acc", "mic"):
    key = {(r[0], r[1], r[2], r[3]): r for r in rows}
    ub = sum(key[("R_U", s, l, ch)][5] > key[("H_H", s, l, ch)][5] for s in (1, 2, 3, 4) for l in (0, 1))
    ms = sum(key[("R_M", s, l, ch)][6] > key[("H_H", s, l, ch)][6] for s in (1, 2, 3, 4) for l in (0, 1))
    print(ch, f"unbalanced 1x > healthy 1x in {ub}/8 speed-load pairs; misaligned 2x/1x > healthy in {ms}/8")
