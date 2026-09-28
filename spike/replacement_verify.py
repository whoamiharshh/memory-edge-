"""Spike: after a REPLACEMENT the machine's healthy signature changes (bench/acoustic_uottawa.py: a new bearing is
outside the old healthy radius). Test a parameter-free rule: a post-action window counts as 'fixed-like' when it is
nearer to the old healthy baseline than to the fault episode's exemplars (nearest-state rule).
Run: .venv\Scripts\python.exe -m spike.replacement_verify
"""
import numpy as np

from bench.acoustic_uottawa import N, bearings, feats
from edge import profiles
from edge.fingerprint import Baseline


def dmin(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)


for profile, col in (("acoustic", 1), ("rotating-hf", 0)):
    prof = profiles.make(profile, bearing="6203-UO")
    B = bearings()
    new_ok = still_ok = 0
    frac_new, frac_still = [], []
    for b, kind in B.items():
        H = feats(prof, f"H_{b}_0", col)
        base = H[: len(H) // 2]
        bl = Baseline.fit(base, prof.fp_version, prof.min_std)
        zb = bl.z(base)
        D2 = bl.z(feats(prof, f"{kind}_{b}_2", col))
        ex = D2[:N]
        other = next(o for o in B if o != b)
        Hn = bl.z(feats(prof, f"H_{other}_0", col))
        rest = D2[N:]
        fn = (dmin(Hn, zb) < dmin(Hn, ex))
        fs = (dmin(rest, zb) < dmin(rest, ex))
        frac_new.append(fn.mean()); frac_still.append(fs.mean())
        run = lambda f: max((len(s) for s in "".join("1" if v else "0" for v in f).split("0")), default=0)
        new_ok += run(fn) >= N
        still_ok += run(fs) < N
    print(profile, f"new bearing judged fixed {new_ok}/20 (mean fraction {np.mean(frac_new):.2f}) | "
                   f"continuing fault judged NOT fixed {still_ok}/20 (fraction fixed-like {np.mean(frac_still):.2f})")
