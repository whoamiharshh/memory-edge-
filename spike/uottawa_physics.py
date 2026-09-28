"""Spike: bearing physics on the University of Ottawa data (natural wear), accelerometer vs MICROPHONE.

Per file: 1 s windows (42 kHz), shaft speed from the file's own speed value (rpm, row 0 of column 3), NSK 6203
geometry from the paper (8 balls, 6.77 mm, pitch 28.50 mm). For each channel and variant, the defect scores and the
argmax class; per recording the majority vote over its 10 windows. Truth = file prefix (I/O/B/C) for stages 1 and 2.
Run: .venv\\Scripts\\python.exe -m spike.uottawa_physics
"""
from __future__ import annotations

import collections
import pathlib

import numpy as np
import scipy.io as sio

from edge import physics as P

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "uottawa"
FS = 42000.0
GEO = P.BearingGeometry(8, 6.77, 28.50, 0.0, "NSK 6203")
TRUTH = {"I": "inner_race", "O": "outer_race", "B": "ball", "C": "cage"}


def load(name: str):
    m = sio.loadmat(str(RAW / f"{name}.mat"))
    d = m[[k for k in m if not k.startswith("__")][0]]
    return d[:, 0].astype(float), d[:, 1].astype(float), float(d[0, 2])


def classify(x, fs, shaft, band, with_cage):
    s = P.bearing_defect_scores(x, fs, shaft, GEO, band=band)
    keys = ("bpfo", "bpfi", "bsf", "ftf") if with_cage else ("bpfo", "bpfi", "bsf")
    best = max(keys, key=lambda k: s[k])
    return P.DEFECT_TO_CLASS[best], s


def main():
    res = collections.defaultdict(lambda: collections.Counter())
    for f in sorted(RAW.glob("*.mat")):
        name = f.stem
        kind, bearing, stage = name.split("_")
        acc, mic, rpm = load(name)
        shaft = rpm / 60.0
        line = [name, f"{rpm:.0f}rpm"]
        for ch, x in (("acc", acc), ("mic", mic)):
            for variant in ("fixed", "kurtogram"):
                votes, maxs = collections.Counter(), []
                for w in np.array_split(x, 10):
                    band = P.kurtogram_band(w, FS) if variant == "kurtogram" else None
                    c, s = classify(w, FS, shaft, band, with_cage=True)
                    votes[c] += 1
                    maxs.append(max(s[k] for k in ("bpfo", "bpfi", "bsf")))
                pred = votes.most_common(1)[0][0]
                if kind != "H":
                    res[(ch, variant, stage)][(TRUTH[kind], pred == TRUTH[kind])] += 1
                line.append(f"{ch}/{variant}:{pred[:5]}({np.median(maxs):.2f})")
        print(" ".join(line), flush=True)
    for key in sorted(res):
        c = res[key]
        tot = sum(c.values())
        ok = sum(v for (t, good), v in c.items() if good)
        per = {t: f"{c[(t, True)]}/{c[(t, True)] + c[(t, False)]}" for t in TRUTH.values()}
        print(key, f"{ok}/{tot}", per)


if __name__ == "__main__":
    main()
