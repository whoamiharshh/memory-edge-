"""Where should the 'fault signature present' threshold be? Distribution of the strongest bearing-defect envelope score
(max of BPFO/BPFI/BSF, log10 ratio over the median envelope level) for healthy vs faulty windows.
Chosen on CWRU (the tuning data, SKF 6205 geometry via rotating-hf), then checked on HUST (held out).
Run: .venv\\Scripts\\python.exe spike\\signature_threshold.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.hust_holdout import CLASS, RAW as HRAW, feats            # noqa: E402
from data.fetch_data import CWRU                                    # noqa: E402
from data.splits import NOMINAL_RPM, RAW                            # noqa: E402
from edge import profiles                                           # noqa: E402
from edge.fingerprint import load_cwru                              # noqa: E402


def cwru_scores():
    p = profiles.make("rotating-hf", bearing="SKF6205-CWRU")
    h, f = [], []
    for fid, (cls, _, load) in CWRU.items():
        x, rpm = load_cwru(RAW / "cwru" / f"{fid}.mat")
        rpm = rpm if np.isfinite(rpm) and rpm > 0 else NOMINAL_RPM[load]
        ws = p.windows(x, 12000.0)[:20]
        s = [max(p.features(w, p.analysis_fs, rpm)[15:18]) for w in ws]
        (h if cls == "normal" else f).extend(s)
    return np.array(h), np.array(f)


def hust_scores():
    h, f = [], []
    for b in "45678":
        p = profiles.make("rotating-hf", bearing=f"620{b}")
        for load in "024":
            h.extend(feats(f"N{b}0{load}", p)[0][:, 15:18].max(1))
            for k in CLASS:
                if (HRAW / f"{k}{b}0{load}.mat").exists():
                    f.extend(feats(f"{k}{b}0{load}", p)[0][:20, 15:18].max(1))
    return np.array(h), np.array(f)


def report(name, h, f):
    q = lambda a: np.percentile(a, [5, 50, 95, 99]).round(2).tolist()
    print(f"{name}: healthy n={len(h)} p5/50/95/99 {q(h)} | faulty n={len(f)} p5/50/95/99 {q(f)}")


hc, fc = cwru_scores()
report("CWRU", hc, fc)
thr = float(np.percentile(hc, 99)) + 0.25          # just above the healthy CWRU maximum region, fixed margin
print(f"threshold chosen on CWRU (healthy p99 + 0.25): {thr:.2f}; CWRU faulty windows above it: {np.mean(fc >= thr):.1%}")
hh, fh = hust_scores()
report("HUST", hh, fh)
print(f"HUST held-out with that threshold: healthy windows above {np.mean(hh >= thr):.1%}, faulty windows above {np.mean(fh >= thr):.1%}")
