"""Does choosing the demodulation band by spectral kurtosis (a simple kurtogram) fix the weak outer-race hint?

Per recording, majority vote of the envelope rule over the first 10 windows, CWRU fault files (tuning data, SKF 6205,
bearing-level accuracy per class) and HUST single faults (held out, 5 geometries):
  fixed      the shipped 2-5.5 kHz band
  kurtogram  per window, the band (from a fixed grid of 1/3-octave-ish bands between 1 kHz and 0.45 fs) whose
             band-passed signal has the highest kurtosis, then the same envelope rule
Adopt only if CWRU improves AND HUST does not get worse.
Run: .venv\\Scripts\\python.exe spike\\kurtogram.py
"""
import collections
import pathlib
import sys

import numpy as np
import scipy.signal as ss
import scipy.stats as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench.hust_holdout import CLASS, FS as HFS, RAW as HRAW              # noqa: E402
from data.fetch_data import CWRU                                          # noqa: E402
from data.splits import NOMINAL_RPM, RAW                                  # noqa: E402
from edge import physics as P                                             # noqa: E402
from edge import profiles                                                 # noqa: E402
from edge.fingerprint import load_cwru                                    # noqa: E402

import scipy.io as sio                                                    # noqa: E402


def bands(fs):
    edges = [1000.0]
    while edges[-1] * 1.4 < 0.45 * fs:
        edges.append(edges[-1] * 1.4)
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])] + [(a, c) for a, _, c in zip(edges[:-2], edges[1:-1], edges[2:])]


def best_band(x, fs):
    best, bk = None, -1.0
    for lo, hi in bands(fs):
        y = ss.sosfiltfilt(ss.butter(4, (lo, hi), btype="bandpass", fs=fs, output="sos"), x)
        k = float(st.kurtosis(y, fisher=True))
        if k > bk:
            best, bk = (lo, hi), k
    return best


def scores(x, fs, shaft, geo, band):
    f, p = P.envelope_spectrum(x, fs, band)
    ref = float(np.median(p[(f > 10) & (f < 1000)])) + 1e-12
    return {n: np.log10(sum(P.peak_near(f, p, h * o * shaft) for h in (1, 2, 3)) / ref + 1e-12)
            for n, o in geo.orders().items()}


def vote(ws, fs, shaft, geo, mode):
    v = collections.Counter()
    for w in ws[:10]:
        band = (2000.0, 5500.0) if mode == "fixed" else best_band(w, fs)
        v[P.bearing_rule(scores(w, fs, shaft, geo, band))["fault_class"]] += 1
    return v.most_common(1)[0][0]


def cwru(mode):
    p, per = profiles.make("rotating-hf", bearing="SKF6205-CWRU"), collections.defaultdict(list)
    for fid, (cls, _, load) in CWRU.items():
        if cls == "normal":
            continue
        x, rpm = load_cwru(RAW / "cwru" / f"{fid}.mat")
        rpm = rpm if np.isfinite(rpm) and rpm > 0 else NOMINAL_RPM[load]
        per[cls].append(vote(p.windows(x, 12000.0), p.analysis_fs, rpm / 60, p.geometry, mode) == cls)
    return {c: round(float(np.mean(v)), 3) for c, v in per.items()} | {"overall": round(float(np.mean(sum(per.values(), []))), 3)}


def hust(mode):
    per = collections.defaultdict(list)
    for b in "45678":
        p = profiles.make("rotating-hf", bearing=f"620{b}")
        for k, cls in CLASS.items():
            for load in "024":
                f = HRAW / f"{k}{b}0{load}.mat"
                if f.exists():
                    m = sio.loadmat(str(f))
                    x, shaft = np.asarray(m["data"]).ravel(), float(np.asarray(m["fs"]).ravel()[0])
                    per[cls].append(vote(p.windows(x, HFS), p.analysis_fs, shaft, p.geometry, mode) == cls)
    return {c: round(float(np.mean(v)), 3) for c, v in per.items()} | {"overall": round(float(np.mean(sum(per.values(), []))), 3)}


for mode in ("fixed", "kurtogram"):
    print(mode, "| CWRU (tuning):", cwru(mode), "| HUST (held out):", hust(mode), flush=True)
