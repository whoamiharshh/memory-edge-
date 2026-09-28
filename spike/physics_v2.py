"""Spike: collision-aware bearing defect scoring (v2) vs the shipped v1, on CWRU, HUST and UOttawa (all real).

v2 changes, each from a failure seen in spike/uottawa_physics.py:
  1. shaft speed refined from the signal: the strongest envelope/raw peak within +-4 % of the nominal speed
     (the UOttawa speed values are 1-3 % off the real 1x line);
  2. a defect harmonic that lies within a guard of an integer shaft harmonic or a mains harmonic (multiples of the
     line frequency) is SKIPPED - there its energy cannot be told apart from looseness/wear or electrical hum;
  3. each peak is compared with its LOCAL floor (median envelope power within +-15 %), not the global median;
  4. score = mean of the two strongest usable harmonics (h = 1..5).
Run: .venv\\Scripts\\python.exe -m spike.physics_v2
"""
from __future__ import annotations

import collections
import math
import pathlib

import numpy as np
import scipy.io as sio

from edge import physics as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
_EPS = 1e-30


def refine_shaft(x, fs, nominal, span=0.04):
    f, a = P.spectrum(x, fs)
    fe, pe = P.envelope_spectrum(x, fs, None)
    best, bv = nominal, 0.0
    for ff, aa in ((f, a), (fe, np.sqrt(pe))):
        sel = (ff >= nominal * (1 - span)) & (ff <= nominal * (1 + span))
        if not sel.any():
            continue
        i = int(np.argmax(aa[sel]))
        loc = (ff >= nominal * 0.7) & (ff <= nominal * 1.3)
        prom = aa[sel][i] / (np.median(aa[loc]) + _EPS)
        if prom > bv:
            best, bv = float(ff[sel][i]), prom
    return best if bv >= 3.0 else nominal


def scores_v2(x, fs, shaft, geo, line_hz=None, band=None, H=5):
    f, p = P.envelope_spectrum(x, fs, band)
    df = f[1] - f[0]
    out, used = {}, {}
    for name, order in geo.orders().items():
        vals = []
        for h in range(1, H + 1):
            fc = h * order * shaft
            if fc > min(1000.0, f[-1] * 0.9):
                break
            guard = max(2.5 * df, 0.006 * fc)
            k = round(fc / shaft)
            if k >= 1 and abs(fc - k * shaft) < guard:
                continue
            if line_hz:
                m = round(fc / line_hz)
                if m >= 1 and abs(fc - m * line_hz) < guard:
                    continue
            w = max(1.0 * df, 0.004 * fc)
            pk = p[(f >= fc - w) & (f <= fc + w)]
            loc = (f >= fc * 0.85) & (f <= fc * 1.15) & (np.abs(f - fc) > 2 * w)
            if not pk.size or not loc.any():
                continue
            vals.append(math.log10(pk.max() / (np.median(p[loc]) + _EPS) + 1e-12))
        vals.sort(reverse=True)
        out[name] = float(np.mean(vals[:2])) if vals else float("nan")
        used[name] = len(vals)
    return out, used


def classify(s, keys=("bpfo", "bpfi", "bsf", "ftf")):
    c = {k: s[k] for k in keys if not math.isnan(s[k])}
    if not c:
        return None, 0.0, 0.0
    order = sorted(c, key=c.get, reverse=True)
    best = order[0]
    margin = c[best] - (c[order[1]] if len(order) > 1 else 0.0)
    return P.DEFECT_TO_CLASS[best], c[best], margin


# ---------------------------------------------------------------- datasets (recording -> segments, fs, rpm, truth)
def uottawa():
    raw = ROOT / "data" / "raw" / "uottawa"
    T = {"I": "inner_race", "O": "outer_race", "B": "ball", "C": "cage", "H": "healthy"}
    geo = P.BearingGeometry(8, 6.77, 28.50, 0.0, "NSK 6203")
    for fp in sorted(raw.glob("*.mat")):
        m = sio.loadmat(str(fp))
        d = m[fp.stem]
        truth = "healthy" if fp.stem.endswith("_0") else T[fp.stem[0]]
        for ch, col in (("acc", 0), ("mic", 1)):
            yield f"uottawa-{ch}", fp.stem, d[:, col].astype(float), 42000.0, float(d[0, 2]), truth, geo, 60.0


def hust():
    raw = ROOT / "data" / "raw" / "hust"
    T = {"I": "inner_race", "O": "outer_race", "B": "ball", "N": "healthy"}
    for fp in sorted(raw.glob("*.mat")):
        n = fp.stem
        if n[0] not in T or n[1] in "IOB":
            continue
        m = sio.loadmat(str(fp))
        x, shaft = np.asarray(m["data"], dtype=float).ravel(), float(np.asarray(m["fs"]).ravel()[0])
        yield "hust", n, x, 51200.0, shaft * 60, T[n[0]], P.BEARINGS[f"620{n[1]}"], 50.0


def cwru():
    from data.fetch_data import CWRU
    from data.splits import NOMINAL_RPM
    from edge.fingerprint import load_cwru
    T = {"normal": "healthy"}
    for fid, (cls, _, load) in CWRU.items():
        x, rpm = load_cwru(ROOT / "data" / "raw" / "cwru" / f"{fid}.mat")
        rpm = rpm if np.isfinite(rpm) and rpm > 0 else NOMINAL_RPM[load]
        yield "cwru", str(fid), x, 12000.0, rpm, T.get(cls, cls), P.BEARINGS["SKF6205-CWRU"], 60.0


def evaluate(source, seg_s=2.0, line=True, refine=True, v1=False):
    res = collections.defaultdict(list)
    for ds, name, x, fs, rpm, truth, geo, line_hz in source:
        n = int(seg_s * fs)
        segs = [x[i:i + n] for i in range(0, len(x) - n + 1, n)][:5]
        votes, strength, margins = collections.Counter(), [], []
        for s in segs:
            shaft = refine_shaft(s, fs, rpm / 60.0) if refine else rpm / 60.0
            band = P.kurtogram_band(s, fs)
            if v1:
                sc = P.bearing_defect_scores(s, fs, shaft, geo, band=band)
            else:
                sc, _ = scores_v2(s, fs, shaft, geo, line_hz if line else None, band)
            c, st, mg = classify(sc, ("bpfo", "bpfi", "bsf", "ftf") if ds.startswith("uottawa") else ("bpfo", "bpfi", "bsf"))
            votes[c] += 1
            strength.append(st)
            margins.append(mg)
        pred = votes.most_common(1)[0][0]
        res[ds].append((name, truth, pred, float(np.median(strength)), float(np.median(margins))))
    return res


def report(res, label):
    for ds, rows in res.items():
        faulty = [r for r in rows if r[1] != "healthy"]
        ok = sum(r[1] == r[2] for r in faulty)
        per = collections.defaultdict(lambda: [0, 0])
        for r in faulty:
            per[r[1]][0] += r[1] == r[2]
            per[r[1]][1] += 1
        h = [r[3] for r in rows if r[1] == "healthy"]
        fstr = [r[3] for r in faulty]
        print(f"{label:18s} {ds:12s} class acc {ok}/{len(faulty)} = {ok / max(1, len(faulty)):.2f} "
              f"{dict((k, f'{a}/{b}') for k, (a, b) in per.items())} | strength healthy med {np.median(h) if h else float('nan'):.2f} "
              f"faulty med {np.median(fstr):.2f}")


if __name__ == "__main__":
    for label, kw in (("v1 (shipped)", dict(v1=True, refine=False)), ("v2 no-line", dict(line=False)),
                      ("v2", {})):
        report(evaluate(uottawa(), **kw), label)
        report(evaluate(hust(), **kw), label)
        report(evaluate(cwru(), **kw), label)
