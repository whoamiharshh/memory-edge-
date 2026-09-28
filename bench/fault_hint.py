"""Fault-TYPE hint on real data: physics rule vs the fleet-learned model vs both agreeing (what the device shows).

Datasets (all real): CWRU (seeded faults, 12 kHz), HUST (5 bearing types, 51.2 kHz), University of Ottawa UODS-VAFDC
(NATURALLY developed faults, 42 kHz; accelerometer AND microphone). Per recording: up to ten 1 s segments.
  physics      the shipped envelope rule (kurtogram band), majority vote over the segments
  fleet model  cloud/hint_model.py on edge/physics.order_features, trained on OTHER bearings only:
               leave-one-bearing-out (UOttawa: 20 bearings), leave-one-fault-size-out (CWRU: 7/14/21 mil are
               different physical bearings), leave-one-bearing-type-out (HUST) - the "fleet of identical machines"
               case; the model never saw the bearing it is asked about
  agree        both name the same class -> shown as "confident"; otherwise "uncertain, inspect"
Two feature scalings are compared (centred only - the shipped choice, the features are already log10 ratios on one
scale - and standardised). Plus a learning curve on UOttawa: accuracy on an unseen bearing vs how many confirmed
bearings the fleet has.
Run: .venv\\Scripts\\python.exe -m bench.fault_hint
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np
import scipy.io as sio

from cloud import hint_model
from edge import physics as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
SEG_S = 1.0


def _uottawa(col):
    raw = ROOT / "data" / "raw" / "uottawa"
    T = {"I": "inner_race", "O": "outer_race", "B": "ball", "C": "cage"}
    for fp in sorted(raw.glob("*_[12].mat")):
        d = sio.loadmat(str(fp))[fp.stem]
        yield fp.stem, d[:, col].astype(float), 42000.0, float(d[0, 2]), T[fp.stem[0]], P.BEARINGS["6203-UO"], fp.stem.split("_")[1]


def _hust():
    raw = ROOT / "data" / "raw" / "hust"
    T = {"I": "inner_race", "O": "outer_race", "B": "ball"}
    for fp in sorted(raw.glob("*.mat")):
        n = fp.stem
        if n[0] not in T or n[1] in "IOB":
            continue
        m = sio.loadmat(str(fp))
        yield (n, np.asarray(m["data"], dtype=float).ravel(), 51200.0, float(np.asarray(m["fs"]).ravel()[0]) * 60,
               T[n[0]], P.BEARINGS[f"620{n[1]}"], n[1])


def _cwru():
    from data.fetch_data import CWRU
    from data.splits import NOMINAL_RPM
    from edge.fingerprint import load_cwru
    for fid, (cls, size, load) in CWRU.items():
        if cls == "normal":
            continue
        x, rpm = load_cwru(ROOT / "data" / "raw" / "cwru" / f"{fid}.mat")
        rpm = rpm if np.isfinite(rpm) and rpm > 0 else NOMINAL_RPM[load]
        yield str(fid), x, 12000.0, rpm, cls, P.BEARINGS["SKF6205-CWRU"], str(size)


def recordings(source, with_cage):
    keys = ("bpfo", "bpfi", "bsf", "ftf") if with_cage else ("bpfo", "bpfi", "bsf")
    out = []
    for name, x, fs, rpm, truth, geo, group in source:
        n = int(SEG_S * fs)
        segs = [x[i:i + n] for i in range(0, min(len(x) - n + 1, 10 * n), n)]
        feats, votes = [], collections.Counter()
        for s in segs:
            feats.append(P.order_features(s, fs, rpm / 60.0, geo))
            sc = P.bearing_defect_scores(s, fs, rpm / 60.0, geo, band=P.kurtogram_band(s, fs))
            votes[P.DEFECT_TO_CLASS[max(keys, key=lambda k: sc[k])]] += 1
        out.append({"name": name, "truth": truth, "group": group, "X": np.asarray(feats),
                    "physics": votes.most_common(1)[0][0]})
    return out


def fit(X, y, scaling):
    from sklearn.linear_model import LogisticRegression
    m = X.mean(axis=0)
    s = X.std(axis=0) if scaling == "standardised" else np.ones(X.shape[1])
    s[s < 1e-6] = 1.0
    return LogisticRegression(max_iter=3000, C=0.5).fit((X - m) / s, y), m, s


def evaluate(R, scaling, groups=None):
    groups = groups or sorted({r["group"] for r in R})
    st = collections.Counter()
    per = collections.defaultdict(lambda: [0, 0])
    for g in groups:
        tr = [r for r in R if r["group"] != g]
        te = [r for r in R if r["group"] == g]
        X = np.concatenate([r["X"] for r in tr]); y = np.concatenate([[r["truth"]] * len(r["X"]) for r in tr])
        clf, m, s = fit(X, y, scaling)
        for r in te:
            mv = collections.Counter(clf.predict((r["X"] - m) / s)).most_common(1)[0][0]
            st["n"] += 1; st["fleet_ok"] += mv == r["truth"]; st["phys_ok"] += r["physics"] == r["truth"]
            per[r["truth"]][0] += mv == r["truth"]; per[r["truth"]][1] += 1
            if mv == r["physics"]:
                st["agree"] += 1; st["agree_ok"] += mv == r["truth"]
    n = st["n"]
    return {"recordings": n, "physics_accuracy": round(st["phys_ok"] / n, 3),
            "fleet_model_accuracy": round(st["fleet_ok"] / n, 3),
            "fleet_per_class": {k: f"{a}/{b}" for k, (a, b) in sorted(per.items())},
            "confident_share": round(st["agree"] / n, 3),
            "confident_precision": round(st["agree_ok"] / max(1, st["agree"]), 3),
            "confident_correct": f"{st['agree_ok']}/{st['agree']}"}


def learning_curve(R, scaling, sizes=(2, 4, 8, 12, 16, 19), reps=200, seed=0):
    """Accuracy on one unseen bearing when the fleet has k OTHER confirmed bearings (random subsets)."""
    rng = np.random.default_rng(seed)
    groups = sorted({r["group"] for r in R})
    out = {}
    for k in sizes:
        acc = []
        for _ in range(reps):
            g = groups[rng.integers(len(groups))]
            pool = [x for x in groups if x != g]
            chosen = set(rng.choice(pool, size=min(k, len(pool)), replace=False))
            tr = [r for r in R if r["group"] in chosen]
            if len({r["truth"] for r in tr}) < 2:
                continue
            X = np.concatenate([r["X"] for r in tr]); y = np.concatenate([[r["truth"]] * len(r["X"]) for r in tr])
            clf, m, s = fit(X, y, scaling)
            for r in (r for r in R if r["group"] == g):
                acc.append(collections.Counter(clf.predict((r["X"] - m) / s)).most_common(1)[0][0] == r["truth"])
        out[str(k)] = round(float(np.mean(acc)), 3) if acc else None
    return out


CACHE = ROOT / "data" / "raw" / "fault_hint_features.json"


def load_data() -> dict:
    """Order features + physics votes per recording (computed once, ~10 min; cached as JSON next to the raw data)."""
    if CACHE.exists():
        d = json.loads(CACHE.read_text())
        return {k: [r | {"X": np.asarray(r["X"])} for r in v] for k, v in d.items()}
    data = {"uottawa_accelerometer": recordings(_uottawa(0), True), "uottawa_microphone": recordings(_uottawa(1), True),
            "cwru": recordings(_cwru(), False), "hust": recordings(_hust(), False)}
    CACHE.write_text(json.dumps({k: [r | {"X": r["X"].tolist()} for r in v] for k, v in data.items()}))
    return data


def main():
    data = load_data()
    res = {"method": __doc__.strip().splitlines()[0], "protocol": __doc__.split("Datasets")[1].split("Run:")[0].strip(),
           "shipped_scaling": "centred", "results": {}}
    for scaling in ("centred", "standardised"):
        for name, R in data.items():
            res["results"].setdefault(name, {})[scaling] = evaluate(R, scaling)
            print(name, scaling, res["results"][name][scaling], flush=True)
    for name in ("uottawa_accelerometer", "uottawa_microphone"):
        res["results"][name]["learning_curve_centred"] = learning_curve(data[name], "centred")
        print(name, "learning curve", res["results"][name]["learning_curve_centred"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fault_hint.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
