"""Vehicles, bigger REAL training set: the SCANIA Component X TRAINING split (time-to-event labels) instead of only the
5,046 validation trucks.

Training examples (all real): for each training truck, up to 3 random cut points in its life (plus up to 2 more inside
the last 48 time steps of trucks that were repaired). Label at a cut = "Component X repaired within 48 time steps"
(repaired AND study end - cut <= 48). Cuts of NOT-repaired trucks within 48 steps of the study end are censored (the
outcome is unknown) and dropped - never guessed. Features at a cut: bench.vehicle_scania.truck_features on the
readouts up to that cut (the same features as the shipped model).
Data note: the training readouts downloaded up to 1,182,793,728 bytes before the SND server began refusing requests
(HTTP 401); the file is sorted by vehicle id, so the first 22,982 of 23,550 trucks are complete - the last, cut-off
truck is dropped (22,981 used). Vehicle ids are anonymised, so the missing 2.4 % should not bias the result.
Model choice: 5-fold GROUP cross-validation over training TRUCKS (a truck's cuts never straddle folds) - logistic
regression vs gradient boosting. The winner is then trained on all training cuts and evaluated ONCE on the test trucks
(each with Scania's own class label of its last readout; target class >= 1), with the alert threshold fixed on the
training scores (riskiest 10 %), exactly like bench/vehicle_scania.py, and compared with that earlier model.
Run: .venv\\Scripts\\python.exe -m bench.vehicle_scania_train
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bench.vehicle_scania import ALERT_SHARE, RAW, load, truck_features, windows

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
CACHE = RAW / "train_cuts_features.npz"
HORIZON = 48.0


def train_cuts(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if CACHE.exists():
        z = np.load(CACHE)
        return z["X"], z["y"], z["g"]
    rng = np.random.default_rng(seed)
    tte = pd.read_csv(RAW / "train_tte.csv").set_index("vehicle_id")
    df = pd.read_csv(RAW / "train_operational_readouts_partial.csv", on_bad_lines="skip")
    last = df["vehicle_id"].iloc[-1]
    df = df[df["vehicle_id"] != last]                        # the cut-off truck at the end of the partial file
    X, y, g = [], [], []
    for vid, grp in df.groupby("vehicle_id"):
        grp = grp.sort_values("time_step")
        W, _ = windows(grp)
        ts = grp["time_step"].to_numpy()
        T, rep = float(tte.loc[vid, "length_of_study_time_step"]), bool(tte.loc[vid, "in_study_repair"])
        idx = np.arange(7, len(ts))                         # >= 6 increments before the cut (baseline)
        if not len(idx):
            continue
        ttr = T - ts[idx]
        lab = np.where(rep & (ttr <= HORIZON), 1, np.where(ttr > HORIZON, 0, -1))   # -1 censored
        ok = idx[lab >= 0]
        if not len(ok):
            continue
        pick = list(rng.choice(ok, size=min(3, len(ok)), replace=False))
        pos = idx[lab == 1]
        if len(pos):
            pick += list(rng.choice(pos, size=min(2, len(pos)), replace=False))
        for j in sorted(set(pick)):
            t = {"W": W[:j], "age": float(ts[j]), "n": int(j + 1)}
            X.append(truck_features(t))
            y.append(int(rep and T - ts[j] <= HORIZON))
            g.append(int(vid))
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
    np.savez_compressed(CACHE, X=X, y=y, g=g)
    return X, y, g


MODELS = {
    "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.1,
                                                                                      max_iter=5000)),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                                l2_regularization=1.0, class_weight="balanced",
                                                                random_state=0),
}


def main() -> dict:
    X, y, g = train_cuts()
    out = {"method": __doc__.strip().splitlines()[0],
           "protocol": __doc__.split("Training examples")[1].split("Run:")[0].strip(),
           "train_cuts": {"examples": int(len(y)), "trucks": int(len(set(g.tolist()))), "positives": int(y.sum())},
           "cv_over_trucks": {}}
    for name, mk in MODELS.items():
        aucs = []
        for tr, te in GroupKFold(5).split(X, y, g):
            m = mk().fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        out["cv_over_trucks"][name] = {"roc_auc_mean": round(float(np.mean(aucs)), 3),
                                       "roc_auc_folds": [round(float(a), 3) for a in aucs]}
        print(name, out["cv_over_trucks"][name], flush=True)
    best = max(out["cv_over_trucks"], key=lambda k: out["cv_over_trucks"][k]["roc_auc_mean"])
    out["chosen_by_cv"] = best
    test = load("test")
    Xte = np.stack([truck_features(t) for t in test])
    yte = np.array([t["label"] >= 1 for t in test])
    c4 = np.array([t["label"] == 4 for t in test])
    out["test"] = {}
    for name, mk in MODELS.items():                          # both reported; the CV choice was made before this
        m = mk().fit(X, y)
        ptr, pte = m.predict_proba(X)[:, 1], m.predict_proba(Xte)[:, 1]
        thr = float(np.quantile(ptr, 1 - ALERT_SHARE))
        alert = pte >= thr
        out["test"][name] = {"roc_auc": round(float(roc_auc_score(yte, pte)), 3),
                             "pr_auc": round(float(average_precision_score(yte, pte)), 3),
                             "alert_rate": round(float(alert.mean()), 3),
                             "recall_repair_within_48": round(float(alert[yte].mean()), 3),
                             "recall_repair_within_6": round(float(alert[c4].mean()), 3),
                             "precision": round(float(yte[alert].mean()), 3) if alert.any() else None,
                             "base_rate": round(float(yte.mean()), 3)}
        print("test", name, out["test"][name], flush=True)
        if name == best:
            out["_model"] = m
    prev = json.loads((OUT / "vehicle_scania.json").read_text())["b_trained_model"]
    out["previous_model_trained_on_validation_only"] = {k: prev[k] for k in ("logistic_regression", "gradient_boosting")}
    model = out.pop("_model")
    OUT.mkdir(exist_ok=True)
    (OUT / "vehicle_scania_train.json").write_text(json.dumps(out, indent=2))
    return out | {"_model": model}


if __name__ == "__main__":
    main()
