"""Vehicles on REAL data: SCANIA Component X (Scania CV AB, CC BY 4.0, DOI 10.5878/jvb5-d390).

Real trucks; cumulative counters + histograms of one anonymised engine component, read out over the trucks' life;
labels from workshop repair records: for each truck's LAST readout, class 0 = more than 48 time steps before a
repair of Component X (or none), 1 = 24-48, 2 = 12-24, 3 = 6-12, 4 = 0-6 steps before the repair.
One window = the increment between two consecutive readouts, fingerprinted by the `telemetry` profile.

A. Memory engine, unsupervised, per truck (the device's own method): the truck's first BASE increments are its
   healthy baseline (shipped gate calibration, no tuning); is its LAST increment flagged abnormal? Compared between
   trucks close to a repair and trucks far from one. Trucks with fewer than BASE + 1 increments are skipped.
   For speed the gate maths (edge.fingerprint.Baseline + edge.gate.calibrate + nearest neighbour) runs in numpy;
   SPOT trucks are also run through the real Device (Qdrant Edge) to show the decisions are identical.
B. Trained early-warning model: TRAINED ON THE VALIDATION SPLIT, TESTED ON THE TEST SPLIT (different trucks; both
   real, both labelled by Scania). Target: repair within 48 time steps (class >= 1). Features per truck: the last
   increment's fingerprint, its deviation from the truck's own early baseline, the distance to that baseline, age
   and number of readouts. Models: logistic regression (transparent; coefficients can ship as JSON) and gradient
   boosting; plus an age-only reference model to show what the readouts add. The alert threshold is chosen on the
   TRAINING split (flag the riskiest 10 %) and then applied unchanged to the test split.
Run: .venv\\Scripts\\python.exe -m bench.vehicle_scania
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edge import profiles
from edge.fingerprint import Baseline
from edge.gate import calibrate

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "scania"
OUT = ROOT / "bench" / "results"
COUNTERS = ["171_0", "666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]
HISTS = {"167": 10, "272": 10, "291": 11, "158": 10, "459": 20, "397": 36}
BASE, SPOT, ALERT_SHARE = 10, 20, 0.10
PROF = profiles.make("telemetry", counters=COUNTERS, histograms=list(HISTS))


def windows(g: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Fingerprints of the increments between consecutive readouts of one truck (+ its last time_step)."""
    g = g.sort_values("time_step")
    v = g.drop(columns=["vehicle_id"]).to_numpy(dtype=np.float64)
    cols = list(g.columns.drop("vehicle_id"))
    ti = cols.index("time_step")
    ci = [cols.index(c) for c in COUNTERS]
    hi = {h: [cols.index(f"{h}_{k}") for k in range(n)] for h, n in HISTS.items()}
    feats = []
    for a, b in zip(v[:-1], v[1:]):
        d = np.nan_to_num(b - a, nan=0.0)
        x = {"dt": d[ti], "counters": {c: d[i] for c, i in zip(COUNTERS, ci)},
             "histograms": {h: d[idx].tolist() for h, idx in hi.items()}}
        feats.append(PROF.features(x))
    return np.asarray(feats), float(v[-1, ti])


def load(split: str) -> list[dict]:
    df = pd.read_csv(RAW / f"{split}_operational_readouts.csv")
    lab = pd.read_csv(RAW / f"{split}_labels.csv").set_index("vehicle_id")["class_label"]
    trucks = []
    for vid, g in df.groupby("vehicle_id"):
        W, age = windows(g)
        trucks.append({"id": int(vid), "label": int(lab[vid]), "W": W, "age": age, "n": len(g)})
    return trucks


def gate_last(W: np.ndarray) -> tuple[bool, float]:
    """The device's gate, numpy form: baseline on the first BASE increments, is the last increment abnormal?"""
    b = Baseline.fit(W[:BASE], PROF.fp_version, PROF.min_std)
    zb, zl = b.z(W[:BASE]), b.z(W[-1])
    tau = calibrate(zb).tau_normal
    d = float(np.sqrt(((zb - zl) ** 2).sum(1)).min())
    return d > tau, d / tau


def part_a(trucks: list[dict]) -> dict:
    groups = {"far (class 0)": [0], "within 48 steps (classes 1-4)": [1, 2, 3, 4], "within 6 steps (class 4)": [4]}
    elig = [t for t in trucks if len(t["W"]) >= BASE + 1]
    res = {}
    for name, cls in groups.items():
        flags = [gate_last(t["W"])[0] for t in elig if t["label"] in cls]
        res[name] = {"trucks": len(flags), "last_readout_flagged": round(float(np.mean(flags)), 3) if flags else None}
    # spot check: the real Device (Qdrant Edge shard, journal, gate) must decide exactly like the numpy maths
    from edge.device import Device, DeviceConfig
    from shared.embed import HashEmbedder
    same, spot = 0, [t for t in elig if t["label"] >= 1][:SPOT // 2] + [t for t in elig if t["label"] == 0][:SPOT // 2]
    for t in spot:
        root = pathlib.Path(tempfile.mkdtemp(prefix="truck_"))
        dev = Device(DeviceConfig("truck", "s", str(t["id"]), root, profile="telemetry",
                                  profile_params={"counters": COUNTERS, "histograms": list(HISTS)}), HashEmbedder())
        try:
            dev.fit_baseline(t["W"][:BASE])
            same += (dev.ingest_window(t["W"][-1])["state"] != "normal") == gate_last(t["W"])[0]
        finally:
            dev.close()
            shutil.rmtree(root, ignore_errors=True)
    return {"eligible_trucks": len(elig), "skipped_too_few_readouts": len(trucks) - len(elig), "groups": res,
            "device_spot_check_identical": f"{same}/{len(spot)}"}


def truck_features(t: dict) -> np.ndarray:
    W = t["W"]
    last = W[-1] if len(W) else np.zeros(27)
    if len(W) >= 6:
        k = min(BASE, len(W) - 1)
        b = Baseline.fit(W[:k], PROF.fp_version, PROF.min_std)
        z = np.clip(b.z(last), -20, 20)
        dist = float(np.sqrt((z ** 2).sum()))
    else:
        z, dist = np.zeros(27), 0.0
    return np.concatenate([last, z, [dist, np.log1p(t["age"]), np.log1p(t["n"]), float(len(W) >= 6)]])


def part_b(train: list[dict], test: list[dict]) -> dict:
    Xtr, Xte = np.stack([truck_features(t) for t in train]), np.stack([truck_features(t) for t in test])
    ytr, yte = np.array([t["label"] >= 1 for t in train]), np.array([t["label"] >= 1 for t in test])
    age_col = 27 + 27 + 1
    models = {
        "age_only_reference": (make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)), [age_col]),
        "logistic_regression": (make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.1, max_iter=5000)), None),
        "gradient_boosting": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                                             random_state=0), None)}
    out = {"train": {"trucks": len(train), "positives": int(ytr.sum())}, "test": {"trucks": len(test), "positives": int(yte.sum())}}
    for name, (m, cols) in models.items():
        sel = (lambda X: X[:, cols]) if cols else (lambda X: X)
        m.fit(sel(Xtr), ytr)
        ptr, pte = m.predict_proba(sel(Xtr))[:, 1], m.predict_proba(sel(Xte))[:, 1]
        thr = float(np.quantile(ptr, 1 - ALERT_SHARE))                 # chosen on TRAINING scores only
        alert = pte >= thr
        c4 = np.array([t["label"] == 4 for t in test])
        out[name] = {"test_roc_auc": round(float(roc_auc_score(yte, pte)), 3),
                     "test_pr_auc": round(float(average_precision_score(yte, pte)), 3),
                     "test_alert_rate": round(float(alert.mean()), 3),
                     "test_recall_repair_within_48": round(float(alert[yte].mean()), 3),
                     "test_recall_repair_within_6": round(float(alert[c4].mean()), 3),
                     "test_precision": round(float(yte[alert].mean()), 3) if alert.any() else None,
                     "base_rate": round(float(yte.mean()), 3)}
        if name == "logistic_regression":
            lr = m.named_steps["logisticregression"]
            sc = m.named_steps["standardscaler"]
            out["shipped_model"] = {"kind": "logistic regression (standardised features)", "threshold": thr,
                                    "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
                                    "coef": lr.coef_[0].round(6).tolist(), "intercept": round(float(lr.intercept_[0]), 6)}
    return out


def main() -> dict:
    val, test = load("validation"), load("test")
    out = {"method": __doc__.strip().splitlines()[0], "dataset": "SCANIA Component X v3, CC BY 4.0",
           "a_memory_engine_per_truck": part_a(test), "b_trained_model": part_b(val, test)}
    shipped = out["b_trained_model"].pop("shipped_model")
    (ROOT / "knowledge").mkdir(exist_ok=True)
    (ROOT / "knowledge" / "vehicle_risk_model.json").write_text(json.dumps(
        {"about": "Early-warning risk hint for SCANIA-style telemetry. Trained on REAL data only: SCANIA Component X "
                  "validation split (Scania CV AB, CC BY 4.0); tested on its test split (bench/results/vehicle_scania.json). "
                  "A hint shown with its measured recall/precision, never an automatic decision.",
         "features": "bench.vehicle_scania.truck_features", "target": "repair of Component X within 48 time steps",
         **shipped}, indent=1))
    OUT.mkdir(exist_ok=True)
    (OUT / "vehicle_scania.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
