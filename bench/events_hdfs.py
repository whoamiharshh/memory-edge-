"""The `events` profile (kiosks / apps / vehicle codes: "which codes, how many, per bucket") on REAL labelled logs.

Data: Loghub HDFS_v1 (CC BY 4.0, DOI 10.5281/zenodo.8196385; labels from Xu et al., SOSP 2009): 575,061 sessions
(block ids), each a count of 29 event templates, labelled Success / Fail. Each session is one "bucket" of codes for
the Events profile, exactly as a kiosk's error codes per hour would be.
Protocol (seed 7, nothing tuned on the test sessions): the device's own pipeline - Events.features -> Baseline.fit ->
gate.calibrate (as shipped for events: radius at a 1 % false-alarm target, set from
healthy data only) - learns "normal" from 5,000 Success sessions; tested on 50,000 other
Success sessions (false alarms) and ALL Fail sessions not used anywhere (detection). A session is flagged when its
distance to the nearest healthy baseline point exceeds tau_normal (the gate's "not normal" decision).
Reference points from the Loghub/loglizer benchmark (He et al., ISSRE 2016, HDFS, unsupervised): PCA F1 0.79,
Invariant Mining F1 0.91, Log Clustering F1 0.80 - different splits, shown for scale only.
Run: .venv\\Scripts\\python.exe -m bench.events_hdfs
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from edge import profiles
from edge.fingerprint import Baseline
from edge.gate import calibrate

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "loghub" / "preprocessed" / "Event_occurrence_matrix.csv"
OUT = ROOT / "bench" / "results"


def nearest(A: np.ndarray, B: np.ndarray, chunk: int = 2000) -> np.ndarray:
    out = []
    bb = (B ** 2).sum(1)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        d2 = (a ** 2).sum(1)[:, None] + bb[None, :] - 2 * a @ B.T
        out.append(np.sqrt(np.maximum(d2.min(1), 0)))
    return np.concatenate(out)


def main() -> dict:
    df = pd.read_csv(CSV)
    ev = [c for c in df.columns if c.startswith("E")]
    rng = np.random.default_rng(7)
    ok = df[df["Label"] == "Success"].sample(frac=1.0, random_state=7)
    fail = df[df["Label"] == "Fail"]
    prof = profiles.make("events")
    feat = lambda rows: np.stack([prof.features({"codes": {c: int(n) for c, n in zip(ev, r) if n}}) for r in rows])
    base_raw = feat(ok[ev].values[:5000])
    bl = Baseline.fit(base_raw, prof.fp_version, prof.min_std)
    zb = bl.z(base_raw)
    g = calibrate(zb, normal_factor=prof.normal_factor, target_false_alarm=prof.target_false_alarm)
    zn = bl.z(feat(ok[ev].values[5000:55000]))
    zf = bl.z(feat(fail[ev].values))
    fa = nearest(zn, zb) > g.tau_normal + 1e-6          # expansion formula: identical rows give ~1e-7, not 0
    det = nearest(zf, zb) > g.tau_normal + 1e-6
    tp, fp, fn = int(det.sum()), int(fa.sum()), int((~det).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res = {"method": __doc__.strip().splitlines()[0], "protocol": __doc__.split("Protocol")[1].split("Run:")[0].strip(),
           "baseline_sessions": 5000, "tau_normal": round(g.tau_normal, 3),
           "normal_sessions_tested": len(zn), "false_alarms": fp, "false_alarm_rate": round(fp / len(zn), 4),
           "fail_sessions": len(zf), "detected": tp, "recall": round(rec, 4),
           "precision_at_dataset_ratio": round(prec, 4), "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
           "note": "precision uses the tested sessions (50,000 normal : all failed); in HDFS ~3 % of sessions fail"}
    print({k: v for k, v in res.items() if k not in ("method", "protocol")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "events_hdfs.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
