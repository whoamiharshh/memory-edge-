"""Held-out "second machine" test (answers the data-realism weakness): the pipeline, with every threshold exactly as
shipped (chosen on CWRU), runs on a DIFFERENT dataset it has never seen: HUST bearing (CC BY 4.0, DOI
10.17632/cbv7jyx4p9.3), another test rig, another sensor (PCB 325C33), 51.2 kHz, and FIVE bearing types.

Profile `rotating-hf` (geometry-derived defect frequencies), shaft speed from each file's `fs` variable (Hz).
Each bearing type (6204..6208) is its own machine with its own healthy baseline:
  baseline          N<b>0 + N<b>2 (healthy, 0 W and 200 W)
  unseen healthy    N<b>4 (400 W): gate false alarms
  single faults     I / O / B at 0, 200, 400 W (no B for 6204 in the dataset): per recording, a fresh device
      detection         first 20 windows flagged abnormal; episodes opened in them (ideal 1)
      K3 fixed          20 fault windows -> action -> unseen healthy N<b>4 -> "symptom resolved"?
      K3 still faulty   the fault continues after the action -> "symptom persists"? (needs >= 25 more windows)
      physics hint      device hint after the first windows (majority vote, as shipped) vs the true class;
                        and the pure envelope rule (argmax of BPFO/BPFI/BSF), forced to a bearing class
  combined faults   IB / IO / OB: detection only (no single true class)
Run: .venv\\Scripts\\python.exe -m bench.hust_holdout
"""
from __future__ import annotations

import collections
import json
import pathlib
import shutil
import tempfile

import numpy as np
import scipy.io as sio

from edge import physics as P
from edge import profiles
from edge.device import Device, DeviceConfig
from edge.verifier import REQUIRED_WINDOWS
from shared.embed import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hust"
OUT = ROOT / "bench" / "results"
FS = 51200.0
CLASS = {"I": "inner_race", "O": "outer_race", "B": "ball"}
_cache: dict[tuple, tuple[np.ndarray, float]] = {}


def feats(name: str, prof) -> tuple[np.ndarray, float]:
    key = (name, prof.params["bearing"])
    if key not in _cache:
        m = sio.loadmat(str(RAW / f"{name}.mat"))
        x, shaft = np.asarray(m["data"], dtype=np.float64).ravel(), float(np.asarray(m["fs"]).ravel()[0])
        ws = prof.windows(x, FS)
        _cache[key] = (np.stack([prof.features(w, prof.analysis_fs, shaft * 60) for w in ws]), shaft)
    return _cache[key]


def envelope_rule_votes(name: str, prof, n: int = 10) -> str:
    m = sio.loadmat(str(RAW / f"{name}.mat"))
    x, shaft = np.asarray(m["data"], dtype=np.float64).ravel(), float(np.asarray(m["fs"]).ravel()[0])
    votes = collections.Counter(P.bearing_rule(P.bearing_defect_scores(w, prof.analysis_fs, shaft, prof.geometry))["fault_class"]
                                for w in prof.windows(x, FS)[:n])
    return votes.most_common(1)[0][0]


def fresh(prof, base: np.ndarray) -> tuple[Device, pathlib.Path]:
    root = pathlib.Path(tempfile.mkdtemp(prefix="hust_"))
    dev = Device(DeviceConfig("bench", "s", "m", root, profile="rotating-hf", profile_params=dict(prof.params)),
                 HashEmbedder())
    dev.fit_baseline(base)
    return dev, root


TEACH = 10     # windows of the untaught healthy state the technician sees before saying "normal operation"


def teach(dev: Device, unseen: np.ndarray) -> int:
    """The technician's one confirmation: the first TEACH windows of the untaught healthy state open an episode;
    it is marked 'normal operation'. Returns how many baseline points were added."""
    for w in unseen[:TEACH]:
        dev.ingest_window(w)
    eps = [e for e in dev.episodes() if e["status"] != "closed"]
    added = 0
    for e in eps:
        dev.mark_normal(e["episode_id"])
        added += dev.episode(e["episode_id"])["dismissed"]["baseline_points_added"]
    for w in unseen[:1]:                  # end the abnormal run cleanly
        dev.ingest_window(w)
    return added


def verify_after_fix(dev: Device, W: np.ndarray, healthy: np.ndarray) -> str:
    for w in W[:20]:
        dev.ingest_window(w)
    ep = max((e for e in dev.episodes() if not e.get("dismissed")), key=lambda e: e["occurrences"])
    dev.record_action(ep["episode_id"], "replace_bearing")
    verdict = "verifying"
    for w in np.concatenate([healthy, healthy])[:REQUIRED_WINDOWS * 2]:
        dev.ingest_window(w)
        verdict = dev.episode(ep["episode_id"])["verify"]["verdict"]
        if verdict != "verifying":
            break
    return verdict


def main() -> dict:
    rows, healthy_rows, combo_rows = [], [], []
    for b in "45678":
        prof = profiles.make("rotating-hf", bearing=f"620{b}")
        base = np.concatenate([feats(f"N{b}00", prof)[0], feats(f"N{b}02", prof)[0]])
        unseen, _ = feats(f"N{b}04", prof)
        dev, root = fresh(prof, base)
        try:
            states = [dev.ingest_window(w)["state"] for w in unseen]
        finally:
            dev.close()
            shutil.rmtree(root, ignore_errors=True)
        dev, root = fresh(prof, base)
        try:
            added = teach(dev, unseen)
            after = [dev.ingest_window(w)["state"] for w in unseen[TEACH:]]
        finally:
            dev.close()
            shutil.rmtree(root, ignore_errors=True)
        healthy_rows.append({"bearing": f"620{b}", "windows": len(states), "false_alarms": sum(s != "normal" for s in states),
                             "after_teach": {"baseline_points_added": added, "windows": len(after),
                                             "false_alarms": sum(s != "normal" for s in after)}})
        for kind in ("I", "O", "B", "IB", "IO", "OB"):
            for load in "024":
                name = f"{kind}{b}0{load}"
                if not (RAW / f"{name}.mat").exists():
                    continue
                W, shaft = feats(name, prof)
                dev, root = fresh(prof, base)
                try:
                    st = [dev.ingest_window(w)["state"] for w in W[:20]]
                    row = {"file": name, "bearing": f"620{b}", "kind": kind, "load_w": int(load) * 100,
                           "shaft_hz": round(shaft, 2), "windows": len(W),
                           "detected": round(sum(s != "normal" for s in st) / len(st), 3),
                           "episodes_first_20": sum(s == "new" for s in st)}
                    if kind in CLASS:
                        ep = max(dev.episodes(), key=lambda e: e["occurrences"])
                        row["true"] = CLASS[kind]
                        row["device_hint"] = ep["fault_hint"]["fault_class"]
                        row["envelope_rule"] = envelope_rule_votes(name, prof)
                        dev.record_action(ep["episode_id"], "replace_bearing")
                        verdict = None
                        for w in np.concatenate([unseen, unseen])[:REQUIRED_WINDOWS * 2]:
                            dev.ingest_window(w)
                            verdict = dev.episode(ep["episode_id"])["verify"]["verdict"]
                            if verdict != "verifying":
                                break
                        row["k3_fixed"] = verdict
                        rest = W[20:]
                        if len(rest) >= 5 + REQUIRED_WINDOWS:
                            dev.confirm_outcome(ep["episode_id"], "worked")
                            for w in rest[:5]:
                                dev.ingest_window(w)
                            eps = [e for e in dev.episodes() if e["episode_id"] != ep["episode_id"]
                                   and e["status"] != "closed"]
                            if not eps:                    # the continuing fault merged into the first episode
                                row["k3_still_faulty_note"] = "no separate episode opened for the continuing fault"
                            else:
                                ep2 = max(eps, key=lambda e: e["occurrences"])["episode_id"]
                                dev.record_action(ep2, "lubricate")
                                for w in rest[5:5 + REQUIRED_WINDOWS]:
                                    dev.ingest_window(w)
                                row["k3_still_faulty"] = dev.episode(ep2)["verify"]["verdict"]
                        dev2, root2 = fresh(prof, base)      # same recording, after the technician taught 400 W
                        try:
                            teach(dev2, unseen)
                            row["k3_fixed_after_teach"] = verify_after_fix(dev2, W, unseen[TEACH:])
                        finally:
                            dev2.close()
                            shutil.rmtree(root2, ignore_errors=True)
                        rows.append(row)
                    else:
                        combo_rows.append(row)
                finally:
                    dev.close()
                    shutil.rmtree(root, ignore_errors=True)
                print({k: v for k, v in row.items() if k not in ("shaft_hz",)}, flush=True)

    per = collections.defaultdict(lambda: [0, 0])
    per_env = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["true"]][0] += r["device_hint"] == r["true"]
        per[r["true"]][1] += 1
        per_env[r["true"]][0] += r["envelope_rule"] == r["true"]
        per_env[r["true"]][1] += 1
    still = [r for r in rows if "k3_still_faulty" in r]
    skipped = sum("k3_still_faulty_note" in r for r in rows)
    out = {
        "method": __doc__.strip().splitlines()[0], "dataset": "HUST bearing v3, CC BY 4.0, DOI 10.17632/cbv7jyx4p9.3",
        "thresholds": "as shipped (NORMAL_FACTOR 2.0, MERGE_FACTOR 1.5, N 20), not re-tuned",
        "gate_false_alarms_unseen_healthy": {"windows": sum(h["windows"] for h in healthy_rows),
                                              "false_alarms": sum(h["false_alarms"] for h in healthy_rows),
                                              "per_bearing": healthy_rows},
        "detection_rate_single_faults": round(float(np.mean([r["detected"] for r in rows])), 4),
        "detection_rate_combined_faults": round(float(np.mean([r["detected"] for r in combo_rows])), 4),
        "episodes_per_recording_mean": round(float(np.mean([r["episodes_first_20"] for r in rows + combo_rows])), 3),
        "gate_false_alarms_after_one_teach": {
            "windows": sum(h["after_teach"]["windows"] for h in healthy_rows),
            "false_alarms": sum(h["after_teach"]["false_alarms"] for h in healthy_rows),
            "protocol": f"technician marks the first {TEACH} windows of the untaught 400 W healthy state as normal "
                        "operation once; false alarms counted on the remaining windows"},
        "k3_fixed_resolved": f"{sum(r['k3_fixed'] == 'symptom_resolved' for r in rows)}/{len(rows)}",
        "k3_fixed_resolved_after_one_teach": f"{sum(r['k3_fixed_after_teach'] == 'symptom_resolved' for r in rows)}/{len(rows)}",
        "k3_still_faulty_persists": f"{sum(r['k3_still_faulty'] == 'symptom_persists' for r in still)}/{len(still)}",
        "k3_still_faulty_not_evaluable": skipped,
        "k3_false_promotions": sum(r.get("k3_still_faulty") == "symptom_resolved" for r in still),
        "physics_hint": {"overall": round(sum(v[0] for v in per.values()) / max(1, sum(v[1] for v in per.values())), 4),
                         "per_class": {k: round(v[0] / v[1], 4) for k, v in sorted(per.items())},
                         "n": {k: v[1] for k, v in sorted(per.items())}},
        "envelope_rule_forced": {"overall": round(sum(v[0] for v in per_env.values()) / max(1, sum(v[1] for v in per_env.values())), 4),
                                 "per_class": {k: round(v[0] / v[1], 4) for k, v in sorted(per_env.items())}},
        "single_fault_rows": rows, "combined_fault_rows": combo_rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "hust_holdout.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_rows")}, indent=2))
    return out


if __name__ == "__main__":
    main()
