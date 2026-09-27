"""Laya experiment (CLAUDE.md "finals": benchmark Laya vs bge-small + logistic regression; keep the winner).

Laya = convaiinnovations/laya (Apache-2.0, ModernBERT-large 421M, ~808 MB, CPU via PyTorch), a "System 1" typed
decision model. Two ADVISORY roles were proposed; neither may enter the share decision:
  1. action-code pre-fill: read the technician's OWN description of what was done and pre-select the action picker
     (classifying a past action, never proposing one: the system does not recommend actions)
  2. a second PII flag that can only make the privacy gate stricter
Data (proxy domain, aviation): Annotated Maintenance Logbook, Zenodo 17903357, CC BY 4.0. Input = ACTION text,
label = annotated action verb (TAGGEDACTION) mapped to 8 families (FAMILIES below; the mapping is ours). Rows with
several tagged verbs are dropped; texts are de-duplicated and the train/test split is by distinct text.
Methods on the SAME held-out subset: majority class; bge-small zero-shot (cosine to the family descriptions);
bge-small + logistic regression (trained on the train split); Laya zero-shot `choice` with the same descriptions.
PII probe (synthetic, stated as such): held-out action texts, half with an appended person name; Laya `noul` vs the
project's regex/denylist redactor with an EMPTY denylist (i.e. what it catches without being told names).
Run in the isolated env:  $env:HF_HOME="models_cache\\hf"; .venv-laya\\Scripts\\python.exe -m bench.laya_experiment
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import pathlib
import random
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "bench" / "results"
N_LAYA = 400            # held-out notes sent to Laya (CPU time); every method is scored on this same subset
N_PII = 120

FAMILIES = {   # family -> (description given to the zero-shot methods, annotated verbs mapped to it)
    "replace": ("a part was removed and replaced with another part",
                ["REMOVED & REPLACED", "REPLACED", "REMOVED & INSTALLED"]),
    "install": ("a part was installed, reinstalled or fabricated and fitted",
                ["INSTALLED", "REINSTALLED", "FABRICATED & INSTALLED", "FABRICATED"]),
    "secure": ("something was tightened, secured, re-secured, torqued or reattached",
               ["TIGHTENED", "RESECURED", "SECURED", "RETORQUED", "REATTACHED"]),
    "inspect_test": ("something was checked, inspected, run or started to test it",
                     ["CHECKED", "INSPECTED", "RAN", "STARTED", "RAN & CHECKED"]),
    "adjust": ("something was adjusted or repositioned", ["ADJUSTED", "REPOSITIONED"]),
    "clean_apply": ("something was cleaned, or a substance was applied", ["CLEANED", "APPLIED"]),
    "remove": ("a part was removed and not replaced", ["REMOVED"]),
    "repair": ("damage was repaired, sealed or a crack was stop-drilled", ["STOP", "REPAIRED", "SEALED"]),
}
VERB_TO_FAMILY = {v: f for f, (_, vs) in FAMILIES.items() for v in vs}
NAMES = ["Ravi Kumar", "Priya Sharma", "Anil Mehta", "John Carter", "Maria Lopez", "Wei Chen", "Fatima Khan",
         "David Miller", "Sunita Rao", "Tom Baker", "Aisha Bello", "Karan Singh"]
TEMPLATES = ["{t} - done by {n}", "{n}: {t}", "{t}. Signed off by {n}.", "{t} per {n}"]


def load() -> tuple[list[str], list[str]]:
    f = glob.glob(str(ROOT / "data" / "raw" / "logbook" / "*.csv"))[0]
    rows = list(csv.reader(open(f, encoding="utf-8", errors="replace")))[1:]
    by_text: dict[str, set[str]] = collections.defaultdict(set)
    for r in rows:
        tag = r[11].strip().upper()
        if tag and "," not in tag and tag in VERB_TO_FAMILY and r[6].strip():
            by_text[" ".join(r[6].split()).upper()].add(VERB_TO_FAMILY[tag])
    items = sorted((t, next(iter(fs))) for t, fs in by_text.items() if len(fs) == 1)   # drop conflicting labels
    return [t for t, _ in items], [y for _, y in items]


def split(texts, labels, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    return idx[:cut], idx[cut:]


def scores(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)}


def main() -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from fastembed import TextEmbedding
    from sklearn.linear_model import LogisticRegression

    texts, labels = load()
    tr, te = split(texts, labels)
    fams = list(FAMILIES)
    out: dict = {"method": __doc__.strip().splitlines()[0], "n_distinct_texts": len(texts),
                 "n_train": len(tr), "n_test": len(te),
                 "label_counts": dict(collections.Counter(labels).most_common())}
    rng = random.Random(1)
    sub = sorted(rng.sample(te, min(N_LAYA, len(te))))
    y_sub = [labels[i] for i in sub]

    emb = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=str(ROOT / "models_cache"))
    E = lambda xs: np.array(list(emb.embed(xs)))
    t = time.perf_counter()
    X_tr, X_te = E([texts[i] for i in tr]), E([texts[i] for i in te])
    embed_ms = (time.perf_counter() - t) * 1000 / (len(tr) + len(te))
    y_tr, y_te = [labels[i] for i in tr], [labels[i] for i in te]
    pos = {i: k for k, i in enumerate(te)}
    X_sub = X_te[[pos[i] for i in sub]]

    maj = collections.Counter(y_tr).most_common(1)[0][0]
    D = E([FAMILIES[f][0] for f in fams])
    zs = lambda X: [fams[j] for j in (X / np.linalg.norm(X, axis=1, keepdims=True) @
                                      (D / np.linalg.norm(D, axis=1, keepdims=True)).T).argmax(1)]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
    t = time.perf_counter()
    for i in range(50):
        lr.predict(X_sub[i:i + 1])
    lr_ms = (time.perf_counter() - t) * 1000 / 50

    out["full_test"] = {"majority": scores(y_te, [maj] * len(y_te)), "bge_zero_shot": scores(y_te, zs(X_te)),
                        "bge_logreg": scores(y_te, list(lr.predict(X_te)))}

    import laya
    t = time.perf_counter()
    agent = laya.load("convaiinnovations/laya", device="cpu")
    load_s = time.perf_counter() - t
    q = {"action": {"type": "choice", "instructions": "What kind of maintenance action does this note describe?",
                    "criteria": {f: FAMILIES[f][0] for f in fams}}}
    laya_pred, laya_conf, laya_ms = [], [], []
    for n, i in enumerate(sub):
        t = time.perf_counter()
        a = agent.predict({"note": texts[i]}, q)["answers"]["action"]
        laya_ms.append((time.perf_counter() - t) * 1000)
        laya_pred.append(a["choice"])
        laya_conf.append(float(a["confidence"]))
        if n % 50 == 0:
            print(f"laya {n}/{len(sub)}", flush=True)
    correct = np.array(laya_pred) == np.array(y_sub)
    conf = np.array(laya_conf)
    out["same_subset"] = {"n": len(sub), "majority": scores(y_sub, [maj] * len(sub)),
                          "bge_zero_shot": scores(y_sub, zs(X_sub)), "bge_logreg": scores(y_sub, list(lr.predict(X_sub))),
                          "laya_zero_shot": scores(y_sub, laya_pred),
                          "laya_mean_confidence": round(float(conf.mean()), 3),
                          "laya_accuracy_when_confidence_ge_0.9": round(float(correct[conf >= 0.9].mean()), 4) if (conf >= 0.9).any() else None,
                          "laya_share_confidence_ge_0.9": round(float((conf >= 0.9).mean()), 3)}
    out["latency_ms_per_note_cpu"] = {"bge_embed": round(embed_ms, 2), "bge_logreg_predict": round(lr_ms, 3),
                                      "laya_p50": round(float(np.percentile(laya_ms, 50)), 1),
                                      "laya_p95": round(float(np.percentile(laya_ms, 95)), 1),
                                      "laya_load_s": round(load_s, 1)}

    # ---- PII probe (synthetic) ----
    from shared.redact import redact
    base = [texts[i] for i in rng.sample(te, N_PII)]
    probe = []
    for k, tx in enumerate(base):
        if k % 2:
            probe.append((TEMPLATES[k % len(TEMPLATES)].format(t=tx, n=NAMES[k % len(NAMES)]), True))
        else:
            probe.append((tx, False))
    qp = {"pii": {"type": "noul", "instructions": "Does the text mention a person by name?"}}
    lay = [agent.predict({"note": tx}, qp)["answers"]["pii"]["noul"] >= 0.5 for tx, _ in probe]
    red = [bool(redact(tx, []).findings) for tx, _ in probe]
    truth = [y for _, y in probe]

    def pr(pred):
        tp = sum(p and y for p, y in zip(pred, truth))
        fp = sum(p and not y for p, y in zip(pred, truth))
        fn = sum(y and not p for p, y in zip(pred, truth))
        return {"recall": round(tp / max(1, tp + fn), 3), "precision": round(tp / max(1, tp + fp), 3),
                "false_positives": fp}
    out["pii_probe_synthetic"] = {"n": len(probe), "with_name": sum(truth), "laya_noul": pr(lay),
                                  "regex_redactor_empty_denylist": pr(red),
                                  "either": pr([a or b for a, b in zip(lay, red)])}
    OUT.mkdir(exist_ok=True)
    (OUT / "laya_experiment.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
