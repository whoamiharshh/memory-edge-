"""'Does this word look like a person's given name?' - a small character n-gram model for the note redactor.

Trained on REAL data only: 10,562 given names (Wikidata, CC0) vs real maintenance/technical words (knowledge/
domain_vocab.json). Character 2-4-grams, hashed (2^15 buckets), logistic regression; exported as plain JSON numbers
(knowledge/name_model.json), applied with numpy. The decision threshold is chosen on a VALIDATION split (20 % of names
and of words held out of training): the lowest threshold whose false-positive rate on held-out words is <= 2 %.
It only ever makes the redactor stricter (a flagged note stays on the device).
"""
from __future__ import annotations

import functools
import json
import pathlib
import random
import zlib

import numpy as np

KNOWLEDGE = pathlib.Path(__file__).resolve().parents[1] / "knowledge"
N_FEATURES = 2 ** 15
GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MAX_VAL_FP = 0.02


def _grams(word: str) -> list[str]:
    w = f" {word.lower()} "
    return [w[i:i + n] for n in (2, 3, 4) for i in range(len(w) - n + 1)]


def features(words: list[str]) -> np.ndarray:
    """Hashed, L2-normalised character n-gram counts (crc32: stable across Python runs and machines)."""
    X = np.zeros((len(words), N_FEATURES), dtype=np.float32)
    for r, w in enumerate(words):
        for g in _grams(w):
            X[r, zlib.crc32(g.encode()) % N_FEATURES] += 1.0
        n = np.linalg.norm(X[r])
        if n:
            X[r] /= n
    return X


def train(names: list[str], words: list[str], seed: int = 7) -> dict:
    from sklearn.linear_model import LogisticRegression
    rng = random.Random(seed)
    names, words = sorted(set(names) - set(words)), sorted(set(words))
    rng.shuffle(names)
    rng.shuffle(words)
    cn, cw = int(0.8 * len(names)), int(0.8 * len(words))
    fit = lambda N, W: LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(
        features(N + W), np.array([1] * len(N) + [0] * len(W)))
    m = fit(names[:cn], words[:cw])
    pn = m.predict_proba(features(names[cn:]))[:, 1]
    pw = m.predict_proba(features(words[cw:]))[:, 1]
    val = [{"threshold": t, "name_recall": round(float(np.mean(pn >= t)), 3),
            "word_false_positive": round(float(np.mean(pw >= t)), 3)} for t in GRID]
    thr = next((v["threshold"] for v in val if v["word_false_positive"] <= MAX_VAL_FP), GRID[-1])
    m = fit(names, words)
    return {"n_features": N_FEATURES, "ngrams": [2, 3, 4], "coef": np.round(m.coef_[0], 4).tolist(),
            "intercept": round(float(m.intercept_[0]), 4), "threshold": thr, "validation": val,
            "trained_on": {"names": len(names), "words": len(words)}}


@functools.lru_cache(maxsize=1)
def load() -> dict | None:
    p = KNOWLEDGE / "name_model.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def probability(model: dict, words: list[str]) -> np.ndarray:
    if not words:
        return np.zeros(0)
    z = features(words) @ np.asarray(model["coef"], dtype=np.float32) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-z))
