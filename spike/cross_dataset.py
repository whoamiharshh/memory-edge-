"""Spike: a PRIOR fault-type model trained on other test rigs (leave-one-DATASET-out). Order features as in
spike/order_features.py; UOttawa accelerometer only (same sensor type as CWRU/HUST); cage class excluded (only UOttawa
has it). Run: .venv\Scripts\python.exe -m spike.cross_dataset
"""
from __future__ import annotations

import collections

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spike.hint_agreement import rows
from spike.physics_v2 import cwru, hust, uottawa

if __name__ == "__main__":
    data = {"uottawa": [r for r in rows(r for r in uottawa() if r[0] == "uottawa-acc") if r[2] != "cage"],
            "cwru": rows(cwru()), "hust": rows(hust())}
    for held in data:
        tr = [r for k, R in data.items() if k != held for r in R]
        X = np.concatenate([r[4] for r in tr]); y = np.concatenate([[r[2]] * len(r[4]) for r in tr])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X, y)
        ok = agree = agree_ok = 0
        for r in data[held]:
            mv = collections.Counter(clf.predict(r[4])).most_common(1)[0][0]
            ok += mv == r[2]
            if mv == r[5]:
                agree += 1; agree_ok += mv == r[2]
        n = len(data[held])
        print(f"held-out {held:8s}: prior model {ok}/{n} = {ok / n:.2f} | physics {sum(r[5] == r[2] for r in data[held]) / n:.2f}"
              f" | agree {agree}/{n} -> correct {agree_ok}/{max(1, agree)}", flush=True)
