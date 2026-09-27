"""Window-level CWRU dataset + train/query splits, including the leakage-free bearing-level split.

Leakage (see docs/RESEARCH.md Part O): windows cut from the same recording, or the same physical bearing
measured at another load, must never appear in both the index and the queries of an honest evaluation.
In CWRU, each (fault class, fault size) is one physically distinct seeded bearing.
"""
from __future__ import annotations

import pathlib

import numpy as np

from data.fetch_data import CWRU
from edge.fingerprint import FP_VERSION, features_batch, load_cwru, windows

RAW = pathlib.Path(__file__).resolve().parent / "raw"
CACHE = RAW / f"cwru_features_{FP_VERSION}.npz"
# Nominal speed per motor load (CWRU normal-baseline page), used only if a file lacks its RPM field.
NOMINAL_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}


def build_dataset(force: bool = False) -> dict[str, np.ndarray]:
    """Raw features for every window of every CWRU file, with labels. Cached on disk."""
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}
    X, cls, size, load, fid = [], [], [], [], []
    for f, (c, s, l) in sorted(CWRU.items()):
        x, rpm = load_cwru(RAW / "cwru" / f"{f}.mat")
        if not np.isfinite(rpm) or rpm <= 0:
            rpm = NOMINAL_RPM[l]
        feats = features_batch(windows(x), rpm=rpm)
        X.append(feats)
        n = len(feats)
        cls += [c] * n; size += [s] * n; load += [l] * n; fid += [f] * n
    data = {"X": np.concatenate(X), "cls": np.array(cls), "size": np.array(size),
            "load": np.array(load), "fid": np.array(fid)}
    np.savez(CACHE, **data)
    return data


def bearing_split(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Leakage-free: faults of 7 & 21 mil bearings (all loads) are indexed; 14 mil bearings are queries.
    Healthy data: loads 0,1 indexed; loads 2,3 queried (the rig has one healthy bearing, so healthy
    windows remain the same physical bearing - stated as a limitation)."""
    fault = d["cls"] != "normal"
    idx = (fault & np.isin(d["size"], [7, 21])) | (~fault & np.isin(d["load"], [0, 1]))
    qry = (fault & (d["size"] == 14)) | (~fault & np.isin(d["load"], [2, 3]))
    return np.where(idx)[0], np.where(qry)[0]


def leaky_split(d: dict[str, np.ndarray], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The common (flawed) protocol: random 50/50 split of windows. Reported only to show the gap."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(d["cls"]))
    half = len(perm) // 2
    return np.sort(perm[:half]), np.sort(perm[half:])
