"""Unit tests for data/splits.py. The core property: the honest split never puts the same physical
bearing in both the index and the queries."""
import itertools

import numpy as np
import pytest

from data import splits
from data.fetch_data import CWRU
from edge.fingerprint import DIM

N_PER_FILE = 3


def synthetic() -> dict[str, np.ndarray]:
    """Label arrays shaped exactly like build_dataset() output, N_PER_FILE windows per CWRU file."""
    cls, size, load, fid = [], [], [], []
    for f, (c, s, l) in sorted(CWRU.items()):
        cls += [c] * N_PER_FILE; size += [s] * N_PER_FILE; load += [l] * N_PER_FILE; fid += [f] * N_PER_FILE
    n = len(cls)
    return {"X": np.zeros((n, DIM)), "cls": np.array(cls), "size": np.array(size),
            "load": np.array(load), "fid": np.array(fid)}


def bearings(d, rows):
    """Physical bearing identity of each row: (class, size) for faults; the one healthy bearing is 'normal'."""
    return {(str(d["cls"][i]), int(d["size"][i])) for i in rows}


def assert_honest(d, idx, qry):
    assert len(np.intersect1d(idx, qry)) == 0
    assert len(np.intersect1d(d["fid"][idx], d["fid"][qry])) == 0            # no recording on both sides
    fault_idx = [i for i in idx if d["cls"][i] != "normal"]
    fault_qry = [i for i in qry if d["cls"][i] != "normal"]
    assert bearings(d, fault_idx).isdisjoint(bearings(d, fault_qry))         # no faulty bearing on both sides
    assert set(d["size"][fault_idx]) <= {7, 21}
    assert set(d["size"][fault_qry]) == {14}


def test_mapping_has_one_bearing_per_class_and_size():
    """Sanity of the file table the split relies on: 3 fault classes x 3 sizes x 4 loads + 4 normal."""
    assert len(CWRU) == 40
    faults = [(c, s) for c, s, _ in CWRU.values() if c != "normal"]
    assert set(faults) == set(itertools.product(["inner_race", "ball", "outer_race"], [7, 14, 21]))
    for key in set(faults):
        assert sorted(l for c, s, l in CWRU.values() if (c, s) == key) == [0, 1, 2, 3]


def test_bearing_split_is_leakage_free_and_complete():
    d = synthetic()
    idx, qry = splits.bearing_split(d)
    assert_honest(d, idx, qry)
    np.testing.assert_array_equal(np.sort(np.concatenate([idx, qry])), np.arange(len(d["cls"])))


def test_bearing_split_covers_every_class_on_both_sides():
    d = synthetic()
    idx, qry = splits.bearing_split(d)
    classes = {"normal", "inner_race", "ball", "outer_race"}
    assert set(d["cls"][idx]) == classes
    assert set(d["cls"][qry]) == classes


def test_bearing_split_healthy_by_load():
    d = synthetic()
    idx, qry = splits.bearing_split(d)
    assert set(d["load"][idx][d["cls"][idx] == "normal"]) == {0, 1}
    assert set(d["load"][qry][d["cls"][qry] == "normal"]) == {2, 3}


def test_leaky_split_is_a_disjoint_partition_and_seeded():
    d = synthetic()
    a_idx, a_qry = splits.leaky_split(d, seed=0)
    b_idx, b_qry = splits.leaky_split(d, seed=0)
    np.testing.assert_array_equal(a_idx, b_idx)
    np.testing.assert_array_equal(a_qry, b_qry)
    assert len(np.intersect1d(a_idx, a_qry)) == 0
    assert len(a_idx) + len(a_qry) == len(d["cls"])
    c_idx, _ = splits.leaky_split(d, seed=1)
    assert not np.array_equal(a_idx, c_idx)


def test_leaky_split_actually_leaks():
    """Documents WHY it is the flawed protocol: the same recordings end up on both sides."""
    d = synthetic()
    idx, qry = splits.leaky_split(d)
    assert len(np.intersect1d(d["fid"][idx], d["fid"][qry])) > 0


def test_build_dataset_reads_cache_without_touching_raw_files(tmp_path, monkeypatch):
    d = synthetic()
    cache = tmp_path / "c.npz"
    np.savez(cache, **d)
    monkeypatch.setattr(splits, "CACHE", cache)
    monkeypatch.setattr(splits, "load_cwru", lambda *_: pytest.fail("raw file read despite cache"))
    out = splits.build_dataset()
    assert set(out) == set(d)
    for k in d:
        np.testing.assert_array_equal(out[k], d[k])


@pytest.mark.skipif(not splits.CACHE.exists(), reason="real CWRU feature cache not built (run data/fetch_data.py + bench)")
def test_real_cwru_dataset_and_honest_split():
    d = splits.build_dataset()
    n = len(d["cls"])
    assert d["X"].shape == (n, DIM)
    assert np.all(np.isfinite(d["X"]))
    assert set(d["fid"].tolist()) == set(CWRU)
    for f, (c, s, l) in CWRU.items():                     # labels match the verified file table
        rows = d["fid"] == f
        assert set(d["cls"][rows]) == {c} and set(d["size"][rows]) == {s} and set(d["load"][rows]) == {l}
    idx, qry = splits.bearing_split(d)
    assert_honest(d, idx, qry)
