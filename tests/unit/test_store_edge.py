"""Unit tests for edge/store_edge.py against a real Qdrant Edge shard (no mocks)."""
import math
import threading
import uuid

import pytest

from edge.fingerprint import DIM as FP_DIM, FP_VERSION
from edge.store_edge import (NOTE, NOTE_BM25, VIB, EdgeStore, StoreConfigError, StorePoint, build_filter,
                             canonical_id)

VD, ND = 4, 3


def pid(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


@pytest.fixture
def store(tmp_path):
    s = EdgeStore(tmp_path / "dev", vib_dim=VD, note_dim=ND)
    yield s
    s.close()


def ep(name, vib=None, note=None, text=None, **payload):
    return StorePoint(pid(name), {"type": "episode", "name": name, **payload}, vib=vib, note=note, bm25_text=text)


# ---- creation / reopen ----------------------------------------------------------------------------
def test_default_dims_follow_fingerprint(tmp_path):
    with EdgeStore(tmp_path / "d") as s:
        assert s.info()["vib_dim"] == FP_DIM and s.info()["fp_version"] == FP_VERSION
        s.upsert([StorePoint(pid("a"), {"type": "baseline"}, vib=[0.0] * FP_DIM)])


def test_points_and_meta_survive_close_and_reopen(tmp_path):
    root = tmp_path / "dev"
    with EdgeStore(root, vib_dim=VD, note_dim=ND, bm25_avg_len=8.0) as s:
        s.upsert([ep("a", vib=[1, 2, 3, 4], text="bearing replaced", machine_id="m1")])
    with EdgeStore(root, vib_dim=VD, note_dim=ND) as s:
        assert s.meta["bm25_avg_len"] == 8.0                      # persisted setting wins on reopen
        assert s.get(pid("a")).payload["machine_id"] == "m1"
        assert s.search(text="bearing")[0].id == pid("a")        # indexes and sparse vectors survived


@pytest.mark.parametrize("kw", [{"vib_dim": VD + 1}, {"note_dim": ND + 1}, {"fp_version": "fp-v0"},
                                {"text_model": "other/model"}])
def test_reopen_with_different_settings_is_refused(tmp_path, kw):
    root = tmp_path / "dev"
    EdgeStore(root, vib_dim=VD, note_dim=ND).close()
    with pytest.raises(StoreConfigError):
        EdgeStore(root, **({"vib_dim": VD, "note_dim": ND} | kw))


# ---- validation -----------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    ep("x", vib=[1, 2, 3]),                                # wrong dim
    ep("x", vib=[1, 2, 3, math.nan]),                      # NaN
    ep("x", note=[1, math.inf, 0]),                        # inf
    ep("x"),                                               # no vectors at all
    StorePoint("not-a-uuid", {}, vib=[0, 0, 0, 0]),        # bad id
])
def test_invalid_points_raise_and_write_nothing(store, bad):
    good = ep("ok", vib=[0, 0, 0, 0])
    with pytest.raises(ValueError):
        store.upsert([good, bad])
    assert store.count() == 0                              # batch validated before any write


def test_canonical_id():
    u = uuid.uuid4()
    assert canonical_id(u) == canonical_id(str(u).upper()) == str(u)


# ---- writes ---------------------------------------------------------------------------------------
def test_partial_vectors_and_retrieve_order(store):
    store.upsert([ep("a", vib=[0, 0, 0, 1]), ep("b", note=[1, 0, 0], text="lubrication added"),
                  ep("c", vib=[1, 1, 1, 1], note=[0, 1, 0], text="bearing replaced")])
    recs = store.retrieve([pid("c"), pid("missing"), pid("a")], with_vectors=True)
    assert [r.id for r in recs] == [pid("c"), pid("a")]
    assert set(recs[0].vectors) == {VIB, NOTE, NOTE_BM25}
    assert set(recs[0].vectors[NOTE_BM25]) == {"indices", "values"}
    assert set(recs[1].vectors) == {VIB}
    assert store.get(pid("missing")) is None


def test_insert_only_is_idempotent_and_reports_new_ids(store):
    first = store.upsert([ep("a", vib=[0] * 4, v=1), ep("b", vib=[0] * 4, v=1)], insert_only=True)
    assert sorted(first) == sorted([pid("a"), pid("b")])
    again = store.upsert([ep("a", vib=[9] * 4, v=2), ep("c", vib=[0] * 4, v=1)], insert_only=True)
    assert again == [pid("c")]
    assert store.get(pid("a")).payload["v"] == 1           # existing point untouched
    assert store.count() == 3


def test_plain_upsert_overwrites(store):
    store.upsert([ep("a", vib=[0] * 4, v=1)])
    store.upsert([ep("a", vib=[0] * 4, v=2)])
    assert store.get(pid("a")).payload["v"] == 2 and store.count() == 1


def test_cas_update(store):
    store.upsert([ep("a", vib=[0] * 4, text="orig", version=1)])
    assert store.cas_update(ep("a", vib=[0] * 4, text="edit one"), expected_version=1)
    assert store.get(pid("a")).payload["version"] == 2
    # a second writer that also read version 1 must be refused, and must not overwrite
    assert not store.cas_update(ep("a", vib=[0] * 4, text="stale", note_by="stale"), expected_version=1)
    rec = store.get(pid("a"))
    assert rec.payload["version"] == 2 and "note_by" not in rec.payload
    assert store.search(text="edit")[0].id == pid("a")
    assert not store.cas_update(ep("missing", vib=[0] * 4), expected_version=0)
    assert store.get(pid("missing")) is None


def test_modify_is_atomic_across_threads(store):
    store.upsert([ep("a", vib=[0] * 4, occurrences=0)])

    def bump():
        for _ in range(20):
            store.modify(pid("a"), lambda p: {"occurrences": p["occurrences"] + 1})

    ts = [threading.Thread(target=bump) for _ in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert store.get(pid("a")).payload["occurrences"] == 100
    assert store.modify(pid("missing"), lambda p: {"x": 1}) is None


# ---- reads ----------------------------------------------------------------------------------------
def test_nearest_is_euclidean_distance_and_filtered(store):
    store.upsert([StorePoint(pid("h"), {"type": "baseline", "machine_id": "m1"}, vib=[0, 0, 0, 0]),
                  StorePoint(pid("h2"), {"type": "baseline", "machine_id": "m2"}, vib=[0.1, 0, 0, 0]),
                  ep("e", vib=[3, 4, 0, 0], machine_id="m1")])
    hits = store.nearest([0.1, 0, 0, 0], filter={"machine_id": "m1"}, limit=5)
    assert [h.id for h in hits] == [pid("h"), pid("e")]
    assert hits[0].score == pytest.approx(0.1, abs=1e-6)
    assert hits[1].score == pytest.approx(math.dist([0.1, 0, 0, 0], [3, 4, 0, 0]), abs=1e-5)
    assert store.nearest([0] * 4, filter={"type": "episode", "machine_id": "m2"}) == []


def corpus(store):
    store.upsert([
        ep("bearing", vib=[5, 0, 0, 0], note=[1, 0, 0], text="inner race spall bearing replaced", component="bearing"),
        ep("lube", vib=[0, 5, 0, 0], note=[0, 1, 0], text="lubrication added grease", component="bearing"),
        ep("align", vib=[0, 0, 6, 0], note=[0, 0, 1], text="coupling misalignment corrected", component="coupling"),  # no distance ties
        ep("retracted", vib=[5, 0, 0, 0.1], note=[1, 0, 0], text="bearing replaced", component="bearing",
           status="retracted"),
    ])


def test_search_single_legs(store):
    corpus(store)
    assert store.search(text="misalignment")[0].id == pid("align")
    assert store.search(note=[0, 1, 0])[0].id == pid("lube")
    assert store.search(vib=[0, 0, 4.8, 0])[0].id == pid("align")


def test_search_hybrid_fuses_legs_and_explains(store):
    corpus(store)
    # vib points at "lube", text and note point at "bearing": fused result favours the 2-of-3 agreement
    hits = store.search(vib=[0, 5, 0, 0], note=[1, 0, 0], text="spall", filter={"!status": "retracted"},
                        limit=3, explain=True)
    assert hits[0].id == pid("bearing")
    assert hits[0].legs == {VIB: 2, NOTE: 1, NOTE_BM25: 1}
    lube = next(h for h in hits if h.id == pid("lube"))
    assert lube.legs[VIB] == 1 and lube.legs[NOTE_BM25] is None     # BM25 never matched "spall" in lube's note
    assert pid("retracted") not in [h.id for h in hits]
    assert all(h.score > 0 for h in hits)


def test_search_weights_change_the_winner(store):
    """prefetch_limit=1: the vib leg returns only "lube", the text leg only "bearing"; both have rank 1, so the
    leg weights alone decide. (With k=60, rank 1 vs 2 barely differs, so a hit found by BOTH legs beats a
    heavier-weighted single-leg hit; that is RRF behaviour, not a bug.)"""
    corpus(store)
    args = dict(vib=[0, 5, 0, 0], text="spall", filter={"!status": "retracted"}, limit=1, prefetch_limit=1)
    assert store.search(**args, weights={VIB: 5.0})[0].id == pid("lube")
    assert store.search(**args, weights={NOTE_BM25: 5.0})[0].id == pid("bearing")


def test_search_filter_applies_to_every_leg(store):
    corpus(store)
    hits = store.search(vib=[5, 0, 0, 0], text="bearing replaced", filter={"component": "coupling"}, limit=10)
    assert [h.id for h in hits] == [pid("align")]


def test_search_needs_a_leg(store):
    with pytest.raises(ValueError):
        store.search(text="")


def test_count_facet_scroll(store):
    corpus(store)
    assert store.count() == 4
    assert store.count({"component": ["bearing", "coupling"], "!status": "retracted"}) == 3
    assert store.facet("component") == {"bearing": 3, "coupling": 1}
    assert store.facet("component", filter={"!status": "retracted"}) == {"bearing": 2, "coupling": 1}
    ids = [r.id for r in store.scroll(batch=1)]
    assert len(ids) == len(set(ids)) == 4
    assert [r.id for r in store.scroll(filter={"component": "coupling"}, with_vectors=True)] == [pid("align")]


def test_build_filter_shapes():
    assert build_filter(None) is None and build_filter({}) is None
    f = build_filter({"a": 1, "b": ["x", "y"], "v": {"gte": 2}, "!s": "r"})
    assert len(f.must) == 3 and len(f.must_not) == 1


def test_closed_store_rejects_use(tmp_path):
    s = EdgeStore(tmp_path / "d", vib_dim=VD, note_dim=ND)
    s.close()
    s.close()                                              # idempotent
    with pytest.raises(Exception):
        s.count()
