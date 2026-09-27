"""Episode edits under optimistic concurrency (G.4 "updated memory"), usefulness feedback (G.1) and retention /
ARCHIVE (G.8), on real CWRU fingerprints."""
import datetime as dt

import pytest
from fastapi.testclient import TestClient

from edge import policy
from edge.api import create_app
from edge.device import Device, DeviceConfig, VersionConflict
from edge.store_edge import StorePoint
from shared import ids
from shared.embed import HashEmbedder
from tests.conftest import needs_cwru
from tests.integration.test_fleet_flow import feed, resolve_episode, setup_machine

pytestmark = needs_cwru
LATER = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=policy.ARCHIVE_AFTER_DAYS + 1)


def exemplars(dev, eid):
    return dev.store.count({"type": "exemplar", "episode_id": eid})


# ---- optimistic concurrency ------------------------------------------------------------------------------
def test_stale_edit_is_refused_and_writes_nothing(make_device):
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    feed(a, 105, n=5)
    ep = a.episodes()[0]
    v = ep["version"]
    a.set_note(ep["episode_id"], "first technician", expected_version=v)       # tab 1 saves
    with pytest.raises(VersionConflict) as err:
        a.set_note(ep["episode_id"], "second technician", expected_version=v)  # tab 2 still shows version v
    assert err.value.current > v
    assert a.episode(ep["episode_id"])["note_text"] == "first technician"      # nothing overwritten
    a.set_note(ep["episode_id"], "second technician, after reload", expected_version=err.value.current)
    assert a.episode(ep["episode_id"])["note_text"] == "second technician, after reload"
    assert any(x["kind"] == "conflict" for x in a.outbox.activity())


def test_api_reports_conflict_as_409(make_device):
    a, w = make_device("devA", "site1")
    setup_machine(a)
    feed(a, 105, n=3)
    eid = a.episodes()[0]["episode_id"]
    c = TestClient(create_app(a, w, "op-secret-123"))
    h = {"X-Operator-Token": "op-secret-123"}
    v = c.get(f"/api/episodes/{eid}", headers=h).json()["version"]
    assert c.post(f"/api/episodes/{eid}/fault_class", json={"fault_class": "inner_race", "expected_version": v},
                  headers=h).status_code == 200
    r = c.post(f"/api/episodes/{eid}/note", json={"text": "late", "expected_version": v}, headers=h)
    assert r.status_code == 409 and "CONFLICT" in r.json()["detail"]
    # without expected_version the old behaviour (last write wins) is kept for scripts
    assert c.post(f"/api/episodes/{eid}/note", json={"text": "script"}, headers=h).status_code == 200


# ---- feedback --------------------------------------------------------------------------------------------
def test_feedback_counts_are_shown_and_do_not_change_ranking(make_device):
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    first = resolve_episode(a, 105, "replace_bearing", "inner_race")
    feed(a, 105, n=3, start=40)                                                 # recurrence -> 2nd episode
    second = a.episodes()[0]["episode_id"]
    before = a.search(episode_id=second)
    assert before["local"][0]["id"] == first and before["local"][0]["feedback"] == {"helped": 0, "not_helped": 0}
    a.record_feedback(first, "local", helped=True)
    a.record_feedback(first, "local", helped=True)
    a.record_feedback(first, "local", helped=False)
    after = a.search(episode_id=second)
    assert after["local"][0]["feedback"] == {"helped": 2, "not_helped": 1}
    assert [h["id"] for h in after["local"]] == [h["id"] for h in before["local"]]
    with pytest.raises(ValueError):
        a.record_feedback("not-a-point-id", "local", True)
    with pytest.raises(ValueError):
        a.record_feedback(first, "cloud", True)


# ---- retention -------------------------------------------------------------------------------------------
def test_retention_archives_only_old_closed_decided_episodes(make_device):
    a, w = make_device("devA", "site1")
    setup_machine(a)
    closed = resolve_episode(a, 105, "replace_bearing", "inner_race")
    w.push_once()                                                               # its share is delivered
    # CWRU 105 is so steady that it needs one exemplar; give it two more (as a noisier fault would get)
    ex0 = a.store.get(ids.make_id("exemplar", closed, 0), with_vectors=True)
    a._upsert([StorePoint(ids.make_id("exemplar", closed, k), ex0.payload, vib=[x + 0.5 * k for x in ex0.vectors["vib"]])
               for k in (1, 2)])
    assert exemplars(a, closed) == 3
    feed(a, 118, n=5)                                                           # a second, still-open episode
    open_ep = a.episodes()[0]["episode_id"]
    assert a.run_retention()["archived"] == []                                  # nothing is 90 days old yet
    r = a.run_retention(now=LATER)
    assert r["archived"] == [closed]
    e = a.episode(closed)
    assert e["archived"] is True and e["status"] == "closed" and e["n_exemplars"] == 1
    assert exemplars(a, closed) == 1                                            # extra fingerprints dropped
    assert "archived" not in a.episode(open_ep)                                 # open episodes never archived
    assert a.run_retention(now=LATER)["archived"] == []                         # idempotent
    # knowledge kept: the archived episode is still found, and a recurrence is still recognised
    feed(a, 99, n=3)                                                            # healthy: ends the open run
    out = feed(a, 105, n=2, start=40)
    assert out[0]["state"] == "new" and out[0]["recurrence_of"] == closed


def test_retention_skips_episode_whose_share_is_not_uploaded(make_device):
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    ep = resolve_episode(a, 105, "replace_bearing", "inner_race")
    assert a.episode(ep)["share_state"] == "queued"                             # never pushed
    assert a.run_retention(now=LATER)["archived"] == []


def test_archive_op_is_replayed_after_a_crash(make_device):
    a, w = make_device("devA", "site1")
    setup_machine(a)
    ep = resolve_episode(a, 105, "replace_bearing", "inner_race")
    w.push_once()
    keep = a.store.get(ids.make_id("exemplar", ep, 0), with_vectors=True)
    a.outbox.journal_put("crash-archive", {"kind": "archive", "episode_id": ep,
                                           "keep": {"id": keep.id, "payload": keep.payload, "vib": keep.vectors["vib"]},
                                           "fields": {"archived": True, "archived_at": "2027-01-01T00:00:00+00:00",
                                                      "n_exemplars": 1}})
    root = a.cfg.root
    a.close()
    b = Device(DeviceConfig(device_id="devA", site_id="site1", machine_id="devA-m1", root=root), HashEmbedder())
    try:
        assert b.episode(ep)["archived"] is True and exemplars(b, ep) == 1
    finally:
        b.close()
