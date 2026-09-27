"""Fleet mirror through Qdrant's dual-shard pattern, against a REAL Qdrant Server (release binary): full shard
snapshot -> partial snapshots -> offline search on Device B. Also the failure paths of docs/RESEARCH.md H.3."""
import pathlib

import pytest

from edge.mirror import Mirror
from edge.outbox import Outbox
from tests.conftest import needs_cwru, needs_qdrant_server
from tests.integration.test_fleet_flow import feed, resolve_episode, setup_machine

pytestmark = [needs_cwru, needs_qdrant_server]


def hdr(tok):
    return {"Authorization": f"Bearer {tok}"}


def share_fix(make, name, site, outcome="worked", fault=105, after=99, action="replace_bearing"):
    dev, w = make(name, site)
    setup_machine(dev)
    resolve_episode(dev, fault, action, "inner_race", outcome=outcome, after=after)
    assert w.push_once()["accepted"] == 1
    return dev, w


def test_full_then_partial_snapshot_and_offline_search(make_server_device, server_cloud):
    share_fix(make_server_device, "devA", "site1")
    b, wb = make_server_device("devB", "site2")
    setup_machine(b)

    first = wb.pull_once()
    assert (first["mode"], first["snapshot"], first["pulled"]) == ("snapshot", "full", 1)
    assert first["wire_bytes"] < 2_000_000                     # gzip: the raw shard snapshot is ~150 MB of pages
    info = b.mirror.info()
    assert info["mode"] == "snapshot" and not info["needs_full"] and info["cases"] == 1
    assert info["last_refresh"]["snapshot_bytes"] > 50 * info["last_refresh"]["wire_bytes"]

    # nothing changed: the head check answers, no snapshot is requested
    assert wb.pull_once()["up_to_date"] is True

    # a second site reports the same action FAILED -> the case changes -> B gets a PARTIAL snapshot
    share_fix(make_server_device, "devC", "site3", outcome="failed", fault=106, after=106)
    second = wb.pull_once()
    assert second["snapshot"] == "partial" and second["pulled"] == 1
    case = b.mirror.scroll()[0].payload
    assert {f["kind"] for f in case["flags"]} == {"DISPUTED"}
    act = case["actions"][0]
    assert (act["worked"], act["failed"], case["n_sites"]) == (1, 1, 2)

    # B offline: its own new episode finds the fleet case through the mirror (dense + BM25 + vib, RRF)
    wb.set_online(False)
    feed(b, 169, n=10)
    res = b.search(episode_id=b.episodes()[0]["episode_id"])
    assert res["fleet_error"] is None
    assert [h["case"]["fault_class"] for h in res["fleet"]] == ["inner_race"]
    assert res["fleet"][0]["legs"]["note_bm25"] == 1               # the server-made BM25 vector matches Edge's query
    assert wb.pull_once() == {"skipped": "offline"}


def test_server_answers_304_when_the_manifest_is_current(make_server_device):
    share_fix(make_server_device, "devA", "site1")
    b, wb = make_server_device("devB", "site2")
    wb.pull_once()
    b.mirror.set_seq(0)                                            # pretend we missed the head: force a partial ask
    r = wb.pull_once()
    assert r["up_to_date"] is True and r["pulled"] == 0 and b.mirror.seq > 0


def test_retraction_reaches_the_mirror(make_server_device, server_cloud):
    a, _ = share_fix(make_server_device, "devA", "site1")
    b, wb = make_server_device("devB", "site2")
    wb.pull_once()
    ev = a.episodes()[0]["event_id"]
    r = server_cloud["client"].post(f"/v1/events/{ev}/retract", json={"reason": "wrong bearing logged"},
                                    headers=hdr(server_cloud["admin"]))
    assert r.status_code == 200
    assert wb.pull_once()["snapshot"] == "partial"
    case = b.mirror.scroll()[0].payload
    assert (case["n_events"], case["n_retracted"], case["status"]) == (0, 1, "retracted")   # tombstone, not deleted
    # and Device B's fleet search no longer offers it (filter !status=retracted)
    setup_machine(b)
    feed(b, 169, n=3)
    assert b.search(episode_id=b.episodes()[0]["episode_id"])["fleet"] == []


def test_corrupt_full_snapshot_leaves_old_mirror_in_use(make_server_device, tmp_path):
    share_fix(make_server_device, "devA", "site1")
    b, wb = make_server_device("devB", "site2")
    wb.pull_once()
    bad = tmp_path / "bad.snapshot"
    bad.write_bytes(b"this is not a tar archive")
    with pytest.raises(Exception):
        b.mirror.replace_from_snapshot(bad, 10**15)
    assert b.mirror.count() == 1 and not (pathlib.Path(b.cfg.root) / "mirror.new").exists()
    assert b.mirror.seq < 10**15                                    # cursor untouched


def test_failed_partial_apply_triggers_full_rebuild(make_server_device, tmp_path):
    share_fix(make_server_device, "devA", "site1")
    b, wb = make_server_device("devB", "site2")
    wb.pull_once()
    bad = tmp_path / "bad.snapshot"
    bad.write_bytes(b"garbage")
    with pytest.raises(Exception):
        b.mirror.apply_partial(bad, 10**15)
    assert b.mirror.needs_full
    share_fix(make_server_device, "devC", "site3", outcome="failed", fault=106, after=106)
    r = wb.pull_once()
    assert r["snapshot"] == "full" and not b.mirror.needs_full and b.mirror.count() == 1


def test_crash_mid_swap_is_repaired_on_boot(tmp_path):
    root = tmp_path / "dev"
    root.mkdir()
    ob = Outbox(str(root / "device.sqlite"))
    m = Mirror(root, ob)
    m.close()
    (root / "mirror").rename(root / "mirror.old")                   # died after the first rename
    (root / "mirror.new").mkdir()                                   # ... with a half-restored candidate
    m2 = Mirror(root, ob)
    try:
        assert (root / "mirror").exists() and not (root / "mirror.old").exists()
        assert not (root / "mirror.new").exists()
        assert m2.count() == 0
    finally:
        m2.close()
        ob.close()


def test_mirror_is_per_tenant(make_server_device, server_cloud):
    share_fix(make_server_device, "devA", "site1")
    evil, we = make_server_device("devX", "siteX", tenant="othertenant")
    r = we.pull_once()
    assert r["mode"] == "snapshot" and evil.mirror.count() == 0     # its own (empty) tenant mirror only


def test_mirror_write_never_goes_back_in_version(server_cloud):
    store, t = server_cloud["store"], server_cloud["tenant"]
    store.ensure_tenant(t)
    cid = "00000000-0000-0000-0000-00000000abcd"
    vib, note = [0.0] * 27, [1.0] + [0.0] * 383
    store._mirror_put(t, cid, {"case_id": cid, "text": "v3", "version": 3, "seq": 3}, vib, note)
    store._mirror_put(t, cid, {"case_id": cid, "text": "v2", "version": 2, "seq": 2}, vib, note)  # late writer
    got = store.client.retrieve(f"mirror_{t}", [cid], with_payload=True)[0].payload
    assert got["version"] == 3 and got["text"] == "v3"


def test_partial_endpoint_validates_manifest(server_cloud):
    tok = server_cloud["registry"].issue("devA", "site1", server_cloud["tenant"])
    c = server_cloud["client"]
    assert c.post("/v1/mirror/snapshot/partial", json={"x": 1}, headers=hdr(tok)).status_code == 422
    assert c.post("/v1/mirror/snapshot/partial", json={}, headers=hdr(tok)).status_code == 422
    assert c.get("/v1/mirror/snapshot").status_code == 401


def test_scroll_mode_still_works_against_a_server(make_server_device):
    share_fix(make_server_device, "devA", "site1")
    b, wb = make_server_device("devB", "site2", mirror_mode="scroll")
    r = wb.pull_once()
    assert (r["mode"], r["pulled"]) == ("scroll", 1) and b.mirror.mode == "scroll"
