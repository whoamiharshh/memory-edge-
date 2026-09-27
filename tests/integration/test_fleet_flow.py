"""End-to-end on REAL CWRU fingerprints: Device A learns -> the machine verifies -> cloud aggregates ->
Device B benefits offline. In-process cloud (Qdrant in-memory), HashEmbedder (fast, lexical)."""
import numpy as np
import pytest

from edge.replay import Recordings
from tests.conftest import needs_cwru

pytestmark = needs_cwru


def feed(dev, fid, n=None, start=0):
    out = [dev.ingest_window(w, f"cwru:{fid}") for w in Recordings.windows(fid)[start:][:n]]
    return out


def setup_machine(dev):
    return dev.fit_baseline(Recordings.baseline())


def test_gate_normal_on_unseen_healthy_and_one_episode_per_fault(make_device):
    a, _ = make_device("devA", "site1")
    g = setup_machine(a)
    assert g["tau_normal"] > 0 and g["tau_merge"] > g["tau_normal"]
    healthy = feed(a, 99) + feed(a, 100)                           # loads 2,3: never seen by the baseline
    assert sum(r["state"] != "normal" for r in healthy) == 0
    fault = feed(a, 105, n=30)
    assert fault[0]["state"] == "new"
    assert all(r["state"] == "merge" for r in fault[1:])
    eps = a.episodes()
    assert len(eps) == 1 and eps[0]["occurrences"] == 30
    assert eps[0]["fault_hint"]["fault_class"] == "inner_race"    # physics hint on a 7-mil inner race file
    assert eps[0]["decision"]["action"] == "KEEP_LOCAL"
    assert "awaiting action" in eps[0]["decision"]["reasons"][-1]["detail"]


def resolve_episode(dev, fid_fault, action, fault_class, note="", opt_in=False, outcome="worked", after=99):
    feed(dev, fid_fault, n=25)
    ep = dev.episodes()[0]["episode_id"]
    if note:
        dev.set_note(ep, note, share_opt_in=opt_in)
    dev.set_fault_class(ep, fault_class)
    dev.record_action(ep, action)
    feed(dev, after, n=20)
    dev.confirm_outcome(ep, outcome)
    return ep


def test_full_fleet_flow_device_a_to_cloud_to_device_b(make_device, cloud):
    a, wa = make_device("devA", "site1", denylist=["Ravi"])
    setup_machine(a)
    ep = resolve_episode(a, 105, "replace_bearing", "inner_race",
                         note="Inner race spall found, bearing replaced by Ravi, vibration normal", opt_in=True)
    e = a.episode(ep)
    assert e["verify"]["verdict"] == "symptom_resolved" and e["status"] == "closed"
    assert e["decision"]["action"] == "SHARE"
    assert e["decision"]["note_shared"] is False                   # redactor found a denylisted name
    assert "note kept local" in e["decision"]["reasons"][0]["detail"]
    assert e["share_state"] == "queued"

    # offline: nothing leaves, nothing is lost
    wa.set_online(False)
    assert wa.push_once() == {"skipped": "offline"}
    assert a.outbox.counts() == {"queued": 1}
    # back online: delivered once
    wa.set_online(True)
    assert wa.push_once()["accepted"] == 1
    assert a.episode(ep)["share_state"] == "synced"
    # replay the same event: the cloud says duplicate, counts unchanged
    a.outbox.requeue(e["event_id"])
    assert wa.push_once()["duplicate"] == 1
    cases = cloud["client"].get("/v1/cases", headers={"Authorization": f"Bearer {cloud['admin']}"}).json()
    assert len(cases) == 1
    c = cases[0]
    assert (c["component"], c["fault_class"], c["n_events"], c["n_sites"]) == ("bearing", "inner_race", 1, 1)
    assert c["actions"][0]["action_code"] == "replace_bearing" and c["actions"][0]["worked"] == 1
    assert c["notes"] == []                                        # the raw note never reached the cloud

    # Device B at another site, same tenant: pull the mirror, go offline, see A's evidence
    b, wb = make_device("devB", "site2")
    setup_machine(b)
    assert wb.pull_once()["pulled"] == 1
    wb.set_online(False)
    feed(b, 169, n=10)                                             # 14-mil inner race: a bearing A never saw
    epb = b.episodes()[0]
    assert epb["fault_hint"]["fault_class"] == "inner_race"
    res = b.search(episode_id=epb["episode_id"])
    assert res["query"]["fleet_filter"]["fault_class"] == "inner_race"
    assert [h["case"]["actions"][0]["action_code"] for h in res["fleet"]] == ["replace_bearing"]


def test_failed_fix_is_shared_as_failed_evidence_and_dispute_flagged(make_device, cloud):
    a, wa = make_device("devA", "site1")
    setup_machine(a)
    resolve_episode(a, 105, "replace_bearing", "inner_race")
    wa.push_once()
    c3, w3 = make_device("devC", "site3")
    setup_machine(c3)
    ep = resolve_episode(c3, 106, "replace_bearing", "inner_race", outcome="failed", after=106)  # fault persists
    e = c3.episode(ep)
    assert e["verify"]["verdict"] == "symptom_persists"
    assert e["decision"]["action"] == "SHARE"
    w3.push_once()
    case = cloud["client"].get("/v1/disputes", headers={"Authorization": f"Bearer {cloud['admin']}"}).json()[0]
    assert {f["kind"] for f in case["flags"]} == {"DISPUTED"}
    act = case["actions"][0]
    assert (act["worked"], act["failed"], case["n_sites"]) == (1, 1, 2)   # both kept, nothing overwritten


def test_technician_sensor_conflict_stays_local(make_device):
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    ep = resolve_episode(a, 105, "lubricate", "inner_race", outcome="worked", after=105)   # says worked, isn't
    e = a.episode(ep)
    assert e["verify"]["verdict"] == "symptom_persists"
    assert e["decision"]["action"] == "KEEP_LOCAL"
    assert "conflict" in e["decision"]["reasons"][-1]["detail"]
    assert a.outbox.counts() == {}


def test_unconfirmed_fault_class_is_not_shared(make_device):
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    feed(a, 105, n=5)
    ep = a.episodes()[0]["episode_id"]
    a.record_action(ep, "replace_bearing")
    feed(a, 99, n=20)
    a.confirm_outcome(ep, "worked")
    d = a.episode(ep)["decision"]
    assert d["action"] == "KEEP_LOCAL" and "physics hint" in d["reasons"][-1]["detail"]
    a.set_fault_class(ep, "inner_race")
    assert a.episode(ep)["decision"]["action"] == "SHARE"


def test_recurrence_after_close_is_recognised(make_device):
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    first = resolve_episode(a, 105, "replace_bearing", "inner_race")
    out = feed(a, 105, n=3, start=40)
    assert out[0]["state"] == "new" and out[0]["recurrence_of"] == first


def test_journal_replay_after_crash_between_sqlite_and_shard(make_device, tmp_path):
    """Outbox-first ordering: a journaled write whose shard write never happened is applied on next boot."""
    from edge.device import Device, DeviceConfig
    from shared.embed import HashEmbedder
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    feed(a, 105, n=2)
    ep = a.episodes()[0]["episode_id"]
    # simulate a crash: the journal row is written, the process dies before the shard write
    a.outbox.journal_put("crash-op", {"kind": "set_payload", "id": ep, "fields": {"occurrences": 999}})
    root = a.cfg.root
    a.close()
    b = Device(DeviceConfig(device_id="devA", site_id="site1", machine_id="devA-m1", root=root), HashEmbedder())
    try:
        assert b.episode(ep)["occurrences"] == 999
        assert b.outbox.journal_pending() == []
        assert b.gate is not None                                   # baseline + gate config restored too
    finally:
        b.close()
