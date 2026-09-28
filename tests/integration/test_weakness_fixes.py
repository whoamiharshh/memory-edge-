"""The weak-point fixes, end to end on REAL CWRU recordings: fleet-learned hint (order features travel with the
confirmed evidence, the cloud trains, devices pull), 'did the fix hold' follow-ups, ISO 15243 damage mode,
replacement-aware verification, machine-card limits in the verifier."""
import datetime as dt

import numpy as np

from edge.replay import Recordings
from tests.conftest import needs_cwru
from tests.integration.test_fleet_flow import feed, resolve_episode, setup_machine

pytestmark = needs_cwru


def order_feats(dev, fid, i=0):
    return dev.profile.order_features(*Recordings.raw_segment(fid, i))


def share_confirmed(dev, fid, cls, damage=None, action="replace_bearing"):
    feed(dev, fid, n=25)
    ep = dev.episodes()[0]["episode_id"]
    for i in (0, 10, 20):
        dev.attach_order_features(ep, order_feats(dev, fid, i))
    dev.set_fault_class(ep, cls, damage_mode=damage)
    dev.record_action(ep, action)
    feed(dev, 99, n=20)
    dev.confirm_outcome(ep, "worked")
    return ep


def test_evidence_carries_order_features_and_damage_mode(make_device, cloud):
    a, wa = make_device("devA", "site1")
    setup_machine(a)
    ep = share_confirmed(a, 105, "inner_race", damage="surface_initiated_fatigue")
    e = a.episode(ep)
    assert len(e["order_features"]) == 20 and e["fleet_hint"]["physics"] == "inner_race"
    assert wa.push_once()["accepted"] == 1
    ev = cloud["store"].get_event("acme", e["event_id"])
    assert ev["damage_mode"] == "surface_initiated_fatigue" and len(ev["order_features"]) == 20
    assert ev["of_version"] == "of-v1" and ev["bearing"].startswith("SKF 6205")


def test_fleet_learns_a_hint_model_from_confirmed_cases_and_devices_pull_it(make_device, cloud):
    workers = []
    for k, (fid_in, fid_out) in enumerate(((105, 130), (106, 131), (107, 132))):
        d, w = make_device(f"dev{k}", f"site{k}")
        setup_machine(d)
        share_confirmed(d, fid_in, "inner_race")
        feed(d, 99, n=3)
        share_confirmed(d, fid_out, "outer_race")
        assert w.push_once()["accepted"] == 2
        workers.append((d, w))
    r = cloud["client"].get("/v1/fleet/hint-model", headers={"authorization": f"Bearer {cloud['admin']}"}).json()
    m = r["model"]
    assert m and m["trained_on"] == {"cases": 6, "devices": 3, "per_class": {"inner_race": 3, "outer_race": 3}}
    assert m["unseen_device_accuracy"] is not None
    b, wb = make_device("devB", "siteB")
    setup_machine(b)
    assert wb.pull_once()["hint_model"] == "updated"
    feed(b, 108, n=5)                                   # an inner-race recording no device shared
    eid = b.episodes()[0]["episode_id"]
    h = b.attach_order_features(eid, order_feats(b, 108))
    assert h["fleet"]["fault_class"] == "inner_race" and h["confidence"] == "confident"
    assert b.stats()["fleet_hint_model"]["trained_on"]["cases"] == 6


def test_fix_held_follow_up_reaches_the_fleet_tally(make_device, cloud):
    a, wa = make_device("devA", "site1", hold_days=30.0)
    setup_machine(a)
    ep = resolve_episode(a, 105, "replace_bearing", "inner_race")
    wa.push_once()
    assert a.check_followups() == []                   # too early
    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=31)
    fu = a.check_followups(now=later)
    assert len(fu) == 1 and fu[0]["status"] == "held"
    assert a.check_followups(now=later) == []          # once only
    assert wa.push_once()["accepted"] == 1
    case = cloud["client"].get("/v1/cases", headers={"authorization": f"Bearer {cloud['admin']}"}).json()[0]
    row = case["actions"][0]
    assert row["held"] == 1 and row["recurred"] == 0 and a.episode(ep)["followup"]["status"] == "held"


def test_recurrence_after_a_verified_fix_is_flagged_in_the_fleet(make_device, cloud):
    a, wa = make_device("devA", "site1")
    setup_machine(a)
    first = resolve_episode(a, 105, "replace_bearing", "inner_race")
    wa.push_once()
    feed(a, 105, n=3, start=40)                        # the same fault comes back
    assert a.episode(first)["followup"]["status"] == "recurred"
    assert wa.push_once()["accepted"] == 1
    case = cloud["client"].get("/v1/cases", headers={"authorization": f"Bearer {cloud['admin']}"}).json()[0]
    assert case["actions"][0]["recurred"] == 1
    assert any(f["kind"] == "RECURRED" for f in case["flags"])


def test_a_device_cannot_report_follow_ups_for_another_devices_fix(make_device, cloud):
    a, wa = make_device("devA", "site1")
    setup_machine(a)
    ep = resolve_episode(a, 105, "replace_bearing", "inner_race")
    wa.push_once()
    event_id = a.episode(ep)["event_id"]
    from shared import ids
    tok = cloud["registry"].issue("evil", "siteX", "acme")
    body = {"kind": "followup", "event_id": ids.followup_id("evil", event_id), "episode_id": ep, "refers_to": event_id,
            "status": "recurred", "days_after_fix": 1.0, "hold_days": 30.0, "occurred_at": "2026-09-28T00:00:00+00:00"}
    r = cloud["client"].post("/v1/sync/push", json={"batch_id": "x", "events": [body]},
                             headers={"authorization": f"Bearer {tok}"}).json()
    assert r["results"][0]["status"] == "rejected" and "no event of this device" in r["results"][0]["reason"]
    forged = body | {"event_id": ids.followup_id("devA", event_id)}
    r = cloud["client"].post("/v1/sync/push", json={"batch_id": "y", "events": [forged]},
                             headers={"authorization": f"Bearer {tok}"}).json()
    assert r["results"][0]["status"] == "rejected"


def test_machine_card_limit_blocks_resolution_above_the_manufacturer_limit(make_device):
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    a.set_machine_card({"power_kw": 1.5, "limits_mm_s": {"acceptable": 0.01, "trip": 0.02}, "source": "test card"})
    feed(a, 105, n=25)
    ep = a.episodes()[0]["episode_id"]
    a.record_action(ep, "lubricate")
    assert a.episode(ep)["verify"]["limit_mm_s"] == 0.01
    x = Recordings.raw_segment(99, 0, seconds=5.0)[0]
    a.ingest_signal(x, 12000.0, 1772.0)                 # healthy by the baseline, but above this absurd limit
    v = a.episode(ep)["verify"]
    assert v["verdict"] != "symptom_resolved" and v["over_limit_windows"] > 0


def test_replacement_is_judged_by_the_nearest_state_rule(make_device):
    a, _ = make_device("devA", "site1")
    setup_machine(a)
    feed(a, 105, n=25)
    ep = a.episodes()[0]["episode_id"]
    a.record_action(ep, "replace_bearing")
    assert a.episode(ep)["verify"]["mode"] == "replacement"
    feed(a, 105, n=20, start=25)                       # the fault continues: never "fixed"
    assert a.episode(ep)["verify"]["verdict"] == "symptom_persists"
