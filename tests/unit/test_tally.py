"""The three disagreement flags (docs/RESEARCH.md F.4): DISPUTED, COMPETING, ALTERNATIVES. Pure tally functions first,
then end-to-end through the Sync API with three sites. Disagreement must be kept and flagged, never averaged away."""
from cloud.tally import aggregate
from edge.fingerprint import DIM
from shared import ids
from tests.security.test_security import hdr


def ev(site, action, outcome, root=None, status="active", n=0):
    return {"event_id": f"{site}-{action}-{outcome}-{n}", "site_id": site, "action_code": action, "outcome": outcome,
            "root_cause_claim": root, "machine_verified": True, "status": status, "fingerprint": [0.0] * 3,
            "occurred_at": f"2026-09-2{n}T10:00:00+00:00"}


def kinds(agg):
    return sorted(f["kind"] for f in agg["flags"])


def test_no_flags_when_everyone_agrees():
    agg = aggregate([ev("s1", "replace_bearing", "worked", "fatigue_wear"), ev("s2", "replace_bearing", "worked", "fatigue_wear")])
    assert kinds(agg) == [] and agg["n_sites"] == 2


def test_disputed_same_action_both_outcomes():
    agg = aggregate([ev("s1", "lubricate", "worked"), ev("s2", "lubricate", "failed")])
    assert kinds(agg) == ["DISPUTED"]
    a = agg["actions"][0]
    assert (a["worked"], a["failed"], a["sites_worked"], a["sites_failed"]) == (1, 1, ["s1"], ["s2"])


def test_competing_root_causes():
    agg = aggregate([ev("s1", "replace_bearing", "worked", "fatigue_wear"),
                     ev("s2", "replace_bearing", "worked", "lubrication_starvation")])
    assert kinds(agg) == ["COMPETING"]
    assert agg["root_causes"] == {"fatigue_wear": ["s1"], "lubrication_starvation": ["s2"]}


def test_unknown_root_cause_does_not_compete():
    agg = aggregate([ev("s1", "replace_bearing", "worked", "fatigue_wear"), ev("s2", "replace_bearing", "worked", "unknown")])
    assert kinds(agg) == []


def test_alternatives_two_different_fixes_both_worked():
    agg = aggregate([ev("s1", "replace_bearing", "worked"), ev("s2", "lubricate", "worked")])
    assert kinds(agg) == ["ALTERNATIVES"]


def test_all_three_at_once_and_nothing_is_lost():
    evs = [ev("s1", "replace_bearing", "worked", "fatigue_wear", n=1), ev("s2", "lubricate", "worked", "lubrication_starvation", n=2),
           ev("s3", "lubricate", "failed", n=3)]
    agg = aggregate(evs)
    assert kinds(agg) == ["ALTERNATIVES", "COMPETING", "DISPUTED"]
    assert sum(a["worked"] + a["failed"] for a in agg["actions"]) == 3 and agg["n_sites"] == 3


def test_retraction_removes_the_flag_it_caused_but_keeps_the_count():
    evs = [ev("s1", "lubricate", "worked"), ev("s2", "lubricate", "failed", status="retracted")]
    agg = aggregate(evs)
    assert kinds(agg) == [] and agg["n_retracted"] == 1 and agg["n_events"] == 1


def _event(device, n, action, outcome, root):
    ep = ids.make_id("ep", device, n)
    return {"event_id": ids.event_id(device, ep, outcome, action), "episode_id": ep, "machine_class": "m",
            "component": "bearing", "fault_class": "outer_race", "fault_class_source": "technician",
            "action_code": action, "outcome": outcome, "root_cause_claim": root, "machine_verified": True,
            "verify_windows_ok": 20, "verify_windows_required": 20, "technician_confirmed": True,
            "fingerprint": [0.1 * n] * DIM, "occurred_at": "2026-09-28T10:00:00+00:00"}


def test_three_sites_through_the_sync_api_raise_all_three_flags(cloud):
    c, reg = cloud["client"], cloud["registry"]
    plan = [("devA", "site1", "replace_bearing", "worked", "fatigue_wear"),
            ("devB", "site2", "lubricate", "worked", "lubrication_starvation"),
            ("devC", "site3", "lubricate", "failed", None)]
    for n, (dev, site, action, outcome, root) in enumerate(plan):
        tok = reg.issue(dev, site, "acme")
        r = c.post("/v1/sync/push", json={"batch_id": f"b{n}", "events": [_event(dev, n, action, outcome, root)]},
                   headers=hdr(tok))
        assert r.json()["results"][0]["status"] == "accepted"
    case = c.get("/v1/cases", headers=hdr(cloud["admin"])).json()[0]
    assert sorted(f["kind"] for f in case["flags"]) == ["ALTERNATIVES", "COMPETING", "DISPUTED"]
    assert case["n_sites"] == 3
    disputes = c.get("/v1/disputes", headers=hdr(cloud["admin"])).json()
    assert [d["case_id"] for d in disputes] == [case["case_id"]]


def test_same_repair_reported_twice_by_one_device_counts_once():
    a = ev("s1", "replace_bearing", "worked", n=1) | {"device_id": "devA", "content_hash": "a" * 64}
    b = ev("s1", "replace_bearing", "worked", n=2) | {"device_id": "devA", "content_hash": "a" * 64, "event_id": "other"}
    other_device = ev("s2", "replace_bearing", "worked", n=3) | {"device_id": "devB", "content_hash": "a" * 64}
    agg = aggregate([a, b, other_device])
    assert agg["n_events"] == 2 and agg["n_duplicates_collapsed"] == 1        # same hash, other device: kept
    assert agg["actions"][0]["worked"] == 2


def test_last_confirmed_is_the_latest_machine_verified_success():
    evs = [ev("s1", "lubricate", "worked", n=1), ev("s2", "lubricate", "worked", n=5),
           ev("s3", "lubricate", "failed", n=8), ev("s4", "lubricate", "worked", n=9) | {"machine_verified": False}]
    agg = aggregate(evs)
    assert agg["actions"][0]["last_confirmed"].startswith("2026-09-25")
    assert agg["last_confirmed_at"].startswith("2026-09-25")
    assert aggregate([ev("s1", "lubricate", "failed")])["last_confirmed_at"] is None
