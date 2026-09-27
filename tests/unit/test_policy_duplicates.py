"""Duplicate gate (docs/RESEARCH.md G.4): the same repair logged on two episodes is shared once."""
from edge import policy
from edge.fingerprint import DIM, FP_VERSION
from shared import ids


def episode(eid, **kw):
    return {"episode_id": eid, "machine_id": "m1", "component": "bearing", "fault_class": "inner_race",
            "fault_class_source": "technician", "action_code": "replace_bearing", "outcome": "worked",
            "action_at": "2026-09-28T10:00:00+00:00", "first_seen": "2026-09-28T09:00:00+00:00",
            "technician_confirmed": True, "fp_version": FP_VERSION,
            "verify": {"verdict": "symptom_resolved", "consecutive_ok": 20, "required": 20}} | kw


def decide(ep, repairs=None):
    return policy.decide(ep, fingerprint=[0.1] * DIM, redaction=None, device_id="devA", machine_class="m",
                         already_queued=set(), already_repairs=repairs)


def test_first_report_is_shared_with_a_repair_hash():
    d = decide(episode(ids.make_id("e1")))
    assert d.action == "SHARE"
    assert d.event["content_hash"] == ids.repair_hash("m1", "inner_race", "replace_bearing", "worked", "2026-09-28")


def test_same_repair_on_a_second_episode_is_merged_not_shared():
    first = decide(episode(ids.make_id("e1")))
    d = decide(episode(ids.make_id("e2"), action_at="2026-09-28T16:30:00+00:00"),
               repairs={first.event["content_hash"]: ids.make_id("e1")})
    assert d.action == "MERGE" and "already shared from episode" in d.reasons[-1].detail


def test_a_different_day_or_outcome_is_a_different_repair():
    first = decide(episode(ids.make_id("e1")))
    repairs = {first.event["content_hash"]: ids.make_id("e1")}
    assert decide(episode(ids.make_id("e2"), action_at="2026-09-29T10:00:00+00:00"), repairs).action == "SHARE"
    failed = episode(ids.make_id("e3"), outcome="failed",
                     verify={"verdict": "symptom_persists", "consecutive_bad": 20, "required": 20})
    assert decide(failed, repairs).action == "SHARE"


def test_repair_hash_normalises_text():
    assert ids.repair_hash(" M1 ", "Inner_Race", "replace_bearing", "WORKED", "2026-09-28T01:00") == \
        ids.repair_hash("m1", "inner_race", "replace_bearing", "worked", "2026-09-28T23:00")
