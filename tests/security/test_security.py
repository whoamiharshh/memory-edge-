"""Security tests mapped to the threat model (docs/RESEARCH.md Part K)."""
import json
import pathlib
import re

import pytest
from fastapi.testclient import TestClient

from cloud.auth import RATE_LIMIT, TokenRegistry
from edge.api import create_app as edge_app
from edge.fingerprint import DIM
from shared import ids
from tests.conftest import needs_cwru


def hdr(tok):
    return {"Authorization": f"Bearer {tok}"}


def event(device="devA", episode=None, outcome="worked", action="replace_bearing", **kw):
    ep = episode or ids.make_id("ep", device)
    return {"event_id": ids.event_id(device, ep, outcome, action), "episode_id": ep, "machine_class": "m",
            "component": "bearing", "fault_class": "inner_race", "fault_class_source": "technician",
            "action_code": action, "outcome": outcome, "machine_verified": True, "verify_windows_ok": 20,
            "verify_windows_required": 20, "technician_confirmed": True, "fingerprint": [0.5] * DIM,
            "occurred_at": "2026-09-27T10:00:00+00:00"} | kw


def push(client, tok, events):
    return client.post("/v1/sync/push", json={"batch_id": "b1", "events": events}, headers=hdr(tok))


# ---- unauthorized sync / credential theft -----------------------------------------------------------
def test_push_without_or_with_bad_token_is_401(cloud):
    c = cloud["client"]
    assert c.post("/v1/sync/push", json={"batch_id": "b", "events": []}).status_code == 401
    assert push(c, "not-a-real-token", []).status_code == 401


def test_revoked_token_is_refused_on_next_request(cloud):
    c, reg = cloud["client"], cloud["registry"]
    tok = reg.issue("devA", "site1", "acme")
    assert push(c, tok, [event()]).status_code == 200
    assert c.post("/v1/admin/devices/devA/revoke", headers=hdr(cloud["admin"])).json()["revoked"] == 1
    assert push(c, tok, [event(outcome="failed")]).status_code == 401


def test_token_registry_stores_only_hashes(tmp_path):
    reg = TokenRegistry(tmp_path / "t.json")
    tok = reg.issue("devA", "site1", "acme")
    raw = (tmp_path / "t.json").read_text()
    assert tok not in raw and len(json.loads(raw)) == 1


def test_device_role_cannot_use_admin_endpoints(cloud):
    c = cloud["client"]
    tok = cloud["registry"].issue("devA", "site1", "acme")
    assert c.post("/v1/admin/devices", json={"device_id": "x", "site_id": "y"}, headers=hdr(tok)).status_code == 403
    assert c.post(f"/v1/events/{ids.make_id('e')}/retract", json={"reason": "abc"}, headers=hdr(tok)).status_code == 403


# ---- tenant isolation -------------------------------------------------------------------------------
def test_tenant_comes_from_token_and_tenants_are_isolated(cloud):
    c, reg = cloud["client"], cloud["registry"]
    a = reg.issue("devA", "site1", "acme")
    evil = reg.issue("devX", "siteX", "evilcorp")
    assert push(c, a, [event()]).json()["results"][0]["status"] == "accepted"
    assert c.get("/v1/cases", headers=hdr(evil)).json() == []                   # cannot see acme's evidence
    case_id = ids.case_id("acme", "bearing", "inner_race")
    assert c.get(f"/v1/cases/{case_id}", headers=hdr(evil)).status_code == 404
    r = push(c, evil, [event(device="devX") | {"tenant_id": "acme"}]).json()    # body cannot pick a tenant
    assert r["results"][0]["status"] == "rejected"
    assert c.get("/v1/mirror/cases", headers=hdr(evil)).json()["items"] == []


# ---- replay / poisoning / impersonation -------------------------------------------------------------
def test_replayed_batch_is_counted_once(cloud):
    c = cloud["client"]
    tok = cloud["registry"].issue("devA", "site1", "acme")
    evs = [event()]
    for _ in range(3):
        push(c, tok, evs)
    case = c.get("/v1/cases", headers=hdr(tok)).json()[0]
    assert case["n_events"] == 1 and case["actions"][0]["worked"] == 1


def test_device_cannot_forge_another_devices_event_id(cloud):
    c = cloud["client"]
    mallory = cloud["registry"].issue("devM", "siteM", "acme")
    stolen = event(device="devA")                   # an id derived from devA's identity
    r = push(c, mallory, [stolen]).json()["results"][0]
    assert r["status"] == "rejected" and "authenticated device" in r["reason"]


def test_one_site_cannot_manufacture_consensus(cloud):
    """A device flooding 'worked' reports still counts as ONE site; the UI shows sites, not just reports."""
    c = cloud["client"]
    tok = cloud["registry"].issue("devM", "siteM", "acme")
    push(c, tok, [event(device="devM", episode=ids.make_id("ep", i)) for i in range(20)])
    case = c.get("/v1/cases", headers=hdr(tok)).json()[0]
    assert case["n_events"] == 20 and case["n_sites"] == 1 and case["actions"][0]["sites_worked"] == ["siteM"]


def test_retraction_is_a_tombstone_and_recomputes_tallies(cloud):
    c = cloud["client"]
    tok = cloud["registry"].issue("devA", "site1", "acme")
    e = event()
    push(c, tok, [e])
    r = c.post(f"/v1/events/{e['event_id']}/retract", json={"reason": "fabricated"}, headers=hdr(cloud["admin"]))
    assert r.status_code == 200 and r.json()["status"] == "active" and r.json()["retraction"].startswith("pending")
    r = c.post(f"/v1/events/{e['event_id']}/retract", json={"reason": "fabricated"}, headers=hdr(cloud["admin"]))
    assert r.status_code == 409                                                 # the same admin cannot approve
    r = c.post(f"/v1/events/{e['event_id']}/retract", json={"reason": "agreed"}, headers=hdr(cloud["admin2"]))
    assert r.status_code == 200 and r.json()["status"] == "retracted" and r.json()["retracted_by"] == ["admin1", "admin2"]
    case = c.get(f"/v1/cases/{ids.case_id('acme', 'bearing', 'inner_race')}", headers=hdr(cloud["admin"])).json()
    assert case["n_events"] == 0 and case["n_retracted"] == 1 and case["status"] == "retracted"
    assert len(case["events"]) == 1                                             # kept, not deleted


# ---- malicious payloads -----------------------------------------------------------------------------
def test_infinity_and_nan_literals_rejected(cloud):
    """Python's json accepts the non-standard literals Infinity/NaN; they must still be refused."""
    c = cloud["client"]
    tok = cloud["registry"].issue("devA", "site1", "acme")
    for lit in ("Infinity", "NaN", "-Infinity"):
        body = json.dumps({"batch_id": "b", "events": [event()]}).replace("0.5", lit, 1)
        r = c.post("/v1/sync/push", content=body, headers=hdr(tok) | {"content-type": "application/json"})
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert r.json()["results"][0]["status"] == "rejected"


@pytest.mark.parametrize("bad", [
    {"fingerprint": [0.1] * 5000}, {"note_redacted": "x" * 100_000},
    {"action_code": "<script>alert(1)</script>"}, {"schema_version": 99}, {"note_vector": [0.1] * 384},
])
def test_malicious_events_rejected_individually(cloud, bad):
    c = cloud["client"]
    tok = cloud["registry"].issue("devA", "site1", "acme")
    good = event(episode=ids.make_id("good"))
    r = push(c, tok, [event() | bad, good])
    assert r.status_code == 200
    st = [x["status"] for x in r.json()["results"]]
    assert st == ["rejected", "accepted"]                                        # one bad event != lost batch


def test_oversized_body_and_batch_are_refused(cloud):
    c = cloud["client"]
    tok = cloud["registry"].issue("devA", "site1", "acme")
    assert c.post("/v1/sync/push", content=b"{" + b" " * 600_000 + b"}", headers=hdr(tok) | {"content-type": "application/json"}).status_code == 413
    assert push(c, tok, [event()] * 101).status_code == 422


def test_rate_limit(cloud):
    c = cloud["client"]
    tok = cloud["registry"].issue("devF", "siteF", "acme")
    codes = [c.get("/v1/whoami", headers=hdr(tok)).status_code for _ in range(RATE_LIMIT + 1)]
    assert codes[-1] == 429 and codes[0] == 200


# ---- privacy: what leaves the device ----------------------------------------------------------------
@needs_cwru
def test_outbound_event_has_no_raw_note_signal_or_text_embedding(make_device):
    from edge.replay import Recordings
    a, _ = make_device("devA", "site1")
    a.fit_baseline(Recordings.baseline())
    for w in Recordings.windows(105)[:5]:
        a.ingest_window(w)
    ep = a.episodes()[0]["episode_id"]
    a.set_note(ep, "spall found, bearing replaced, call 98765 43210", share_opt_in=True)
    a.set_fault_class(ep, "inner_race")
    a.record_action(ep, "replace_bearing")
    for w in Recordings.windows(99)[:20]:
        a.ingest_window(w)
    a.confirm_outcome(ep, "worked")
    body = json.loads(a.outbox.rows()[0] and a.outbox._exec("SELECT body FROM outbox").fetchone()[0])
    assert set(body) >= {"fingerprint", "action_code", "outcome"}
    assert body["note_redacted"] is None                                         # phone number -> note stays local
    assert not {"note_text", "note", "note_vector", "signal", "tenant_id", "device_id", "site_id"} & set(body)
    assert len(body["fingerprint"]) == DIM


# ---- device API ------------------------------------------------------------------------------------
@needs_cwru
def test_device_api_requires_operator_token_and_escapes_nothing_server_side(make_device):
    a, w = make_device("devA", "site1")
    c = TestClient(edge_app(a, w, "s3cret-operator"))
    assert c.get("/api/episodes").status_code == 401
    assert c.get("/api/episodes", headers={"X-Operator-Token": "wrong"}).status_code == 401
    assert c.get("/api/episodes", headers={"X-Operator-Token": "s3cret-operator"}).status_code == 200
    assert c.get("/api/health").status_code == 200


def test_ui_never_uses_innerhtml():
    """XSS guard: notes are untrusted text; both UIs must render via textContent only."""
    root = pathlib.Path(__file__).resolve().parents[2]
    for f in (root / "edge" / "ui" / "app.js", root / "cloud" / "ui" / "app.js"):
        src = f.read_text(encoding="utf-8")
        assert not re.search(r"\.(innerHTML|outerHTML)\b|insertAdjacentHTML|document\.write\(", src), f
