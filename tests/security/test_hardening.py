"""Security gaps closed in the weak-point pass: two-person retraction, device quarantine, tamper-evident audit log,
plausibility checks on evidence, code-integrity status, browser security headers, weak operator tokens refused on the
network, certificate revocation for mutual TLS."""
import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from cloud.audit import AuditLog
from edge.api import create_app as create_device_app
from edge.sync_worker import SyncWorker
from shared import ids
from shared.integrity import code_hash
from tests.security.test_security import event, hdr, push

PY = sys.executable


def case(c, admin):
    return c.get(f"/v1/cases/{ids.case_id('acme', 'bearing', 'inner_race')}", headers=hdr(admin)).json()


def test_quarantine_pulls_all_evidence_of_one_device_and_is_reversible(cloud):
    c, reg = cloud["client"], cloud["registry"]
    good, bad = reg.issue("devG", "siteG", "acme"), reg.issue("devX", "siteX", "acme")
    push(c, good, [event(device="devG")])
    push(c, bad, [event(device="devX", episode=ids.make_id("ep", i), outcome="failed") for i in range(3)])
    assert any(f["kind"] == "DISPUTED" for f in case(c, cloud["admin"])["flags"])
    r = c.post("/v1/admin/devices/devX/quarantine", json={"reason": "fabricated failures"}, headers=hdr(cloud["admin"]))
    assert r.status_code == 200 and r.json()["quarantined"] == 3 and r.json()["tokens_revoked"] == 1
    k = case(c, cloud["admin"])
    assert k["n_events"] == 1 and k["n_quarantined"] == 3 and not any(f["kind"] == "DISPUTED" for f in k["flags"])
    assert push(c, bad, [event(device="devX", episode=ids.make_id("ep", 9))]).status_code == 401
    r = c.post("/v1/admin/devices/devX/unquarantine", json={"reason": "investigated"}, headers=hdr(cloud["admin"]))
    assert r.json()["restored"] == 3 and case(c, cloud["admin"])["n_events"] == 4
    assert c.post("/v1/admin/devices/devX/quarantine", json={"reason": "x"}, headers=hdr(good)).status_code == 403


def test_admin_actions_are_in_a_hash_chained_audit_log(cloud, tmp_path):
    c, reg = cloud["client"], cloud["registry"]
    tok = reg.issue("devA", "site1", "acme")
    e = event()
    push(c, tok, [e])
    c.post(f"/v1/events/{e['event_id']}/retract", json={"reason": "wrong machine"}, headers=hdr(cloud["admin"]))
    c.post(f"/v1/events/{e['event_id']}/retract", json={"reason": "agreed"}, headers=hdr(cloud["admin2"]))
    c.post("/v1/admin/devices/devA/revoke", headers=hdr(cloud["admin"]))
    a = c.get("/v1/admin/audit", headers=hdr(cloud["admin"])).json()
    assert a["chain"]["ok"] and [x["action"] for x in a["entries"]] == ["retract_requested", "retract_approved",
                                                                          "token_revoked"]
    log = AuditLog(tmp_path / "audit.log")                   # file-backed: tampering breaks the chain
    for i in range(3):
        log.append("admin1", "acme", "token_issued", device_id=f"d{i}")
    assert log.verify()["ok"]
    lines = (tmp_path / "audit.log").read_text().splitlines()
    forged = json.loads(lines[1]) | {"actor": "someone-else"}
    (tmp_path / "audit.log").write_text("\n".join([lines[0], json.dumps(forged, sort_keys=True), lines[2]]) + "\n")
    assert AuditLog(tmp_path / "audit.log").verify() == {"ok": False, "broken_at": 1, "entries": 1}


@pytest.mark.parametrize("kw,why", [
    ({"verify_windows_ok": 3}, "shorter than required"),
    ({"occurred_at": "2099-01-01T00:00:00+00:00"}, "future"),
    ({"technician_confirmed": False}, "technician-confirmed"),
    ({"severity_mm_s": 9.0, "limit_mm_s": 2.8}, "above its own vibration limit"),
])
def test_implausible_evidence_is_rejected(cloud, kw, why):
    tok = cloud["registry"].issue("devA", "site1", "acme")
    r = push(cloud["client"], tok, [event(**kw)]).json()["results"][0]
    assert r["status"] == "rejected" and why in r["reason"]


def test_code_integrity_status_per_device(cloud):
    c, reg = cloud["client"], cloud["registry"]
    t1, t2 = reg.issue("devOK", "s1", "acme"), reg.issue("devMod", "s2", "acme")
    c.post("/v1/sync/push", json={"batch_id": "b", "events": []}, headers=hdr(t1) | {"X-Code-Hash": code_hash()})
    c.post("/v1/sync/push", json={"batch_id": "b", "events": []}, headers=hdr(t2) | {"X-Code-Hash": "0" * 64})
    ds = {d["device_id"]: d for d in c.get("/v1/admin/devices", headers=hdr(cloud["admin"])).json()}
    assert ds["devOK"]["code"] == "matches this release" and ds["devMod"]["code"] == "DIFFERS"


def test_devices_send_their_code_hash(make_device, cloud):
    d, w = make_device("devA", "site1")
    assert w._headers()["X-Code-Hash"] == code_hash() and len(code_hash()) == 64


def test_security_headers_on_both_apps(cloud, make_device):
    r = cloud["client"].get("/v1/health")
    assert "frame-ancestors 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"
    d, _ = make_device("devA", "site1")
    dc = TestClient(create_device_app(d, SyncWorker(d, None, None), "op-1234567"))
    h = dc.get("/api/health").headers
    assert "script-src 'self'" in h["content-security-policy"] and "microphone=(self)" in h["permissions-policy"]
    assert h["cache-control"] == "no-store"
    assert dc.post("/api/ingest/audio", content=b"x", headers={"content-length": "999999999",
                                                                "X-Operator-Token": "op-1234567"}).status_code == 413


def test_weak_operator_token_refused_on_a_network_address():
    r = subprocess.run([PY, "-m", "edge.main", "--name", "x", "--site", "s", "--host", "0.0.0.0", "--tls",
                        "--operator-token", "operator-devA", "--hash-embedder", "--no-llm", "--no-sync"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "weak operator token" in r.stderr
