"""Expiring device tokens with automatic renewal (docs/THREATS.md 'Credential theft')."""
import time

from cloud import auth
from edge.sync_worker import SyncWorker
from tests.security.test_security import hdr


def test_expired_token_is_refused(cloud):
    tok = cloud["registry"].issue("devX", "s1", "acme", ttl_s=-1)
    assert cloud["client"].get("/v1/whoami", headers=hdr(tok)).status_code == 401


def test_renew_gives_a_new_token_and_the_old_one_only_lives_for_the_grace(cloud, monkeypatch):
    reg, c = cloud["registry"], cloud["client"]
    old = reg.issue("devX", "s1", "acme")
    r = c.post("/v1/token/renew", headers=hdr(old))
    assert r.status_code == 200
    new = r.json()["token"]
    who = c.get("/v1/whoami", headers=hdr(new)).json()
    assert who["device_id"] == "devX" and who["expires_at"] > time.time() + 29 * 24 * 3600
    assert c.get("/v1/whoami", headers=hdr(old)).status_code == 200          # grace period
    real = time.time
    monkeypatch.setattr(auth.time, "time", lambda: real() + auth.RENEW_GRACE_S + 1)
    assert c.get("/v1/whoami", headers=hdr(old)).status_code == 401          # after the grace
    assert c.get("/v1/whoami", headers=hdr(new)).status_code == 200


def test_device_renews_by_itself_and_keeps_the_new_token_across_a_restart(cloud, make_device):
    reg = cloud["registry"]
    cli_token = reg.issue("devA", "site1", "acme", ttl_s=2 * 24 * 3600)      # 2 days left < 7 days
    d, _ = make_device("devA", "site1", token=cli_token)
    w = SyncWorker(d, None, cli_token, client=cloud["client"])
    out = w.maybe_renew_token(force_check=True)
    assert out["renewed"] is True and w.token != cli_token
    assert cloud["client"].get("/v1/whoami", headers=hdr(w.token)).status_code == 200
    w2 = SyncWorker(d, None, cli_token, client=cloud["client"])            # restart with the same start token
    assert w2.token == w.token
    fresh = reg.issue("devA", "site1", "acme")                                # admin issues a brand-new token
    w3 = SyncWorker(d, None, fresh, client=cloud["client"])
    assert w3.token == fresh


def test_no_renewal_when_far_from_expiry(cloud, make_device):
    tok = cloud["registry"].issue("devB", "site2", "acme")
    d, _ = make_device("devB", "site2", token=tok)
    w = SyncWorker(d, None, tok, client=cloud["client"])
    assert w.maybe_renew_token(force_check=True)["renewed"] is False and w.token == tok
