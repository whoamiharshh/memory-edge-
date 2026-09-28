"""Coalesced recompute (the production cloud): a push returns once events are durable; any read right after it
still sees the new tallies, because reads flush their tenant's pending cases first."""
from fastapi.testclient import TestClient

from cloud.api import create_app
from cloud.auth import TokenRegistry
from cloud.store_server import CloudStore
from shared import ids
from shared.embed import HashEmbedder
from tests.security.test_security import event, hdr, push


def test_reads_after_a_push_are_never_stale():
    reg = TokenRegistry()
    app = create_app(CloudStore(location=":memory:"), reg, HashEmbedder(), coalesce=True)
    rc = app.state.recomputer
    rc.stop()                                              # no background pass: the READ must do the work
    c = TestClient(app)
    tok, admin = reg.issue("devA", "site1", "acme"), reg.issue("adm", "hq", "acme", role="admin")
    r = push(c, tok, [event(episode=ids.make_id("ep", i)) for i in range(3)]).json()
    assert [x["status"] for x in r["results"]] == ["accepted"] * 3
    assert rc.pending() == 1                               # one case, marked once for three events
    cases = c.get("/v1/cases", headers=hdr(admin)).json()
    assert cases[0]["n_events"] == 3 and rc.pending() == 0
    push(c, tok, [event(episode=ids.make_id("ep", 9), outcome="failed")])
    assert rc.pending() == 1                               # mirror reads do not force it (background keeps it ~0.5 s)
    c.get("/v1/mirror/head", headers=hdr(tok))
    assert rc.pending() == 1
    assert any(f["kind"] == "DISPUTED" for f in c.get("/v1/cases", headers=hdr(admin)).json()[0]["flags"])
