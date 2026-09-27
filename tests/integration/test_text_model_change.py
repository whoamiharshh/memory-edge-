"""Changing the device's text model (docs/RESEARCH.md H.3): a new named vector, re-embedded memory, then the switch.
The fleet mirror keeps the cloud's model, so a mismatched device drops only the fleet dense-text leg."""
import hashlib
import re

import numpy as np

from edge.device import Device, DeviceConfig
from edge.store_edge import NOTE
from shared.embed import DIM, HashEmbedder
from tests.conftest import needs_cwru
from tests.integration.test_fleet_flow import feed, setup_machine

pytestmark = needs_cwru


class SaltedEmbedder(HashEmbedder):
    """A different (deterministic, lexical) 'model': same idea, other hashing, so its vectors differ."""
    name = "salted-bow-384 (test model v2)"

    def _vec(self, text):
        v = np.zeros(DIM)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.sha1(b"v2" + tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v) or 1.0
        return (v / n).tolist()


def test_new_text_model_reembeds_memory_and_search_still_works(tmp_path):
    cfg = DeviceConfig(device_id="devA", site_id="s1", machine_id="m1", root=tmp_path / "devA")
    a = Device(cfg, HashEmbedder())
    setup_machine(a)
    feed(a, 105, n=5)
    eid = a.episodes()[0]["episode_id"]
    a.set_note(eid, "grease seal leaking near the drive end housing")
    assert a.store.note_name == NOTE
    a.close()

    b = Device(cfg, SaltedEmbedder())                      # same device, new model
    try:
        assert b.store.meta["text_model"] == SaltedEmbedder.name
        assert b.store.note_name != NOTE and b.store.meta["previous_text_models"] == [HashEmbedder.name]
        assert any(x["kind"] == "model" and "re-embedded 1 episode" in x["message"] for x in b.outbox.activity())
        rec = b.store.get(eid, with_vectors=True)
        assert NOTE not in rec.vectors and b.store.note_name in rec.vectors
        assert np.allclose(rec.vectors[b.store.note_name], SaltedEmbedder().embed_documents([b._doc_text(rec.payload)])[0])
        res = b.search(text="grease seal leaking", use_fleet=False)
        assert res["local"][0]["id"] == eid and res["local"][0]["legs"]["note"] == 1
    finally:
        b.close()

    c = Device(cfg, SaltedEmbedder())                      # third boot: nothing left to migrate
    try:
        assert c.store.pending_text_model is None
        assert sum(x["kind"] == "model" for x in c.outbox.activity()) == 1
    finally:
        c.close()


def test_fleet_dense_leg_is_skipped_when_the_fleet_uses_another_model(make_device, cloud):
    from tests.integration.test_fleet_flow import resolve_episode
    a, wa = make_device("devA", "site1")
    setup_machine(a)
    resolve_episode(a, 105, "replace_bearing", "inner_race")
    wa.push_once()
    b = Device(DeviceConfig(device_id="devB", site_id="site2", machine_id="devB-m1", root=a.cfg.root.parent / "devB"),
               SaltedEmbedder())
    from edge.sync_worker import SyncWorker
    wb = SyncWorker(b, None, cloud["registry"].issue("devB", "site2", "acme"), client=cloud["client"])
    try:
        setup_machine(b)
        assert wb.pull_once()["pulled"] == 1
        assert b.outbox.kv_get("fleet_text_model") == HashEmbedder.name         # announced by the cloud
        feed(b, 169, n=5)
        res = b.search(episode_id=b.episodes()[0]["episode_id"])
        assert res["fleet"] and "dense-text leg skipped" in res["fleet_error"]
        assert "note" not in res["fleet"][0]["legs"]                            # BM25 + fingerprint only
    finally:
        b.close()
