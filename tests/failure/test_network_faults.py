"""Bad networks and wrong clocks (the full measurement is bench/network_faults.py):
- a device whose clock is days off still has its evidence accepted, with times corrected onto the cloud's clock, and
  it warns about its clock;
- through a proxy that cuts half the connections (requests AND acknowledgements), every event arrives exactly once;
- a stalled connection costs a timeout, not data."""
import datetime as dt
import threading
import time

import httpx
import pytest
import uvicorn

from cloud.api import create_app
from cloud.auth import TokenRegistry
from cloud.store_server import CloudStore
from edge.device import Device, DeviceConfig
from edge.sync_worker import SyncWorker
from shared import ids
from shared.embed import HashEmbedder
from tests.security.test_security import event, hdr, push
from tools.netem_proxy import Faults, NetemProxy
from tools.qdrant_local import free_port


def ev_for(dev, k):
    ep = ids.make_id("nf", dev.cfg.device_id, k)
    return event(device=dev.cfg.device_id, episode=ep, occurred_at=dev._now_iso())


@pytest.mark.parametrize("offset_h", [72, -48])
def test_wrong_device_clock_is_corrected_not_rejected(make_device, cloud, offset_h):
    d, w = make_device("devA", "site1", clock_offset_s=offset_h * 3600)
    e = ev_for(d, 1)
    d.outbox.enqueue(e)
    assert w.push_once()["accepted"] == 1
    stored = cloud["store"].get_event("acme", e["event_id"])
    true_t = dt.datetime.now(dt.timezone.utc)
    assert abs((dt.datetime.fromisoformat(stored["occurred_at"]) - true_t).total_seconds()) < 120
    assert stored["occurred_at_device"] == e["occurred_at"] and abs(stored["clock_offset_s"] / 3600 - offset_h) < 0.1
    w.pull_once()                                            # the cloud tells the device its time
    st = d.clock_status()
    assert st and not st["ok"] and ("ahead" if offset_h > 0 else "behind") in st["text"]
    devs = {x["device_id"]: x for x in cloud["client"].get("/v1/admin/devices", headers=hdr(cloud["admin"])).json()}
    assert abs(devs["devA"]["clock_offset_s"] / 3600 - offset_h) < 0.1


def test_future_event_without_a_device_clock_is_still_refused(cloud):
    tok = cloud["registry"].issue("devX", "s", "acme")
    r = push(cloud["client"], tok, [event(device="devX", occurred_at="2099-01-01T00:00:00+00:00")]).json()
    assert r["results"][0]["status"] == "rejected"


@pytest.fixture
def http_cloud():
    reg = TokenRegistry()
    app = create_app(CloudStore(location=":memory:"), reg, HashEmbedder(), coalesce=True)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield reg, port, app
    server.should_exit = True


def test_every_event_arrives_exactly_once_through_a_lossy_link(tmp_path, http_cloud):
    reg, port, app = http_cloud
    proxy = NetemProxy("127.0.0.1", port, Faults(drop_prob=0.5, latency_ms=5), seed=3)
    proxy.start()
    devs = []
    try:
        for i in range(2):
            d = Device(DeviceConfig(f"lossy{i}", "s", f"m{i}", tmp_path / f"d{i}"), HashEmbedder())
            w = SyncWorker(d, None, reg.issue(f"lossy{i}", "s", "acme"),
                           client=httpx.Client(base_url=proxy.url, timeout=10), mirror_mode="scroll", push_batch=2)
            evs = [ev_for(d, k) for k in range(10)]
            for e in evs:
                d.outbox.enqueue(e)
            devs.append((d, w, evs))
        for _ in range(300):
            busy = False
            for d, w, _ in devs:
                c = d.outbox.counts()
                if c.get("queued") or c.get("failed") or c.get("uploading"):
                    busy = True
                    d.outbox.retry_now()
                    w.push_once()
            if not busy:
                break
        store = app.state.store
        stored = {e["event_id"] for e in store.events("acme")}
        assert all(e["event_id"] in stored for _, _, evs in devs for e in evs)             # nothing lost
        app.state.recomputer.flush("acme")
        counted = sum(c["n_events"] for c in store.cases("acme", since=0, limit=100))
        assert counted == 20                                                                # nothing counted twice
        assert proxy.stats.dropped > 0                                                     # the faults really happened
    finally:
        proxy.stop()
        for d, _, _ in devs:
            d.close()


def test_a_stalled_connection_costs_a_timeout_not_data(tmp_path, http_cloud):
    reg, port, app = http_cloud
    proxy = NetemProxy("127.0.0.1", port, Faults(stall_prob=1.0))
    proxy.start()
    d = Device(DeviceConfig("stall", "s", "m", tmp_path / "d"), HashEmbedder())
    try:
        w = SyncWorker(d, None, reg.issue("stall", "s", "acme"), client=httpx.Client(base_url=proxy.url, timeout=1.0),
                       mirror_mode="scroll")
        e = ev_for(d, 1)
        d.outbox.enqueue(e)
        r = w.push_once()
        assert "error" in r and d.outbox.status_of(e["event_id"]) == "failed"            # kept for retry
        proxy.set(stall_prob=0.0)
        d.outbox.retry_now()
        assert w.push_once()["accepted"] == 1
    finally:
        proxy.stop()
        d.close()
