"""Partition / crash harness (docs/RESEARCH.md Part O, "Sync" metric): several devices, random network
partitions, requests dropped before they reach the cloud, acks lost AFTER the cloud applied the batch, and
device restarts mid-run. Invariant: every queued event reaches the cloud exactly once (0 lost, 0 double-counted)."""
import pathlib
import random

import httpx
import pytest

from edge.device import Device, DeviceConfig
from edge.fingerprint import DIM
from edge.sync_worker import SyncWorker
from shared import ids
from shared.embed import HashEmbedder

N_DEVICES, EVENTS_PER_DEVICE = 3, 15


class Flaky:
    """Wraps the cloud client: drops some requests before sending and loses some acks after the server acted."""

    def __init__(self, inner, rng, p_before=0.3, p_after=0.3):
        self.inner, self.rng, self.p_before, self.p_after = inner, rng, p_before, p_after
        self.stats = {"dropped_before": 0, "ack_lost": 0, "ok": 0}

    def _call(self, fn, *a, **kw):
        if self.rng.random() < self.p_before:
            self.stats["dropped_before"] += 1
            raise httpx.ConnectError("partition: request never left")
        resp = fn(*a, **kw)
        if self.rng.random() < self.p_after:
            self.stats["ack_lost"] += 1
            raise httpx.ReadTimeout("partition: server applied it, ack lost")
        self.stats["ok"] += 1
        return resp

    def post(self, *a, **kw):
        return self._call(self.inner.post, *a, **kw)

    def get(self, *a, **kw):
        return self._call(self.inner.get, *a, **kw)


def make_event(device: str, i: int, outcome: str) -> dict:
    ep = ids.make_id("ep", device, i)
    action = ["replace_bearing", "lubricate", "realign"][i % 3]
    fc = ["inner_race", "outer_race", "ball"][i % 3]
    return {"event_id": ids.event_id(device, ep, outcome, action), "episode_id": ep, "machine_class": "m",
            "component": "bearing", "fault_class": fc, "fault_class_source": "technician", "action_code": action,
            "outcome": outcome, "machine_verified": True, "verify_windows_ok": 20, "verify_windows_required": 20,
            "technician_confirmed": True, "fingerprint": [float(i)] * DIM, "occurred_at": "2026-09-27T10:00:00+00:00"}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_lost_and_no_duplicate_events_under_partitions_and_restarts(tmp_path, cloud, seed):
    rng = random.Random(seed)
    reg, client = cloud["registry"], cloud["client"]
    devs = {}
    expected = set()

    def boot(name):
        d = Device(DeviceConfig(device_id=name, site_id=f"site-{name}", machine_id=f"{name}-m",
                                root=pathlib.Path(tmp_path) / name), HashEmbedder())
        w = SyncWorker(d, None, tokens[name], client=Flaky(client, rng))
        return d, w

    tokens = {f"dev{i}": reg.issue(f"dev{i}", f"site-dev{i}", "acme") for i in range(N_DEVICES)}
    for name in tokens:
        devs[name] = boot(name)
        for i in range(EVENTS_PER_DEVICE):
            e = make_event(name, i, "worked" if rng.random() < 0.7 else "failed")
            devs[name][0].outbox.enqueue(e)
            expected.add(e["event_id"])

    restarts = 0
    for _round in range(400):
        name = rng.choice(list(devs))
        forced = {3: "dev0", 6: "dev1", 9: "dev2"}.get(_round)   # every device is killed at least once mid-sync
        if forced:
            devs[forced][0].close()
            devs[forced] = boot(forced)
            restarts += 1
        d, w = devs[name]
        roll = rng.random()
        if roll < 0.15:
            w.set_online(not w.online)
        elif roll < 0.22:                                   # crash + restart (no graceful anything)
            d.close()
            devs[name] = boot(name)
            restarts += 1
            continue
        d.outbox.retry_now()                                # skip backoff sleeps in the test
        w.push_once()
        if _round > 30 and all(dv.outbox.counts().keys() <= {"synced"} for dv, _ in devs.values()):
            break
    for name, (d, w) in devs.items():                       # heal the partition and drain
        w.set_online(True)
        w.http.p_before = w.http.p_after = 0.0
        for _ in range(10):
            d.outbox.retry_now()
            w.push_once()

    stored = cloud["store"].events("acme")
    stored_ids = [e["event_id"] for e in stored]
    assert len(stored_ids) == len(set(stored_ids)), "duplicate events stored"
    assert set(stored_ids) == expected, f"lost: {len(expected - set(stored_ids))}"
    total = sum(a["worked"] + a["failed"] for c in cloud["store"].cases("acme") for a in c["actions"])
    assert total == len(expected), "tallies double-counted or missed events"
    for d, _ in devs.values():
        assert d.outbox.counts() == {"synced": EVENTS_PER_DEVICE}
        d.close()
    assert restarts >= N_DEVICES
