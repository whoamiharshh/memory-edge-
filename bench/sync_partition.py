"""Sync under partitions and crashes, K runs (docs/RESEARCH.md Part O, "Sync").

Per run (seed logged): N devices each queue E share events, then a random schedule of:
  - network partitions toggled on/off per device,
  - requests dropped BEFORE they reach the cloud (p=0.3),
  - acks lost AFTER the cloud applied the batch (p=0.3)  -> the device resends, the cloud must say "duplicate",
  - hard device restarts (process state dropped, no graceful close of the worker) incl. one forced per device.
Then the network heals and the outboxes drain. Checked against the cloud (in-process Qdrant + the real Sync API):
  lost events = expected - stored        duplicates = stored ids appearing twice
  tally error = |sum of worked+failed over all cases - expected|   (double counting would show here)
Run: .venv\\Scripts\\python.exe -m bench.sync_partition [runs] [devices] [events_per_device]
"""
from __future__ import annotations

import json
import pathlib
import random
import shutil
import sys
import tempfile
import time

from fastapi.testclient import TestClient

from cloud import auth as cloud_auth
from cloud.api import create_app
from cloud.auth import TokenRegistry
from cloud.store_server import CloudStore
from edge.device import Device, DeviceConfig
from edge.sync_worker import SyncWorker
from shared.embed import HashEmbedder
from tests.failure.test_sync_partition import Flaky, make_event

OUT = pathlib.Path(__file__).resolve().parent / "results"


def one_run(seed: int, n_devices: int, events: int) -> dict:
    # This loop fires requests far faster than a real device; lift the per-device rate limit so a 429 is not
    # mistaken for a sync fault (the rate limit itself is tested in tests/security).
    cloud_auth.RATE_LIMIT = 10 ** 9
    rng = random.Random(seed)
    store, reg = CloudStore(location=":memory:"), TokenRegistry()
    client = TestClient(create_app(store, reg, HashEmbedder()))
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="part_"))
    tokens = {f"dev{i}": reg.issue(f"dev{i}", f"site-dev{i}", "acme") for i in range(n_devices)}
    flaky_stats = {"dropped_before": 0, "ack_lost": 0, "ok": 0}
    devs, expected, restarts, toggles = {}, set(), 0, 0

    def boot(name):
        d = Device(DeviceConfig(device_id=name, site_id=f"site-{name}", machine_id=f"{name}-m", root=tmp / name),
                   HashEmbedder())
        return d, SyncWorker(d, None, tokens[name], client=Flaky(client, rng))

    def harvest(w):
        for k, v in w.http.stats.items():
            flaky_stats[k] += v

    t0 = time.perf_counter()
    try:
        made = {name: 0 for name in tokens}                      # events are created DURING the chaos, a few at
        for name in tokens:                                      # a time, so there are many small pushes
            devs[name] = boot(name)
        forced = {7 * (k + 1): f"dev{k}" for k in range(n_devices)}
        rounds = 0
        for rounds in range(1, 40 * n_devices * events):
            if rounds in forced:
                harvest(devs[forced[rounds]][1])
                devs[forced[rounds]][0].close()
                devs[forced[rounds]] = boot(forced[rounds])
                restarts += 1
            name = rng.choice(list(devs))
            d, w = devs[name]
            if made[name] < events and rng.random() < 0.6:
                e = make_event(name, made[name], "worked" if rng.random() < 0.7 else "failed")
                d.outbox.enqueue(e)
                expected.add(e["event_id"])
                made[name] += 1
            roll = rng.random()
            if roll < 0.15:
                w.set_online(not w.online)
                toggles += 1
            elif roll < 0.22:
                harvest(w)
                d.close()
                devs[name] = boot(name)
                restarts += 1
                continue
            d.outbox.retry_now()
            w.push_once()
            if all(m == events for m in made.values()) and \
                    all(dv.outbox.counts().keys() <= {"synced"} for dv, _ in devs.values()):
                break
        for d, w in devs.values():                               # heal and drain
            w.set_online(True)
            w.http.p_before = w.http.p_after = 0.0
            for _ in range(20):
                d.outbox.retry_now()
                w.push_once()
            harvest(w)
        stored = [e["event_id"] for e in store.events("acme")]
        tallied = sum(a["worked"] + a["failed"] for c in store.cases("acme") for a in c["actions"])
        not_synced = sum(sum(v for k, v in d.outbox.counts().items() if k != "synced") for d, _ in devs.values())
    finally:
        for d, _ in devs.values():
            d.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return {"seed": seed, "expected": len(expected), "stored": len(stored),
            "lost": len(expected - set(stored)), "duplicates": len(stored) - len(set(stored)),
            "tally_error": abs(tallied - len(expected)), "left_unsynced": not_synced, "restarts": restarts,
            "partition_toggles": toggles, "rounds": rounds, **flaky_stats,
            "seconds": round(time.perf_counter() - t0, 1)}


def main(runs: int = 20, n_devices: int = 5, events: int = 30) -> dict:
    rows = []
    for seed in range(1, runs + 1):
        rows.append(one_run(seed, n_devices, events))
        print(rows[-1], flush=True)
    out = {"method": __doc__.strip().splitlines()[0], "runs": runs, "devices": n_devices, "events_per_device": events,
           "total_events": sum(r["expected"] for r in rows),
           "total_lost": sum(r["lost"] for r in rows), "total_duplicates": sum(r["duplicates"] for r in rows),
           "total_tally_error": sum(r["tally_error"] for r in rows),
           "total_restarts": sum(r["restarts"] for r in rows),
           "total_requests_dropped": sum(r["dropped_before"] for r in rows),
           "total_acks_lost": sum(r["ack_lost"] for r in rows), "per_run": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "sync_partition.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_run"}, indent=2))
    return out


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a)
