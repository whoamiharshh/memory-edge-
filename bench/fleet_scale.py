"""Fleet scale: 1,000+ devices syncing with one cloud, on one laptop.

A laptop cannot hold 1,000 COMPLETE devices (each has two Qdrant Edge shards of ~200 MB pre-allocated), so the fleet
has two kinds of member, all talking to the SAME cloud at the same time:
  protocol devices  (default 1,000) speak the exact device protocol - the wire schema, deterministic event ids, batches
                    of <= 50, per-event acks, retry with exponential backoff + jitter, a mirror pull by scroll - but
                    keep their outbox in memory instead of an Edge shard. They load the cloud like real devices.
  complete devices  (default 50) are the real thing: Qdrant Edge shards, SQLite outbox, the sync worker (auto mirror).
The cloud runs as its OWN process (`python -m cloud.main`, coalesced recompute), against Qdrant Server 1.19.1.
The protocol devices are simulated by LOAD_PROCS processes (one process alone saturated a CPU core at 1,000 devices).
Devices wake up at random times within the first WAKE_S seconds (a fleet does not sync in lock-step).
Measured: throughput, request latency, retries, rejected/lost/double-counted events, every mirror equal to the cloud,
cloud and Qdrant memory. The evidence records are GENERATED (a load test; nothing is trained).
Run: .venv\\Scripts\\python.exe -m bench.fleet_scale [protocol_devices] [complete_devices] [events_per_device] [wake_s]
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
import psutil

from cloud.auth import TokenRegistry
from edge.device import Device, DeviceConfig
from edge.fingerprint import DIM
from edge.sync_worker import SyncWorker
from shared import ids
from shared.embed import HashEmbedder
from tools import qdrant_local

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
TENANT, WAKE_S, IN_FLIGHT = "acme", 20.0, 64     # IN_FLIGHT per load process
COMPONENTS = ["bearing", "pump", "fan", "motor", "gearbox"]
FAULTS = ["inner_race", "outer_race", "ball", "imbalance", "misalignment"]


def make_event(dev: str, k: int, rng: random.Random) -> dict:
    ep = ids.make_id("fleet-ep", dev, k)
    outcome = "worked" if rng.random() < 0.8 else "failed"
    action = rng.choice(["replace_bearing", "lubricate", "realign", "tighten", "rebalance"])
    return {"event_id": ids.event_id(dev, ep, outcome, action), "episode_id": ep, "machine_class": "motor",
            "component": rng.choice(COMPONENTS), "fault_class": rng.choice(FAULTS), "fault_class_source": "technician",
            "action_code": action, "outcome": outcome, "machine_verified": True, "verify_windows_ok": 20,
            "verify_windows_required": 20, "technician_confirmed": True,
            "fingerprint": [round(rng.gauss(0, 1), 3) for _ in range(DIM)], "occurred_at": "2026-09-28T10:00:00+00:00"}


async def protocol_device(name: str, token: str, events: list[dict], client: httpx.AsyncClient, sem: asyncio.Semaphore,
                          stats: dict, rng: random.Random) -> None:
    await asyncio.sleep(rng.uniform(0, WAKE_S))
    hdr = {"authorization": f"Bearer {token}", "X-Device-Time": ""}
    queue, tries = list(events), 0
    while queue and tries < 60:
        batch, tries = queue[:50], tries + 1
        async with sem:
            t = time.perf_counter()
            try:
                r = await client.post("/v1/sync/push", json={"batch_id": f"{name}-{tries}", "events": batch}, headers=hdr)
                stats["latency_ms"].append((time.perf_counter() - t) * 1000)
            except httpx.HTTPError as e:
                stats["errors"][type(e).__name__] = stats["errors"].get(type(e).__name__, 0) + 1
                r = None
        if r is not None and r.status_code == 200:
            res = {x["event_id"]: x["status"] for x in r.json()["results"]}
            stats["accepted"] += sum(v == "accepted" for v in res.values())
            stats["duplicate"] += sum(v == "duplicate" for v in res.values())
            stats["rejected"] += sum(v == "rejected" for v in res.values())
            done = {k for k, v in res.items() if v in ("accepted", "duplicate", "rejected")}
            queue = [e for e in queue if e["event_id"] not in done]
            continue
        if r is not None:
            stats["errors"][f"HTTP {r.status_code}"] = stats["errors"].get(f"HTTP {r.status_code}", 0) + 1
        stats["retries"] += 1
        await asyncio.sleep(min(30.0, 0.5 * 2 ** min(tries, 6)) * rng.uniform(0.5, 1.5))   # backoff + jitter
    stats["undelivered"] += len(queue)
    # pull the mirror by scroll (what a device with a small mirror does)
    since, rows, t = 0, 0, time.perf_counter()
    for attempt in range(60):
        async with sem:
            try:
                r = await client.get("/v1/mirror/cases", params={"since": since, "limit": 500}, headers=hdr)
            except httpx.HTTPError as e:                 # e.g. keep-alive race: the server closed an idle connection
                stats["errors"][f"pull {type(e).__name__}"] = stats["errors"].get(f"pull {type(e).__name__}", 0) + 1
                r = None
        if r is None:
            await asyncio.sleep(min(10.0, 0.5 * 2 ** min(attempt, 4)) * rng.uniform(0.5, 1.5))
            continue
        if r.status_code != 200:
            stats["errors"][f"pull HTTP {r.status_code}"] = stats["errors"].get(f"pull HTTP {r.status_code}", 0) + 1
            await asyncio.sleep(1.0)
            continue
        items = r.json()["items"]
        rows += len(items)
        if items:
            since = max(int(it["payload"]["seq"]) for it in items)
        if not r.json()["more"]:
            break
    stats["pull_ms"].append((time.perf_counter() - t) * 1000)
    stats["mirror_rows"].append(rows)


async def final_pulls(cloud: str, tokens: dict[str, str]) -> list[int]:
    sem = asyncio.Semaphore(IN_FLIGHT)

    async def one(tok):
        for attempt in range(10):
            async with sem:
                try:
                    r = await c.get("/v1/mirror/cases", params={"since": 0, "limit": 500},
                                    headers={"authorization": f"Bearer {tok}"})
                    if r.status_code == 200:
                        return len(r.json()["items"])
                except httpx.HTTPError:
                    pass
            await asyncio.sleep(0.5 * (attempt + 1))
        return -1

    async with httpx.AsyncClient(base_url=cloud, timeout=60.0,
                                 limits=httpx.Limits(max_connections=IN_FLIGHT)) as c:
        return list(await asyncio.gather(*(one(t) for t in tokens.values())))


async def run_protocol(cloud: str, tokens: dict[str, str], events: dict[str, list[dict]], seed: int) -> dict:
    stats = {"latency_ms": [], "pull_ms": [], "mirror_rows": [], "accepted": 0, "duplicate": 0, "rejected": 0,
             "retries": 0, "undelivered": 0, "errors": {}}
    sem = asyncio.Semaphore(IN_FLIGHT)
    limits = httpx.Limits(max_connections=IN_FLIGHT, max_keepalive_connections=IN_FLIGHT)
    async with httpx.AsyncClient(base_url=cloud, timeout=httpx.Timeout(60.0, connect=10.0), limits=limits) as c:
        rng = random.Random(seed)
        await asyncio.gather(*(protocol_device(n, tokens[n], events[n], c, sem, stats, random.Random(rng.random()))
                               for n in events))
    return stats


LOAD_PROCS = 6      # the load generator itself saturated one CPU core at 1,000 devices: spread it over processes


def _load_worker(args) -> dict:
    cloud, tokens, proto, seed, wake_s = args
    global WAKE_S
    WAKE_S = wake_s
    return asyncio.run(run_protocol(cloud, tokens, proto, seed))


def run_protocol_parallel(cloud: str, tokens: dict, proto: dict, procs: int) -> dict:
    names = list(proto)
    parts = [names[i::procs] for i in range(procs)]
    jobs = [(cloud, {n: tokens[n] for n in part}, {n: proto[n] for n in part}, 7 + i, WAKE_S)
            for i, part in enumerate(parts) if part]
    with cf.ProcessPoolExecutor(len(jobs)) as ex:
        outs = list(ex.map(_load_worker, jobs))
    merged = {"latency_ms": [], "pull_ms": [], "mirror_rows": [], "accepted": 0, "duplicate": 0, "rejected": 0,
              "retries": 0, "undelivered": 0, "errors": {}}
    for o in outs:
        for k in ("latency_ms", "pull_ms", "mirror_rows"):
            merged[k] += o[k]
        for k in ("accepted", "duplicate", "rejected", "retries", "undelivered"):
            merged[k] += o[k]
        for k, v in o["errors"].items():
            merged["errors"][k] = merged["errors"].get(k, 0) + v
    return merged


def main(n_proto: int = 1000, n_full: int = 50, per: int = 10, wake_s: float = 20.0) -> dict:
    global WAKE_S
    WAKE_S = float(wake_s)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fleetscale_"))
    rng = random.Random(7)
    cloud_proc, full = None, []
    try:
        with qdrant_local.server(tmp / "qdrant") as qurl:
            reg = TokenRegistry(tmp / "cloud" / "tokens.json")
            admin = reg.issue("admin", "hq", TENANT, role="admin")
            proto = {f"p{i:05d}": [make_event(f"p{i:05d}", k, rng) for k in range(per)] for i in range(n_proto)}
            tokens = {n: reg.issue(n, f"site{i % 200}", TENANT, save=False) for i, n in enumerate(proto)}
            full_tokens = {f"f{i:03d}": reg.issue(f"f{i:03d}", f"site{i % 200}", TENANT, save=False) for i in range(n_full)}
            reg.save()                                              # one write for the whole enrolment
            port = qdrant_local.free_port()
            cloud_proc = subprocess.Popen([sys.executable, "-m", "cloud.main", "--qdrant-url", qurl, "--hash-embedder",
                                           "--port", str(port), "--runtime-dir", str(tmp / "cloud")], cwd=ROOT,
                                          stdout=subprocess.DEVNULL, stderr=open(tmp / "cloud.log", "w"))
            cloud = f"http://127.0.0.1:{port}"
            for _ in range(240):
                try:
                    if httpx.get(cloud + "/v1/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.25)
            t0 = time.perf_counter()
            for n, tok in full_tokens.items():
                d = Device(DeviceConfig(n, "s", f"{n}-m", tmp / "dev" / n), HashEmbedder())
                w = SyncWorker(d, cloud, tok, mirror_mode="auto")
                evs = [make_event(n, k, rng) for k in range(per)]
                for e in evs:
                    d.outbox.enqueue(e)
                full.append((d, w, evs))
            setup_s = time.perf_counter() - t0
            print(f"[fleet] {n_proto} protocol + {n_full} complete devices ready ({setup_s:.0f} s)", flush=True)

            def full_device(dw):
                d, w, _ = dw
                time.sleep(random.uniform(0, WAKE_S))
                t, tries = time.perf_counter(), 0
                while tries < 200:
                    c = d.outbox.counts()
                    if not (c.get("queued") or c.get("failed") or c.get("uploading")):
                        break
                    d.outbox.retry_now()
                    w.push_once()
                    tries += 1
                push_ms = (time.perf_counter() - t) * 1000
                return push_ms, tries

            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max(1, n_full)) as ex:
                fut_full = ex.map(full_device, full)
                pstats = run_protocol_parallel(cloud, tokens, proto, LOAD_PROCS)
                full_push = list(fut_full)
            wall = time.perf_counter() - t0
            print(f"[fleet] pushes done in {wall:.0f} s", flush=True)
            hdr = {"authorization": f"Bearer {admin}"}
            cases = httpx.get(cloud + "/v1/cases", headers=hdr, timeout=120).json()
            counted = sum(c["n_events"] for c in cases)
            t1 = time.perf_counter()

            def pull(dw):
                d, w, _ = dw
                t = time.perf_counter()
                for _ in range(20):
                    w.pull_once()
                    if d.mirror.count() == len(cases):
                        break
                return (time.perf_counter() - t) * 1000, d.mirror.count()

            with cf.ThreadPoolExecutor(max(1, n_full)) as ex:
                full_pull = list(ex.map(pull, full))
            pull_wall = time.perf_counter() - t1
            final_rows = asyncio.run(final_pulls(cloud, tokens))      # eventual consistency: after all pushes
            cp, qp = psutil.Process(cloud_proc.pid), None
            cloud_rss = cp.memory_info().rss + sum(ch.memory_info().rss for ch in cp.children(recursive=True))
            for pr in psutil.process_iter(["name"]):
                if (pr.info["name"] or "").lower().startswith("qdrant"):
                    qp = pr
            total = (n_proto + n_full) * per
            lat = np.asarray(pstats["latency_ms"])
            res = {
                "method": __doc__.strip().splitlines()[0],
                "protocol": __doc__.split("A laptop")[1].split("Run:")[0].strip(),
                "devices": {"protocol": n_proto, "complete": n_full}, "events_per_device": per, "events_total": total,
                "wake_window_s": WAKE_S,
                "wall_s_all_pushes": round(wall, 1), "events_per_s": round(total / wall, 1),
                "push_request_ms": {"p50": round(float(np.percentile(lat, 50)), 1),
                                    "p95": round(float(np.percentile(lat, 95)), 1),
                                    "p99": round(float(np.percentile(lat, 99)), 1), "requests": int(len(lat))},
                "retries": pstats["retries"], "errors": pstats["errors"],
                "consistency": {"accepted_or_duplicate_acks": pstats["accepted"] + pstats["duplicate"],
                                "rejected": pstats["rejected"], "undelivered": pstats["undelivered"],
                                "counted_in_cases": counted, "lost": total - counted if counted <= total else 0,
                                "double_counted": max(0, counted - total), "cases": len(cases),
                                "protocol_mirrors_equal_cloud_after_all_pushes": sum(r == len(cases) for r in final_rows),
                                "protocol_mirrors_equal_cloud_during_load": sum(r == len(cases) for r in pstats["mirror_rows"]),
                                "complete_mirrors_equal_cloud": sum(m == len(cases) for _, m in full_pull)},
                "complete_devices": {"push_ms_p50": round(float(np.percentile([p for p, _ in full_push], 50)), 1),
                                     "pull_ms_p50": round(float(np.percentile([p for p, _ in full_pull], 50)), 1),
                                     "pull_wall_s": round(pull_wall, 1)},
                "protocol_pull_ms": {"p50": round(float(np.percentile(pstats["pull_ms"], 50)), 1),
                                     "p95": round(float(np.percentile(pstats["pull_ms"], 95)), 1)},
                "memory_mb": {"cloud_process": round(cloud_rss / 2 ** 20),
                              "qdrant_server": round(qp.memory_info().rss / 2 ** 20) if qp else None},
                "machine": "Intel i5-1335U laptop, 15.7 GB; cloud, Qdrant and every device on the same machine (loopback)"}
            print(json.dumps(res, indent=1), flush=True)
            OUT.mkdir(exist_ok=True)
            (OUT / f"fleet_scale_{n_proto}p_{n_full}c.json").write_text(json.dumps(res, indent=1))
            return res
    finally:
        for d, _, _ in full:
            d.close()
        if cloud_proc:
            cloud_proc.kill()
            cloud_proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:4]), *(float(a) for a in sys.argv[4:5]))
