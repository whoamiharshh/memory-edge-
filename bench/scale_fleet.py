"""Fleet scale test: N devices pushing and pulling AT THE SAME TIME against the real stack - Qdrant Server 1.19.1
(the release binary), the cloud Sync API served over real HTTP (uvicorn), the device outbox, sync worker and Qdrant
Edge mirror. Measured: push throughput and latency, cloud recompute, mirror pull time and bytes, and consistency
(no event lost or counted twice, every mirror identical).
The evidence records are GENERATED (valid ShareEvents with random component / fault / action / outcome): this is a
load test of the software path, not a data result; nothing is trained on them.
Run: .venv\\Scripts\\python.exe -m bench.scale_fleet [n_devices] [events_per_device]
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import pathlib
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import numpy as np
import uvicorn

from cloud.api import create_app
from cloud.auth import TokenRegistry
from cloud.store_server import CloudStore
from edge.device import Device, DeviceConfig
from edge.fingerprint import DIM
from edge.sync_worker import SyncWorker
from shared import ids
from shared.embed import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
from tools.qdrant_local import binary as _qbin

QDRANT = _qbin()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(url: str) -> None:
    for _ in range(240):
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise SystemExit(f"{url} did not come up")


def event(device: str, k: int, rng: random.Random) -> dict:
    ep = ids.make_id("scale-ep", device, k)
    outcome = "worked" if rng.random() < 0.8 else "failed"
    action = rng.choice(["replace_bearing", "lubricate", "realign", "tighten"])
    return {"event_id": ids.event_id(device, ep, outcome, action), "episode_id": ep, "machine_class": "motor",
            "component": rng.choice(["bearing", "pump", "fan", "motor"]),
            "fault_class": rng.choice(["inner_race", "outer_race", "ball", "imbalance", "misalignment"]),
            "fault_class_source": "technician", "action_code": action, "outcome": outcome, "machine_verified": True,
            "verify_windows_ok": 20, "verify_windows_required": 20, "technician_confirmed": True,
            "fingerprint": [round(rng.gauss(0, 1), 3) for _ in range(DIM)],
            "occurred_at": "2026-09-28T10:00:00+00:00"}


def main(n: int = 20, per: int = 50) -> dict:
    import faulthandler
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)    # a hang prints every thread's stack
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="scale_"))
    http, grpc = free_port(), free_port()
    env = os.environ | {"QDRANT__STORAGE__STORAGE_PATH": str(tmp / "q" / "storage"),
                        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(tmp / "q" / "snapshots"),
                        "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
                        "QDRANT__TELEMETRY_DISABLED": "true"}
    (tmp / "q").mkdir()
    q = subprocess.Popen([str(QDRANT)], cwd=tmp / "q", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    devices = []
    try:
        wait(f"http://127.0.0.1:{http}/readyz")
        reg = TokenRegistry()
        app = create_app(CloudStore(url=f"http://127.0.0.1:{http}"), reg, HashEmbedder(), coalesce=True)
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        cloud = f"http://127.0.0.1:{port}"
        wait(cloud + "/v1/health")
        admin = reg.issue("admin", "hq", "acme", role="admin")
        rng = random.Random(7)
        t0 = time.perf_counter()
        for i in range(n):
            name = f"dev{i:03d}"
            d = Device(DeviceConfig(name, f"site{i % 10}", f"{name}-m1", tmp / name), HashEmbedder())
            w = SyncWorker(d, cloud, reg.issue(name, f"site{i % 10}", "acme"), mirror_mode="auto")
            for k in range(per):
                d.outbox.enqueue(event(name, k, rng))
            devices.append((d, w))
        setup_s = time.perf_counter() - t0

        lat = []

        errors = []

        def push_all(dw):
            d, w = dw
            sent = 0
            for _ in range(500):
                c = d.outbox.counts()
                if not (c.get("queued") or c.get("failed")):
                    break
                t = time.perf_counter()
                r = w.push_once()
                lat.append((time.perf_counter() - t) * 1000)
                sent += r.get("accepted", 0) + r.get("duplicate", 0)
                if "error" in r:
                    errors.append(r["error"])
                    d.outbox.retry_now()           # the worker's backoff, shortened for the benchmark
                    time.sleep(0.2)
            return sent

        print(f"[scale] {n} devices ready ({setup_s:.0f} s); pushing {n * per} events", flush=True)
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(n) as ex:
            sent = sum(ex.map(push_all, devices))
        print(f"[scale] push done in {time.perf_counter() - t0:.1f} s", flush=True)
        push_s = time.perf_counter() - t0
        cases = httpx.get(cloud + "/v1/cases", headers={"authorization": f"Bearer {admin}"}).json()
        counted = sum(c["n_events"] for c in cases)

        pull_s, pull_bytes = [], []

        def pull(dw):
            d, w = dw
            b0, t = w.bytes_received, time.perf_counter()
            r = w.pull_once()
            return (time.perf_counter() - t) * 1000, w.bytes_received - b0, d.mirror.count(), r

        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(n) as ex:
            pulls = list(ex.map(pull, devices))
        print(f"[scale] pull done in {time.perf_counter() - t0:.1f} s", flush=True)
        pull_wall = time.perf_counter() - t0
        res = {
            "method": __doc__.strip().splitlines()[0], "devices": n, "events_per_device": per,
            "events_total": n * per, "device_setup_s": round(setup_s, 1),
            "push": {"wall_s": round(push_s, 2), "events_per_s": round(n * per / push_s, 1),
                     "request_ms_p50": round(float(np.percentile(lat, 50)), 1),
                     "request_ms_p95": round(float(np.percentile(lat, 95)), 1), "requests": len(lat),
                     "transient_errors_retried": len(errors)},
            "consistency": {"acknowledged": sent, "counted_in_cases": counted, "cases": len(cases),
                            "lost": n * per - counted, "double_counted": max(0, counted - n * per),
                            "mirrors_identical": len({p[2] for p in pulls}) == 1 and pulls[0][2] == len(cases)},
            "pull": {"wall_s_all_devices": round(pull_wall, 2),
                     "per_device_ms_p50": round(float(np.percentile([p[0] for p in pulls], 50)), 1),
                     "per_device_ms_p95": round(float(np.percentile([p[0] for p in pulls], 95)), 1),
                     "bytes_per_device_p50": int(np.percentile([p[1] for p in pulls], 50)),
                     "mode": pulls[0][3].get("mode")},
            "machine": "Intel i5-1335U laptop, all processes on one machine (network = loopback)"}
        print(json.dumps(res, indent=1))
        OUT.mkdir(exist_ok=True)
        (OUT / f"scale_fleet_{n}x{per}.json").write_text(json.dumps(res, indent=1))
        server.should_exit = True
        return res
    finally:
        for d, w in devices:
            d.close()
        q.kill()
        q.wait()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
