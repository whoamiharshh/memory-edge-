"""Fleet-mirror pull cost: Qdrant partial snapshots vs the scroll fallback (docs/RESEARCH.md H.2, Part L).

A throwaway Qdrant Server (release binary, free ports) + the real Sync API (in-process) + one device per method.
For fleets of N cases (synthetic case payloads of the real shape, 384-d note vectors, Edge-BM25 text):
  initial   first pull of all N cases            snapshot: full shard snapshot   scroll: all rows
  one       pull after ONE case changed          snapshot: partial snapshot      scroll: rows with seq > cursor
  none      pull when nothing changed            both: the /v1/mirror/head check only
Measured: bytes on the wire (compressed, what the device downloads), wall time, and for snapshots the raw
snapshot size before gzip. Single laptop, localhost: times exclude any real network.
Run: .venv\\Scripts\\python.exe -m bench.mirror_sync
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import subprocess
import tempfile
import time

import httpx
import numpy as np
from fastapi.testclient import TestClient

from cloud import auth as cloud_auth
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
SIZES = (10, 100, 1000)
ACTIONS = ["replace_bearing", "lubricate", "realign", "rebalance", "tighten_mounting"]
CLASSES = ["inner_race", "outer_race", "ball"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(tmp: pathlib.Path) -> tuple[subprocess.Popen, str]:
    http, grpc = free_port(), free_port()
    env = os.environ | {"QDRANT__STORAGE__STORAGE_PATH": str(tmp / "storage"),
                        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(tmp / "snapshots"),
                        "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
                        "QDRANT__TELEMETRY_DISABLED": "true"}
    p = subprocess.Popen([str(ROOT / "qdrant_server" / "qdrant.exe")], cwd=tmp, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{http}"
    for _ in range(120):
        try:
            if httpx.get(url + "/readyz", timeout=1).status_code == 200:
                return p, url
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    p.kill()
    raise RuntimeError("Qdrant Server did not start")


def fake_case(i: int, rng, emb: HashEmbedder, version_bump: int = 0) -> tuple[str, dict, list, list]:
    fc, act = CLASSES[i % 3], ACTIONS[i % 5]
    cid = ids.make_id("benchcase", i)
    text = f"bearing {fc.replace('_', ' ')} fault. action {act.replace('_', ' ')} worked at {1 + i % 4} site(s)" + \
           (f" update {version_bump}" if version_bump else "")
    payload = {"type": "fleet_case", "case_id": cid, "component": f"component{i}", "fault_class": fc, "text": text,
               "n_events": 1 + i % 4, "n_retracted": 0, "n_sites": 1 + i % 4, "sites": ["site1"], "notes": [],
               "status": "active", "flags": [], "updated_at": "2026-09-28T10:00:00+00:00",
               "actions": [{"action_code": act, "worked": 1 + i % 4, "failed": 0, "machine_verified": 1,
                            "sites_worked": ["site1"], "sites_failed": []}]}
    return cid, payload, rng.normal(size=DIM).tolist(), emb.embed_documents([text])[0]


def timed_pull(w: SyncWorker) -> dict:
    before = w.bytes_received
    t = time.perf_counter()
    r = w.pull_once()
    return {"ms": round((time.perf_counter() - t) * 1000, 1), "wire_bytes": w.bytes_received - before,
            "pulled": r.get("pulled"), "kind": r.get("snapshot") or ("head only" if r.get("up_to_date") else r.get("mode")),
            "snapshot_raw_bytes": (w.device.mirror.last_refresh or {}).get("snapshot_bytes") if r.get("snapshot") else None}


def main() -> dict:
    cloud_auth.RATE_LIMIT = 10 ** 9
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mirrorbench_"))
    proc, url = start_server(tmp)
    rng, emb = np.random.default_rng(0), HashEmbedder()
    rows = []
    try:
        store, reg = CloudStore(url=url), TokenRegistry()
        client = TestClient(create_app(store, reg, emb))
        for n in SIZES:
            tenant = f"bench{n}"
            store.ensure_tenant(tenant)
            for i in range(n):
                cid, p, vib, note = fake_case(i, rng, emb)
                assert store.put_case(tenant, cid, p, vib, note, None)
            for mode in ("snapshot", "scroll"):
                dev = Device(DeviceConfig(f"d{mode}{n}", "s", "m", tmp / f"{mode}{n}"), emb)
                w = SyncWorker(dev, None, reg.issue(f"d{mode}{n}", "s", tenant), client=client, mirror_mode=mode)
                initial = timed_pull(w)
                none = timed_pull(w)
                cid, p, vib, note = fake_case(0, rng, emb, version_bump=1)            # one case changes
                cur = store.get_case(tenant, cid)
                assert store.put_case(tenant, cid, p, vib, note, cur["version"])
                one = timed_pull(w)
                assert dev.mirror.count() == n
                disk = sum(f.stat().st_size for f in (tmp / f"{mode}{n}" / "mirror").rglob("*") if f.is_file())
                rows.append({"cases": n, "method": mode, "initial": initial, "one_change": one, "no_change": none,
                             "mirror_disk_bytes": disk})
                print(rows[-1], flush=True)
                dev.close()
    finally:
        proc.kill()
        proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)
    out = {"method": __doc__.strip().splitlines()[0], "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "mirror_sync.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
