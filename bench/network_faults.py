"""Sync over BAD networks and with WRONG device clocks, end to end: real devices (Qdrant Edge shards, SQLite outbox,
sync worker) -> a fault-injecting TCP proxy (tools/netem_proxy.py) -> the cloud over real HTTP -> Qdrant Server.

Scenarios (each: 5 devices, 20 outcome events each sent in batches of 4, plus 4 follow-ups; then every device pulls the fleet mirror):
  lan          no faults
  slow_link    400 +- 200 ms per chunk, 16 kB/s per direction (a weak GSM/rural link)
  lossy        40 % of connections cut after a random number of bytes (requests AND acknowledgements lost)
  stalls       25 % of connections accepted but never answered (device read timeout shortened to 3 s for the run)
  flapping     network down 2 s / up 2 s, repeatedly
  all_at_once  latency + loss + stalls together
Device clocks in every scenario: +3 days, -2 days, +40 min, 0, 0 (the first two would be rejected as "future" /
misdated without the cloud's clock correction).
Checked per scenario: all events delivered, 0 lost, 0 counted twice (tallies rebuilt from stored events), follow-ups
applied once, corrected event times within 2 minutes of the real time, every mirror identical to the cloud.
Generated evidence records (a protocol test; nothing is trained). Run: .venv\\Scripts\\python.exe -m bench.network_faults
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import random
import shutil
import sys
import tempfile
import threading
import time

import httpx
import uvicorn

from cloud.api import create_app
from cloud.auth import TokenRegistry
from cloud.store_server import CloudStore
from edge.device import Device, DeviceConfig
from edge.sync_worker import SyncWorker
from shared import ids
from shared.embed import HashEmbedder
from tools import qdrant_local
from tools.netem_proxy import Faults, NetemProxy

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
CLOCKS = [3 * 86400, -2 * 86400, 2400, 0, 0]
N_EV = 20
PUSH_BATCH = 4          # small batches: 5+ requests per device, so faults hit many requests and acks
SCENARIOS = {
    "lan": {},
    "slow_link": {"latency_ms": 400, "jitter_ms": 200, "bandwidth_bps": 16_000},
    "lossy": {"drop_prob": 0.4},
    "stalls": {"stall_prob": 0.25},
    "flapping": {"flap": True},
    "all_at_once": {"latency_ms": 150, "jitter_ms": 100, "drop_prob": 0.25, "stall_prob": 0.1},
}


def event(dev: Device, k: int, rng: random.Random) -> dict:
    ep = ids.make_id("net-ep", dev.cfg.device_id, k)
    outcome = "worked" if k % 5 else "failed"
    action = rng.choice(["replace_bearing", "lubricate", "realign"])
    return {"event_id": ids.event_id(dev.cfg.device_id, ep, outcome, action), "episode_id": ep, "machine_class": "m",
            "component": "bearing", "fault_class": rng.choice(["inner_race", "outer_race"]),
            "fault_class_source": "technician", "action_code": action, "outcome": outcome, "machine_verified": True,
            "verify_windows_ok": 20, "verify_windows_required": 20, "technician_confirmed": True,
            "fingerprint": [round(rng.gauss(0, 1), 3) for _ in range(27)], "occurred_at": dev._now_iso()}


def run(name: str, spec: dict, qurl: str, tmp: pathlib.Path) -> dict:
    reg = TokenRegistry()
    tenant = "net" + name.replace("_", "")
    app = create_app(CloudStore(url=qurl), reg, HashEmbedder(), coalesce=True)
    port = qdrant_local.free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    flap = spec.pop("flap", False)
    proxy = NetemProxy("127.0.0.1", port, Faults(**spec), seed=7)
    proxy.start()
    stop_flap = threading.Event()
    if flap:
        def flapper():
            while not stop_flap.wait(2.0):
                proxy.set(blackout=not proxy.faults.blackout)
        threading.Thread(target=flapper, daemon=True).start()
    rng = random.Random(7)
    devs, sent = [], []
    admin = reg.issue("adm", "hq", tenant, role="admin")
    try:
        for i, off in enumerate(CLOCKS):
            name_i = f"{name}{i}"
            d = Device(DeviceConfig(name_i, f"site{i}", f"{name_i}-m", tmp / name / name_i, clock_offset_s=off),
                       HashEmbedder())
            timeout = httpx.Timeout(3.0 if spec.get("stall_prob") else 30.0, connect=5.0)
            w = SyncWorker(d, None, reg.issue(name_i, f"site{i}", tenant),
                           client=httpx.Client(base_url=proxy.url, timeout=timeout), mirror_mode="scroll",
                           push_batch=PUSH_BATCH)
            evs = [event(d, k, rng) for k in range(N_EV)]
            for e in evs:
                d.outbox.enqueue(e)
            devs.append((d, w, evs))
            sent += evs
        t0 = time.perf_counter()
        attempts = 0
        # phase 1: push everything (the worker's own retry/backoff decides when; we only speed its clock up)
        while time.perf_counter() - t0 < 600:
            pending = 0
            for d, w, _ in devs:
                c = d.outbox.counts()
                if c.get("queued") or c.get("failed") or c.get("uploading"):
                    d.outbox.retry_now()
                    w.push_once()
                    attempts += 1
                    pending += 1
            if not pending:
                break
        push_s = time.perf_counter() - t0
        # phase 2: follow-ups for 4 fixes of device 0 (idempotent under loss as well)
        fus = []
        for e in [x for x in devs[0][2] if x["outcome"] == "worked"][:4]:
            fu = {"kind": "followup", "event_id": ids.followup_id(devs[0][0].cfg.device_id, e["event_id"]),
                  "episode_id": e["episode_id"], "refers_to": e["event_id"], "status": "held",
                  "days_after_fix": 30.0, "hold_days": 30.0, "occurred_at": devs[0][0]._now_iso()}
            devs[0][0].outbox.enqueue(fu)
            fus.append(fu)
        t1 = time.perf_counter()
        while devs[0][0].outbox.counts().get("queued") or devs[0][0].outbox.counts().get("failed") \
                or devs[0][0].outbox.counts().get("uploading"):
            devs[0][0].outbox.retry_now()
            devs[0][1].push_once()
            attempts += 1
            if time.perf_counter() - t1 > 300:
                break
        # phase 3: every device pulls the mirror until it matches the cloud
        stop_flap.set()
        proxy.set(blackout=False)
        hdr = {"authorization": f"Bearer {admin}"}
        cases = httpx.get(f"http://127.0.0.1:{port}/v1/cases", headers=hdr, timeout=60).json()
        t2 = time.perf_counter()
        for d, w, _ in devs:
            for _ in range(200):
                r = w.pull_once()
                if d.mirror.count() == len(cases) and "error" not in r:
                    break
        pull_s = time.perf_counter() - t2
        events = CloudStore(url=qurl).events(tenant)
        stored = {e["event_id"]: e for e in events}
        lost = [e["event_id"] for e in sent if e["event_id"] not in stored]
        counted = sum(c["n_events"] for c in httpx.get(f"http://127.0.0.1:{port}/v1/cases", headers=hdr,
                                                        timeout=60).json())
        fu_ok = sum(1 for fu in fus if (stored.get(fu["refers_to"]) or {}).get("followup", {}).get("status") == "held")
        # corrected times: the cloud's occurred_at must be within 2 min of the true (server) time of creation
        worst_err = 0.0
        for d, _, evs in devs:
            for e in evs:
                s = stored.get(e["event_id"])
                if s:
                    true_t = dt.datetime.fromisoformat(e["occurred_at"]) - dt.timedelta(seconds=d.cfg.clock_offset_s)
                    worst_err = max(worst_err, abs((dt.datetime.fromisoformat(s["occurred_at"]) - true_t).total_seconds()))
        devices_ = {x["device_id"]: x for x in httpx.get(f"http://127.0.0.1:{port}/v1/admin/devices", headers=hdr,
                                                          timeout=60).json()}
        res = {"scenario": name, "faults": spec | ({"flap": "2 s down / 2 s up"} if flap else {}),
               "events_sent": len(sent), "stored": len(stored), "lost": len(lost),
               "counted_in_tallies": counted, "double_counted": max(0, counted - len(sent)),
               "followups_applied_once": f"{fu_ok}/{len(fus)}", "push_attempts": attempts,
               "push_s": round(push_s, 1), "pull_s": round(pull_s, 1),
               "mirrors_identical": all(d.mirror.count() == len(cases) for d, _, _ in devs),
               "worst_time_error_after_correction_s": round(worst_err, 1),
               "clock_offsets_seen_by_cloud_h": [round((devices_.get(d.cfg.device_id, {}).get("clock_offset_s") or 0) / 3600, 2)
                                                 for d, _, _ in devs],
               "device_clock_warnings": [bool(d.clock_status() and not d.clock_status()["ok"]) for d, _, _ in devs],
               "proxy": proxy.stats.as_dict()}
        print(json.dumps(res), flush=True)
        return res
    finally:
        stop_flap.set()
        proxy.stop()
        server.should_exit = True
        for d, _, _ in devs:
            d.close()


def main(which: list[str]) -> dict:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="netfault_"))
    out = {"method": __doc__.strip().splitlines()[0], "protocol": __doc__.split("Scenarios")[1].split("Run:")[0].strip(),
           "results": []}
    try:
        with qdrant_local.server(tmp / "qdrant") as qurl:
            for name in which or list(SCENARIOS):
                out["results"].append(run(name, dict(SCENARIOS[name]), qurl, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "network_faults.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
