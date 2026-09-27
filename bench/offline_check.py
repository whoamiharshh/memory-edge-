"""Offline checklist (docs/RESEARCH.md M.4 + Part O "Offline"): every device function with the network switched off,
while a socket guard BLOCKS AND RECORDS every connection attempt the process makes (to any address, loopback too).

Setup (network allowed): an in-process cloud; Device A shares a verified fix; Device B pulls the fleet mirror.
Then the guard is armed and, with Device B's network switch OFF, the script runs:
  load text model + local LLM from disk, fit baseline, gate on healthy + fault windows, episode opened, note saved,
  fault class confirmed, action recorded, outcome verified by the sensor, policy decision, SHARE queued (not sent),
  local + fleet-mirror hybrid search, evidence brief, sync-worker ticks (must not try to connect).
PASS = the function worked AND zero connection attempts were recorded.
Wi-Fi-off is a manual rehearsal step (docs/DEMO.md); this guard is the stricter, scripted check of the same thing.
Run: .venv\\Scripts\\python.exe -m bench.offline_check [--hash]   (--hash: lexical test embedder, no model files)
"""
from __future__ import annotations

import json
import pathlib
import shutil
import socket
import sys
import tempfile
import time

from fastapi.testclient import TestClient

from cloud.api import create_app
from cloud.auth import TokenRegistry
from cloud.store_server import CloudStore
from edge.device import Device, DeviceConfig
from edge.replay import Recordings
from edge.sync_worker import SyncWorker
from shared.embed import HashEmbedder

OUT = pathlib.Path(__file__).resolve().parent / "results"


class SocketGuard:
    """Replaces socket connect paths with a recorder that refuses the connection."""

    def __init__(self):
        self.attempts: list[str] = []
        self._saved = {}

    def _refuse(self, what):
        def f(*a, **kw):
            self.attempts.append(f"{what} {a[1:] if what.startswith('socket.') else a}"[:200])
            raise OSError("offline check: network access blocked")
        return f

    def __enter__(self):
        self._saved = {"connect": socket.socket.connect, "connect_ex": socket.socket.connect_ex,
                       "create_connection": socket.create_connection, "getaddrinfo": socket.getaddrinfo}
        socket.socket.connect = self._refuse("socket.connect")
        socket.socket.connect_ex = self._refuse("socket.connect_ex")
        socket.create_connection = self._refuse("create_connection")
        socket.getaddrinfo = self._refuse("getaddrinfo")
        return self

    def __exit__(self, *exc):
        socket.socket.connect = self._saved["connect"]
        socket.socket.connect_ex = self._saved["connect_ex"]
        socket.create_connection = self._saved["create_connection"]
        socket.getaddrinfo = self._saved["getaddrinfo"]


def run(real_models: bool = True) -> dict:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="offline_"))
    checks: list[dict] = []
    try:
        # ---------------- setup, network allowed ----------------
        store, reg = CloudStore(location=":memory:"), TokenRegistry()
        client = TestClient(create_app(store, reg, HashEmbedder()))
        a = Device(DeviceConfig("devA", "site1", "mA", tmp / "a"), HashEmbedder())
        wa = SyncWorker(a, None, reg.issue("devA", "site1", "acme"), client=client)
        a.fit_baseline(Recordings.baseline())
        for w in Recordings.windows(105)[:25]:
            a.ingest_window(w)
        ea = a.episodes()[0]["episode_id"]
        a.set_fault_class(ea, "inner_race")
        a.record_action(ea, "replace_bearing")
        for w in Recordings.windows(99)[:20]:
            a.ingest_window(w)
        a.confirm_outcome(ea, "worked")
        assert wa.push_once()["accepted"] == 1
        a.close()
        b_tok = reg.issue("devB", "site2", "acme")
        pre = Device(DeviceConfig("devB", "site2", "mB", tmp / "b"), HashEmbedder())
        SyncWorker(pre, None, b_tok, client=client).pull_once()
        mirror_cases = pre.mirror.count()
        pre.close()
        shutil.rmtree(tmp / "b" / "local")      # setup only fills B's MIRROR; B's (empty) local shard is rebuilt
                                                # offline with the real text model (a store refuses a model change)

        # ---------------- the offline run ----------------
        with SocketGuard() as guard:
            def check(name, fn):
                before = len(guard.attempts)
                t = time.perf_counter()
                try:
                    detail = fn()
                    ok = True
                except Exception as e:                       # a failing function is a FAIL, never a crash
                    detail, ok = f"{type(e).__name__}: {e}"[:200], False
                new = guard.attempts[before:]
                checks.append({"check": name, "ok": ok and not new, "detail": detail,
                               "connection_attempts": new, "ms": round((time.perf_counter() - t) * 1000, 1)})
                return detail

            if real_models:
                from edge import rag
                from shared.embed import load_embedder
                loaded = {}
                check("load bge-small text model from disk",
                      lambda: loaded.setdefault("emb", load_embedder()).name)
                check("load local LLM from disk",
                      lambda: _expect(loaded.setdefault("llm", rag.LocalLLM()).available, rag.MODEL_NAME))
                emb, llm = loaded.get("emb") or HashEmbedder(), loaded.get("llm")
            else:
                emb, llm = HashEmbedder(), None
            b = Device(DeviceConfig("devB", "site2", "mB", tmp / "b"), emb)
            wb = SyncWorker(b, "http://127.0.0.1:8100", b_tok)       # a REAL http client this time
            wb.set_online(False)
            check("mirror from last sync is present",
                  lambda: _expect(b.mirror.count() == mirror_cases > 0, f"{mirror_cases} fleet case(s)"))
            check("fit healthy baseline", lambda: f"tau_normal {b.fit_baseline(Recordings.baseline())['tau_normal']:.2f}")
            check("gate: unseen healthy windows are normal",
                  lambda: _expect(all(b.ingest_window(w)["state"] == "normal" for w in Recordings.windows(100)[:20]), "20/20 normal"))
            check("gate: fault opens ONE episode (unseen 14-mil bearing, CWRU 169)",
                  lambda: _expect([b.ingest_window(w)["state"] for w in Recordings.windows(169)[:15]].count("new") == 1, "1 new + 14 merged"))
            eb = b.episodes()[0]["episode_id"]
            check("technician note saved", lambda: b.set_note(eb, "inner race noise, bearing swapped")["action"])
            check("fault class confirmed", lambda: b.set_fault_class(eb, "inner_race")["action"])
            check("hybrid search: local + fleet mirror",
                  lambda: _expect(len(b.search(episode_id=eb)["fleet"]) >= 1, "fleet evidence found offline"))
            if llm is not None and getattr(llm, "available", False):
                check("evidence brief with the local LLM",
                      lambda: rag.brief("What has been tried?", b.search(episode_id=eb), llm)["mode"])
            check("action recorded", lambda: b.record_action(eb, "replace_bearing")["action"])
            check("outcome verified by the sensor",
                  lambda: [b.ingest_window(w) for w in Recordings.windows(99)[:20]] and b.episode(eb)["verify"]["verdict"])
            check("policy decides SHARE with reasons",
                  lambda: _expect(b.confirm_outcome(eb, "worked")["action"] == "SHARE", "SHARE"))
            check("SHARE waits in the outbox (QUEUED)", lambda: _expect(b.outbox.counts() == {"queued": 1}, "1 queued"))
            check("sync worker ticks while offline: no connection attempt",
                  lambda: [wb.tick() for _ in range(3)] and wb.push_once())
            b.close()
            attempts = list(guard.attempts)
            # guard self-test: a deliberate request MUST be recorded and refused, or "0 attempts" would prove nothing
            import httpx
            try:
                httpx.get("http://127.0.0.1:9/", timeout=1)
                self_test = False
            except Exception:
                self_test = len(guard.attempts) > len(attempts)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = {"method": __doc__.strip().splitlines()[0], "real_models": real_models,
           "passed": f"{sum(c['ok'] for c in checks)}/{len(checks)}", "connection_attempts_total": len(attempts),
           "guard_self_test_caught_a_deliberate_request": self_test, "checks": checks}
    return out


def _expect(cond: bool, detail: str) -> str:
    if not cond:
        raise AssertionError("expected: " + detail)
    return detail


if __name__ == "__main__":
    res = run(real_models="--hash" not in sys.argv)
    OUT.mkdir(exist_ok=True)
    (OUT / "offline_check.json").write_text(json.dumps(res, indent=2, default=str))
    for c in res["checks"]:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"], "-", c["detail"], c["connection_attempts"] or "")
    print("passed", res["passed"], "| connection attempts:", res["connection_attempts_total"],
          "| guard self-test:", res["guard_self_test_caught_a_deliberate_request"])
