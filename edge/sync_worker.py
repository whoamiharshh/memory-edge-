"""Sync worker: pushes queued share events to the cloud and pulls the fleet mirror.

Push: claim due outbox rows (-> uploading), POST one batch, apply per-event acks:
  accepted | duplicate -> synced      rejected -> rejected (reason kept, visible in the UI)
  transport error / 5xx / 429 -> failed, retried with capped exponential backoff + full jitter
  401 / 403 -> rows back to queued, worker pauses with "auth required" (no data lost)
Pull (scroll-based mirror; see docs/RESEARCH.md H.2 / kill test K5): GET cases with seq > cursor, upsert them
into the read-only mirror shard, advance the cursor only after the shard write returned (durable).
The `online` switch simulates a network partition for the demo: when off, no request is attempted at all.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import httpx

from edge.device import Device
from edge.store_edge import StorePoint

PULL_EVERY_S = 5.0


class SyncWorker:
    def __init__(self, device: Device, cloud_url: str | None, token: str | None,
                 client: httpx.Client | None = None, interval: float = 2.0):
        self.device, self.cloud_url, self.token = device, (cloud_url or "").rstrip("/"), token
        self.http = client or httpx.Client(base_url=self.cloud_url, timeout=5.0)
        self.interval = interval
        self.auth_required = False
        self.last: dict[str, Any] = {"push": None, "pull": None, "error": None}
        self.bytes_sent = 0
        self.bytes_received = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_pull = 0.0

    # ---- partition switch -------------------------------------------------------------------------------
    @property
    def online(self) -> bool:
        return bool(self.device.outbox.kv_get("online", True))

    def set_online(self, on: bool) -> None:
        self.device.outbox.kv_set("online", bool(on))
        self.device.outbox.log("network", "network ONLINE" if on else "network OFFLINE (simulated partition)")
        if on:
            self.device.outbox.retry_now()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # ---- push -------------------------------------------------------------------------------------------
    def push_once(self) -> dict:
        with self._lock:
            if not self.online:
                return {"skipped": "offline"}
            ob = self.device.outbox
            events = ob.claim_due(limit=50)
            if not events:
                return {"sent": 0}
            body = {"batch_id": uuid.uuid4().hex, "events": events}
            ids_ = [e["event_id"] for e in events]
            try:
                r = self.http.post("/v1/sync/push", json=body, headers=self._headers())
            except httpx.HTTPError as e:
                ob.mark_failed(ids_, f"transport: {type(e).__name__}")
                self.last["error"] = f"push transport error: {type(e).__name__}"
                return {"failed": len(ids_), "error": self.last["error"]}
            self.bytes_sent += len(r.request.content or b"")
            if r.status_code in (401, 403):
                for i in ids_:
                    ob.mark(i, "queued", f"auth: HTTP {r.status_code}")
                self.auth_required = True
                self.last["error"] = f"auth required (HTTP {r.status_code})"
                ob.log("sync", f"push refused: HTTP {r.status_code}; paused until a valid token is set")
                return {"auth_required": True}
            if r.status_code != 200:
                ob.mark_failed(ids_, f"HTTP {r.status_code}")
                self.last["error"] = f"push HTTP {r.status_code}"
                return {"failed": len(ids_), "error": self.last["error"]}
            self.auth_required = False
            results = {x["event_id"]: x for x in r.json()["results"]}
            by_ep = {e["event_id"]: e["episode_id"] for e in events}
            summary = {"accepted": 0, "duplicate": 0, "rejected": 0, "missing": 0}
            for eid in ids_:
                res = results.get(eid)
                if res is None:
                    ob.mark_failed([eid], "no ack for event")
                    summary["missing"] += 1
                    continue
                summary[res["status"]] += 1
                if res["status"] in ("accepted", "duplicate"):
                    ob.mark(eid, "synced")
                    self.device.set_share_state(eid, "synced", by_ep[eid])
                else:
                    ob.mark(eid, "rejected", res.get("reason"))
                    self.device.set_share_state(eid, "rejected", by_ep[eid])
            ob.log("sync", "push: " + ", ".join(f"{v} {k}" for k, v in summary.items() if v))
            self.last["push"] = time.time()
            self.last["error"] = None
            return summary

    # ---- pull (fleet mirror) ------------------------------------------------------------------------------
    def pull_once(self) -> dict:
        with self._lock:
            if not self.online:
                return {"skipped": "offline"}
            ob = self.device.outbox
            since = int(ob.kv_get("mirror_seq", 0))
            total = 0
            while True:
                try:
                    r = self.http.get("/v1/mirror/cases", params={"since": since, "limit": 200}, headers=self._headers())
                except httpx.HTTPError as e:
                    self.last["error"] = f"pull transport error: {type(e).__name__}"
                    return {"error": self.last["error"], "pulled": total}
                if r.status_code in (401, 403):
                    self.auth_required = True
                    self.last["error"] = f"auth required (HTTP {r.status_code})"
                    return {"auth_required": True, "pulled": total}
                if r.status_code != 200:
                    self.last["error"] = f"pull HTTP {r.status_code}"
                    return {"error": self.last["error"], "pulled": total}
                self.bytes_received += len(r.content)
                data = r.json()
                items = data["items"]
                if items:
                    self.device.mirror.upsert([StorePoint(it["id"], it["payload"], vib=it["vib"], note=it["note"],
                                                          bm25_text=it["text"]) for it in items])
                    since = max(since, max(int(it["payload"]["seq"]) for it in items))
                    ob.kv_set("mirror_seq", since)            # cursor advances only after the durable write
                    total += len(items)
                if not data.get("more"):
                    break
            if total:
                ob.log("sync", f"mirror: pulled {total} fleet case update(s)")
            self.last["pull"] = time.time()
            self._last_pull = time.time()
            return {"pulled": total, "cursor": since}

    # ---- background loop --------------------------------------------------------------------------------
    def tick(self) -> None:
        if self.online and not self.auth_required:
            self.push_once()
            if time.time() - self._last_pull >= PULL_EVERY_S:
                self.pull_once()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception as e:           # never let the worker die; show the error in the UI
                    self.last["error"] = f"worker: {e!r}"
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, name="sync-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def set_token(self, token: str) -> None:
        self.token = token
        self.auth_required = False
        self.device.outbox.log("sync", "device token updated")

    def status(self) -> dict:
        return {"online": self.online, "auth_required": self.auth_required, "cloud_url": self.cloud_url or None,
                "last_push": self.last["push"], "last_pull": self.last["pull"], "last_error": self.last["error"],
                "bytes_sent": self.bytes_sent, "bytes_received": self.bytes_received,
                "mirror_seq": self.device.outbox.kv_get("mirror_seq", 0), "outbox": self.device.outbox.counts()}
