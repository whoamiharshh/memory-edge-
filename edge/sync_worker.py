"""Sync worker: pushes queued share events to the cloud and pulls the fleet mirror.

Push: claim due outbox rows (-> uploading), POST one batch, apply per-event acks:
  accepted | duplicate -> synced      rejected -> rejected (reason kept, visible in the UI)
  transport error / 5xx / 429 -> failed, retried with capped exponential backoff + full jitter
  401 / 403 -> rows back to queued, worker pauses with "auth required" (no data lost)
Pull (fleet mirror, edge/mirror.py): GET /v1/mirror/head?since=cursor first. If the mirror is already at the
head, nothing else is sent. Otherwise:
  auto mode (default; cloud has a Qdrant Server): full shard snapshot to bootstrap or rebuild; after that each
      delta goes by scroll or by a new full snapshot, whichever this device's measured costs say is cheaper
  snapshot mode (Qdrant's dual-shard pattern verbatim, docs/RESEARCH.md H.2):
      first time / after a failed partial  -> GET full shard snapshot (gzip), restore beside the mirror, swap
      afterwards                            -> POST our snapshot_manifest, apply the partial snapshot (304 = current)
  scroll mode (in-process cloud, or mirror_mode="scroll"; kill test K5 fallback): GET cases with seq > cursor and
      upsert them; the cursor advances only after the durable shard write.
The `online` switch simulates a network partition for the demo: when off, no request is attempted at all.
"""
from __future__ import annotations

import pathlib
import shutil
import threading
import time
import uuid
import zlib
from typing import Any

import httpx

from edge.device import Device
from edge.store_edge import StorePoint

PULL_EVERY_S = 5.0
SNAPSHOT_TIMEOUT_S = 120.0
RENEW_CHECK_S = 3600.0            # how often to look at the token's expiry
RENEW_BEFORE_S = 7 * 24 * 3600    # renew when less than this is left


def _tls_context(ca: str | None, client_cert: tuple[str, str] | None):
    """True (system CAs), or an SSL context that trusts our private CA and, for mutual TLS, presents this device's
    certificate."""
    if not ca and not client_cert:
        return True
    import ssl
    ctx = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
    if client_cert:
        ctx.load_cert_chain(*client_cert)
    return ctx


def _fingerprint(token: str) -> str:
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()[:16]
ROW_BYTES_DEFAULT = 2600          # bench/mirror_sync.py: ~2.5-2.7 kB per case row (JSON incl. two vectors)
SNAPSHOT_BYTES_DEFAULT = 400_000  # bench/mirror_sync.py: gzip full snapshot, 188 kB (10 cases) - 398 kB (1,000)


class SyncWorker:
    def __init__(self, device: Device, cloud_url: str | None, token: str | None,
                 client: httpx.Client | None = None, interval: float = 2.0, mirror_mode: str = "auto",
                 ca: str | None = None, client_cert: tuple[str, str] | None = None, push_batch: int = 50):
        self.device, self.cloud_url = device, (cloud_url or "").rstrip("/")
        self.push_batch = max(1, min(int(push_batch), 100))      # events per push request (the cloud accepts <= 100)
        # A token this device renewed itself is kept in its SQLite; it wins unless the operator started the device
        # with a DIFFERENT token (e.g. a new one issued by the admin), which then starts a new chain.
        kv = device.outbox
        origin = _fingerprint(token) if token else None
        saved, saved_origin = kv.kv_get("device_token"), kv.kv_get("device_token_origin")
        self.token = saved if saved and (token is None or saved_origin == origin) else token
        if token and saved_origin != origin:
            kv.kv_set("device_token", None)
            kv.kv_set("device_token_origin", origin)
        self._last_renew_check = 0.0
        self.mirror_mode = mirror_mode              # preferred; falls back to scroll if the cloud cannot snapshot
        # https cloud: verify its certificate against our private CA (tools/make_certs.py) or the system store
        # mutual TLS: client_cert = (pem, key) proves THIS device at the TLS layer, in addition to its bearer token
        # 30 s read timeout: a busy cloud answering a 50-event push in ~6 s made a 5 s timeout retry every push
        # (bench/scale_fleet.py); retries stayed correct (counted once) but wasted work
        self.http = client or httpx.Client(base_url=self.cloud_url, timeout=httpx.Timeout(30.0, connect=5.0),
                                           verify=_tls_context(ca, client_cert))
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
        from shared.integrity import code_hash      # tamper evidence: the cloud compares it with its release
        h = {"X-Code-Hash": code_hash(), "X-Device-Time": self.device._now_iso()}
        return h | ({"Authorization": f"Bearer {self.token}"} if self.token else {})

    # ---- push -------------------------------------------------------------------------------------------
    def push_once(self) -> dict:
        with self._lock:
            if not self.online:
                return {"skipped": "offline"}
            ob = self.device.outbox
            events = ob.claim_due(limit=self.push_batch)
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
    def _refused(self, r: httpx.Response, pulled: int = 0) -> dict | None:
        """Common handling of a non-200 pull response; None if the response is usable."""
        if r.status_code in (401, 403):
            self.auth_required = True
            self.last["error"] = f"auth required (HTTP {r.status_code})"
            return {"auth_required": True, "pulled": pulled}
        if r.status_code not in (200, 304):
            self.last["error"] = f"pull HTTP {r.status_code}"
            return {"error": self.last["error"], "pulled": pulled}
        return None

    def pull_once(self) -> dict:
        with self._lock:
            if not self.online:
                return {"skipped": "offline"}
            try:
                r = self.http.get("/v1/mirror/head", params={"since": self.device.mirror.seq}, headers=self._headers())
            except httpx.HTTPError as e:
                self.last["error"] = f"pull transport error: {type(e).__name__}"
                return {"error": self.last["error"], "pulled": 0}
            if (bad := self._refused(r)) is not None:
                return bad
            self.bytes_received += r.num_bytes_downloaded
            head = r.json()
            if head.get("server_time"):
                self._note_clock(head["server_time"])
            if head.get("text_model"):
                self.device.outbox.kv_set("fleet_text_model", head["text_model"])
            mode = self.mirror_mode if self.mirror_mode != "scroll" and head.get("snapshots") else "scroll"
            self.device.mirror.ensure_mode(mode)
            out = {"scroll": self._pull_scroll, "snapshot": lambda: self._pull_snapshot(head),
                   "auto": lambda: self._pull_auto(head)}[mode]()
            if "error" not in out and "auth_required" not in out:
                self.last["pull"] = self._last_pull = time.time()
                self.last["error"] = None
                out["hint_model"] = self._pull_hint_model()
            return out | {"mode": mode}

    def _note_clock(self, server_time: str) -> None:
        import datetime as dt
        try:
            off = (self.device._now() - dt.datetime.fromisoformat(server_time)).total_seconds()
        except (TypeError, ValueError):
            return
        ob = self.device.outbox
        old = ob.kv_get("clock_offset_vs_cloud_s")
        ob.kv_set("clock_offset_vs_cloud_s", off)
        from edge.device import CLOCK_WARN_S
        if abs(off) > CLOCK_WARN_S and (old is None or abs(old - off) > CLOCK_WARN_S):
            ob.log("clock", f"device clock differs from the fleet cloud by {off / 3600:+.2f} h; the cloud corrects the "
                            "times of this device's evidence - set the device clock (NTP)")

    def _pull_hint_model(self) -> str:
        """The fleet-learned fault hint (plain JSON numbers). A failure here never fails the mirror pull."""
        try:
            r = self.http.get("/v1/fleet/hint-model", headers=self._headers())
        except httpx.HTTPError as e:
            return f"not fetched ({type(e).__name__})"
        if r.status_code != 200:
            return f"not fetched (HTTP {r.status_code})"
        self.bytes_received += r.num_bytes_downloaded
        from edge import fleet_hint
        m = (r.json() or {}).get("model")
        ob = self.device.outbox
        old = ob.kv_get("fleet_hint_model")
        if m is None or not fleet_hint.valid(m):
            return "none yet: " + str((r.json() or {}).get("why", ""))[:200]
        if old != m:
            ob.kv_set("fleet_hint_model", m)
            n = self.device.refresh_fleet_hints()
            ob.log("sync", f"fleet fault-hint model updated: {m['trained_on']['cases']} confirmed cases from "
                           f"{m['trained_on']['devices']} devices, {m.get('unseen_device_accuracy')} accuracy on "
                           f"unseen devices; {n} open episode hint(s) refreshed")
            return "updated"
        return "unchanged"

    def _pull_auto(self, head: dict) -> dict:
        """Full snapshot to bootstrap/rebuild; otherwise the cheaper of a scroll delta and a full snapshot, by
        bytes estimated from THIS device's last measured costs (defaults from bench/mirror_sync.py)."""
        mirror = self.device.mirror
        if mirror.needs_full:
            # bootstrap by measured bytes too: a small fleet is cheaper as rows than as a full shard snapshot
            # (20 cases ~ 52 kB of rows vs ~400 kB snapshot that also unpacks to a pre-allocated shard)
            rows = int(head.get("cases", 0)) * float(mirror.kv.kv_get("mirror_row_bytes", ROW_BYTES_DEFAULT))
            fresh = mirror.seq == 0 and mirror.count() == 0 and not mirror.kv.kv_get("mirror_applying", False)
            if fresh and rows < float(mirror.kv.kv_get("mirror_full_wire_bytes", SNAPSHOT_BYTES_DEFAULT)):
                out = self._pull_scroll()
                if "error" not in out and "auth_required" not in out:
                    mirror.kv.kv_set("mirror_needs_full", False)
                return out | {"bootstrap": "scroll", "estimate": {"rows_bytes_est": int(rows)}}
            return self._pull_snapshot(head)
        if int(head["seq"]) <= mirror.seq:
            return {"pulled": 0, "cursor": mirror.seq, "up_to_date": True}
        est_rows = int(head.get("changed", 1)) * float(mirror.kv.kv_get("mirror_row_bytes", ROW_BYTES_DEFAULT))
        est_snap = float(mirror.kv.kv_get("mirror_full_wire_bytes", SNAPSHOT_BYTES_DEFAULT))
        est = {"changed": head.get("changed"), "scroll_bytes_est": int(est_rows), "snapshot_bytes_est": int(est_snap)}
        if est_rows <= est_snap:
            return self._pull_scroll() | {"delta": "scroll", "estimate": est}
        mirror.request_full()
        return self._pull_snapshot(head) | {"estimate": est}

    def _pull_scroll(self) -> dict:
        mirror = self.device.mirror
        since = mirror.seq
        total = 0
        wire0 = self.bytes_received
        while True:
            try:
                r = self.http.get("/v1/mirror/cases", params={"since": since, "limit": 200}, headers=self._headers())
            except httpx.HTTPError as e:
                self.last["error"] = f"pull transport error: {type(e).__name__}"
                return {"error": self.last["error"], "pulled": total}
            if (bad := self._refused(r, total)) is not None:
                return bad
            self.bytes_received += r.num_bytes_downloaded
            data = r.json()
            items = data["items"]
            if items:
                since = max(since, max(int(it["payload"]["seq"]) for it in items))
                mirror.upsert_rows([StorePoint(it["id"], it["payload"], vib=it["vib"], note=it["note"],
                                               bm25_text=it["text"]) for it in items], since)
                total += len(items)
            if not data.get("more"):
                break
        if total:
            wire = self.bytes_received - wire0
            mirror.kv.kv_set("mirror_row_bytes", round(wire / total, 1))       # measured cost for _pull_auto
            mirror.last_refresh = {"kind": "scroll", "wire_bytes": wire, "snapshot_bytes": None,
                                   "cases_changed": total, "at": time.time()}
            self.device.outbox.log("sync", f"mirror (scroll): pulled {total} fleet case update(s), {wire:,} bytes")
        return {"pulled": total, "cursor": since}

    def _pull_snapshot(self, head: dict) -> dict:
        mirror = self.device.mirror
        before = mirror.seq
        full = mirror.needs_full
        if not full and int(head["seq"]) <= before:
            return {"pulled": 0, "cursor": before, "up_to_date": True}
        dl = pathlib.Path(self.device.cfg.root) / "mirror.dl"
        shutil.rmtree(dl, ignore_errors=True)
        dl.mkdir(parents=True)
        path = dl / "shard.snapshot"
        t0 = time.perf_counter()
        try:
            req = (self.http.build_request("GET", "/v1/mirror/snapshot", headers=self._headers(),
                                           timeout=SNAPSHOT_TIMEOUT_S) if full else
                   self.http.build_request("POST", "/v1/mirror/snapshot/partial", json=mirror.manifest(),
                                           headers=self._headers(), timeout=SNAPSHOT_TIMEOUT_S))
            r = self.http.send(req, stream=True)
            try:
                if (bad := self._refused(r)) is not None:
                    if not full and "error" in bad:
                        self.device.mirror.kv.kv_set("mirror_needs_full", True)   # next pull: full rebuild
                    return bad
                if r.status_code == 304:                   # server: nothing changed since our manifest
                    mirror.set_seq(int(head["seq"]))
                    return {"pulled": 0, "cursor": mirror.seq, "up_to_date": True}
                gunzip = zlib.decompressobj(31)
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(gunzip.decompress(chunk))
                    f.write(gunzip.flush())
                wire = r.num_bytes_downloaded
            finally:
                r.close()
            self.bytes_received += wire
            raw = path.stat().st_size
            t_apply = time.perf_counter()
            if full:
                mirror.replace_from_snapshot(path, int(head["seq"]))
                mirror.kv.kv_set("mirror_full_wire_bytes", wire)                 # measured cost for _pull_auto
            else:
                mirror.apply_partial(path, int(head["seq"]))
            apply_ms = round((time.perf_counter() - t_apply) * 1000, 1)
        except (httpx.HTTPError, zlib.error, OSError) as e:
            self.last["error"] = f"mirror {'full' if full else 'partial'} snapshot failed: {type(e).__name__}"
            return {"error": self.last["error"], "pulled": 0}
        except Exception as e:                             # Edge refused the snapshot: rebuild next time
            mirror.kv.kv_set("mirror_needs_full", True)
            self.last["error"] = f"mirror snapshot apply failed: {e!r}"[:300]
            return {"error": self.last["error"], "pulled": 0}
        finally:
            shutil.rmtree(dl, ignore_errors=True)
        pulled = mirror.changed_since(before)
        kind = "full" if full else "partial"
        mirror.last_refresh = {"kind": kind, "wire_bytes": wire, "snapshot_bytes": raw, "cases_changed": pulled,
                               "ms": round((time.perf_counter() - t0) * 1000, 1), "apply_ms": apply_ms,
                               "at": time.time()}
        self.device.outbox.log("sync", f"mirror ({kind} snapshot): {pulled} fleet case update(s); "
                                       f"{wire:,} bytes on the wire for a {raw:,}-byte shard snapshot")
        return {"pulled": pulled, "cursor": mirror.seq, "snapshot": kind, "wire_bytes": wire}

    # ---- background loop --------------------------------------------------------------------------------
    def maybe_renew_token(self, now: float | None = None, force_check: bool = False) -> dict | None:
        """Hourly: ask the cloud when this token expires; with less than RENEW_BEFORE_S left, swap it for a new one
        (saved locally, so a restart keeps it). Offline: nothing happens; the token just keeps its date."""
        now = time.time() if now is None else now
        if not self.token or not self.online or (not force_check and now - self._last_renew_check < RENEW_CHECK_S):
            return None
        self._last_renew_check = now
        try:
            r = self.http.get("/v1/whoami", headers=self._headers())
            if r.status_code != 200:
                return {"checked": False, "status": r.status_code}
            exp = r.json().get("expires_at")
            self.device.outbox.kv_set("token_expires_at", exp)
            if exp is None or exp - now > RENEW_BEFORE_S:
                return {"checked": True, "renewed": False, "expires_at": exp}
            r = self.http.post("/v1/token/renew", headers=self._headers())
            if r.status_code != 200:
                return {"checked": True, "renewed": False, "status": r.status_code}
        except httpx.HTTPError as e:
            return {"checked": False, "error": type(e).__name__}
        body = r.json()
        self.token = body["token"]
        self.device.outbox.kv_set("device_token", self.token)
        self.device.outbox.kv_set("token_expires_at", body["expires_at"])
        self.device.outbox.log("sync", "device token renewed before expiry (the old one stops working in 10 min)")
        return {"checked": True, "renewed": True, "expires_at": body["expires_at"]}

    def tick(self) -> None:
        self.device.maybe_run_retention()               # local housekeeping; runs offline too (hourly)
        self.maybe_renew_token()
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
        self.device.outbox.kv_set("device_token", token)
        self.device.outbox.kv_set("device_token_origin", _fingerprint(token))
        self.auth_required = False
        self.device.outbox.log("sync", "device token updated")

    # ---- shared free-text knowledge (cloud/knowledge.py) -------------------------------------------------
    def publish_knowledge(self, text: str, topic: str = "general", audience: str = "everyone",
                          recipients: list[str] | None = None) -> dict:
        """Send one text record to the cloud for other devices to pull. Unlike evidence this is not queued
        in the outbox: publishing is a deliberate act a person takes while looking at the screen, so it
        either succeeds now or reports why it did not."""
        if not self.online:
            raise RuntimeError("offline: reconnect to publish (what you typed is already saved on the device)")
        r = self.http.post("/v1/knowledge", headers=self._headers(),
                           json={"text": text, "topic": topic, "audience": audience,
                                 "recipients": recipients or []})
        if r.status_code >= 400:
            raise RuntimeError(f"cloud refused it ({r.status_code}): {r.text[:200]}")
        self.bytes_sent += len(r.request.content or b"")
        out = r.json()
        self.device.outbox.log("knowledge", f"published to {audience}: {text[:60]}")
        return out

    def my_knowledge(self) -> list[dict]:
        """What this device has published, asked with this device's own token.

        It has to be the device's token: the cloud decides ownership from the caller's identity, so asking
        as anyone else — the gateway's admin token, for instance — truthfully reports that the caller has
        published nothing, and the screen looks broken.
        """
        if not self.online:
            return []
        try:
            r = self.http.get("/v1/knowledge/mine", headers=self._headers())
            return r.json() if r.status_code == 200 else []
        except Exception:
            return []

    def withdraw_knowledge(self, record_id: str) -> dict:
        """Stop handing a published record to anyone who has not pulled it yet. Devices that already
        have a copy delete it on their next check; a record already read cannot be recalled."""
        if not self.online:
            raise RuntimeError("offline: reconnect to withdraw")
        r = self.http.post(f"/v1/knowledge/{record_id}/withdraw", headers=self._headers())
        if r.status_code >= 400:
            raise RuntimeError(f"cloud refused it ({r.status_code})")
        self.device.outbox.log("knowledge", f"withdrew {record_id[:8]}")
        return r.json()

    def pull_knowledge(self) -> dict:
        """Fetch records addressed to this device and store them locally so they stay searchable offline.
        The cloud decides what this device may see; the device never filters other people's records itself."""
        if not self.online:
            return {"skipped": "offline"}
        since = int(self.device.outbox.kv_get("knowledge_seq", 0) or 0)
        try:
            r = self.http.get("/v1/knowledge", headers=self._headers(), params={"since": since, "limit": 200})
            if r.status_code >= 400:
                return {"error": f"HTTP {r.status_code}"}
            self.bytes_received += len(r.content)
            body = r.json()
        except Exception as e:
            return {"error": type(e).__name__}
        n = self.device.store_shared_knowledge(body.get("items", []))
        if int(body.get("seq", since)) > since:
            self.device.outbox.kv_set("knowledge_seq", int(body["seq"]))
        return {"received": n, "seq": body.get("seq", since), "more": body.get("more", False)}

    def pending(self) -> dict:
        """What is waiting to leave this device, and what it would cost to send.

        Reported before anything is sent rather than after: on a metered or intermittent link, the person
        deciding whether to sync now wants the size first. Nothing here sends anything.
        """
        rows = [r for r in self.device.outbox.rows(500) if r.get("status") in ("queued", "failed")]
        # a rough per-event size — the real payload is the redacted note plus the fingerprint, and the
        # fingerprint dominates, so this is close enough to inform a decision without building the batch
        approx = sum(len(str(r.get("payload") or "")) or 900 for r in rows)
        by_state: dict[str, int] = {}
        for r in rows:
            key = r.get("last_error") or "ready"
            by_state[key] = by_state.get(key, 0) + 1
        return {"online": self.online, "count": len(rows), "approx_bytes": approx,
                "retrying": sum(1 for r in rows if r.get("status") == "failed"),
                "by_state": by_state,
                "advice": ("nothing waiting" if not rows else
                           "offline — these go automatically when the network returns" if not self.online
                           else f"{len(rows)} item(s) ready, about {max(1, approx // 1024)} kB")}

    def status(self) -> dict:
        return {"online": self.online, "auth_required": self.auth_required, "cloud_url": self.cloud_url or None,
                "last_push": self.last["push"], "last_pull": self.last["pull"], "last_error": self.last["error"],
                "bytes_sent": self.bytes_sent, "bytes_received": self.bytes_received,
                "mirror_seq": self.device.mirror.seq, "mirror": self.device.mirror.info(),
                "outbox": self.device.outbox.counts()}
