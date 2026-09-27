"""One edge device: the orchestration of store, journal, novelty gate, outcome verifier, policy and sync.

Point types in the local shard (all carry machine_id):
  baseline  healthy fingerprints of this machine (vib only). Used by the gate and the verifier.
  exemplar  up to MAX_EXEMPLARS fingerprints per episode (vib only). The gate merges against these.
  episode   one abnormal-state occurrence: fingerprint + note vectors + all lifecycle fields.
Fleet knowledge lives in a SEPARATE read-only mirror shard (Qdrant's documented dual-shard pattern).

Episode lifecycle:  open -> verifying (action recorded) -> closed (verdict final AND technician outcome set)
Every write goes journal (SQLite) -> shard (flushed) -> journal marked applied.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from edge import fault_hint, policy
from edge import verifier as V
from edge.fingerprint import FP_VERSION, Baseline
from edge.gate import GateConfig, NoveltyGate, calibrate
from edge.mirror import Mirror
from edge.outbox import Outbox
from edge.store_edge import EdgeStore, StorePoint
from shared import ids
from shared.embed import Embedder
from shared.redact import redact
from shared.schema import ActionCode, FaultClass, RootCause

MAX_EXEMPLARS = 30
HINT_WINDOWS = 10          # physics-hint votes collected from the first windows of an episode
TERMINAL = ("closed",)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class DeviceConfig:
    device_id: str
    site_id: str
    machine_id: str
    root: pathlib.Path
    machine_class: str = "2hp-induction-motor/SKF6205-DE"
    component: str = "bearing"
    denylist: list[str] = field(default_factory=list)
    cloud_url: str | None = None
    device_token: str | None = None


class Device:
    def __init__(self, cfg: DeviceConfig, embedder: Embedder):
        self.cfg, self.embedder = cfg, embedder
        root = pathlib.Path(cfg.root)
        root.mkdir(parents=True, exist_ok=True)
        self.store = EdgeStore(root / "local", text_model=embedder.name)
        self.outbox = Outbox(str(root / "device.sqlite"))
        self.mirror = Mirror(root, self.outbox, text_model="BAAI/bge-small-en-v1.5")
        self._lock = threading.RLock()
        self.baseline: Baseline | None = None
        self.gate: NoveltyGate | None = None
        self.counters = collections.Counter()
        self.gate_ms: collections.deque[float] = collections.deque(maxlen=2000)
        self.search_ms: collections.deque[float] = collections.deque(maxlen=500)
        self.last_gate: dict | None = None
        self.recent: collections.deque[dict] = collections.deque(maxlen=240)   # recent window results (UI chart)
        self._run_episode: str | None = None       # episode of the current uninterrupted abnormal run
        replayed = self._replay_journal()
        requeued = self.outbox.recover_uploading()
        bpath = root / "baseline.json"
        if bpath.exists():
            b = json.loads(bpath.read_text())
            self.baseline = Baseline.from_dict(b["baseline"])
            self.gate = NoveltyGate(self.store, cfg.machine_id, GateConfig(**b["gate"]))
        self.outbox.log("boot", f"device started; journal re-applied {replayed} op(s); {requeued} upload(s) re-queued")

    # ---- persistence: journal first, then shard ---------------------------------------------------------
    def _apply(self, body: dict) -> None:
        kind = body["kind"]
        if kind == "upsert":
            self.store.upsert([StorePoint(**p) for p in body["points"]])
        elif kind == "set_payload":
            self.store.modify(body["id"], lambda _p: body["fields"])
        elif kind == "set_payload_where":
            self.store.set_payload_where(body["filter"], body["fields"])
        else:
            raise ValueError(kind)

    def _write(self, body: dict) -> None:
        op = ids.make_id("op", self.cfg.device_id, time.time_ns(), id(body))
        self.outbox.journal_put(op, body)
        self._apply(body)
        self.outbox.journal_applied(op)

    def _replay_journal(self) -> int:
        pending = self.outbox.journal_pending()
        for op, body in pending:
            self._apply(body)
            self.outbox.journal_applied(op)
        return len(pending)

    def _upsert(self, points: list[StorePoint]) -> None:
        self._write({"kind": "upsert", "points": [p.__dict__ | {"vib": list(p.vib) if p.vib is not None else None,
                                                                 "note": list(p.note) if p.note is not None else None}
                                                  for p in points]})

    def _set(self, pid: str, **fields) -> None:
        self._write({"kind": "set_payload", "id": pid, "fields": fields})

    # ---- baseline ---------------------------------------------------------------------------------------
    def fit_baseline(self, healthy_raw: np.ndarray) -> dict:
        """Fit z-scoring stats on this machine's healthy windows, store them as baseline points, calibrate the
        gate from them. Replaces any previous baseline of this machine."""
        with self._lock:
            b = Baseline.fit(healthy_raw)
            z = b.z(healthy_raw)
            g = calibrate(z)
            pts = [StorePoint(ids.baseline_point_id(self.cfg.machine_id, i),
                              {"type": "baseline", "machine_id": self.cfg.machine_id, "fp_version": FP_VERSION},
                              vib=z[i].tolist()) for i in range(len(z))]
            for s in range(0, len(pts), 256):
                self._upsert(pts[s:s + 256])
            (pathlib.Path(self.cfg.root) / "baseline.json").write_text(
                json.dumps({"baseline": b.to_dict(), "gate": g.to_dict()}))
            self.baseline, self.gate = b, NoveltyGate(self.store, self.cfg.machine_id, g)
            self.outbox.log("baseline", f"baseline fitted on {len(z)} healthy windows; tau_normal={g.tau_normal:.2f}, "
                                        f"tau_merge={g.tau_merge:.2f} (calibration q99 {g.calib_q99:.2f})")
            return g.to_dict() | {"n_windows": len(z)}

    # ---- the per-window path ----------------------------------------------------------------------------
    def ingest_window(self, raw: np.ndarray, source: str | None = None) -> dict:
        if self.gate is None or self.baseline is None:
            raise RuntimeError("fit a healthy baseline first")
        with self._lock:
            z = self.baseline.z(np.asarray(raw, dtype=float))
            r = self.gate.classify(z)
            self.gate_ms.append(r.latency_ms)
            self.counters["windows"] += 1
            self.counters[r.state] += 1
            ts = now_iso()
            out: dict[str, Any] = {"state": r.state, "d_baseline": round(r.d_baseline, 2),
                                   "latency_ms": round(r.latency_ms, 3), "episode_id": r.episode_id, "source": source}
            healthy = r.state == "normal"
            # feed the verifier of every episode that is waiting for its post-action verdict
            for ep in self.episodes(status="verifying"):
                vs = V.step(V.VerifyState.from_dict(ep.get("verify")), healthy)
                self._set(ep["episode_id"], verify=vs.to_dict())
                if vs.done:
                    self.outbox.log("verify", f"{vs.label()}", ep["episode_id"])
                    self._after_verdict(ep["episode_id"])
            cont = self._run_episode
            if r.state == "new" and cont and self.store.get(cont).payload["status"] != "closed":
                # Continuity rule: an uninterrupted abnormal run is ONE episode, even if a noisy window lands
                # outside tau_merge of every exemplar (measured: 14-mil ball faults spread up to ~35 z-units).
                r.state, r.episode_id = "merge", cont
                out.update(state="merge", episode_id=cont, merge_reason="contiguous abnormal run")
                self.counters["new"] -= 1
                self.counters["merge"] += 1
            if r.state == "merge":
                self._merge(r.episode_id, z, raw, r.d_episode if r.d_episode is not None else r.d_baseline, ts)
                self._run_episode = r.episode_id
            elif r.state == "new":
                out["episode_id"] = self._open_episode(z, raw, ts, r.recurrence_of)
                out["recurrence_of"] = r.recurrence_of
                self._run_episode = out["episode_id"]
            else:
                self._run_episode = None                         # a healthy window ends the abnormal run
            self.last_gate = out | {"ts": ts}
            self.recent.append({"ts": ts, "state": r.state, "d": round(r.d_baseline, 2)})
            return out

    def _open_episode(self, z: np.ndarray, raw: np.ndarray, ts: str, recurrence_of: str | None) -> str:
        seq = int(self.outbox.kv_get("episode_seq", 0)) + 1
        self.outbox.kv_set("episode_seq", seq)
        eid = ids.episode_id(self.cfg.device_id, seq)
        hint = fault_hint.suggest(raw)
        payload = {
            "type": "episode", "episode_id": eid, "seq": seq, "machine_id": self.cfg.machine_id,
            "device_id": self.cfg.device_id, "site_id": self.cfg.site_id, "component": self.cfg.component,
            "first_seen": ts, "last_seen": ts, "occurrences": 1, "status": "open", "n_exemplars": 1,
            "fault_hint": hint, "hint_votes": {hint["fault_class"]: 1}, "fault_class": None, "fault_class_source": None,
            "action_code": None, "root_cause_claim": None, "action_at": None, "note_text": "", "note_share_opt_in": False,
            "outcome": "pending", "technician_confirmed": False, "verify": None, "share_state": "local",
            "decision": None, "recurrence_of": recurrence_of, "schema_version": 1, "fp_version": FP_VERSION,
            "text_model": self.embedder.name, "version": 1,
        }
        text = self._doc_text(payload)
        self._upsert([
            StorePoint(eid, payload, vib=z.tolist(), note=self.embedder.embed_documents([text])[0], bm25_text=text),
            StorePoint(ids.make_id("exemplar", eid, 0), {"type": "exemplar", "machine_id": self.cfg.machine_id,
                                                         "episode_id": eid, "episode_active": True}, vib=z.tolist()),
        ])
        msg = "unfamiliar vibration state: new episode opened"
        if recurrence_of:
            msg += f" (resembles closed episode {recurrence_of[:8]} on this machine)"
        self.outbox.log("gate", msg + f"; physics hint: {hint['fault_class']}", eid)
        self._decide(eid)
        return eid

    def _merge(self, eid: str, z: np.ndarray, raw: np.ndarray, d_episode: float | None, ts: str) -> None:
        ep = self.store.get(eid).payload
        fields: dict[str, Any] = {"occurrences": ep["occurrences"] + 1, "last_seen": ts}
        if ep["occurrences"] < HINT_WINDOWS:                   # majority vote of the first windows' hints
            votes = dict(ep.get("hint_votes") or {})
            h = fault_hint.suggest(raw)["fault_class"]
            votes[h] = votes.get(h, 0) + 1
            best = max(votes, key=votes.get)
            fields["hint_votes"] = votes
            fields["fault_hint"] = ep["fault_hint"] | {
                "fault_class": best, "measured_accuracy": fault_hint.MEASURED["per_class"].get(best),
                "why": f"dominant defect frequency in {votes[best]} of the first {sum(votes.values())} windows"}
        if d_episode is not None and d_episode > self.gate.cfg.tau_normal and ep["n_exemplars"] < MAX_EXEMPLARS:
            self._upsert([StorePoint(ids.make_id("exemplar", eid, ep["n_exemplars"]),
                                     {"type": "exemplar", "machine_id": self.cfg.machine_id, "episode_id": eid,
                                      "episode_active": True}, vib=z.tolist())])
            fields["n_exemplars"] = ep["n_exemplars"] + 1
        self._set(eid, **fields)

    # ---- technician actions -----------------------------------------------------------------------------
    def _doc_text(self, ep: dict) -> str:
        fc = ep.get("fault_class") or (ep.get("fault_hint") or {}).get("fault_class", "unknown")
        parts = [ep.get("component", ""), fc.replace("_", " "), "fault"]
        if ep.get("action_code"):
            parts += ["action", ep["action_code"].replace("_", " ")]
        if ep.get("outcome") and ep["outcome"] != "pending":
            parts += ["outcome", ep["outcome"]]
        if ep.get("note_text"):
            parts.append(ep["note_text"])
        return " ".join(parts)

    def _rewrite(self, eid: str, **fields) -> dict:
        """Full re-write of an episode point (payload + refreshed text vectors). Returns the new payload."""
        rec = self.store.get(eid, with_vectors=True)
        if rec is None:
            raise KeyError(eid)
        p = rec.payload | fields | {"version": rec.payload.get("version", 1) + 1}
        text = self._doc_text(p)
        self._upsert([StorePoint(eid, p, vib=rec.vectors["vib"], note=self.embedder.embed_documents([text])[0],
                                 bm25_text=text)])
        return p

    def _require(self, eid: str) -> dict:
        rec = self.store.get(eid)
        if rec is None or rec.payload.get("type") != "episode":
            raise KeyError(f"no episode {eid}")
        return rec.payload

    def set_note(self, eid: str, text: str, share_opt_in: bool = False) -> dict:
        with self._lock:
            self._require(eid)
            if len(text) > 2000:
                raise ValueError("note too long (max 2000 characters)")
            self._rewrite(eid, note_text=text.strip(), note_share_opt_in=bool(share_opt_in))
            self.outbox.log("note", f"note saved ({len(text)} chars, share opt-in={bool(share_opt_in)})", eid)
            return self._decide(eid)

    def set_fault_class(self, eid: str, fault_class: str) -> dict:
        with self._lock:
            self._require(eid)
            FaultClass(fault_class)
            self._rewrite(eid, fault_class=fault_class, fault_class_source="technician")
            self.outbox.log("fault_class", f"technician confirmed fault class {fault_class}", eid)
            return self._decide(eid)

    def record_action(self, eid: str, action_code: str, root_cause: str | None = None,
                      required_windows: int = V.REQUIRED_WINDOWS) -> dict:
        """An intervention was made: from now on, windows are fed to this episode's outcome verifier."""
        with self._lock:
            ep = self._require(eid)
            if ep["status"] in TERMINAL:
                raise ValueError("episode is closed")
            ActionCode(action_code)
            if root_cause:
                RootCause(root_cause)
            self._rewrite(eid, action_code=action_code, root_cause_claim=root_cause or None, action_at=now_iso(),
                          status="verifying", verify=V.VerifyState(required=int(required_windows)).to_dict(),
                          outcome="pending", technician_confirmed=False)
            self.store.set_payload_where({"type": "exemplar", "episode_id": eid}, {"episode_active": True})
            self.outbox.log("action", f"action {action_code} recorded; verifying over {required_windows} windows", eid)
            return self._decide(eid)

    def confirm_outcome(self, eid: str, outcome: str) -> dict:
        with self._lock:
            ep = self._require(eid)
            if outcome not in ("worked", "failed"):
                raise ValueError("outcome must be worked or failed")
            if not ep.get("action_code"):
                raise ValueError("record an action before confirming an outcome")
            self._rewrite(eid, outcome=outcome, technician_confirmed=True)
            self.outbox.log("outcome", f"technician reports the action {outcome}", eid)
            self._after_verdict(eid)
            return self._decide(eid)

    def _after_verdict(self, eid: str) -> None:
        ep = self.store.get(eid).payload
        vs = V.VerifyState.from_dict(ep.get("verify"))
        if vs.done and ep.get("technician_confirmed") and ep["status"] != "closed":
            self._set(eid, status="closed", closed_at=now_iso())
            self.store.set_payload_where({"type": "exemplar", "episode_id": eid}, {"episode_active": False})
            self.outbox.log("episode", f"episode closed ({vs.label()})", eid)
        self._decide(eid)

    # ---- policy -----------------------------------------------------------------------------------------
    def _decide(self, eid: str) -> dict:
        rec = self.store.get(eid, with_vectors=True)
        ep = rec.payload
        red = redact(ep["note_text"], self.cfg.denylist) if ep.get("note_text") else None
        d = policy.decide(ep, fingerprint=rec.vectors["vib"], redaction=red, device_id=self.cfg.device_id,
                          machine_class=self.cfg.machine_class,
                          already_queued=self.outbox.known_event_ids() - {ep.get("event_id")})   # own event is not a duplicate
        fields: dict[str, Any] = {"decision": d.to_dict()}
        if d.action == "SHARE" and self.outbox.enqueue(d.event):
            fields["share_state"] = "queued"
            fields["event_id"] = d.event["event_id"]
            self.outbox.log("policy", "SHARE: outcome evidence queued for the fleet" +
                            (" (with redacted note)" if d.note_shared else " (note kept local)"), eid)
        elif ep.get("decision") is None or ep["decision"].get("action") != d.action:
            last = next((r for r in reversed(d.reasons) if not r.ok), d.reasons[-1])
            self.outbox.log("policy", f"{d.action}: {last.detail}", eid)
        self._set(eid, **fields)
        return fields["decision"]

    # ---- reads ------------------------------------------------------------------------------------------
    def episodes(self, status: str | None = None) -> list[dict]:
        flt: dict[str, Any] = {"type": "episode", "machine_id": self.cfg.machine_id}
        if status:
            flt["status"] = status
        eps = [r.payload for r in self.store.scroll(filter=flt)]
        return sorted(eps, key=lambda e: e["seq"], reverse=True)

    def episode(self, eid: str) -> dict:
        ep = self._require(eid)
        return ep | {"outbox_status": self.outbox.status_of(ep["event_id"]) if ep.get("event_id") else None}

    # ---- retrieval (offline: local shard + fleet mirror shard) ------------------------------------------
    def search(self, text: str | None = None, episode_id: str | None = None, use_fleet: bool = True,
               limit: int = 5) -> dict:
        """Hybrid search. Local: this machine's episodes by fingerprint + note (dense) + note (BM25), fused with RRF.
        Fleet (K2 decision): filtered by component and the episode's fault class (technician-confirmed, else the
        physics hint), ranked by text; the fingerprint is only a low-weight tie-break across machines."""
        t0 = time.perf_counter()
        vib, fc, src = None, None, None
        if episode_id:
            rec = self.store.get(episode_id, with_vectors=True)
            if rec is None:
                raise KeyError(episode_id)
            vib = rec.vectors.get("vib")
            fc = rec.payload.get("fault_class") or (rec.payload.get("fault_hint") or {}).get("fault_class")
            src = "technician" if rec.payload.get("fault_class") else "physics_hint"
            if not text:
                text = self._doc_text(rec.payload)
        if not text and vib is None:
            raise ValueError("give a text query or an episode")
        note = self.embedder.embed_query(text) if text else None
        local = self.store.search(vib=vib, note=note, text=text, limit=limit + 1, explain=True,
                                  filter={"type": "episode", "machine_id": self.cfg.machine_id})
        local = [h for h in local if h.id != episode_id][:limit]
        fleet, fleet_filter, fleet_error = [], None, None
        try:
            if use_fleet and self.mirror.count():
                fleet_filter = {"component": self.cfg.component, "!status": "retracted"}
                if fc and fc != "unknown":
                    fleet_filter["fault_class"] = fc
                fleet = self.mirror.search(vib=vib, note=note, text=text, filter=fleet_filter, limit=limit,
                                           weights={"vib": 0.25}, explain=True)
        except Exception as e:              # a broken mirror must never take local memory down with it
            fleet, fleet_error = [], f"fleet mirror unavailable ({type(e).__name__}); showing local memory only"
        ms = (time.perf_counter() - t0) * 1000
        self.search_ms.append(ms)
        slim = lambda p: {k: v for k, v in p.items() if k not in ("hint_votes",)}
        return {"query": {"text": text, "episode_id": episode_id, "fault_class": fc, "fault_class_source": src,
                          "fleet_filter": fleet_filter},
                "local": [{"id": h.id, "rrf": round(h.score, 4), "legs": h.legs, "episode": slim(h.payload)} for h in local],
                "fleet": [{"id": h.id, "rrf": round(h.score, 4), "legs": h.legs, "case": h.payload} for h in fleet],
                "fleet_error": fleet_error, "latency_ms": round(ms, 2)}

    def set_share_state(self, event_id: str, state: str, episode_id: str) -> None:
        with self._lock:
            if self.store.get(episode_id) is not None:
                self._set(episode_id, share_state=state)

    def stats(self) -> dict:
        g = list(self.gate_ms)
        s = list(self.search_ms)
        pct = lambda xs, q: round(float(np.percentile(xs, q)), 3) if xs else None
        return {
            "device_id": self.cfg.device_id, "site_id": self.cfg.site_id, "machine_id": self.cfg.machine_id,
            "machine_class": self.cfg.machine_class, "component": self.cfg.component,
            "baseline_ready": self.gate is not None, "gate": self.gate.cfg.to_dict() if self.gate else None,
            "windows": dict(self.counters), "last_gate": self.last_gate, "recent": list(self.recent),
            "gate_ms": {"p50": pct(g, 50), "p95": pct(g, 95), "n": len(g)},
            "search_ms": {"p50": pct(s, 50), "p95": pct(s, 95), "n": len(s)},
            "local_points": self.store.facet("type"), "episodes_by_status": self.store.facet("status", filter={"type": "episode"}),
            "share_states": self.store.facet("share_state", filter={"type": "episode"}),
            "mirror_cases": self.mirror.count(), "outbox": self.outbox.counts(),
            "raw_bytes_kept_local": int(self.counters["windows"] * 2048 * 4),
            "embedder": self.embedder.name,
        }

    def close(self) -> None:
        with self._lock:
            self.store.close()
            self.mirror.close()
            self.outbox.close()
