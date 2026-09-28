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

from edge import fleet_hint, local_detector, machine_card, policy, profiles, storage_os, vehicle_risk
from edge import physics as P
from edge import verifier as V
from edge.fingerprint import Baseline
from edge.gate import GateConfig, NoveltyGate, calibrate
from edge.mirror import Mirror
from edge.crypto import NoteCipher, load_or_create_key
from edge.outbox import Outbox
from edge.store_edge import EdgeStore, StorePoint, canonical_id
from shared import ids
from shared.embed import Embedder
from shared.redact import redact
from shared.schema import ActionCode, DamageMode, FaultClass, FollowUp, RootCause

MAX_EXEMPLARS = 30
HINT_WINDOWS = 10          # physics-hint votes collected from the first windows of an episode
TERMINAL = ("closed",)
RETENTION_EVERY_S = 3600.0
FLEET_TEXT_MODEL = "BAAI/bge-small-en-v1.5"   # default until the cloud announces its model (/v1/mirror/head)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


class _NoteVault:
    """The device's view of its store with technician notes encrypted at rest (edge/crypto.py): every payload
    written carries an encrypted note_text, every payload read comes back decrypted. Everything else passes through."""

    def __init__(self, store: EdgeStore, cipher: NoteCipher):
        self._s, self._c = store, cipher

    def __getattr__(self, name):
        return getattr(self._s, name)

    def _enc(self, p: dict) -> dict:
        return p | {"note_text": self._c.encrypt(p["note_text"])} if p.get("note_text") else p

    def _dec(self, p: dict) -> dict:
        return p | {"note_text": self._c.decrypt(p["note_text"])} if p.get("note_text") else p

    def upsert(self, points, **kw):
        return self._s.upsert([StorePoint(p.id, self._enc(p.payload), p.vib, p.note, p.bm25_text) for p in points], **kw)

    def modify(self, pid, fn):
        out = self._s.modify(pid, lambda cur: self._enc(dict(fn(self._dec(cur)))))
        return None if out is None else self._dec(out)

    def get(self, pid, **kw):
        r = self._s.get(pid, **kw)
        if r is not None:
            r.payload = self._dec(r.payload)
        return r

    def retrieve(self, ids_, **kw):
        out = self._s.retrieve(ids_, **kw)
        for r in out:
            r.payload = self._dec(r.payload)
        return out

    def scroll(self, **kw):
        for r in self._s.scroll(**kw):
            r.payload = self._dec(r.payload)
            yield r

    def search(self, **kw):
        out = self._s.search(**kw)
        for h in out:
            h.payload = self._dec(h.payload)
        return out


def _json_safe(x):
    """Round floats and drop NaN/inf so physics diagnoses can live in a payload."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, float):
        return round(x, 4) if np.isfinite(x) else None
    return x


class VersionConflict(RuntimeError):
    """The episode changed since the caller read it (another technician or tab). Nothing was written."""

    def __init__(self, eid: str, expected: int, current: int):
        super().__init__(f"CONFLICT: episode {eid[:8]} is at version {current}, you edited version {expected}; "
                         "reload it and re-apply your change")
        self.current = current


@dataclass
class DeviceConfig:
    device_id: str
    site_id: str
    machine_id: str
    root: pathlib.Path
    machine_class: str = "2hp-induction-motor/SKF6205-DE"
    component: str | None = None                # None = the profile's default component
    denylist: list[str] = field(default_factory=list)
    cloud_url: str | None = None
    device_token: str | None = None
    profile: str = "bearing-12k"                # edge/profiles.py: what kind of signal this device watches
    profile_params: dict = field(default_factory=dict)
    compress_storage: bool = False              # Windows: NTFS-compress the device folder at start + hourly
    encrypt_notes: bool = True                  # technician notes encrypted at rest (edge/crypto.py)
    hold_days: float = 30.0                     # after a verified fix: report 'held' after this, or 'recurred'


class Device:
    def __init__(self, cfg: DeviceConfig, embedder: Embedder):
        self.cfg, self.embedder = cfg, embedder
        self.profile = profiles.make(cfg.profile, **cfg.profile_params)
        self.component = cfg.component or self.profile.default_component
        root = pathlib.Path(cfg.root)
        root.mkdir(parents=True, exist_ok=True)
        self.card = machine_card.load(root)          # manufacturer data (bearing, limits, mains), if entered
        self.profile.apply_card(self.card)
        self.store = EdgeStore(root / "local", text_model=embedder.name, note_dim=embedder.dim,
                               fp_version=self.profile.fp_version, allow_text_model_change=True)
        self.note_protection = "none (encrypt_notes off)"
        if cfg.encrypt_notes:
            key, how = load_or_create_key(root)
            self._cipher = NoteCipher(key, how)
            self.store = _NoteVault(self.store, self._cipher)
            self.note_protection = f"AES-256-GCM, key protected by {how}"
        self.outbox = Outbox(str(root / "device.sqlite"))
        self.mirror = Mirror(root, self.outbox, text_model=FLEET_TEXT_MODEL)
        if self.store.pending_text_model:            # H.3: the device got a new text model -> re-embed its memory
            m = self.store.migrate_text_model(embedder.embed_documents,
                                              lambda p: self._doc_text(self._plain(p)) if p.get("type") == "episode" else None)
            self.outbox.log("model", f"text model changed {m['from']} -> {m['to']}: re-embedded {m['migrated']} "
                                     f"episode(s) into a new vector; BM25 and fingerprints unchanged")
        self._lock = threading.RLock()
        self.baseline: Baseline | None = None
        self.gate: NoveltyGate | None = None
        self.counters = collections.Counter()
        self.gate_ms: collections.deque[float] = collections.deque(maxlen=2000)
        self.search_ms: collections.deque[float] = collections.deque(maxlen=500)
        self.last_gate: dict | None = None
        self.recent: collections.deque[dict] = collections.deque(maxlen=240)   # recent window results (UI chart)
        self._run_episode: str | None = None       # episode of the current uninterrupted abnormal run
        self._capture: dict | None = None          # healthy-baseline capture from a live sensor (ingest_signal)
        self.taught: dict[str, list[float]] = {}   # operating variable -> [min, max] the healthy baseline covers
        self.last_diagnosis: dict | None = None    # physics diagnosis of the latest abnormal signal window
        replayed = self._replay_journal()
        requeued = self.outbox.recover_uploading()
        bpath = root / "baseline.json"
        if bpath.exists():
            b = json.loads(bpath.read_text())
            self.taught = b.get("taught", {})
            self.baseline = Baseline.from_dict(b["baseline"], self.profile.fp_version)
            self.gate = NoveltyGate(self.store, cfg.machine_id, GateConfig(**b["gate"]))
        self.outbox.log("boot", f"device started; journal re-applied {replayed} op(s); {requeued} upload(s) re-queued")
        self._last_retention = 0.0
        self.maybe_run_retention()                 # also compresses the folder when compress_storage is on

    # ---- persistence: journal first, then shard ---------------------------------------------------------
    def _apply(self, body: dict) -> None:
        kind = body["kind"]
        if kind == "upsert":
            c = getattr(self, "_cipher", None)
            pts = [StorePoint(**p) for p in body["points"]]
            if c:
                pts = [StorePoint(p.id, p.payload, p.vib, p.note, c.decrypt(p.bm25_text)) for p in pts]
            self.store.upsert(pts)
        elif kind == "set_payload":
            self.store.modify(body["id"], lambda _p: body["fields"])
        elif kind == "set_payload_where":
            self.store.set_payload_where(body["filter"], body["fields"])
        elif kind == "archive":                  # one journal op, so a crash cannot lose the kept exemplar
            self.store.delete_where({"type": "exemplar", "episode_id": body["episode_id"]})
            if body["keep"]:
                self.store.upsert([StorePoint(**body["keep"])])
            self.store.modify(body["episode_id"], lambda _p: body["fields"])
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

    def _plain(self, p: dict) -> dict:
        c = getattr(self, "_cipher", None)
        return p | {"note_text": c.decrypt(p["note_text"])} if c and p.get("note_text") else p

    def _seal(self, fields: dict) -> dict:
        """Encrypt note_text BEFORE it reaches the journal (the store wrapper encrypts the shard copy too)."""
        c = getattr(self, "_cipher", None)
        return fields | {"note_text": c.encrypt(fields["note_text"])} if c and fields.get("note_text") else fields

    def _upsert(self, points: list[StorePoint]) -> None:
        c = getattr(self, "_cipher", None)
        # the BM25 text contains the note, so it is journaled encrypted too; _apply decrypts it in memory only
        points = [StorePoint(p.id, self._seal(p.payload), p.vib, p.note,
                             c.encrypt(p.bm25_text) if c and p.bm25_text else p.bm25_text) for p in points]
        self._write({"kind": "upsert", "points": [p.__dict__ | {"vib": list(p.vib) if p.vib is not None else None,
                                                                 "note": list(p.note) if p.note is not None else None}
                                                  for p in points]})

    def _set(self, pid: str, **fields) -> None:
        self._write({"kind": "set_payload", "id": pid, "fields": self._seal(fields)})

    # ---- operating points (load / speed) the healthy baseline covers ------------------------------------
    OP_TOLERANCE = 0.02         # relative margin around a taught range (speed jitter, slip)

    def _save_baseline_file(self) -> None:
        (pathlib.Path(self.cfg.root) / "baseline.json").write_text(json.dumps(
            {"baseline": self.baseline.to_dict(), "gate": self.gate.cfg.to_dict(), "taught": self.taught}))

    def _teach_ops(self, ops: list[dict | None]) -> None:
        """Widen the taught range of each operating variable to include these operating points."""
        for op in ops:
            for k, v in (op or {}).items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    lo, hi = self.taught.get(k, [float(v), float(v)])
                    self.taught[k] = [min(lo, float(v)), max(hi, float(v))]

    def _op_suggestion(self, u: dict) -> str:
        """Physics decides the wording: the median fault-signature score of the episode's first windows against the
        profile's threshold (chosen on CWRU; bench/operating_point.py measures it on HUST)."""
        scores = u.get("sig_scores") or []
        if not scores:
            return "this operating point was never taught: check whether the machine is healthy"
        med = float(np.median(scores))
        if med >= self.profile.SIGNATURE_THRESHOLD:
            return (f"a fault signature is present (defect score {med:.2f} >= {self.profile.SIGNATURE_THRESHOLD}) even "
                    "though this operating point was never taught")
        return (f"probably a new normal operating point (no fault signature, score {med:.2f}): if the machine is "
                "healthy, press 'Not a fault: normal operation'")

    def untaught(self, op: dict | None) -> dict:
        """Operating variables whose value lies outside what the healthy baseline was taught (with a margin).
        Empty if nothing is known: no data, no claim."""
        out = {}
        for k, v in (op or {}).items():
            if k in self.taught and isinstance(v, (int, float)):
                lo, hi = self.taught[k]
                m = self.OP_TOLERANCE * max(abs(lo), abs(hi), 1e-9)
                if v < lo - m or v > hi + m:
                    out[k] = {"value": round(float(v), 3), "taught": [round(lo, 3), round(hi, 3)]}
        return out

    # ---- baseline ---------------------------------------------------------------------------------------
    def fit_baseline(self, healthy_raw: np.ndarray, ops: list[dict | None] | None = None) -> dict:
        """Fit z-scoring stats on this machine's healthy windows, store them as baseline points, calibrate the
        gate from them. Replaces any previous baseline of this machine. `ops` (one per window, optional): the
        operating point of each window, e.g. {"speed_hz": 29.9, "load_kw": 1.5}, remembered as taught ranges."""
        with self._lock:
            self.taught = {}
            self._teach_ops(ops or [])
            b = Baseline.fit(healthy_raw, self.profile.fp_version, self.profile.min_std)
            z = b.z(np.asarray(healthy_raw, dtype=np.float64))
            g = calibrate(z, normal_factor=self.profile.normal_factor, target_false_alarm=self.profile.target_false_alarm)
            pts = [StorePoint(ids.baseline_point_id(self.cfg.machine_id, i),
                              {"type": "baseline", "machine_id": self.cfg.machine_id,
                               "fp_version": self.profile.fp_version}, vib=z[i].tolist()) for i in range(len(z))]
            for s in range(0, len(pts), 256):
                self._upsert(pts[s:s + 256])
            self.baseline, self.gate = b, NoveltyGate(self.store, self.cfg.machine_id, g)
            self._save_baseline_file()
            self.outbox.log("baseline", f"baseline fitted on {len(z)} healthy windows; tau_normal={g.tau_normal:.2f}, "
                                        f"tau_merge={g.tau_merge:.2f} (calibration q99 {g.calib_q99:.2f})"
                                        + (f"; taught operating points {self.taught}" if self.taught else ""))
            return g.to_dict() | {"n_windows": len(z), "taught": self.taught}

    # ---- live sensor input (any profile) ----------------------------------------------------------------
    def start_baseline_capture(self, n_windows: int) -> dict:
        """The next n_windows signal windows are treated as THIS machine's healthy state (the technician says the
        machine is known-good now); then the baseline is fitted from them."""
        if n_windows < 10:
            raise ValueError("capture at least 10 windows (the gate calibration needs them)")
        with self._lock:
            self._capture = {"want": int(n_windows), "features": [], "ops": []}
        self.outbox.log("baseline", f"capturing {n_windows} healthy windows from the live sensor")
        return self.capture_state()

    def capture_state(self) -> dict | None:
        c = self._capture
        return None if c is None else {"want": c["want"], "have": len(c["features"])}

    def ingest_signal(self, x, fs: float, rpm: float | None = None, source: str | None = None,
                      operating_point: dict | None = None) -> list[dict]:
        """Raw sensor data (any length; the profile cuts it into windows) -> fingerprint -> gate. While a baseline
        capture runs, windows only feed the capture. Returns one gate result per window. The operating point is
        the shaft speed (from rpm) plus anything the caller knows, e.g. {"load_kw": 1.2}."""
        results = []
        op = {k: float(v) for k, v in (operating_point or {}).items() if isinstance(v, (int, float))}
        if rpm and rpm > 0:
            op["speed_hz"] = rpm / 60.0
        fs_w = self.profile.analysis_fs or fs            # profile.windows() already resampled to the analysis rate
        # order features of the whole chunk at its native rate (fleet-learned hint; needs >= 0.5 s)
        of = None if isinstance(x, dict) else self.profile.order_features(np.asarray(x, dtype=float), fs, rpm)
        touched: set[str] = set()
        for w in self.profile.windows(x, fs) if not isinstance(x, dict) else [x]:
            f = self.profile.features(w, fs_w, rpm)
            self._tm_windows = getattr(self, "_tm_windows", 0) + 1    # readouts seen = windows + 1 (telemetry)
            if not np.all(np.isfinite(f)):
                raise ValueError("signal produced non-finite features (flat or corrupt input?)")
            with self._lock:
                if self._capture is not None:
                    self._capture["features"].append(f)
                    self._capture["ops"].append(op or None)
                    if len(self._capture["features"]) >= self._capture["want"]:
                        feats, ops = np.asarray(self._capture["features"]), self._capture["ops"]
                        self._capture = None
                        results.append({"state": "baseline", "fitted": self.fit_baseline(feats, ops)})
                    else:
                        results.append({"state": "capturing", **self.capture_state()})
                    continue
            r = self.ingest_window(f, source, op or None, velocity=self._velocity(w, fs_w))
            if self.profile.name == "telemetry":             # vehicle early-warning hint (trained on real data)
                self.risk_hint = vehicle_risk.risk(f, self.baseline, float(op.get("age", 0.0)), self._tm_windows + 1)
                if self.risk_hint and r.get("episode_id"):
                    self._set(r["episode_id"], risk_hint=self.risk_hint)
            if r["state"] != "normal" and not isinstance(w, dict):
                self.attach_diagnosis(r.get("episode_id"), w, fs_w, rpm)
            if r["state"] != "normal" and isinstance(w, dict) and r.get("episode_id") and w.get("codes"):
                self._note_codes(r["episode_id"], w.get("codes") or {})
            if r["state"] != "normal" and r.get("episode_id"):
                touched.add(r["episode_id"])
            results.append(r)
        for eid in touched:
            self.attach_order_features(eid, of)
        return results

    def _velocity(self, w, fs: float) -> float | None:
        """Vibration velocity of one window, only while an episode is verifying against a machine-card limit."""
        if self.card is None or isinstance(w, dict) or not self.verifying_with_limit():
            return None
        try:
            v = P.velocity_rms_mm_s(np.asarray(w, dtype=float), fs)
        except (TypeError, ValueError):
            return None
        return float(v) if np.isfinite(v) else None

    def verifying_with_limit(self) -> bool:
        return any((ep.get("verify") or {}).get("limit_mm_s") for ep in self.episodes(status="verifying"))

    def limit_for_verification(self) -> tuple[float | None, str | None]:
        """The acceptable velocity a fix must get below, from the machine card; None when this profile's signal is not
        a calibrated acceleration (phone, microphone, robots, events) or there is no card."""
        if self.card is None or self.profile.name not in ("bearing-12k", "rotating-hf"):
            return None, None
        bounds, ref = self.card.severity_bounds()
        return float(bounds[1]), ref

    # ---- the machine's own learned detector (edge/local_detector.py) --------------------------------------
    def _learned_alarm(self, z) -> float | None:
        if not self.profile.learned_detector:
            return None
        m = self.outbox.kv_get("local_detector")
        if not m:
            return None
        p = local_detector.probability(m, z)
        return p if p > 0.5 else None

    def train_local_detector(self) -> dict:
        """Healthy baseline fingerprints vs exemplars of technician-confirmed fault episodes of this machine."""
        with self._lock:
            normal = [r.vectors["vib"] for r in self.store.scroll(filter={"type": "baseline",
                                                                          "machine_id": self.cfg.machine_id}, with_vectors=True)]
            confirmed = {e["episode_id"] for e in self.episodes() if e.get("fault_class_source") == "technician"
                         and not e.get("dismissed") and e.get("fault_class") not in (None, "unknown")}
            faults = [r.vectors["vib"] for r in self.store.scroll(filter={"type": "exemplar",
                                                                          "machine_id": self.cfg.machine_id}, with_vectors=True)
                      if r.payload.get("episode_id") in confirmed]
            m = local_detector.train(np.asarray(normal, dtype=float), np.asarray(faults, dtype=float))
            self.outbox.kv_set("local_detector", m)
            cv = m["cross_validated"]
            self.outbox.log("detector", f"learned detector trained on this machine: {len(normal)} healthy vs "
                                        f"{len(faults)} confirmed-fault fingerprints ({len(confirmed)} episodes); "
                                        f"cross-validated detection {cv['detection_rate']:.0%}, false alarms "
                                        f"{cv['false_alarm_rate']:.0%}")
            return m

    # ---- fleet-learned fault hint -------------------------------------------------------------------------
    def fleet_model(self) -> dict | None:
        m = self.outbox.kv_get("fleet_hint_model")
        return m if fleet_hint.valid(m) else None

    def attach_order_features(self, eid: str | None, feats: list[float] | None) -> dict | None:
        """Keep the order features of an episode's first chunks (median = robust) and refresh the combined hint."""
        if not eid or feats is None:
            return None
        with self._lock:
            ep = self.store.get(eid)
            if ep is None or ep.payload.get("type") != "episode" or ep.payload["status"] == "closed":
                return None
            seen = list(ep.payload.get("order_feats_seen") or [])
            if len(seen) >= HINT_WINDOWS:
                return ep.payload.get("fleet_hint")
            seen.append([round(float(v), 4) for v in feats])
            med = np.median(np.asarray(seen), axis=0).round(4).tolist()
            h = fleet_hint.combine(ep.payload.get("fault_hint"), med, self.fleet_model())
            self._set(eid, order_feats_seen=seen, order_features=med, fleet_hint=h,
                      bearing=getattr(getattr(self.profile, "geometry", None), "name", None))
            return h

    def refresh_fleet_hints(self) -> int:
        """After a new fleet model arrives: recompute the combined hint of open episodes."""
        n = 0
        with self._lock:
            for ep in self.episodes():
                if ep["status"] != "closed" and ep.get("order_features"):
                    self._set(ep["episode_id"], fleet_hint=fleet_hint.combine(ep.get("fault_hint"), ep["order_features"],
                                                                              self.fleet_model()))
                    n += 1
        return n

    # ---- machine card (manufacturer data) -----------------------------------------------------------------
    def set_machine_card(self, data: dict) -> dict:
        card = machine_card.MachineCard.from_dict(data)
        with self._lock:
            old_geo = getattr(self.profile, "geometry", None)
            prof = profiles.make(self.cfg.profile, **self.cfg.profile_params)
            prof.apply_card(card)
            machine_card.save(self.cfg.root, card)
            self.card, self.profile = card, prof
            new_geo = getattr(prof, "geometry", None)
            recapture = bool(self.baseline is not None and old_geo != new_geo and prof.name != "bearing-12k")
            self.outbox.log("machine_card", f"machine card saved ({card.manufacturer} {card.model}); severity: "
                            f"{card.severity_bounds()[1]}" + ("; bearing geometry changed: RE-CAPTURE the healthy "
                                                               "baseline (the fingerprint uses defect frequencies)"
                                                               if recapture else ""))
            return card.to_dict() | {"baseline_recapture_recommended": recapture}

    def _note_codes(self, eid: str, codes: dict) -> None:
        """Event profile: remember which error codes this episode showed (for the code dictionary and search)."""
        with self._lock:
            ep = self.store.get(eid)
            if ep is None:
                return
            seen = dict(ep.payload.get("codes") or {})
            for c, n in codes.items():
                seen[str(c)[:40]] = seen.get(str(c)[:40], 0) + int(n) if isinstance(n, (int, float)) else 1
            if len(seen) <= 50:
                self._set(eid, codes=seen)

    def attach_diagnosis(self, eid: str | None, raw_window, fs: float, rpm: float | None) -> dict | None:
        """Physics diagnosis of one raw abnormal window, kept on the episode while it is young (first HINT_WINDOWS
        windows), and as `last_diagnosis` for the UI. Used by live input and by the recording replay."""
        d = self.profile.diagnose(raw_window, fs, rpm)
        if not d:
            return None
        self.last_diagnosis = _json_safe(d) | {"episode_id": eid}
        with self._lock:
            ep = self.store.get(eid) if eid else None
            if ep and ep.payload.get("type") == "episode" and ep.payload["occurrences"] <= HINT_WINDOWS:
                self._set(eid, physics=_json_safe(d))
        return d

    # ---- the per-window path ----------------------------------------------------------------------------
    def ingest_window(self, raw: np.ndarray, source: str | None = None, op: dict | None = None,
                      velocity: float | None = None) -> dict:
        if self.gate is None or self.baseline is None:
            raise RuntimeError("fit a healthy baseline first")
        with self._lock:
            self._op = op
            z = self.baseline.z(np.asarray(raw, dtype=float))
            r = self.gate.classify(z)
            learned = self._learned_alarm(z) if r.state == "normal" else None
            if learned is not None:                  # the machine's own learned detector adds an alarm
                r = self.gate.classify_abnormal(z, r)
            self.gate_ms.append(r.latency_ms)
            self.counters["windows"] += 1
            self.counters[r.state] += 1
            ts = now_iso()
            out: dict[str, Any] = {"state": r.state, "d_baseline": round(r.d_baseline, 2),
                                   "latency_ms": round(r.latency_ms, 3), "episode_id": r.episode_id, "source": source}
            if learned is not None:
                out["learned_detector"] = round(learned, 3)
            healthy = r.state == "normal"
            new_part = False
            # feed the verifier of every episode that is waiting for its post-action verdict
            for ep in self.episodes(status="verifying"):
                h = healthy
                vs = V.VerifyState.from_dict(ep.get("verify"))
                extra: dict[str, Any] = {}
                if not h and vs.mode == "replacement" and not vs.done:
                    hits = self.store.nearest(z, filter={"type": "exemplar", "episode_id": ep["episode_id"]}, limit=1)
                    if hits and r.d_baseline < hits[0].score:      # nearer to healthy than to the fault
                        h = new_part = True
                        vs.new_part_windows += 1
                        extra["new_part_z"] = [*(ep.get("new_part_z") or [])[-199:], [round(float(v), 5) for v in z]]
                vs = V.step(vs, h, velocity)
                self._set(ep["episode_id"], verify=vs.to_dict(), **extra)
                if vs.done:
                    self.outbox.log("verify", f"{vs.label()}", ep["episode_id"])
                    if vs.verdict == "symptom_resolved" and vs.mode == "replacement":
                        self._adopt_new_part(ep["episode_id"])
                    self._after_verdict(ep["episode_id"])
            if new_part and r.state != "normal":          # a new part's normal: do not open or grow an episode
                self.counters[r.state] -= 1
                self.counters["normal"] += 1
                r.state, r.episode_id = "normal", None
                out.update(state="normal", episode_id=None, new_part=True)
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
        hint = self.profile.hint(raw, z)
        payload = {
            "type": "episode", "episode_id": eid, "seq": seq, "machine_id": self.cfg.machine_id,
            "device_id": self.cfg.device_id, "site_id": self.cfg.site_id, "component": self.component,
            "profile": self.profile.name,
            "first_seen": ts, "last_seen": ts, "occurrences": 1, "status": "open", "n_exemplars": 1,
            "fault_hint": hint, "hint_votes": {hint["fault_class"]: 1}, "fault_class": None, "fault_class_source": None,
            "action_code": None, "root_cause_claim": None, "action_at": None, "note_text": "", "note_share_opt_in": False,
            "outcome": "pending", "technician_confirmed": False, "verify": None, "share_state": "local",
            "decision": None, "recurrence_of": recurrence_of, "schema_version": 1, "fp_version": self.profile.fp_version,
            "text_model": self.embedder.name, "version": 1,
        }
        op = getattr(self, "_op", None)
        if op:
            payload["operating_point"] = {k: round(v, 4) for k, v in op.items()}
            new_op = self.untaught(op)
            if new_op:
                s = self.profile.signature_score(raw)
                payload["untaught_operating_point"] = new_op | {"sig_scores": [] if s is None else [round(s, 3)]}
                payload["untaught_operating_point"]["suggestion"] = self._op_suggestion(payload["untaught_operating_point"])
        text = self._doc_text(payload)
        self._upsert([
            StorePoint(eid, payload, vib=z.tolist(), note=self.embedder.embed_documents([text])[0], bm25_text=text),
            StorePoint(ids.make_id("exemplar", eid, 0), {"type": "exemplar", "machine_id": self.cfg.machine_id,
                                                         "episode_id": eid, "episode_active": True}, vib=z.tolist()),
        ])
        msg = "unfamiliar state: new episode opened"
        if recurrence_of:
            msg += f" (resembles closed episode {recurrence_of[:8]} on this machine)"
        if payload.get("untaught_operating_point"):
            msg += "; UNTAUGHT operating point: " + payload["untaught_operating_point"]["suggestion"]
        self.outbox.log("gate", msg + f"; physics hint: {hint['fault_class']}", eid)
        self._decide(eid)
        self.check_followups()
        return eid

    def _merge(self, eid: str, z: np.ndarray, raw: np.ndarray, d_episode: float | None, ts: str) -> None:
        ep = self.store.get(eid).payload
        fields: dict[str, Any] = {"occurrences": ep["occurrences"] + 1, "last_seen": ts}
        u = ep.get("untaught_operating_point")
        if u and ep["occurrences"] < HINT_WINDOWS:              # refine the operating-point suggestion (median)
            s = self.profile.signature_score(raw)
            if s is not None:
                u = u | {"sig_scores": [*u.get("sig_scores", []), round(s, 3)]}
                fields["untaught_operating_point"] = u | {"suggestion": self._op_suggestion(u)}
        if ep["occurrences"] < HINT_WINDOWS:                   # majority vote of the first windows' hints
            votes = dict(ep.get("hint_votes") or {})
            h = self.profile.hint(raw, z)
            votes[h["fault_class"]] = votes.get(h["fault_class"], 0) + 1
            best = max(votes, key=votes.get)
            fields["hint_votes"] = votes
            fields["fault_hint"] = ep["fault_hint"] | {
                "fault_class": best, "measured_accuracy": self.profile.measured(best),
                "why": f"{h['why'] if best == h['fault_class'] else 'physics rule'}; "
                       f"voted in {votes[best]} of the first {sum(votes.values())} windows"}
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

    def _require(self, eid: str, expected_version: int | None = None) -> dict:
        """The episode payload. With expected_version (the version the technician's screen showed), refuse the
        edit if the episode has moved on: an optimistic compare-and-set, checked under the device lock."""
        rec = self.store.get(eid)
        if rec is None or rec.payload.get("type") != "episode":
            raise KeyError(f"no episode {eid}")
        cur = int(rec.payload.get("version", 1))
        if expected_version is not None and int(expected_version) != cur:
            self.outbox.log("conflict", f"edit refused: expected version {expected_version}, current {cur}", eid)
            raise VersionConflict(eid, int(expected_version), cur)
        return rec.payload

    def set_note(self, eid: str, text: str, share_opt_in: bool = False, expected_version: int | None = None) -> dict:
        with self._lock:
            self._require(eid, expected_version)
            if len(text) > 2000:
                raise ValueError("note too long (max 2000 characters)")
            self._rewrite(eid, note_text=text.strip(), note_share_opt_in=bool(share_opt_in))
            self.outbox.log("note", f"note saved ({len(text)} chars, share opt-in={bool(share_opt_in)})", eid)
            return self._decide(eid)

    def set_fault_class(self, eid: str, fault_class: str, expected_version: int | None = None,
                        damage_mode: str | None = None) -> dict:
        """The technician confirms the fault class, ideally from what they SAW (ISO 15243 damage mode on the removed
        bearing). This confirmed class - never the hint - is what the fleet groups by and learns from."""
        with self._lock:
            self._require(eid, expected_version)
            FaultClass(fault_class)
            if damage_mode:
                DamageMode(damage_mode)
            self._rewrite(eid, fault_class=fault_class, fault_class_source="technician", damage_mode=damage_mode or None)
            self.outbox.log("fault_class", f"technician confirmed fault class {fault_class}"
                            + (f" (seen: {damage_mode.replace('_', ' ')}, ISO 15243)" if damage_mode else ""), eid)
            return self._decide(eid)

    def record_action(self, eid: str, action_code: str, root_cause: str | None = None,
                      required_windows: int = V.REQUIRED_WINDOWS, expected_version: int | None = None) -> dict:
        """An intervention was made: from now on, windows are fed to this episode's outcome verifier."""
        with self._lock:
            ep = self._require(eid, expected_version)
            if ep["status"] in TERMINAL:
                raise ValueError("episode is closed")
            ActionCode(action_code)
            if root_cause:
                RootCause(root_cause)
            lim, src = self.limit_for_verification()
            mode = "replacement" if action_code in V.REPLACEMENT_ACTIONS else "same_part"
            self._rewrite(eid, action_code=action_code, root_cause_claim=root_cause or None, action_at=now_iso(),
                          status="verifying", verify=V.VerifyState(required=int(required_windows), limit_mm_s=lim,
                                                                   limit_source=src, mode=mode).to_dict(),
                          outcome="pending", technician_confirmed=False)
            self.store.set_payload_where({"type": "exemplar", "episode_id": eid}, {"episode_active": True})
            self.outbox.log("action", f"action {action_code} recorded; verifying over {required_windows} windows", eid)
            return self._decide(eid)

    def confirm_outcome(self, eid: str, outcome: str, expected_version: int | None = None) -> dict:
        with self._lock:
            ep = self._require(eid, expected_version)
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

    def _adopt_new_part(self, eid: str) -> int:
        """A replacement verified by the nearest-state rule: its windows become extra healthy-baseline points, so the
        new part's normal is normal from now on (z-normalisation not refitted, like mark_normal)."""
        ep = self.store.get(eid).payload
        zs = ep.get("new_part_z") or []
        pts = [StorePoint(ids.make_id("baseline-newpart", self.cfg.machine_id, eid, k),
                          {"type": "baseline", "machine_id": self.cfg.machine_id, "fp_version": self.profile.fp_version,
                           "regime_from_episode": eid, "new_part": True}, vib=list(v)) for k, v in enumerate(zs)]
        if pts:
            self._upsert(pts)
            self._set(eid, new_part_z=None, new_part_adopted=len(pts))
            self.outbox.log("baseline", f"new part verified: {len(pts)} of its windows added to this machine's healthy "
                                        "baseline", eid)
        return len(pts)

    # ---- policy -----------------------------------------------------------------------------------------
    def _decide(self, eid: str) -> dict:
        rec = self.store.get(eid, with_vectors=True)
        ep = rec.payload
        if ep.get("dismissed"):                           # not a fault: nothing to share, ever
            d = {"action": "KEEP_LOCAL", "event_id": None, "note_shared": False,
                 "reasons": [{"gate": "human", "ok": True, "detail": "marked as normal operation by the technician; "
                              "its fingerprints now extend the healthy baseline"}]}
            self._set(eid, decision=d)
            return d
        red = redact(ep["note_text"], self.cfg.denylist) if ep.get("note_text") else None
        d = policy.decide(ep, fingerprint=rec.vectors["vib"], redaction=red, device_id=self.cfg.device_id,
                          machine_class=self.cfg.machine_class,
                          already_queued=self.outbox.known_event_ids() - {ep.get("event_id")},   # own event is not a duplicate
                          already_repairs={h: e for h, e in self.outbox.known_repairs().items() if e != eid},
                          fp_version=self.profile.fp_version)
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

    # ---- "this WAS a fault": teach a failure the gate did not flag ----------------------------------------
    def teach_fault(self, raw: np.ndarray, fault_class: str, note: str = "") -> dict:
        """The technician saw a failure (e.g. the robot collided) in a cycle the gate called normal. The window is
        stored as a confirmed-fault example (a closed, 'taught' episode: never shared, no action), so the machine's
        learned detector (edge/local_detector.py) can learn it and the gate recognises it as a recurrence later.
        The mirror image of mark_normal()."""
        if self.gate is None or self.baseline is None:
            raise RuntimeError("fit a healthy baseline first")
        FaultClass(fault_class)
        with self._lock:
            z = self.baseline.z(np.asarray(raw, dtype=float))
            seq = int(self.outbox.kv_get("episode_seq", 0)) + 1
            self.outbox.kv_set("episode_seq", seq)
            eid = ids.episode_id(self.cfg.device_id, seq)
            ts = now_iso()
            payload = {"type": "episode", "episode_id": eid, "seq": seq, "machine_id": self.cfg.machine_id,
                       "device_id": self.cfg.device_id, "site_id": self.cfg.site_id, "component": self.component,
                       "profile": self.profile.name, "first_seen": ts, "last_seen": ts, "occurrences": 1,
                       "status": "closed", "closed_at": ts, "n_exemplars": 1,
                       "fault_hint": {"fault_class": "unknown", "why": "taught by the technician"}, "hint_votes": {},
                       "fault_class": fault_class, "fault_class_source": "technician", "action_code": None,
                       "root_cause_claim": None, "action_at": None, "note_text": note.strip()[:2000],
                       "note_share_opt_in": False, "outcome": "pending", "technician_confirmed": True, "verify": None,
                       "share_state": "local", "decision": None, "recurrence_of": None, "schema_version": 1,
                       "fp_version": self.profile.fp_version, "text_model": self.embedder.name, "version": 1,
                       "taught_fault": True}
            text = self._doc_text(payload)
            self._upsert([
                StorePoint(eid, payload, vib=z.tolist(), note=self.embedder.embed_documents([text])[0], bm25_text=text),
                StorePoint(ids.make_id("exemplar", eid, 0), {"type": "exemplar", "machine_id": self.cfg.machine_id,
                                                             "episode_id": eid, "episode_active": False}, vib=z.tolist())])
            self.outbox.log("teach", f"technician taught a {fault_class.replace('_', ' ')} the gate had called normal "
                                     "(kept on this device as a learning example)", eid)
            return self._decide(eid)

    # ---- "not a fault": teach a new healthy operating state ----------------------------------------------
    def mark_normal(self, eid: str, expected_version: int | None = None) -> dict:
        """The technician says this episode is normal operation (a new load, speed or a re-mounted sensor), not a
        fault. Its stored fingerprints become extra healthy-baseline points of this machine (the z-normalisation is
        NOT refitted, so every stored vector stays comparable); the episode closes as dismissed and is never shared.
        Found necessary by the HUST held-out test: healthy data at an untaught load looked abnormal."""
        with self._lock:
            ep = self._require(eid, expected_version)
            if ep["status"] == "closed":
                raise ValueError("episode is closed")
            ex = [r for r in self.store.scroll(filter={"type": "exemplar", "episode_id": eid}, with_vectors=True)]
            pts = [StorePoint(ids.make_id("baseline-extra", self.cfg.machine_id, eid, k),
                              {"type": "baseline", "machine_id": self.cfg.machine_id, "fp_version": self.profile.fp_version,
                               "regime_from_episode": eid}, vib=r.vectors["vib"]) for k, r in enumerate(ex)]
            if pts:
                self._upsert(pts)
            self._write({"kind": "archive", "episode_id": eid, "keep": None,
                         "fields": {"status": "closed", "closed_at": now_iso(), "n_exemplars": 0,
                                    "dismissed": {"reason": "normal operation (not a fault)", "at": now_iso(),
                                                  "baseline_points_added": len(pts)}}})
            self._teach_ops([ep.get("operating_point"), getattr(self, "_op", None) if self._run_episode == eid else None])
            if self.gate is not None:
                self._save_baseline_file()
            self.outbox.log("baseline", f"technician: normal operation, not a fault; {len(pts)} fingerprint(s) added "
                                        f"to this machine's healthy baseline"
                                        + (f"; taught operating points now {self.taught}" if self.taught else ""), eid)
            self._run_episode = None
            return self._decide(eid)

    # ---- retention (docs/RESEARCH.md G.1 / G.8: separate from the share decision) -----------------------
    def run_retention(self, now: dt.datetime | None = None) -> dict:
        """ARCHIVE closed, decided episodes older than policy.ARCHIVE_AFTER_DAYS: keep the episode point and its
        first exemplar (so a recurrence is still recognised), drop the other exemplar fingerprints. Knowledge is
        never deleted; open, undecided or not-yet-uploaded episodes are never touched."""
        now = now or dt.datetime.now(dt.timezone.utc)
        archived, checked = [], 0
        with self._lock:
            for ep in self.episodes(status="closed"):
                checked += 1
                if ep.get("archived") or ep.get("share_state") in ("queued", "uploading"):
                    continue
                if policy.retention(ep, now) != "ARCHIVE":
                    continue
                eid = ep["episode_id"]
                keep = self.store.get(ids.make_id("exemplar", eid, 0), with_vectors=True)
                self._write({"kind": "archive", "episode_id": eid,
                             "keep": {"id": keep.id, "payload": keep.payload, "vib": keep.vectors["vib"]} if keep else None,
                             "fields": {"archived": True, "archived_at": now.isoformat(timespec="seconds"),
                                        "n_exemplars": 1 if keep else 0}})
                self.outbox.log("retention", f"ARCHIVE: closed + decided + last seen over {policy.ARCHIVE_AFTER_DAYS} "
                                             f"days ago; kept the episode and 1 exemplar", eid)
                archived.append(eid)
        self._last_retention = time.time()
        return {"checked": checked, "archived": archived}

    # ---- did the fix HOLD? (follow-ups weeks after a verified fix) ------------------------------------------
    def check_followups(self, now: dt.datetime | None = None) -> list[dict]:
        """For every shared 'worked' fix without a follow-up: 'recurred' if this machine opened a new episode that
        resembles it (the gate's recurrence match) or carries the same confirmed fault class within hold_days of the
        action; 'held' once hold_days passed without that. One follow-up per fix, queued in the outbox."""
        now = now or dt.datetime.now(dt.timezone.utc)
        out = []
        with self._lock:
            eps = self.episodes()
            for ep in eps:
                if (ep.get("outcome") != "worked" or not ep.get("event_id") or ep.get("followup")
                        or not ep.get("action_at") or self.outbox.status_of(ep["event_id"]) in (None, "rejected")):
                    continue
                t0 = dt.datetime.fromisoformat(ep["action_at"])
                later = [e for e in eps if e["seq"] > ep["seq"] and not e.get("dismissed") and (
                    e.get("recurrence_of") == ep["episode_id"]
                    or (ep.get("fault_class") and e.get("fault_class") == ep["fault_class"]))]
                recur = next((e for e in sorted(later, key=lambda e: e["first_seen"])
                              if dt.datetime.fromisoformat(e["first_seen"]) >= t0), None)
                days_recur = ((dt.datetime.fromisoformat(recur["first_seen"]) - t0).total_seconds() / 86400
                              if recur else None)
                if recur is not None and days_recur <= self.cfg.hold_days:
                    status, days = "recurred", days_recur
                elif (now - t0).total_seconds() / 86400 >= self.cfg.hold_days:
                    status, days = "held", self.cfg.hold_days
                else:
                    continue
                fu = FollowUp(event_id=ids.followup_id(self.cfg.device_id, ep["event_id"]), episode_id=ep["episode_id"],
                              refers_to=ep["event_id"], status=status, days_after_fix=round(max(days, 0.0), 3),
                              hold_days=self.cfg.hold_days, occurred_at=now_iso()).model_dump(mode="json")
                self.outbox.enqueue(fu)
                self._set(ep["episode_id"], followup={
                    "status": status, "days_after_fix": fu["days_after_fix"], "hold_days": self.cfg.hold_days,
                    "event_id": fu["event_id"],
                    "recurrence_episode": recur["episode_id"] if recur is not None and status == "recurred" else None})
                self.outbox.log("followup", (f"fix HELD for {self.cfg.hold_days:g} days" if status == "held" else
                                             f"fault RECURRED {fu['days_after_fix']:.1f} days after the fix")
                                + "; follow-up queued for the fleet", ep["episode_id"])
                out.append(fu)
        return out

    def maybe_run_retention(self) -> dict | None:
        """Hourly housekeeping: retention, follow-ups, and (if enabled) re-compressing files Edge created since last
        time."""
        if time.time() - self._last_retention >= RETENTION_EVERY_S:
            self.check_followups()
            out = self.run_retention()
            if self.cfg.compress_storage:
                out["compression"] = storage_os.enable_compression(self.cfg.root)
            return out
        return None

    # ---- usefulness feedback (G.1 "Evaluate usefulness": a counter, no learned model) ---------------------
    def record_feedback(self, result_id: str, kind: str, helped: bool) -> dict:
        """The technician marks a search result as helpful or not. Stored on this device only (SQLite kv) and
        shown next to the result; it never changes ranking and is not shared."""
        key = self._feedback_key(result_id, kind)
        with self._lock:
            fb = self.outbox.kv_get(key, {"helped": 0, "not_helped": 0})
            fb["helped" if helped else "not_helped"] += 1
            self.outbox.kv_set(key, fb)
        self.outbox.log("feedback", f"{kind} result {result_id[:8]} marked {'helpful' if helped else 'not helpful'}")
        return fb

    @staticmethod
    def _feedback_key(result_id: str, kind: str) -> str:
        if kind not in ("local", "fleet"):
            raise ValueError("kind must be local or fleet")
        return f"feedback:{kind}:{canonical_id(result_id)}"          # ValueError if not a point id

    def feedback_for(self, result_id: str, kind: str) -> dict:
        return self.outbox.kv_get(self._feedback_key(result_id, kind), {"helped": 0, "not_helped": 0})

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
                fleet_filter = {"component": self.component, "!status": "retracted"}
                if fc and fc != "unknown":
                    fleet_filter["fault_class"] = fc
                # the mirror's text vectors come from the cloud's model; a query vector from another model would
                # compare meaningless numbers, so that leg is dropped and BM25 + fingerprint carry the fleet search
                fleet_model = self.outbox.kv_get("fleet_text_model", FLEET_TEXT_MODEL)   # announced by the cloud
                same_model = self.embedder.name == fleet_model
                fleet = self.mirror.search(vib=vib, note=note if same_model else None, text=text, filter=fleet_filter,
                                           limit=limit, weights={"vib": 0.25}, explain=True)
                if not same_model:
                    fleet_error = (f"fleet dense-text leg skipped: this device embeds with {self.embedder.name}, "
                                   f"the fleet with {fleet_model}; BM25 + fingerprint used")
        except Exception as e:              # a broken mirror must never take local memory down with it
            fleet, fleet_error = [], f"fleet mirror unavailable ({type(e).__name__}); showing local memory only"
        ms = (time.perf_counter() - t0) * 1000
        self.search_ms.append(ms)
        slim = lambda p: {k: v for k, v in p.items() if k not in ("hint_votes",)}
        return {"query": {"text": text, "episode_id": episode_id, "fault_class": fc, "fault_class_source": src,
                          "fleet_filter": fleet_filter},
                "local": [{"id": h.id, "rrf": round(h.score, 4), "legs": h.legs, "episode": slim(h.payload),
                           "feedback": self.feedback_for(h.id, "local")} for h in local],
                "fleet": [{"id": h.id, "rrf": round(h.score, 4), "legs": h.legs, "case": h.payload,
                           "feedback": self.feedback_for(h.id, "fleet")} for h in fleet],
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
            "machine_class": self.cfg.machine_class, "component": self.component, "profile": self.profile.describe(),
            "baseline_capture": self.capture_state(), "last_diagnosis": self.last_diagnosis,
            "risk_hint": getattr(self, "risk_hint", None), "note_protection": self.note_protection,
            "machine_card": self.card.to_dict() if self.card else None,
            "fleet_hint_model": self._model_summary(), "hold_days": self.cfg.hold_days,
            "local_detector": ({k: v for k, v in (self.outbox.kv_get("local_detector") or {}).items()
                                if k in ("trained_on", "cross_validated", "trained_at")} or None)
            if self.profile.learned_detector else None,
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

    def _model_summary(self) -> dict | None:
        m = self.fleet_model()
        return None if m is None else {k: m.get(k) for k in ("classes", "trained_on", "unseen_device_accuracy",
                                                             "unseen_device_cases", "trained_at")}

    def close(self) -> None:
        with self._lock:
            self.store.close()
            self.mirror.close()
            self.outbox.close()
