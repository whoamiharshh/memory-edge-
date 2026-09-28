"""Edge-local API + UI for one device (FastAPI). Binds to localhost by default.

Every /api route requires the operator token (header X-Operator-Token): the device holds raw technician
notes, so even local reads are authenticated. The token is compared in constant time against its sha256.
"""
from __future__ import annotations

import base64
import contextlib
import binascii
import hashlib
import hmac
import pathlib

import numpy as np

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from edge import procedures as procedures_mod
from edge import machine_card as machine_card_mod
from edge import manuals as manuals_mod
from edge import physics as P
from edge import rag
from edge.device import Device
from edge.replay import HEALTHY_BASELINE_FILES, Recordings, ReplayRunner
from edge.sync_worker import SyncWorker
from shared.schema import ActionCode, Component, DamageMode, FaultClass, RootCause

UI = pathlib.Path(__file__).resolve().parent / "ui"
MAX_BODY = 8 * 1024 * 1024                  # a signal chunk (600k values as JSON) or 5 s of audio fit
MAX_MANUAL_BODY = 30 * 1024 * 1024          # a 20 MB PDF, base64-encoded
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                               "img-src 'self' data:; connect-src 'self'; worker-src 'self'; manifest-src 'self'; "
                               "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), geolocation=(), microphone=(self), accelerometer=(self), gyroscope=(self)",
}


class Versioned(BaseModel):
    expected_version: int | None = Field(default=None, ge=1)   # the version the screen showed; 409 if it moved


class NoteBody(Versioned):
    text: str = Field(max_length=2000)
    share_opt_in: bool = False


class FaultBody(Versioned):
    fault_class: FaultClass
    damage_mode: DamageMode | None = None       # ISO 15243: what was SEEN on the removed part


class ActionBody(Versioned):
    action_code: ActionCode
    root_cause: RootCause | None = None
    required_windows: int = Field(default=20, ge=3, le=500)


class ConfirmBody(Versioned):
    outcome: str = Field(pattern=r"^(worked|failed)$")


MAX_SIGNAL_VALUES = 600_000


class SignalBody(BaseModel):
    """A chunk of raw sensor data for this device's profile: `samples` (1 channel), `axes` (n rows x k channels,
    e.g. phone x/y/z or robot Fx..Tz) or `events` (one bucket of error codes)."""
    samples: list[float] | None = None
    axes: list[list[float]] | None = None
    events: dict | None = None
    fs: float = Field(default=1.0, gt=0, le=200_000)
    rpm: float | None = Field(default=None, gt=0, le=100_000)
    source: str | None = Field(default=None, max_length=80)
    operating_point: dict[str, float] | None = Field(default=None, max_length=10)   # e.g. {"load_kw": 1.2}


MAX_AUDIO_SECONDS = 5


class AudioBody(BaseModel):
    """Microphone audio for the `acoustic` profile: 16-bit little-endian mono PCM, base64 (compact: a phone sends
    ~1 s chunks at 44.1/48 kHz)."""
    pcm16_b64: str = Field(min_length=4, max_length=int(96_000 * 2 * MAX_AUDIO_SECONDS * 4 / 3) + 8)
    fs: float = Field(ge=8000, le=96_000)
    rpm: float | None = Field(default=None, gt=0, le=100_000)
    source: str | None = Field(default=None, max_length=80)


class ManualBody(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    source: str = Field(default="", max_length=300)
    pdf_b64: str = Field(min_length=8, max_length=manuals_mod.MAX_BYTES * 4 // 3 + 8)


class ProcedureBody(BaseModel):
    """A site SOP typed in by a technician. The source (the manual or SOP it follows) is mandatory."""
    title: str = Field(min_length=3, max_length=120)
    fault_classes: list[FaultClass] = Field(min_length=1, max_length=5)
    components: list[Component] = Field(min_length=1, max_length=5)
    confirm_first: list[str] = Field(default_factory=list, max_length=10)
    steps: list[str] = Field(min_length=1, max_length=20)
    source_title: str = Field(min_length=3, max_length=160)


class CaptureBody(BaseModel):
    windows: int = Field(ge=10, le=5000)


class FeedbackBody(BaseModel):
    result_id: str = Field(min_length=36, max_length=36)
    kind: str = Field(pattern=r"^(local|fleet)$")
    helped: bool


class SearchBody(BaseModel):
    text: str | None = Field(default=None, max_length=500)
    episode_id: str | None = None
    use_fleet: bool = True
    limit: int = Field(default=5, ge=1, le=20)


class BriefBody(SearchBody):
    question: str = Field(default="What has been tried for this fault, and how did it turn out?", max_length=300)


class NetBody(BaseModel):
    online: bool


class ReplayBody(BaseModel):
    fid: int
    n: int | None = Field(default=None, ge=1, le=5000)
    interval: float = Field(default=0.15, ge=0.0, le=5.0)
    start: int = Field(default=0, ge=0)


class TokenBody(BaseModel):
    token: str = Field(min_length=10, max_length=200)


def create_app(device: Device, worker: SyncWorker, operator_token: str, llm: rag.LocalLLM | None = None) -> FastAPI:
    app = FastAPI(title=f"Machine Memory - {device.cfg.device_id}", docs_url="/docs", lifespan=_lifespan)
    SITE_SOPS = pathlib.Path(device.cfg.root) / "site_procedures.json"
    def _replay_physics(result: dict, fid: int, i: int) -> None:
        """Replayed recordings carry cached fingerprints; the physics panel and the fleet hint need raw signal, read
        on demand (a window for the diagnosis, a 1 s segment for the order features)."""
        if device.profile.name != "bearing-12k" or not result.get("episode_id"):
            return
        ep = device.store.get(result["episode_id"])
        if ep is None or ep.payload.get("physics") and ep.payload["occurrences"] > 3:
            return                                   # enough: only the first windows of an episode are diagnosed
        rw = Recordings.raw_window(fid, i)
        if rw is not None:
            device.attach_diagnosis(result["episode_id"], *rw)
        seg = Recordings.raw_segment(fid, i)
        if seg is not None:
            device.attach_order_features(result["episode_id"], device.profile.order_features(*seg))

    replay = ReplayRunner(device.ingest_window, on_abnormal=_replay_physics)
    op_hash = hashlib.sha256(operator_token.encode()).hexdigest()
    app.state.device, app.state.worker, app.state.replay = device, worker, replay

    def operator(request: Request) -> None:
        tok = request.headers.get("x-operator-token", "")
        if not hmac.compare_digest(hashlib.sha256(tok.encode()).hexdigest(), op_hash):
            raise HTTPException(401, "operator token required")

    def guard(fn):
        try:
            return fn()
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    api = Depends(operator)

    @app.middleware("http")
    async def harden(request: Request, call_next):
        """Body-size limit (manual PDFs may be large, everything else small) + browser security headers. The device UI
        may use the microphone and motion sensor of the page's own origin only."""
        cl = request.headers.get("content-length")
        cap = MAX_MANUAL_BODY if request.url.path == "/api/manuals" else MAX_BODY
        if cl and (not cl.isdigit() or int(cl) > cap):
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        resp = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        if request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/api/health")
    def health():
        return {"ok": True, "device_id": device.cfg.device_id}

    @app.get("/api/stats", dependencies=[api])
    def stats():
        return device.stats() | {"sync": worker.status(), "replay": replay.state,
                                 "llm": {"available": bool(llm and llm.available), "model": rag.MODEL_NAME}}

    @app.get("/api/enums", dependencies=[api])
    def enums():
        return {"action_codes": [a.value for a in ActionCode], "fault_classes": [f.value for f in FaultClass],
                "components": [c.value for c in Component],
                "root_causes": [r.value for r in RootCause], "damage_modes": [d.value for d in DamageMode]}

    @app.post("/api/baseline/fit", dependencies=[api])
    def fit():
        return guard(lambda: device.fit_baseline(Recordings.baseline(HEALTHY_BASELINE_FILES)))

    @app.get("/api/episodes", dependencies=[api])
    def episodes(status: str | None = None):
        return device.episodes(status)

    @app.get("/api/episodes/{eid}", dependencies=[api])
    def episode(eid: str):
        return guard(lambda: device.episode(eid))

    @app.post("/api/episodes/{eid}/note", dependencies=[api])
    def note(eid: str, b: NoteBody):
        return guard(lambda: device.set_note(eid, b.text, b.share_opt_in, b.expected_version))

    @app.get("/api/machine-card", dependencies=[api])
    def get_machine_card():
        return {"card": device.card.to_dict() if device.card else None,
                "bearings": {k: g.name for k, g in P.BEARINGS.items()},
                "machine_types": list(machine_card_mod.MACHINE_TYPES)}

    @app.put("/api/machine-card", dependencies=[api])
    def put_machine_card(card: dict):
        if len(str(card)) > 5000:
            raise HTTPException(413, "machine card too large")
        return guard(lambda: device.set_machine_card(card))

    @app.get("/api/manuals", dependencies=[api])
    def list_manuals():
        return manuals_mod.listing(device)

    @app.post("/api/manuals", dependencies=[api])
    def add_manual(b: ManualBody):
        try:
            pdf = base64.b64decode(b.pdf_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(422, "pdf_b64 is not valid base64")
        return guard(lambda: manuals_mod.add(device, b.title, pdf, b.source))

    @app.delete("/api/manuals/{doc_id}", dependencies=[api])
    def delete_manual(doc_id: str):
        if not manuals_mod.remove(device, doc_id):
            raise HTTPException(404, "no such manual")
        return {"removed": doc_id}

    @app.get("/api/manuals/search", dependencies=[api])
    def search_manuals(q: str | None = None, episode_id: str | None = None, limit: int = 5):
        if episode_id:
            q = manuals_mod.query_for(guard(lambda: device.episode(episode_id)))
        if not q or len(q) > 500:
            raise HTTPException(422, "give q (<= 500 characters) or episode_id")
        return {"query": q, "hits": manuals_mod.search(device, q, max(1, min(limit, 20)))}

    class TeachFaultBody(SignalBody):
        fault_class: FaultClass
        note: str = Field(default="", max_length=2000)

    @app.post("/api/teach/fault", dependencies=[api])
    def teach_fault(b: TeachFaultBody):
        """A failure the operator SAW in a cycle the gate called normal (e.g. a robot collision): send that cycle's
        signal with the fault class; each window becomes a confirmed learning example on this device."""
        x = np.asarray(b.samples if b.samples is not None else b.axes, dtype=np.float64) if b.events is None else b.events
        ws = device.profile.windows(x, b.fs) if not isinstance(x, dict) else [x]
        fs_w = device.profile.analysis_fs or b.fs
        return {"taught": [guard(lambda w=w: device.teach_fault(device.profile.features(w, fs_w, b.rpm),
                                                                 b.fault_class.value, b.note)) for w in ws[:50]]}

    @app.post("/api/detector/train", dependencies=[api])
    def train_detector():
        if not device.profile.learned_detector:
            raise HTTPException(409, f"the {device.profile.name} profile does not use a learned detector")
        return guard(lambda: device.train_local_detector())

    @app.post("/api/followups/check", dependencies=[api])
    def followups():
        return {"queued": guard(lambda: device.check_followups())}

    @app.post("/api/ingest/audio", dependencies=[api])
    def ingest_audio(b: AudioBody):
        """Phone / any microphone -> the acoustic profile (sound carries the bearing impacts at a rate a phone CAN
        deliver; its motion sensor is capped at 60 Hz in browsers)."""
        if device.profile.name != "acoustic":
            raise HTTPException(409, f"this device watches '{device.profile.name}'; start it with --profile acoustic "
                                     "to listen with a microphone")
        try:
            raw = base64.b64decode(b.pcm16_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(422, "pcm16_b64 is not valid base64")
        if len(raw) % 2 or not (0.5 * b.fs * 2 <= len(raw) <= MAX_AUDIO_SECONDS * b.fs * 2):
            raise HTTPException(422, f"send 0.5-{MAX_AUDIO_SECONDS} s of 16-bit mono PCM")
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        if float(np.std(x)) < 1e-5:
            raise HTTPException(422, "silent audio (microphone muted or blocked?)")
        if device.gate is None and device.capture_state() is None:
            raise HTTPException(409, "no baseline yet: POST /api/baseline/capture first (machine known-good)")
        res = guard(lambda: device.ingest_signal(x, b.fs, b.rpm, b.source or "microphone"))
        states: dict[str, int] = {}
        for r in res:
            states[r["state"]] = states.get(r["state"], 0) + 1
        return {"windows": len(res), "states": states, "last": res[-1] if res else None,
                "capture": device.capture_state(), "diagnosis": device.last_diagnosis,
                "level_dbfs": round(20 * float(np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)), 1)}

    @app.post("/api/episodes/{eid}/fault_class", dependencies=[api])
    def fault(eid: str, b: FaultBody):
        return guard(lambda: device.set_fault_class(eid, b.fault_class.value, b.expected_version,
                                                        b.damage_mode.value if b.damage_mode else None))

    @app.post("/api/episodes/{eid}/action", dependencies=[api])
    def action(eid: str, b: ActionBody):
        return guard(lambda: device.record_action(eid, b.action_code.value, b.root_cause.value if b.root_cause else None,
                                                  b.required_windows, b.expected_version))

    @app.post("/api/episodes/{eid}/confirm", dependencies=[api])
    def confirm(eid: str, b: ConfirmBody):
        return guard(lambda: device.confirm_outcome(eid, b.outcome, b.expected_version))

    @app.post("/api/search", dependencies=[api])
    def search(b: SearchBody):
        return guard(lambda: device.search(b.text, b.episode_id, b.use_fleet, b.limit))

    @app.get("/api/procedures", dependencies=[api])
    def procedures(fault_class: str | None = None, episode_id: str | None = None):
        """Documented reference procedures for a fault class (or for an episode's confirmed class / physics hint)."""
        component = device.component
        if episode_id:
            ep = guard(lambda: device.episode(episode_id))
            fault_class = ep.get("fault_class") or (ep.get("fault_hint") or {}).get("fault_class")
            component = ep.get("component") or component
        codes = []
        if episode_id:
            for c in sorted((ep.get("codes") or {}), key=lambda c: -(ep["codes"][c]))[:10]:
                codes.append(procedures_mod.code_info(c) or {"code": c, "title": None,
                                                              "source_note": "not a standard vehicle code: needs a site SOP"})
        return {"fault_class": fault_class, "component": component, "codes": codes,
                "procedures": procedures_mod.lookup(fault_class, component, SITE_SOPS)}

    @app.post("/api/procedures", dependencies=[api])
    def add_procedure(b: ProcedureBody):
        return guard(lambda: procedures_mod.add_site_procedure(SITE_SOPS, b.model_dump(mode="json")))

    @app.get("/api/codes/{code}", dependencies=[api])
    def code(code: str):
        info = procedures_mod.code_info(code)
        if info is None:
            raise HTTPException(404, "not a known standard vehicle fault code")
        return info

    @app.get("/api/profile", dependencies=[api])
    def profile():
        return device.profile.describe() | {"component": device.component, "baseline_ready": device.gate is not None,
                                            "capture": device.capture_state()}

    @app.post("/api/baseline/capture", dependencies=[api])
    def capture(b: CaptureBody):
        return guard(lambda: device.start_baseline_capture(b.windows))

    @app.post("/api/ingest/signal", dependencies=[api])
    def ingest_signal(b: SignalBody):
        """Live sensor input (M.3 'POST /ingest/window', generalised to any profile)."""
        given = [v is not None for v in (b.samples, b.axes, b.events)]
        if sum(given) != 1:
            raise HTTPException(422, "give exactly one of samples, axes, events")
        if b.events is not None:
            x = b.events
        else:
            x = np.asarray(b.samples if b.samples is not None else b.axes, dtype=np.float64)
            if x.size > MAX_SIGNAL_VALUES or x.size == 0 or not np.all(np.isfinite(x)):
                raise HTTPException(422, f"signal must be 1..{MAX_SIGNAL_VALUES} finite values")
            if b.axes is not None and (x.ndim != 2 or len({len(r) for r in b.axes}) != 1):
                raise HTTPException(422, "axes must be a rectangular list of rows")
        if device.gate is None and device.capture_state() is None:
            raise HTTPException(409, "no baseline yet: POST /api/baseline/capture first (machine known-good)")
        res = guard(lambda: device.ingest_signal(x, b.fs, b.rpm, b.source or "live-sensor", b.operating_point))
        states: dict[str, int] = {}
        for r in res:
            states[r["state"]] = states.get(r["state"], 0) + 1
        return {"windows": len(res), "states": states, "last": res[-1] if res else None,
                "capture": device.capture_state(), "diagnosis": device.last_diagnosis}

    @app.post("/api/episodes/{eid}/normal", dependencies=[api])
    def mark_normal(eid: str, b: Versioned):
        return guard(lambda: device.mark_normal(eid, b.expected_version))

    @app.post("/api/search/feedback", dependencies=[api])
    def feedback(b: FeedbackBody):
        return guard(lambda: device.record_feedback(b.result_id, b.kind, b.helped))

    @app.post("/api/retention/run", dependencies=[api])
    def retention():
        return device.run_retention()

    @app.post("/api/brief", dependencies=[api])
    def brief(b: BriefBody):
        res = guard(lambda: device.search(b.text, b.episode_id, b.use_fleet, b.limit))
        return rag.brief(b.question, res, llm) | {"retrieval": res}

    @app.get("/api/mirror", dependencies=[api])
    def mirror():
        return [r.payload for r in device.mirror.scroll()]

    @app.get("/api/sync/status", dependencies=[api])
    def sync_status():
        return worker.status()

    @app.post("/api/sync/now", dependencies=[api])
    def sync_now():
        return {"push": worker.push_once(), "pull": worker.pull_once()}

    @app.post("/api/network", dependencies=[api])
    def network(b: NetBody):
        worker.set_online(b.online)
        return worker.status()

    @app.post("/api/sync/token", dependencies=[api])
    def set_token(b: TokenBody):
        worker.set_token(b.token)
        return worker.status()

    @app.post("/api/outbox/{event_id}/resend", dependencies=[api])
    def resend(event_id: str):
        device.outbox.requeue(event_id)
        device.outbox.log("sync", f"event {event_id[:8]} re-queued by operator (idempotency check)")
        return {"requeued": event_id}

    @app.get("/api/outbox", dependencies=[api])
    def outbox():
        return device.outbox.rows(200)

    @app.get("/api/activity", dependencies=[api])
    def activity(limit: int = 100):
        return device.outbox.activity(min(limit, 500))

    @app.get("/api/replay/catalogue", dependencies=[api])
    def catalogue():
        return Recordings.catalogue()

    @app.post("/api/replay", dependencies=[api])
    def play(b: ReplayBody):
        if device.gate is None:
            raise HTTPException(409, "fit the healthy baseline first")
        return guard(lambda: replay.play(b.fid, b.n, b.interval, b.start))

    @app.post("/api/replay/stop", dependencies=[api])
    def stop():
        replay.stop()
        return replay.state

    if UI.exists():
        app.mount("/static", StaticFiles(directory=UI), name="static")

        @app.get("/")
        def index():
            return FileResponse(UI / "index.html")

        @app.get("/sw.js")
        def service_worker():
            """At the site root so it may control the whole app (a worker under /static/ could only see /static/)."""
            return FileResponse(UI / "sw.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"})

        @app.get("/sensor")
        def sensor_page():
            """Phone sensor page: streams the phone's accelerometer to /api/ingest/signal (needs HTTPS on phones)."""
            return FileResponse(UI / "sensor.html")

    return app


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan (replaces the deprecated on_event hook): stop background threads on shutdown."""
    yield
    for k in ("replay", "worker"):
        obj = getattr(app.state, k, None)
        if obj is not None:
            obj.stop()
