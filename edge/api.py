"""Edge-local API + UI for one device (FastAPI). Binds to localhost by default.

Every /api route requires the operator token (header X-Operator-Token): the device holds raw technician
notes, so even local reads are authenticated. The token is compared in constant time against its sha256.
"""
from __future__ import annotations

import hashlib
import hmac
import pathlib

import numpy as np

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from edge import procedures as procedures_mod
from edge import rag
from edge.device import Device
from edge.replay import HEALTHY_BASELINE_FILES, Recordings, ReplayRunner
from edge.sync_worker import SyncWorker
from shared.schema import ActionCode, FaultClass, RootCause

UI = pathlib.Path(__file__).resolve().parent / "ui"


class Versioned(BaseModel):
    expected_version: int | None = Field(default=None, ge=1)   # the version the screen showed; 409 if it moved


class NoteBody(Versioned):
    text: str = Field(max_length=2000)
    share_opt_in: bool = False


class FaultBody(Versioned):
    fault_class: FaultClass


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
    app = FastAPI(title=f"Machine Memory - {device.cfg.device_id}", docs_url="/docs")
    def _replay_physics(result: dict, fid: int, i: int) -> None:
        """Replayed recordings carry cached fingerprints; the physics panel needs the raw window, read on demand."""
        if device.profile.name != "bearing-12k" or not result.get("episode_id"):
            return
        ep = device.store.get(result["episode_id"])
        if ep is None or ep.payload.get("physics") and ep.payload["occurrences"] > 3:
            return                                   # enough: only the first windows of an episode are diagnosed
        rw = Recordings.raw_window(fid, i)
        if rw is not None:
            device.attach_diagnosis(result["episode_id"], *rw)

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
                "root_causes": [r.value for r in RootCause]}

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

    @app.post("/api/episodes/{eid}/fault_class", dependencies=[api])
    def fault(eid: str, b: FaultBody):
        return guard(lambda: device.set_fault_class(eid, b.fault_class.value, b.expected_version))

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
        return {"fault_class": fault_class, "component": component,
                "procedures": procedures_mod.lookup(fault_class, component)}

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
        res = guard(lambda: device.ingest_signal(x, b.fs, b.rpm, b.source or "live-sensor"))
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

        @app.get("/sensor")
        def sensor_page():
            """Phone sensor page: streams the phone's accelerometer to /api/ingest/signal (needs HTTPS on phones)."""
            return FileResponse(UI / "sensor.html")

    @app.on_event("shutdown")
    def _shutdown():
        replay.stop()
        worker.stop()

    return app
