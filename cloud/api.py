"""Cloud Sync API (FastAPI).

  POST /v1/sync/push                 device  batch of ShareEvents -> per-event accepted|duplicate|rejected
  GET  /v1/mirror/head?since=        device  {seq, cases, snapshots[, changed]}: newest change, snapshot support
  GET  /v1/mirror/snapshot           device  full Qdrant shard snapshot of the tenant's mirror (gzip stream)
  POST /v1/mirror/snapshot/partial   device  body = the device's snapshot_manifest -> partial snapshot (gzip), or 304
  GET  /v1/mirror/cases?since=&limit device  case groups with seq > since (scroll fallback, kill test K5)
  GET  /v1/cases, /v1/cases/{id}     device|admin  fleet evidence (admin also sees the events)
  GET  /v1/disputes                  device|admin  cases carrying DISPUTED / COMPETING / RECURRED flags
  GET  /v1/fleet/hint-model          device|admin  fleet-learned fault hint (plain JSON coefficients + its
                                                   leave-one-device-out accuracy), trained on confirmed cases
  POST /v1/events/{id}/retract       admin   request a tombstone; a SECOND, different admin's call carries it out
  POST /v1/admin/devices             admin   issue a device token      POST /v1/admin/devices/{id}/revoke
  POST /v1/admin/devices/{id}/quarantine | /unquarantine   admin   pull / restore ALL evidence of one device
  GET  /v1/admin/devices             admin   tokens + per-device evidence counts + code-integrity status
  GET  /v1/admin/audit               admin   hash-chained audit log of admin actions + chain check
  GET  /v1/health                    public  GET /  fleet UI
Tenant always comes from the token.
"""
from __future__ import annotations

import collections
import json
import pathlib
import time
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cloud import hint_model, ingest, knowledge
from cloud.audit import AuditLog
from cloud.recompute import Recomputer
from shared.integrity import code_hash
from cloud.auth import AuthContext, TokenRegistry
from cloud.store_server import CloudStore, SnapshotError
from shared.embed import Embedder
from shared.schema import PushRequest

MAX_BODY = 512 * 1024
HINT_RETRAIN_S = 30.0
UI = pathlib.Path(__file__).resolve().parent / "ui"


class RetractBody(BaseModel):
    reason: str = Field(min_length=3, max_length=200)


class IssueBody(BaseModel):
    device_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    site_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    role: str = Field(default="device", pattern=r"^(device|admin)$")


class QuarantineBody(BaseModel):
    reason: str = Field(min_length=3, max_length=200)


class KnowledgeBody(BaseModel):
    """Text a person deliberately publishes for other devices. Carries no sensor claim."""
    text: str = Field(min_length=1, max_length=knowledge.MAX_TEXT)
    topic: str = Field(default="general", max_length=80)
    audience: str = Field(default="everyone", pattern=r"^(everyone|device|site)$")
    recipients: list[str] = Field(default_factory=list, max_length=knowledge.MAX_RECIPIENTS)
    record_id: str | None = Field(default=None, min_length=36, max_length=36)


_FRAME_ANCESTORS = "http://127.0.0.1:9000 http://127.0.0.1:8000"
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                               "img-src 'self' data:; connect-src 'self'; "
                               f"frame-ancestors 'self' {_FRAME_ANCESTORS}; base-uri 'none'; "
                               "form-action 'self'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), geolocation=(), microphone=(), accelerometer=()",
}


def create_app(store: CloudStore, registry: TokenRegistry, embedder: Embedder, audit: AuditLog | None = None,
               retract_approvals: int = 2, coalesce: bool = False, bind_cert: bool = False) -> FastAPI:
    """coalesce=True (the production launcher): pushes return once events are durable and case tallies are recomputed
    by one background worker (cloud/recompute.py); every read flushes its tenant's pending cases first.
    bind_cert=True (mutual TLS): the client certificate's name must equal the token's device id (cloud/tls.py), so a
    token copied from one device cannot be used through another device's certificate."""
    app = FastAPI(title="Machine Memory - Fleet Cloud", docs_url="/docs")
    audit = audit or AuditLog()
    app.state.store, app.state.registry, app.state.audit = store, registry, audit
    code_seen: dict[str, dict] = {}
    rc = Recomputer(store, embedder) if coalesce else None
    app.state.recomputer = rc
    fresh = (lambda tenant: rc.flush(tenant)) if rc else (lambda tenant: 0)
    release = code_hash()

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and (not cl.isdigit() or int(cl) > MAX_BODY):
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        resp = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        if request.url.path.startswith("/v1/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    def auth(request: Request) -> AuthContext:
        h = request.headers.get("authorization", "")
        ctx = registry.verify(h[7:] if h.lower().startswith("bearer ") else None)
        if ctx is None:
            raise HTTPException(401, "missing, invalid or revoked token")
        if bind_cert:
            cn = request.scope.get("state", {}).get("tls_client_cn")
            if cn != ctx.device_id:
                raise HTTPException(403, f"client certificate '{cn}' does not belong to this token's device")
        if not registry.allow(ctx):
            raise HTTPException(429, "rate limit exceeded")
        return ctx

    def admin(ctx: AuthContext = Depends(auth)) -> AuthContext:
        if ctx.role != "admin":
            raise HTTPException(403, "admin role required")
        return ctx

    @app.get("/v1/health")
    def health():
        return {"ok": True, "backend": store.backend}

    @app.post("/v1/sync/push")
    def push(req: PushRequest, request: Request, ctx: AuthContext = Depends(auth)):
        h = request.headers.get("x-code-hash", "")[:64]
        dev_time = request.headers.get("x-device-time", "")[:40] or None
        off = ingest.clock_offset(dev_time)
        code_seen[ctx.device_id] = {"code_hash": h or None, "at": ingest._now(),
                                    "clock_offset_s": None if off is None else round(off, 1)}
        return ingest.push(store, embedder, ctx, req, defer=rc.mark if rc else None, device_time=dev_time)

    @app.get("/v1/mirror/head")
    def mirror_head(since: int | None = None, ctx: AuthContext = Depends(auth)):
        # mirror reads do NOT force a recompute: the background worker keeps the mirror within ~0.5 s of the events
        # (a device mirror is eventually consistent anyway); forcing it here cost 27 % of the cloud's time under load
        return store.mirror_head(ctx.tenant_id, since) | {"text_model": embedder.name, "server_time": ingest._now()}   # model of the mirror's text vectors

    def _snapshot_response(tenant: str, manifest: dict | None):
        if not store.supports_snapshots:
            raise HTTPException(501, "this cloud has no Qdrant Server behind it; use /v1/mirror/cases")
        try:
            status, stream = store.open_mirror_snapshot(tenant, manifest)
        except SnapshotError as e:
            raise HTTPException(502, str(e)[:300])
        if status == 304:
            return Response(status_code=304)
        return StreamingResponse(stream, media_type="application/gzip")

    @app.get("/v1/mirror/snapshot")
    def mirror_snapshot(ctx: AuthContext = Depends(auth)):
        return _snapshot_response(ctx.tenant_id, None)

    @app.post("/v1/mirror/snapshot/partial")
    def mirror_partial(manifest: dict[str, Any] = Body(...), ctx: AuthContext = Depends(auth)):
        if not manifest or not all(isinstance(v, dict) for v in manifest.values()):
            raise HTTPException(422, "body must be an Edge snapshot manifest (segment id -> segment manifest)")
        return _snapshot_response(ctx.tenant_id, manifest)

    @app.get("/v1/mirror/cases")
    def mirror(since: int = 0, limit: int = 200, ctx: AuthContext = Depends(auth)):
        limit = max(1, min(limit, 500))
        # every device asks for the same pages: encode each page once per mirror version and serve the bytes
        key = (ctx.tenant_id, since, limit, store.mirror_version.get(ctx.tenant_id, 0))
        body = page_cache.get(key)
        if body is None:
            rows = store.cases(ctx.tenant_id, since=since, limit=limit + 1, with_vectors=True)
            items = [{"id": r["case_id"], "payload": {k: v for k, v in r.items() if k != "_vectors"},
                      "vib": r["_vectors"]["vib"], "note": r["_vectors"]["note"], "text": r["text"]} for r in rows[:limit]]
            body = json.dumps({"items": items, "more": len(rows) > limit}, separators=(",", ":")).encode()
            if len(page_cache) > 512:
                page_cache.clear()
            page_cache[key] = body
        return Response(content=body, media_type="application/json")

    @app.get("/v1/cases")
    def cases(ctx: AuthContext = Depends(auth)):
        fresh(ctx.tenant_id)
        return sorted(store.cases(ctx.tenant_id, since=0, limit=1000), key=lambda c: c.get("updated_at", ""), reverse=True)

    @app.get("/v1/cases/{case_id}")
    def case(case_id: str, ctx: AuthContext = Depends(auth)):
        fresh(ctx.tenant_id)
        c = store.get_case(ctx.tenant_id, case_id)
        if c is None:
            raise HTTPException(404, "no such case in your tenant")
        if ctx.role == "admin":
            c["events"] = store.events(ctx.tenant_id, case_id)
        return c

    @app.get("/v1/disputes")
    def disputes(ctx: AuthContext = Depends(auth)):
        fresh(ctx.tenant_id)
        return [c for c in store.cases(ctx.tenant_id, since=0, limit=1000)
                if any(f["kind"] in ("DISPUTED", "COMPETING", "RECURRED") for f in c.get("flags", []))]

    hint_cache: dict[str, tuple[int, dict, float]] = {}
    page_cache: dict[tuple, bytes] = {}

    @app.get("/v1/fleet/hint-model")
    def fleet_hint_model(ctx: AuthContext = Depends(auth)):
        """Retrained only when the tenant's evidence changed, and at most every HINT_RETRAIN_S under a push storm:
        it reads every event, and doing that for each device's pull dominated the cloud at 5,000 devices."""
        ver = store.events_version.get(ctx.tenant_id, 0)
        hit = hint_cache.get(ctx.tenant_id)
        now = time.monotonic()
        if hit is None or (hit[0] != ver and now - hit[2] >= HINT_RETRAIN_S):
            hit = (ver, hint_model.train(store.events(ctx.tenant_id)), now)
            hint_cache[ctx.tenant_id] = hit
        return hit[1]

    @app.post("/v1/events/{event_id}/retract")
    def retract(event_id: str, body: RetractBody, ctx: AuthContext = Depends(admin)):
        try:
            ev = ingest.retract(store, embedder, ctx, event_id, body.reason, retract_approvals)
        except KeyError:
            raise HTTPException(404, "no such event in your tenant")
        except ingest.SameApprover as e:
            raise HTTPException(409, str(e))
        done = ev.get("status") == "retracted"
        audit.append(ctx.device_id, ctx.tenant_id, "retract_approved" if done else "retract_requested",
                     event_id=event_id, reason=body.reason)
        return ev | {"retraction": "done" if done else "pending: a second admin must approve"}

    @app.post("/v1/admin/devices")
    def issue(body: IssueBody, ctx: AuthContext = Depends(admin)):
        token = registry.issue(body.device_id, body.site_id, ctx.tenant_id, body.role)
        audit.append(ctx.device_id, ctx.tenant_id, "token_issued", device_id=body.device_id, role=body.role)
        return {"device_id": body.device_id, "token": token, "note": "shown once; only its hash is stored"}

    @app.post("/v1/admin/devices/{device_id}/revoke")
    def revoke(device_id: str, ctx: AuthContext = Depends(admin)):
        n = registry.revoke(device_id)
        audit.append(ctx.device_id, ctx.tenant_id, "token_revoked", device_id=device_id, tokens=n)
        return {"revoked": n}

    @app.post("/v1/admin/devices/{device_id}/quarantine")
    def quarantine(device_id: str, body: QuarantineBody, ctx: AuthContext = Depends(admin)):
        out = ingest.quarantine(store, embedder, ctx.tenant_id, device_id, True, body.reason)
        out["tokens_revoked"] = registry.revoke(device_id)
        audit.append(ctx.device_id, ctx.tenant_id, "device_quarantined", device_id=device_id, reason=body.reason,
                     events=out["quarantined"])
        return out

    @app.post("/v1/admin/devices/{device_id}/unquarantine")
    def unquarantine(device_id: str, body: QuarantineBody, ctx: AuthContext = Depends(admin)):
        out = ingest.quarantine(store, embedder, ctx.tenant_id, device_id, False)
        audit.append(ctx.device_id, ctx.tenant_id, "device_restored", device_id=device_id, reason=body.reason,
                     events=out["restored"], note="its old tokens stay revoked: issue a new one")
        return out

    @app.get("/v1/admin/devices")
    def devices(ctx: AuthContext = Depends(admin)):
        """Tokens plus, per device, COUNTS of its evidence (never a trust score) and whether its code matches this
        release (shared/integrity.py: tamper evidence, not attestation)."""
        counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for e in store.events(ctx.tenant_id):
            c = counts[e.get("device_id")]
            c[e.get("status", "active")] += 1
            c[e.get("outcome", "?")] += 1
            fu = (e.get("followup") or {}).get("status")
            if fu:
                c[fu] += 1
        out = []
        for d in registry.devices(ctx.tenant_id):
            seen = code_seen.get(d["device_id"])
            out.append({k: v for k, v in d.items()} | {
                "evidence": dict(counts.get(d["device_id"], {})),
                "code": None if not seen else ("matches this release" if seen["code_hash"] == release else "DIFFERS"),
                "clock_offset_s": None if not seen else seen.get("clock_offset_s"),
                "code_seen": seen})
        return out

    @app.get("/v1/admin/audit")
    def audit_log(limit: int = 200, ctx: AuthContext = Depends(admin)):
        return {"chain": audit.verify(),
                "entries": [e for e in audit.entries() if e["tenant"] == ctx.tenant_id][-max(1, min(limit, 5000)):]}

    @app.post("/v1/token/renew")
    def renew(request: Request, ctx: AuthContext = Depends(auth)):
        """The calling device swaps its valid token for a fresh one (the old one lives 10 more minutes at most)."""
        h = request.headers.get("authorization", "")
        out = registry.renew(h[7:])
        if out is None:
            raise HTTPException(401, "missing, invalid, expired or revoked token")
        return {"token": out[0], "expires_at": out[1], "note": "shown once; only its hash is stored"}

    @app.get("/v1/whoami")
    def whoami(ctx: AuthContext = Depends(auth)):
        return ctx.__dict__

    # ---- shared free-text knowledge (cloud/knowledge.py): a second channel, kept out of the case groups ----
    @app.post("/v1/knowledge")
    def publish_knowledge(b: KnowledgeBody, ctx: AuthContext = Depends(auth)):
        try:
            return knowledge.publish(store, embedder, ctx.tenant_id, ctx.device_id, text=b.text,
                                     topic=b.topic, audience=b.audience, recipients=b.recipients,
                                     record_id=b.record_id)
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/v1/knowledge")
    def pull_knowledge(since: int = 0, limit: int = 200, ctx: AuthContext = Depends(auth)):
        """Only what this token may read: the audience filter runs in the query, so a device is never
        handed a record addressed to somebody else."""
        return knowledge.fetch(store, ctx.tenant_id, ctx.device_id, ctx.site_id, since, limit)

    @app.get("/v1/knowledge/mine")
    def my_knowledge(ctx: AuthContext = Depends(auth)):
        return knowledge.mine(store, ctx.tenant_id, ctx.device_id)

    @app.post("/v1/knowledge/{record_id}/withdraw")
    def withdraw_knowledge(record_id: str, ctx: AuthContext = Depends(auth)):
        if not knowledge.withdraw(store, ctx.tenant_id, record_id, ctx.device_id):
            raise HTTPException(404, "no such record published by this device")
        audit.append(ctx.device_id, ctx.tenant_id, "knowledge_withdrawn", record_id=record_id)
        return {"withdrawn": record_id,
                "note": "devices that already pulled it keep their copy"}

    if UI.exists():
        app.mount("/static", StaticFiles(directory=UI), name="static")

        @app.get("/")
        def index():
            return FileResponse(UI / "index.html")

    return app
