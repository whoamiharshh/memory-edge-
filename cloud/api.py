"""Cloud Sync API (FastAPI).

  POST /v1/sync/push                 device  batch of ShareEvents -> per-event accepted|duplicate|rejected
  GET  /v1/mirror/cases?since=&limit device  case groups with seq > since (scroll-based fleet mirror pull)
  GET  /v1/cases, /v1/cases/{id}     device|admin  fleet evidence (admin also sees the events)
  GET  /v1/disputes                  device|admin  cases carrying DISPUTED / COMPETING flags
  POST /v1/events/{id}/retract       admin   tombstone one piece of evidence (never hard-deleted)
  POST /v1/admin/devices             admin   issue a device token      POST /v1/admin/devices/{id}/revoke
  GET  /v1/health                    public  GET /  fleet UI
Tenant always comes from the token.
"""
from __future__ import annotations

import pathlib

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cloud import ingest
from cloud.auth import AuthContext, TokenRegistry
from cloud.store_server import CloudStore
from shared.embed import Embedder
from shared.schema import PushRequest

MAX_BODY = 512 * 1024
UI = pathlib.Path(__file__).resolve().parent / "ui"


class RetractBody(BaseModel):
    reason: str = Field(min_length=3, max_length=200)


class IssueBody(BaseModel):
    device_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    site_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    role: str = Field(default="device", pattern=r"^(device|admin)$")


def create_app(store: CloudStore, registry: TokenRegistry, embedder: Embedder) -> FastAPI:
    app = FastAPI(title="Machine Memory - Fleet Cloud", docs_url="/docs")
    app.state.store, app.state.registry = store, registry

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_BODY:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    def auth(request: Request) -> AuthContext:
        h = request.headers.get("authorization", "")
        ctx = registry.verify(h[7:] if h.lower().startswith("bearer ") else None)
        if ctx is None:
            raise HTTPException(401, "missing, invalid or revoked token")
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
    def push(req: PushRequest, ctx: AuthContext = Depends(auth)):
        return ingest.push(store, embedder, ctx, req)

    @app.get("/v1/mirror/cases")
    def mirror(since: int = 0, limit: int = 200, ctx: AuthContext = Depends(auth)):
        limit = max(1, min(limit, 500))
        rows = store.cases(ctx.tenant_id, since=since, limit=limit + 1, with_vectors=True)
        items = [{"id": r["case_id"], "payload": {k: v for k, v in r.items() if k != "_vectors"},
                  "vib": r["_vectors"]["vib"], "note": r["_vectors"]["note"], "text": r["text"]} for r in rows[:limit]]
        return {"items": items, "more": len(rows) > limit}

    @app.get("/v1/cases")
    def cases(ctx: AuthContext = Depends(auth)):
        return sorted(store.cases(ctx.tenant_id, since=0, limit=1000), key=lambda c: c.get("updated_at", ""), reverse=True)

    @app.get("/v1/cases/{case_id}")
    def case(case_id: str, ctx: AuthContext = Depends(auth)):
        c = store.get_case(ctx.tenant_id, case_id)
        if c is None:
            raise HTTPException(404, "no such case in your tenant")
        if ctx.role == "admin":
            c["events"] = store.events(ctx.tenant_id, case_id)
        return c

    @app.get("/v1/disputes")
    def disputes(ctx: AuthContext = Depends(auth)):
        return [c for c in store.cases(ctx.tenant_id, since=0, limit=1000)
                if any(f["kind"] in ("DISPUTED", "COMPETING") for f in c.get("flags", []))]

    @app.post("/v1/events/{event_id}/retract")
    def retract(event_id: str, body: RetractBody, ctx: AuthContext = Depends(admin)):
        try:
            return ingest.retract(store, embedder, ctx, event_id, body.reason)
        except KeyError:
            raise HTTPException(404, "no such event in your tenant")

    @app.post("/v1/admin/devices")
    def issue(body: IssueBody, ctx: AuthContext = Depends(admin)):
        token = registry.issue(body.device_id, body.site_id, ctx.tenant_id, body.role)
        return {"device_id": body.device_id, "token": token, "note": "shown once; only its hash is stored"}

    @app.post("/v1/admin/devices/{device_id}/revoke")
    def revoke(device_id: str, ctx: AuthContext = Depends(admin)):
        return {"revoked": registry.revoke(device_id)}

    @app.get("/v1/admin/devices")
    def devices(ctx: AuthContext = Depends(admin)):
        return [{k: v for k, v in d.items()} for d in registry.devices(ctx.tenant_id)]

    @app.get("/v1/whoami")
    def whoami(ctx: AuthContext = Depends(auth)):
        return ctx.__dict__

    if UI.exists():
        app.mount("/static", StaticFiles(directory=UI), name="static")

        @app.get("/")
        def index():
            return FileResponse(UI / "index.html")

    return app
