"""Unified gateway: one clean URL for Machine Memory.

Proxies API calls to the edge device and cloud, injecting auth tokens
automatically so the user never has to deal with them.

  .venv\\Scripts\\python.exe -m app.unified --port 9000
  Then open http://127.0.0.1:9000
"""
from __future__ import annotations

import argparse
import json
import pathlib

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def _read_tokens() -> dict:
    boot = ROOT / "runtime" / "cloud" / "bootstrap.json"
    if boot.exists():
        b = json.loads(boot.read_text())
        return {"admin": b.get("admin", ""), "device": (b.get("devices", {}).get("devA", {}) or {}).get("token", "")}
    return {"admin": "", "device": ""}


def create_app(cloud_port: int = 8100, device_port: int = 8101,
               operator_token: str = "operator-devA") -> FastAPI:
    app = FastAPI(title="Machine Memory", docs_url=None, redoc_url=None)

    cloud_base = f"http://127.0.0.1:{cloud_port}"
    device_base = f"http://127.0.0.1:{device_port}"

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "index.html").read_text(encoding="utf-8")

    @app.get("/app.js")
    def app_js():
        return HTMLResponse((HERE / "app.js").read_text(encoding="utf-8"), media_type="application/javascript")

    @app.get("/style.css")
    def app_css():
        return HTMLResponse((HERE / "style.css").read_text(encoding="utf-8"), media_type="text/css")

    # ── proxy: device API ──
    @app.api_route("/proxy/device/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_device(path: str, request: Request):
        return await _proxy(f"{device_base}/api/{path}", request,
                            {"X-Operator-Token": operator_token})

    # ── proxy: cloud API ──
    @app.api_route("/proxy/cloud/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_cloud(path: str, request: Request):
        tokens = _read_tokens()
        return await _proxy(f"{cloud_base}/v1/{path}", request,
                            {"Authorization": f"Bearer {tokens['admin']}"})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


async def _proxy(url: str, request: Request, extra_headers: dict) -> JSONResponse:
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "connection", "content-length", "transfer-encoding")}
    headers.update(extra_headers)
    qs = str(request.url.query)
    if qs:
        url = f"{url}?{qs}"
    # Generous, because the slowest call behind this proxy is the first Ask after a start: it loads a
    # 1.1 GB model from disk and may also be waiting on a web search. 30s timed that out and the browser
    # saw a bare 500. Everything here is a loopback call to our own process, so a long read is harmless.
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
        try:
            resp = await client.request(request.method, url, content=body if body else None, headers=headers)
        except httpx.ConnectError:
            return JSONResponse({"detail": "backend service not reachable"}, status_code=502)
    ct = resp.headers.get("content-type", "")
    if "json" in ct:
        return JSONResponse(resp.json(), status_code=resp.status_code)
    # anything else passes through as-is: photos come back as image bytes, and wrapping those in JSON
    # left every <img> in the UI broken
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=ct or "application/octet-stream")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
