"""Run the fleet cloud (Sync API + fleet UI) against Qdrant Server.

  .venv\\Scripts\\python.exe -m cloud.main --qdrant-url http://127.0.0.1:6333 --port 8100 --bootstrap
--bootstrap issues an admin token and device tokens (devA/site1, devB/site2, devC/site3) once and writes them to
runtime/cloud/bootstrap.json (local demo secrets; the registry itself stores only hashes).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import uvicorn

from cloud.api import create_app
from cloud.auth import TokenRegistry
from cloud.store_server import CloudStore
from shared.embed import HashEmbedder, load_embedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime" / "cloud"


def bootstrap(reg: TokenRegistry, tenant: str) -> dict:
    path = RUNTIME / "bootstrap.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {"tenant": tenant, "admin": reg.issue("admin", "hq", tenant, role="admin"), "devices": {}}
    for dev, site in (("devA", "site1"), ("devB", "site2"), ("devC", "site3")):
        out["devices"][dev] = {"site": site, "token": reg.issue(dev, site, tenant)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--memory", action="store_true", help="in-process Qdrant (no server) - tests/dev only")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--tenant", default="acme")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--hash-embedder", action="store_true")
    p.add_argument("--tls", action="store_true", help="serve HTTPS with runtime/tls/server.pem (tools/make_certs.py)")
    p.add_argument("--insecure-lan", action="store_true", help="allow plain HTTP on a non-localhost address (NOT advised)")
    a = p.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.tls and not a.insecure_lan:
        raise SystemExit("refusing to serve plain HTTP on the network: add --tls (run tools/make_certs.py first); "
                         "device tokens would otherwise cross the network unencrypted")
    reg = TokenRegistry(RUNTIME / "tokens.json")
    if a.bootstrap:
        b = bootstrap(reg, a.tenant)
        print(f"[cloud] admin token: {b['admin']}  (all tokens: {RUNTIME / 'bootstrap.json'})", flush=True)
    store = CloudStore(location=":memory:") if a.memory else CloudStore(url=a.qdrant_url)
    embedder = HashEmbedder() if a.hash_embedder else load_embedder()
    scheme = "https" if a.tls else "http"
    print(f"[cloud] fleet UI: {scheme}://{a.host}:{a.port}/   Qdrant: {store.backend}", flush=True)
    tls = ROOT / "runtime" / "tls"
    ssl = {"ssl_certfile": str(tls / "server.pem"), "ssl_keyfile": str(tls / "server.key")} if a.tls else {}
    uvicorn.run(create_app(store, reg, embedder), host=a.host, port=a.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
