"""Run the fleet cloud (Sync API + fleet UI) against Qdrant Server.

  .venv\\Scripts\\python.exe -m cloud.main --qdrant-url http://127.0.0.1:6333 --port 8100 --bootstrap
--bootstrap issues two admin tokens (retraction needs two different admins) and device tokens (devA/site1,
devB/site2, devC/site3) once and writes them to runtime/cloud/bootstrap.json (local demo secrets; the registry itself
stores only hashes). Admin actions are appended to runtime/cloud/audit.log (hash-chained).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import uvicorn

from cloud.api import create_app
from cloud.audit import AuditLog
from cloud.auth import TokenRegistry
from cloud.tls import cert_bound_protocol, server_context
from cloud.store_server import CloudStore
from shared.embed import HashEmbedder, load_embedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime" / "cloud"


def bootstrap(reg: TokenRegistry, tenant: str, runtime: pathlib.Path = RUNTIME) -> dict:
    path = runtime / "bootstrap.json"
    if path.exists():
        b = json.loads(path.read_text())
        if "admin2" not in b:                          # bootstrap files from before the two-person rule
            b["admin2"] = reg.issue("admin2", "hq", tenant, role="admin")
            path.write_text(json.dumps(b, indent=2))
        return b
    out = {"tenant": tenant, "admin": reg.issue("admin", "hq", tenant, role="admin"),
           "admin2": reg.issue("admin2", "hq", tenant, role="admin"), "devices": {}}
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
    p.add_argument("--mtls", action="store_true", help="mutual TLS: only devices with a CA-signed client certificate may "
                   "connect (tools/make_certs.py device <id>); implies --tls; revoked certificates (tools/make_certs.py "
                   "revoke <id> -> runtime/tls/crl.pem) are refused")
    p.add_argument("--single-admin-retract", action="store_true",
                   help="one admin may retract evidence alone (default: two different admins, the two-person rule)")
    p.add_argument("--runtime-dir", default=str(RUNTIME), help="token registry, bootstrap and audit log folder")
    a = p.parse_args()
    runtime = pathlib.Path(a.runtime_dir)
    a.tls = a.tls or a.mtls
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.tls and not a.insecure_lan:
        raise SystemExit("refusing to serve plain HTTP on the network: add --tls (run tools/make_certs.py first); "
                         "device tokens would otherwise cross the network unencrypted")
    reg = TokenRegistry(runtime / "tokens.json")
    if a.bootstrap:
        b = bootstrap(reg, a.tenant, runtime)
        print(f"[cloud] admin token: {b['admin']}  (all tokens: {runtime / 'bootstrap.json'})", flush=True)
    store = CloudStore(location=":memory:") if a.memory else CloudStore(url=a.qdrant_url)
    embedder = HashEmbedder() if a.hash_embedder else load_embedder()
    scheme = "https" if a.tls else "http"
    print(f"[cloud] fleet UI: {scheme}://{a.host}:{a.port}/   Qdrant: {store.backend}", flush=True)
    tls = ROOT / "runtime" / "tls"
    app = create_app(store, reg, embedder, AuditLog(runtime / "audit.log"), 1 if a.single_admin_retract else 2,
                     coalesce=True, bind_cert=a.mtls)
    # our own TLS context (uvicorn's ssl_context_factory hook): TLS >= 1.2; with mTLS, client certificates + CRL
    # (reloaded when it changes) and the certificate's name handed to the app (it must match the token's device)
    factory = (lambda _cfg, _default: server_context(tls / "server.pem", tls / "server.key",
                                                     tls / "ca.pem" if a.mtls else None,
                                                     tls / "crl.pem" if a.mtls else None)) if a.tls else None
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", ssl_context_factory=factory,
                **({"http": cert_bound_protocol()} if a.mtls else {}))

if __name__ == "__main__":
    main()
