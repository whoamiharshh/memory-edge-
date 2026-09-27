"""Run one edge device (API + UI + sync worker).

  .venv\\Scripts\\python.exe -m edge.main --name devA --site site1 --port 8101 --cloud http://127.0.0.1:8100 \\
      --device-token <token> --operator-token <secret>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import secrets

import uvicorn

from edge import profiles, rag, storage_os
from edge.api import create_app
from edge.device import Device, DeviceConfig
from edge.sync_worker import SyncWorker
from shared.embed import HashEmbedder, load_embedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
TLS = ROOT / "runtime" / "tls"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--machine", default=None)
    p.add_argument("--port", type=int, default=8101)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--cloud", default="http://127.0.0.1:8100")
    p.add_argument("--device-token", default=None)
    p.add_argument("--operator-token", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--denylist", default="", help="comma-separated staff/site names the redactor must catch")
    p.add_argument("--hash-embedder", action="store_true", help="use the lexical test embedder instead of bge-small")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--mirror-mode", choices=("auto", "snapshot", "scroll"), default="auto",
                   help="fleet mirror fill: auto (full Qdrant snapshot to bootstrap, then the cheaper of scroll delta "
                        "or snapshot; default), snapshot (Qdrant partial snapshots), scroll (fallback)")
    p.add_argument("--fit-baseline", action="store_true", help="fit the healthy baseline at start if missing")
    p.add_argument("--profile", default="bearing-12k", help="signal profile: " + ", ".join(sorted(profiles.REGISTRY)))
    p.add_argument("--profile-params", default="{}", help='JSON, e.g. {"shaft_hz": 17} or {"bearing": "6206"}')
    p.add_argument("--component", default=None, help="what this device watches (default: the profile's)")
    p.add_argument("--tls", action="store_true", help="serve HTTPS with runtime/tls/server.pem (tools/make_certs.py)")
    p.add_argument("--ca", default=None, help="CA file to verify an https cloud (default runtime/tls/ca.pem if it exists)")
    p.add_argument("--insecure-lan", action="store_true", help="allow plain HTTP on a non-localhost address (NOT advised)")
    p.add_argument("--client-cert", default=None, help="this device's mutual-TLS certificate (runtime/tls/devices/<id>.pem)")
    p.add_argument("--client-key", default=None, help="its private key (runtime/tls/devices/<id>.key)")
    p.add_argument("--compress-storage", action="store_true",
                   help="Windows: NTFS-compress the device folder (Edge pre-allocates ~200 MB of zero pages per shard)")
    a = p.parse_args()

    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.tls and not a.insecure_lan:
        raise SystemExit("refusing to serve plain HTTP on the network: add --tls (run tools/make_certs.py first); the "
                         "operator token and technician notes would otherwise cross the network unencrypted")
    root = pathlib.Path(a.root) if a.root else ROOT / "runtime" / a.name
    embedder = HashEmbedder() if a.hash_embedder else load_embedder()
    dev = Device(DeviceConfig(device_id=a.name, site_id=a.site, machine_id=a.machine or f"{a.name}-motor1", root=root,
                              denylist=[w.strip() for w in a.denylist.split(",") if w.strip()], component=a.component, compress_storage=a.compress_storage,
                              profile=a.profile, profile_params=json.loads(a.profile_params)), embedder)
    if a.fit_baseline and dev.gate is None and a.profile == "bearing-12k":
        from edge.replay import Recordings
        dev.fit_baseline(Recordings.baseline())
    ca = a.ca or (str(TLS / "ca.pem") if (TLS / "ca.pem").exists() else None)
    cert = (a.client_cert, a.client_key) if a.client_cert else None
    worker = SyncWorker(dev, a.cloud, a.device_token, mirror_mode=a.mirror_mode, ca=ca, client_cert=cert)
    if not a.no_sync:
        worker.start()
    op = a.operator_token or secrets.token_urlsafe(12)
    llm = None if a.no_llm else rag.LocalLLM()
    scheme = "https" if a.tls else "http"
    print(f"[{a.name}] UI: {scheme}://{a.host}:{a.port}/   phone sensor: {scheme}://<this-ip>:{a.port}/sensor   "
          f"profile: {dev.profile.name}   operator token: {op}", flush=True)
    ssl = {"ssl_certfile": str(TLS / "server.pem"), "ssl_keyfile": str(TLS / "server.key")} if a.tls else {}
    uvicorn.run(create_app(dev, worker, op, llm), host=a.host, port=a.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
