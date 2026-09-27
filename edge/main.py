"""Run one edge device (API + UI + sync worker).

  .venv\\Scripts\\python.exe -m edge.main --name devA --site site1 --port 8101 --cloud http://127.0.0.1:8100 \\
      --device-token <token> --operator-token <secret>
"""
from __future__ import annotations

import argparse
import pathlib
import secrets

import uvicorn

from edge import rag
from edge.api import create_app
from edge.device import Device, DeviceConfig
from edge.sync_worker import SyncWorker
from shared.embed import HashEmbedder, load_embedder

ROOT = pathlib.Path(__file__).resolve().parents[1]


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
    p.add_argument("--fit-baseline", action="store_true", help="fit the healthy baseline at start if missing")
    a = p.parse_args()

    root = pathlib.Path(a.root) if a.root else ROOT / "runtime" / a.name
    embedder = HashEmbedder() if a.hash_embedder else load_embedder()
    dev = Device(DeviceConfig(device_id=a.name, site_id=a.site, machine_id=a.machine or f"{a.name}-motor1", root=root,
                              denylist=[w.strip() for w in a.denylist.split(",") if w.strip()]), embedder)
    if a.fit_baseline and dev.gate is None:
        from edge.replay import Recordings
        dev.fit_baseline(Recordings.baseline())
    worker = SyncWorker(dev, a.cloud, a.device_token)
    if not a.no_sync:
        worker.start()
    op = a.operator_token or secrets.token_urlsafe(12)
    llm = None if a.no_llm else rag.LocalLLM()
    print(f"[{a.name}] UI: http://{a.host}:{a.port}/   operator token: {op}", flush=True)
    uvicorn.run(create_app(dev, worker, op, llm), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
