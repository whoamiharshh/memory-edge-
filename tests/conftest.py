"""Shared fixtures: an in-process cloud (Qdrant in-memory + FastAPI TestClient) and devices wired to it, plus a
REAL Qdrant Server (the release binary in qdrant_server/, started once per session on free ports) for the tests
that need what only a server has, such as shard snapshots."""
import os
import pathlib
import socket
import subprocess
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from cloud.api import create_app
from cloud.auth import TokenRegistry
from cloud.store_server import CloudStore
from data.splits import CACHE
from edge.device import Device, DeviceConfig
from edge.sync_worker import SyncWorker
from shared.embed import HashEmbedder

needs_cwru = pytest.mark.skipif(not CACHE.exists(), reason="CWRU feature cache missing (run data/fetch_data.py)")
QDRANT_EXE = pathlib.Path(__file__).resolve().parents[1] / "qdrant_server" / "qdrant.exe"
needs_qdrant_server = pytest.mark.skipif(not QDRANT_EXE.exists(), reason="qdrant_server/qdrant.exe missing (docs/SETUP.md)")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def qdrant_url(tmp_path_factory):
    if not QDRANT_EXE.exists():
        pytest.skip("qdrant_server/qdrant.exe missing")
    tmp = tmp_path_factory.mktemp("qdrant")
    http, grpc = _free_port(), _free_port()
    env = os.environ | {"QDRANT__STORAGE__STORAGE_PATH": str(tmp / "storage"),
                        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(tmp / "snapshots"),
                        "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
                        "QDRANT__TELEMETRY_DISABLED": "true"}
    proc = subprocess.Popen([str(QDRANT_EXE)], cwd=tmp, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{http}"
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{url}/readyz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        else:
            pytest.fail("Qdrant Server did not become ready")
        yield url
    finally:
        proc.kill()
        proc.wait()


@pytest.fixture
def server_cloud(qdrant_url):
    """Like `cloud`, but backed by the real Qdrant Server. Each test gets its own tenant (own collections)."""
    store = CloudStore(url=qdrant_url)
    reg = TokenRegistry()
    tenant = "t" + uuid.uuid4().hex[:10]
    client = TestClient(create_app(store, reg, HashEmbedder()))
    return {"store": store, "registry": reg, "client": client, "tenant": tenant,
            "admin": reg.issue("admin1", "hq", tenant, role="admin")}


@pytest.fixture
def make_server_device(tmp_path, server_cloud):
    made = []

    def _make(name: str, site: str, tenant: str | None = None, mirror_mode: str = "snapshot",
              **kw) -> tuple[Device, SyncWorker]:
        tok = server_cloud["registry"].issue(name, site, tenant or server_cloud["tenant"])
        dev = Device(DeviceConfig(device_id=name, site_id=site, machine_id=f"{name}-m1",
                                  root=pathlib.Path(tmp_path) / name, **kw), HashEmbedder())
        worker = SyncWorker(dev, None, tok, client=server_cloud["client"], mirror_mode=mirror_mode)
        made.append(dev)
        return dev, worker

    yield _make
    for d in made:
        d.close()


@pytest.fixture
def cloud():
    store = CloudStore(location=":memory:")
    reg = TokenRegistry()
    app = create_app(store, reg, HashEmbedder())
    client = TestClient(app)
    admin = reg.issue("admin1", "hq", "acme", role="admin")
    return {"store": store, "registry": reg, "client": client, "admin": admin}


@pytest.fixture
def make_device(tmp_path, cloud):
    made = []

    def _make(name: str, site: str, tenant: str = "acme", token: str | None = None, **kw) -> tuple[Device, SyncWorker]:
        tok = token or cloud["registry"].issue(name, site, tenant)
        dev = Device(DeviceConfig(device_id=name, site_id=site, machine_id=f"{name}-m1",
                                  root=pathlib.Path(tmp_path) / name, **kw), HashEmbedder())
        worker = SyncWorker(dev, None, tok, client=cloud["client"])
        made.append(dev)
        return dev, worker

    yield _make
    for d in made:
        d.close()
