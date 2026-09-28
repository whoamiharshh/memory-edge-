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

needs_cwru = pytest.mark.skipif(not CACHE.exists(), reason="CWRU feature cache missing (run data/fetch_data.py, then "
                                "python -c \"from data.splits import build_dataset; build_dataset()\")")
from tools import qdrant_local

QDRANT_EXE = qdrant_local.binary()          # qdrant.exe on Windows, qdrant elsewhere
needs_qdrant_server = pytest.mark.skipif(not QDRANT_EXE.exists(),
                                         reason="Qdrant Server binary missing (python -m tools.qdrant_local download)")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def qdrant_url(tmp_path_factory):
    if not QDRANT_EXE.exists():
        pytest.skip("Qdrant Server binary missing (python -m tools.qdrant_local download)")
    with qdrant_local.server(tmp_path_factory.mktemp("qdrant")) as url:
        yield url


@pytest.fixture
def server_cloud(qdrant_url):
    """Like `cloud`, but backed by the real Qdrant Server. Each test gets its own tenant (own collections)."""
    store = CloudStore(url=qdrant_url)
    reg = TokenRegistry()
    tenant = "t" + uuid.uuid4().hex[:10]
    client = TestClient(create_app(store, reg, HashEmbedder()))
    return {"store": store, "registry": reg, "client": client, "tenant": tenant,
            "admin": reg.issue("admin1", "hq", tenant, role="admin"),
            "admin2": reg.issue("admin2", "hq", tenant, role="admin")}


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
    admin2 = reg.issue("admin2", "hq", "acme", role="admin")               # retraction needs two different admins
    return {"store": store, "registry": reg, "client": client, "admin": admin, "admin2": admin2}


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
