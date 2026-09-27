"""Shared fixtures: an in-process cloud (Qdrant in-memory + FastAPI TestClient) and devices wired to it."""
import pathlib

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
