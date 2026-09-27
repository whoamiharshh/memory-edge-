"""The installable web app: the service worker is served from the site root, the manifest is valid, and the worker
never caches /api responses (technician notes must not sit in a phone browser's cache)."""
import json
import pathlib
import re

from fastapi.testclient import TestClient

from edge.api import create_app
from edge.sync_worker import SyncWorker

UI = pathlib.Path(__file__).resolve().parents[2] / "edge" / "ui"


def test_service_worker_skips_api_and_non_get():
    sw = (UI / "sw.js").read_text(encoding="utf-8")
    assert re.search(r'url\.pathname\.startsWith\("/api/"\)\) return', sw)
    assert 'e.request.method !== "GET"' in sw


def test_manifest_is_valid():
    m = json.loads((UI / "manifest.webmanifest").read_text(encoding="utf-8-sig"))
    assert m["start_url"] == "/" and m["display"] == "standalone" and m["icons"]


def test_worker_and_manifest_are_served(make_device):
    d, _ = make_device("devA", "site1")
    c = TestClient(create_app(d, SyncWorker(d, None, None), "op-1234567"))
    r = c.get("/sw.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert c.get("/static/manifest.webmanifest").status_code == 200
    assert 'rel="manifest"' in c.get("/").text
