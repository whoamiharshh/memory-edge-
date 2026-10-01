"""Find, download and start the Qdrant Server release binary on any OS (Windows, Linux, macOS; x86-64 or ARM64).

  .venv/bin/python -m tools.qdrant_local download        # puts qdrant(.exe) for this OS into qdrant_server/
Release assets of v1.19.1 (checked 28 Sep 2026 on github.com/qdrant/qdrant): windows-msvc zip, linux-gnu / musl
tar.gz (x86-64), linux-musl aarch64, apple-darwin x86-64 / aarch64.
"""
from __future__ import annotations

import io
import os
import pathlib
import platform
import socket
import subprocess
import sys
import tarfile
import time
import zipfile
from contextlib import contextmanager

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / "qdrant_server"
VERSION = "v1.19.1"


def binary() -> pathlib.Path:
    return DIR / ("qdrant.exe" if os.name == "nt" else "qdrant")


def asset() -> str:
    m = platform.machine().lower()
    arm = m in ("arm64", "aarch64")
    if sys.platform.startswith("win"):
        return "qdrant-x86_64-pc-windows-msvc.zip"
    if sys.platform == "darwin":
        return f"qdrant-{'aarch64' if arm else 'x86_64'}-apple-darwin.tar.gz"
    return "qdrant-aarch64-unknown-linux-musl.tar.gz" if arm else "qdrant-x86_64-unknown-linux-gnu.tar.gz"


def download() -> pathlib.Path:
    url = f"https://github.com/qdrant/qdrant/releases/download/{VERSION}/{asset()}"
    data = httpx.get(url, follow_redirects=True, timeout=300).raise_for_status().content
    DIR.mkdir(exist_ok=True)
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extract("qdrant.exe", DIR)
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("qdrant"))
            member.name = "qdrant"
            t.extract(member, DIR, filter="data")
        binary().chmod(0o755)
    return binary()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def server(storage: pathlib.Path):
    """A throwaway Qdrant Server on free ports; yields its URL."""
    http, grpc = free_port(), free_port()
    storage.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"QDRANT__STORAGE__STORAGE_PATH": str(storage / "storage"),
                        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(storage / "snapshots"),
                        "QDRANT__SERVICE__HTTP_PORT": str(http), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
                        "QDRANT__TELEMETRY_DISABLED": "true",
                        # A throwaway server holds a handful of points per collection, but Qdrant sizes a
                        # new collection for a real workload: one segment per CPU thread, each with a
                        # preallocated write-ahead log. Measured here, a collection holding THREE events
                        # produced a 513 MB snapshot, and one full test run left 98 GB behind and then
                        # failed with StorageFull. One segment and a small WAL cost nothing at this size.
                        "QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER": "1",
                        "QDRANT__STORAGE__WAL__WAL_CAPACITY_MB": "4"}
    proc = subprocess.Popen([str(binary())], cwd=storage, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{http}"
    try:
        for _ in range(240):
            try:
                if httpx.get(f"{url}/readyz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("Qdrant Server did not become ready")
        yield url
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        print(download())
    else:
        print(binary(), "exists" if binary().exists() else "missing (run: python -m tools.qdrant_local download)")
