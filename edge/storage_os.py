"""Operating-system help for the device's disk footprint.

Qdrant Edge pre-allocates its storage pages (~200 MB per shard even when empty; bench/footprint.py). The pages are
mostly zeros, so filesystem compression removes almost all of it: on Windows (NTFS) a 1,000-point shard went from
267 MB to 3.8 MB allocated and still opened and answered queries (measured 28 Sep 2026). On Linux, Android and macOS,
filesystems usually keep such pre-allocated-but-unwritten ranges sparse; that could not be measured here (no Linux
machine available) and is labelled as unverified in docs/BENCHMARKS.md.
"""
from __future__ import annotations

import os
import pathlib
import subprocess


def enable_compression(path: str | os.PathLike) -> dict:
    """Windows: compress the files that exist now (measured: works while the shard is open; writes, search and
    reopen keep working; 267 MB -> 1.9 MB). Files Edge creates LATER are not compressed automatically (measured:
    they do not inherit the folder attribute), so the device calls this at start-up and in its hourly housekeeping.
    Elsewhere a no-op that says so. Never raises: a device must start even if compression is unavailable."""
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        return {"compressed": False, "reason": "not Windows: rely on the filesystem's sparse files (unverified here)"}
    try:
        r = subprocess.run(["compact", "/c", f"/s:{p}", "/i", "/q"], capture_output=True, text=True, timeout=600)
        return {"compressed": r.returncode == 0, "reason": (r.stdout or r.stderr).strip().splitlines()[-1:] or ""}
    except (OSError, subprocess.SubprocessError) as e:
        return {"compressed": False, "reason": f"{type(e).__name__}: {e}"}


def allocated_bytes(path: str | os.PathLike) -> int:
    """Bytes the files really occupy (compressed/sparse aware)."""
    total = 0
    for f in pathlib.Path(path).rglob("*"):
        if not f.is_file():
            continue
        if os.name == "nt":
            import ctypes
            high = ctypes.c_ulong(0)
            low = ctypes.windll.kernel32.GetCompressedFileSizeW(str(f), ctypes.byref(high))
            total += (high.value << 32) + low
        else:
            total += f.stat().st_blocks * 512
    return total
