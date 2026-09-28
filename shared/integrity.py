"""Code-integrity fingerprint (tamper EVIDENCE, not secure boot).

The device hashes its own program files (edge/, shared/, the device UI) at start and sends the digest with every
push; the cloud compares it with the digest of the release it runs and shows "matches" / "DIFFERS" per device in the
admin view. This catches an accidentally or naively modified device. It cannot stop a fully compromised device from
sending the expected digest - that needs hardware attestation (secure boot / TPM), which this prototype does not have.
"""
from __future__ import annotations

import functools
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARTS = ("edge", "shared")
SUFFIXES = {".py", ".js", ".html", ".css", ".webmanifest", ".svg"}


@functools.lru_cache(maxsize=1)
def code_hash(root: pathlib.Path = ROOT) -> str:
    h = hashlib.sha256()
    for part in PARTS:
        for p in sorted((root / part).rglob("*")):
            if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts:
                h.update(p.relative_to(root).as_posix().encode() + b"\0")
                h.update(p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()
