"""Tamper-evident audit log of admin actions (token issue/revoke, retraction requests and approvals, quarantine).

Append-only JSON lines; every entry carries the SHA-256 of the previous entry, so editing or deleting a past line
breaks the chain and `verify()` reports where. (Evidence of tampering, not prevention: an attacker with write
access to the file can rewrite the whole chain - keep a copy of the latest hash elsewhere, e.g. printed in reports.)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import threading

GENESIS = "0" * 64


def _digest(entry: dict) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | pathlib.Path | None = None):
        self.path = pathlib.Path(path) if path else None
        self._lock = threading.Lock()
        self._mem: list[dict] = []
        self._last = GENESIS
        for e in self.entries():
            self._last = e["hash"]

    def entries(self) -> list[dict]:
        if self.path is None:
            return list(self._mem)
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, actor: str, tenant: str, action: str, **detail) -> dict:
        with self._lock:
            body = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "actor": actor,
                    "tenant": tenant, "action": action, "detail": detail, "prev": self._last}
            entry = body | {"hash": _digest(body)}
            if self.path is None:
                self._mem.append(entry)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")
            self._last = entry["hash"]
            return entry

    def verify(self) -> dict:
        prev = GENESIS
        for i, e in enumerate(self.entries()):
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("prev") != prev or _digest(body) != e.get("hash"):
                return {"ok": False, "broken_at": i, "entries": i}
            prev = e["hash"]
        return {"ok": True, "entries": len(self.entries()), "head": prev}
