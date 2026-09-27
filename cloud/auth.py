"""Device and admin authentication for the cloud Sync API.

- Every device has its own bearer token; only its sha256 is stored (a stolen tokens file reveals no tokens).
- Tenant, device and site are taken from the token record, never from a request body.
- Tokens are revocable (revocation list = `revoked` flag), checked on every request.
- Roles: "device" (push, pull mirror, read own tenant's cases) and "admin" (also retract evidence, issue tokens).
- Per-token rate limit (sliding window) against sync floods.
Prototype limits (documented): no token expiry/rotation, no mTLS; demo runs on localhost HTTP.
"""
from __future__ import annotations

import collections
import hashlib
import hmac
import json
import pathlib
import secrets
import threading
import time
from dataclasses import asdict, dataclass

RATE_LIMIT = 120          # requests per window per token
RATE_WINDOW_S = 60.0


@dataclass(frozen=True)
class AuthContext:
    device_id: str
    site_id: str
    tenant_id: str
    role: str                 # device | admin


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TokenRegistry:
    def __init__(self, path: str | pathlib.Path | None = None):
        self.path = pathlib.Path(path) if path else None
        self._lock = threading.Lock()
        self._tokens: dict[str, dict] = {}
        self._hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        if self.path and self.path.exists():
            self._tokens = json.loads(self.path.read_text())

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._tokens, indent=2))
            tmp.replace(self.path)

    def issue(self, device_id: str, site_id: str, tenant_id: str, role: str = "device", token: str | None = None) -> str:
        if role not in ("device", "admin"):
            raise ValueError("role must be device or admin")
        token = token or secrets.token_urlsafe(32)
        with self._lock:
            self._tokens[_h(token)] = asdict(AuthContext(device_id, site_id, tenant_id, role)) | {
                "revoked": False, "issued_at": time.time()}
            self._save()
        return token

    def revoke(self, device_id: str) -> int:
        with self._lock:
            n = 0
            for rec in self._tokens.values():
                if rec["device_id"] == device_id and not rec["revoked"]:
                    rec["revoked"] = True
                    n += 1
            self._save()
            return n

    def verify(self, token: str | None) -> AuthContext | None:
        if not token:
            return None
        h = _h(token)
        with self._lock:
            for k, rec in self._tokens.items():          # constant-time compare against each stored hash
                if hmac.compare_digest(k, h):
                    if rec["revoked"]:
                        return None
                    return AuthContext(rec["device_id"], rec["site_id"], rec["tenant_id"], rec["role"])
        return None

    def allow(self, ctx: AuthContext, now: float | None = None) -> bool:
        """Sliding-window rate limit per device."""
        now = time.time() if now is None else now
        with self._lock:
            q = self._hits[ctx.device_id]
            while q and q[0] <= now - RATE_WINDOW_S:
                q.popleft()
            if len(q) >= RATE_LIMIT:
                return False
            q.append(now)
            return True

    def devices(self, tenant_id: str) -> list[dict]:
        with self._lock:
            return [{k: v for k, v in r.items()} for r in self._tokens.values() if r["tenant_id"] == tenant_id]
