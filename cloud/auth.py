"""Device and admin authentication for the cloud Sync API.

- Every device has its own bearer token; only its sha256 is stored (a stolen tokens file reveals no tokens).
- Tenant, device and site are taken from the token record, never from a request body.
- Tokens are revocable (revocation list = `revoked` flag), checked on every request.
- Roles: "device" (push, pull mirror, read own tenant's cases) and "admin" (also retract evidence, issue tokens).
- Per-token rate limit (sliding window) against sync floods.
- Tokens EXPIRE (TOKEN_TTL_S, default 30 days). A device renews its own token before expiry (POST /v1/token/renew);
  the old token keeps working for RENEW_GRACE_S so a sync in flight never breaks. Expired = refused.
Prototype limits (documented): mTLS is optional (--mtls, cloud/tls.py; it binds the certificate to the token's device);
the localhost demo is plain HTTP (the launchers enforce HTTPS off-localhost).
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import secrets
import threading
import time
from dataclasses import asdict, dataclass

RATE_LIMIT = 120          # requests per window per token
RATE_WINDOW_S = 60.0
TOKEN_TTL_S = 30 * 24 * 3600
RENEW_GRACE_S = 600


@dataclass(frozen=True)
class AuthContext:
    device_id: str
    site_id: str
    tenant_id: str
    role: str                 # device | admin
    expires_at: float | None = None


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

    def issue(self, device_id: str, site_id: str, tenant_id: str, role: str = "device", token: str | None = None,
              ttl_s: float = TOKEN_TTL_S, save: bool = True) -> str:
        """save=False: for bulk enrolment (call save() once at the end) - the file is rewritten per token otherwise."""
        if role not in ("device", "admin"):
            raise ValueError("role must be device or admin")
        token = token or secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._tokens[_h(token)] = asdict(AuthContext(device_id, site_id, tenant_id, role)) | {
                "revoked": False, "issued_at": now, "expires_at": now + ttl_s}
            if save:
                self._save()
        return token

    def save(self) -> None:
        with self._lock:
            self._save()

    def renew(self, token: str, ttl_s: float = TOKEN_TTL_S) -> tuple[str, float] | None:
        """A still-valid token gets a successor with the same identity; the old one lives RENEW_GRACE_S longer at
        most. Returns (new_token, expires_at) or None if the token is not valid."""
        ctx = self.verify(token)
        if ctx is None:
            return None
        new = self.issue(ctx.device_id, ctx.site_id, ctx.tenant_id, ctx.role, ttl_s=ttl_s)
        with self._lock:
            old = self._tokens[_h(token)]
            old["expires_at"] = min(self._expiry(old), time.time() + RENEW_GRACE_S)
            old["renewed"] = True
            self._save()
            return new, self._tokens[_h(new)]["expires_at"]

    @staticmethod
    def _expiry(rec: dict) -> float:
        return rec.get("expires_at") or rec.get("issued_at", 0.0) + TOKEN_TTL_S   # records from before expiry existed

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
            # Lookup by the token's SHA-256: O(1) for fleets of thousands (the old loop compared against EVERY stored
            # hash per request). Timing does not leak the token: an attacker cannot choose the hash of a guess.
            rec = self._tokens.get(h)
            if rec is None:
                return None
            if rec["revoked"] or time.time() >= self._expiry(rec):
                return None
            return AuthContext(rec["device_id"], rec["site_id"], rec["tenant_id"], rec["role"], self._expiry(rec))

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
