"""Encryption at rest for technician notes (docs/THREATS.md 'Stolen device').

Notes are the most sensitive thing on a device (names, site details). They are encrypted with AES-256-GCM before
they reach the Edge shard or the SQLite journal, as "enc1:" + base64(nonce | ciphertext+tag). The 32-byte key:
  Windows  protected with DPAPI (CryptProtectData, current user): a copied disk or folder alone cannot decrypt it.
  elsewhere a key file readable by the owner only (0600). WEAKER: anyone who can read the device user's files can
           read the key; use OS disk encryption (LUKS / FileVault) there. Documented, not hidden.
Residual (documented): the note's embedding and BM25 vectors are stored in the shard for search; they are not the text,
but embeddings can leak parts of it (the reason they are never shipped to the cloud).
"""
from __future__ import annotations

import base64
import os
import pathlib
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PREFIX = "enc1:"
UNREADABLE = "[encrypted note: this device's key is not available]"


def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    blob_in, blob_out = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), BLOB()
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = (fn(ctypes.byref(blob_in), "machine-memory note key" if protect else None, None, None, None, 0x1,
             ctypes.byref(blob_out)))                                        # 0x1 = CRYPTPROTECT_UI_FORBIDDEN
    if not ok:
        raise OSError("DPAPI call failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def load_or_create_key(root: str | os.PathLike) -> tuple[bytes, str]:
    """(key, how it is protected)."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        f = root / "note_key.dpapi"
        if f.exists():
            return _dpapi(f.read_bytes(), protect=False), "Windows DPAPI"
        key = secrets.token_bytes(32)
        f.write_bytes(_dpapi(key, protect=True))
        return key, "Windows DPAPI"
    f = root / "note_key.bin"
    if f.exists():
        return f.read_bytes(), "owner-only key file (weaker; use disk encryption)"
    key = secrets.token_bytes(32)
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(key)
    return key, "owner-only key file (weaker; use disk encryption)"


class NoteCipher:
    def __init__(self, key: bytes, how: str = ""):
        self._aes, self.how = AESGCM(key), how

    def encrypt(self, text: str | None) -> str | None:
        if not text or text.startswith(PREFIX):
            return text
        nonce = secrets.token_bytes(12)
        return PREFIX + base64.b64encode(nonce + self._aes.encrypt(nonce, text.encode("utf-8"), b"note")).decode()

    def decrypt(self, value: str | None) -> str | None:
        if not value or not value.startswith(PREFIX):
            return value                                   # empty, or a note written before encryption existed
        raw = base64.b64decode(value[len(PREFIX):])
        try:
            return self._aes.decrypt(raw[:12], raw[12:], b"note").decode("utf-8")
        except Exception:
            return UNREADABLE
