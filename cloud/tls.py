"""Server TLS context shared by the cloud and the device UI.

TLS 1.2 minimum. With `ca` (mutual TLS) every client must present a certificate signed by our CA; with `crl` the
certificate revocation list written by `tools/make_certs.py revoke <device>` is loaded and checked for the client
(leaf) certificate, so a stolen device certificate is refused. The CRL is read at start: after revoking, restart the
cloud (the device's TOKEN can be revoked instantly through the admin API; both are needed for a full cut-off).
"""
from __future__ import annotations

import pathlib
import ssl


def server_context(certfile: str | pathlib.Path, keyfile: str | pathlib.Path, ca: str | pathlib.Path | None = None,
                   crl: str | pathlib.Path | None = None) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(certfile), str(keyfile))
    if ca:
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile=str(ca))
        if crl:
            if not pathlib.Path(crl).exists():
                raise FileNotFoundError(f"{crl} missing: run tools/make_certs.py crl (an empty list is fine)")
            ctx.load_verify_locations(cafile=str(crl))
            ctx.verify_flags |= ssl.VERIFY_CRL_CHECK_LEAF
    return ctx
