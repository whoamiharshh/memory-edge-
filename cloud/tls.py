"""Server TLS context shared by the cloud and the device UI.

TLS 1.2 minimum. With `ca` (mutual TLS) every client must present a certificate signed by our CA; with `crl` the
certificate revocation list written by `tools/make_certs.py revoke <device>` is loaded and checked for the client
(leaf) certificate, so a stolen device certificate is refused.
  CrlReloader     re-reads the CRL when the file changes (every RELOAD_S): a revocation takes effect for every NEW
                  connection within seconds, without restarting the cloud. OpenSSL keeps every CRL it was given and
                  uses the most recent one of the issuer, so the newest list always wins.
  CertBoundH11    uvicorn's HTTP protocol that also reads the verified client certificate's name when a connection
                  opens and hands it to the app (request.state.tls_client_cn): the cloud then refuses a token used with
                  ANOTHER device's certificate (a stolen token alone, or a certificate alone, is not enough).
"""
from __future__ import annotations

import logging
import pathlib
import ssl
import threading

log = logging.getLogger("cloud.tls")
RELOAD_S = 2.0


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
            CrlReloader(ctx, pathlib.Path(crl)).start()
    return ctx


class CrlReloader(threading.Thread):
    def __init__(self, ctx: ssl.SSLContext, crl: pathlib.Path, interval_s: float = RELOAD_S):
        super().__init__(name="crl-reloader", daemon=True)
        self.ctx, self.crl, self.interval_s = ctx, crl, interval_s
        self._seen = self._stamp()
        self._stop = threading.Event()

    def _stamp(self):
        try:
            st = self.crl.stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            stamp = self._stamp()
            if stamp is None or stamp == self._seen:
                continue
            try:
                self.ctx.load_verify_locations(cafile=str(self.crl))
                self._seen = stamp
                log.warning("reloaded certificate revocation list %s", self.crl)
            except (ssl.SSLError, OSError) as e:        # half-written file: try again next round
                log.warning("CRL reload failed (%s); keeping the previous list", e)

    def stop(self) -> None:
        self._stop.set()


def peer_common_name(cert: dict | None) -> str | None:
    """The subject CN of a certificate as returned by SSLSocket.getpeercert() (None without a verified cert)."""
    for rdn in (cert or {}).get("subject", ()):
        for k, v in rdn:
            if k == "commonName":
                return v
    return None


def cert_bound_protocol():
    """uvicorn's h11 protocol class, extended to expose the verified client certificate's CN per connection."""
    from uvicorn.protocols.http.h11_impl import H11Protocol

    class CertBoundH11(H11Protocol):
        def connection_made(self, transport) -> None:          # called after the TLS handshake completed
            super().connection_made(transport)
            cn = peer_common_name(transport.get_extra_info("peercert"))
            # scope["state"] is copied from app_state for every request: a per-connection copy carries the CN
            self.app_state = {**self.app_state, "tls_client_cn": cn}

    return CertBoundH11
