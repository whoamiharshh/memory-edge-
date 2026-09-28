"""HTTPS between devices and the cloud (docs/THREATS.md "Insecure transport"): the cloud serves TLS with a
certificate from our private CA; a client that trusts the CA connects, a client that does not is refused, plain HTTP
to the TLS port fails, and the launchers refuse plain HTTP on a network address."""
import os
import pathlib
import socket
import subprocess
import sys
import time

import httpx
import pytest

from edge.sync_worker import _tls_context

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def certs(tmp_path_factory):
    tls = ROOT / "runtime" / "tls"
    if not (tls / "ca.pem").exists():
        subprocess.run([PY, str(ROOT / "tools" / "make_certs.py")], check=True, cwd=ROOT, capture_output=True)
    return tls


@pytest.fixture(scope="module")
def tls_cloud(certs):
    port = free_port()
    p = subprocess.Popen([PY, "-m", "cloud.main", "--memory", "--hash-embedder", "--tls", "--port", str(port)],
                         cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"https://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(url + "/v1/health", verify=_tls_context(str(certs / "ca.pem"), None), timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        else:
            pytest.fail("TLS cloud did not start")
        yield url
    finally:
        p.kill()
        p.wait()


def test_client_trusting_our_ca_connects(tls_cloud, certs):
    r = httpx.get(tls_cloud + "/v1/health", verify=_tls_context(str(certs / "ca.pem"), None))
    assert r.status_code == 200 and r.json()["ok"] is True


def test_client_without_our_ca_is_refused(tls_cloud):
    with pytest.raises(httpx.ConnectError):                 # certificate verify failed
        httpx.get(tls_cloud + "/v1/health", verify=True)


def test_plain_http_to_the_tls_port_fails(tls_cloud):
    with pytest.raises(httpx.HTTPError):
        httpx.get(tls_cloud.replace("https://", "http://") + "/v1/health", timeout=3)


@pytest.mark.parametrize("module", ["cloud.main", "edge.main"])
def test_launchers_refuse_plain_http_on_the_network(module):
    args = [PY, "-m", module, "--host", "0.0.0.0", "--port", str(free_port())]
    if module == "edge.main":
        args += ["--name", "x", "--site", "s", "--hash-embedder", "--no-llm", "--no-sync"]
    else:
        args += ["--memory", "--hash-embedder"]
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "refusing to serve plain HTTP" in r.stderr


MTLS_TOKENS: dict[str, str] = {}


@pytest.fixture(scope="module")
def mtls_cloud(certs, tmp_path_factory):
    for args in (["device", "devtest"], ["device", "devstolen"], ["revoke", "devstolen"], ["device", "devother"],
                 ["device", "devlate"]):
        subprocess.run([PY, str(ROOT / "tools" / "make_certs.py"), *args], check=True, cwd=ROOT, capture_output=True)
    rt = tmp_path_factory.mktemp("mtls_cloud")
    from cloud.auth import TokenRegistry
    reg = TokenRegistry(rt / "tokens.json")
    for d in ("devtest", "devother", "devlate"):
        MTLS_TOKENS[d] = reg.issue(d, "s1", "acme")
    port = free_port()
    p = subprocess.Popen([PY, "-m", "cloud.main", "--memory", "--hash-embedder", "--mtls", "--port", str(port),
                          "--runtime-dir", str(rt)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"https://127.0.0.1:{port}"
    cert = (str(certs / "devices" / "devtest.pem"), str(certs / "devices" / "devtest.key"))
    try:
        for _ in range(120):
            try:
                if httpx.get(url + "/v1/health", verify=_tls_context(str(certs / "ca.pem"), cert), timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        else:
            pytest.fail("mTLS cloud did not start")
        yield url, cert
    finally:
        p.kill()
        p.wait()


def test_mtls_accepts_a_device_certificate_and_refuses_without(mtls_cloud, certs):
    url, cert = mtls_cloud
    assert httpx.get(url + "/v1/health", verify=_tls_context(str(certs / "ca.pem"), cert)).status_code == 200
    with pytest.raises(httpx.HTTPError):                       # right CA, but no client certificate
        httpx.get(url + "/v1/health", verify=_tls_context(str(certs / "ca.pem"), None))


def test_mtls_refuses_a_revoked_device_certificate(mtls_cloud, certs):
    """tools/make_certs.py revoke puts the certificate on the CRL; the cloud checks it for every client."""
    url, cert = mtls_cloud
    stolen = (str(certs / "devices" / "devstolen.pem"), str(certs / "devices" / "devstolen.key"))
    with pytest.raises(httpx.HTTPError):
        httpx.get(url + "/v1/health", verify=_tls_context(str(certs / "ca.pem"), stolen))
    assert httpx.get(url + "/v1/health", verify=_tls_context(str(certs / "ca.pem"), cert)).status_code == 200


def _dev_cert(certs, name):
    return str(certs / "devices" / f"{name}.pem"), str(certs / "devices" / f"{name}.key")


def test_mtls_token_must_match_the_certificate(mtls_cloud, certs):
    """A token copied from devtest is refused when presented through devother's (valid) certificate."""
    url, _ = mtls_cloud
    ca = str(certs / "ca.pem")
    hdr = {"authorization": f"Bearer {MTLS_TOKENS['devtest']}"}
    ok = httpx.get(url + "/v1/cases", headers=hdr, verify=_tls_context(ca, _dev_cert(certs, "devtest")))
    assert ok.status_code == 200
    bad = httpx.get(url + "/v1/cases", headers=hdr, verify=_tls_context(ca, _dev_cert(certs, "devother")))
    assert bad.status_code == 403 and "does not belong" in bad.json()["detail"]


def test_mtls_revocation_takes_effect_without_restart(mtls_cloud, certs):
    """tools/make_certs.py revoke rewrites crl.pem; the running cloud reloads it (cloud/tls.py CrlReloader)."""
    url, _ = mtls_cloud
    ca, late = str(certs / "ca.pem"), _dev_cert(certs, "devlate")
    assert httpx.get(url + "/v1/health", verify=_tls_context(ca, late)).status_code == 200
    subprocess.run([PY, str(ROOT / "tools" / "make_certs.py"), "revoke", "devlate"], check=True, cwd=ROOT,
                   capture_output=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15:
        try:
            httpx.get(url + "/v1/health", verify=_tls_context(ca, late), timeout=3)
        except httpx.HTTPError:
            break                                            # refused: the new list is in force
        time.sleep(0.5)
    else:
        pytest.fail("revoked certificate still accepted 15 s after revocation")
    assert time.monotonic() - t0 < 10
    good = _dev_cert(certs, "devtest")                       # other devices are unaffected
    assert httpx.get(url + "/v1/health", verify=_tls_context(ca, good)).status_code == 200


def test_server_refuses_tls_below_1_2(tls_cloud, certs):
    import ssl
    ctx = ssl.create_default_context(cafile=str(certs / "ca.pem"))
    ctx.minimum_version = ssl.TLSVersion.TLSv1
    ctx.maximum_version = ssl.TLSVersion.TLSv1_1
    with pytest.raises(Exception):
        httpx.get(tls_cloud + "/v1/health", verify=ctx)
