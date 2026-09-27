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
                if httpx.get(url + "/v1/health", verify=str(certs / "ca.pem"), timeout=1).status_code == 200:
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
    r = httpx.get(tls_cloud + "/v1/health", verify=str(certs / "ca.pem"))
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
