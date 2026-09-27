"""Create a private certificate authority + a server certificate for HTTPS between devices, the cloud and phones.

  .venv\\Scripts\\python.exe tools\\make_certs.py [extra-hostname-or-ip ...]      (CA + server certificate)
  .venv\\Scripts\\python.exe tools\\make_certs.py device <device-id>             (a device's client certificate)
Writes runtime/tls/: ca.pem (share with devices/phones to trust), ca.key (keep private), server.pem, server.key.
The server certificate covers localhost, 127.0.0.1, this computer's hostname and every IPv4 address it has now
(so a phone on the same Wi-Fi can open https://<laptop-ip>:8101/sensor). Re-run if the laptop's IP changes.

Why a private CA and not a public one: the demo has no public domain name, and a phone or second PC on the same
network can be told to trust ca.pem once. Browsers otherwise show a warning you can click through; the connection is
still encrypted, but only a trusted CA also proves you reached the right machine.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import pathlib
import socket
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

OUT = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "tls"


def local_names() -> tuple[list[str], list[str]]:
    names, ips = {"localhost", socket.gethostname()}, {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(names), sorted(ips)


def write_key(key, path: pathlib.Path) -> None:
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))


def main(extra: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Machine Memory local CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
          .not_valid_after(now + dt.timedelta(days=825))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                       key_encipherment=False, data_encipherment=False, key_agreement=False,
                                       encipher_only=False, decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    names, ips = local_names()
    for e in extra:
        try:
            ipaddress.ip_address(e)
            ips.append(e)
        except ValueError:
            names.append(e)
    san = [x509.DNSName(n) for n in sorted(set(names))] + [x509.IPAddress(ipaddress.ip_address(i)) for i in sorted(set(ips))]
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])]))
            .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=397))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    (OUT / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    write_key(ca_key, OUT / "ca.key")
    (OUT / "server.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM))
    write_key(key, OUT / "server.key")
    print(f"wrote {OUT}: ca.pem ca.key server.pem server.key")
    print("server certificate valid for:", ", ".join(sorted(set(names)) + sorted(set(ips))))


def device_cert(device_id: str) -> None:
    """Issue a CLIENT certificate for one device, signed by the existing CA (mutual TLS: the cloud started with
    --mtls only accepts connections that present such a certificate). Writes runtime/tls/devices/<id>.pem/.key.
    Copy those two files (and ca.pem) to the device; never copy ca.key."""
    if not device_id.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("device id: letters, digits, - and _ only")
    ca_pem, ca_key_pem = OUT / "ca.pem", OUT / "ca.key"
    if not ca_pem.exists():
        raise SystemExit("run tools/make_certs.py first (it creates the CA)")
    ca = x509.load_pem_x509_certificate(ca_pem.read_bytes())
    ca_key = serialization.load_pem_private_key(ca_key_pem.read_bytes(), password=None)
    now = dt.datetime.now(dt.timezone.utc)
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, device_id)]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=397))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    d = OUT / "devices"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{device_id}.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    write_key(key, d / f"{device_id}.key")
    print(f"wrote {d / (device_id + '.pem')} and .key (client certificate for mutual TLS)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "device":
        device_cert(sys.argv[2])
    else:
        main(sys.argv[1:])
