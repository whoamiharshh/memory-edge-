"""Technician notes are encrypted at rest (docs/THREATS.md 'Stolen device'): the plain text is nowhere in the
device folder (shard files, SQLite journal), the device still reads, searches and shares normally, a restart keeps
the key, and another device's key cannot read them."""
import pathlib

from edge import crypto
from edge.device import Device, DeviceConfig
from shared.embed import HashEmbedder
from tests.conftest import needs_cwru
from tests.integration.test_fleet_flow import feed, setup_machine

pytestmark = needs_cwru
SECRET = "Quixotic zebra saw Ravi fix the drive-end seal"


def folder_contains(root: pathlib.Path, needle: str) -> list[str]:
    hits = []
    for f in root.rglob("*"):
        if f.is_file() and f.stat().st_size < 400 * 2 ** 20:
            with open(f, "rb") as h:
                data = h.read()
            if needle.encode() in data or needle.encode("utf-16-le") in data:
                hits.append(str(f.relative_to(root)))
    return hits


def test_note_is_encrypted_on_disk_but_works_everywhere(tmp_path):
    cfg = DeviceConfig(device_id="d", site_id="s", machine_id="m", root=tmp_path / "d")
    d = Device(cfg, HashEmbedder())
    try:
        setup_machine(d)
        feed(d, 105, n=5)
        eid = d.episodes()[0]["episode_id"]
        d.set_note(eid, SECRET)
        assert d.episode(eid)["note_text"] == SECRET                         # readable through the device
        assert d.search(text="quixotic zebra", use_fleet=False)["local"][0]["id"] == eid   # still searchable
        assert "DPAPI" in d.note_protection or "key file" in d.note_protection
    finally:
        d.close()
    assert folder_contains(tmp_path / "d", "Quixotic zebra") == []            # nowhere in plain text on disk
    d2 = Device(cfg, HashEmbedder())                                          # restart: same key, still readable
    try:
        assert d2.episode(eid)["note_text"] == SECRET
    finally:
        d2.close()


def test_another_devices_key_cannot_read_the_note(tmp_path):
    k1, _ = crypto.load_or_create_key(tmp_path / "a")
    k2, _ = crypto.load_or_create_key(tmp_path / "b")
    token = crypto.NoteCipher(k1).encrypt(SECRET)
    assert token.startswith(crypto.PREFIX) and SECRET not in token
    assert crypto.NoteCipher(k2).decrypt(token) == crypto.UNREADABLE
    assert crypto.NoteCipher(k1).decrypt(token) == SECRET
    assert crypto.NoteCipher(k1).decrypt("an old plain note") == "an old plain note"     # before encryption existed
