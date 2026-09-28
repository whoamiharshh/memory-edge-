"""Backup and restore: a device's memory survives being zipped and restored elsewhere (verified by reopening it), a
damaged archive is refused, and a cloud tenant's collections come back from Qdrant Server snapshots."""
import zipfile

import numpy as np
import pytest

from tools import backup
from tests.conftest import needs_cwru
from edge.replay import Recordings


@needs_cwru
def test_device_backup_restore_roundtrip(make_device, tmp_path):
    from edge.device import Device, DeviceConfig
    from shared.embed import HashEmbedder
    d, _ = make_device("devA", "site1")
    d.fit_baseline(Recordings.baseline())
    for w in Recordings.windows(105)[:5]:
        d.ingest_window(w)
    eid = d.episodes()[0]["episode_id"]
    d.set_note(eid, "Inner race spall, bearing replaced")
    root = d.cfg.root
    d.close()
    info = backup.device_backup(root, tmp_path / "b.zip")
    assert info["files"] > 5
    out = backup.device_restore(tmp_path / "b.zip", tmp_path / "restored")
    assert out["check"]["points_by_type"]["episode"] == 1
    d2 = Device(DeviceConfig("devA", "site1", "devA-m1", tmp_path / "restored"), HashEmbedder())
    try:
        assert d2.episode(eid)["note_text"] == "Inner race spall, bearing replaced"      # key restored with the data
        assert d2.ingest_window(Recordings.windows(99)[0])["state"] == "normal"          # baseline restored
    finally:
        d2.close()
    with pytest.raises(SystemExit, match="not empty"):
        backup.device_restore(tmp_path / "b.zip", tmp_path / "restored")


def test_damaged_archive_is_refused(tmp_path):
    src = tmp_path / "dev"
    src.mkdir()
    (src / "device.sqlite").write_bytes(b"x" * 100)
    (src / "baseline.json").write_text("{}")
    backup.device_backup(src, tmp_path / "b.zip")
    with zipfile.ZipFile(tmp_path / "b.zip") as z:
        items = {n: z.read(n) for n in z.namelist()}
    items["baseline.json"] = b'{"tampered": true}'
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as z:
        for n, b in items.items():
            z.writestr(n, b)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        backup.device_restore(tmp_path / "bad.zip", tmp_path / "r", verify=False)
    assert not (tmp_path / "r").exists()


def test_cloud_backup_and_restore_with_qdrant_server(qdrant_url, server_cloud, tmp_path):
    from tests.security.test_security import event, push
    tok = server_cloud["registry"].issue("devA", "site1", server_cloud["tenant"])
    from shared import ids
    push(server_cloud["client"], tok, [event(episode=ids.make_id("ep", i)) for i in range(3)])
    b = backup.cloud_backup(qdrant_url, tmp_path / "cb", server_cloud["tenant"])
    ev_col = f"events_{server_cloud['tenant']}"
    assert ev_col in b["collections"]
    import httpx
    httpx.delete(f"{qdrant_url}/collections/{ev_col}", timeout=60)
    r = backup.cloud_restore(qdrant_url, tmp_path / "cb")
    assert r["restored_points"][ev_col] == 3
