"""Backup and restore for a device and for the fleet cloud.

  python -m tools.backup device-backup  <device folder> <backup.zip>      (stop the device first)
  python -m tools.backup device-restore <backup.zip> <device folder>      (then start the device on that folder)
  python -m tools.backup cloud-backup   <qdrant url> <out folder> [--tenant acme]
  python -m tools.backup cloud-restore  <qdrant url> <backup folder>

Device: the whole device folder (Qdrant Edge shards, the SQLite journal/outbox, baseline, machine card, manuals index)
is zipped with a manifest (file list + SHA-256); restore refuses a damaged archive and a non-empty target, then opens
the restored device and counts its points. The note key inside is DPAPI-protected on Windows: technician notes in a
backup can only be decrypted on the same Windows account (on Linux/macOS the key file is copied as is - protect the
archive).
Cloud: a Qdrant Server snapshot of every collection of the tenant (events, cases, mirror) downloaded next to the token
registry (hashes only) and the audit log; restore uploads the snapshots back (recovers the collections).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import shutil
import sys
import zipfile

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = "backup_manifest.json"


def _sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def device_backup(folder: str | pathlib.Path, out: str | pathlib.Path) -> dict:
    folder, out = pathlib.Path(folder), pathlib.Path(out)
    if not (folder / "device.sqlite").exists():
        raise SystemExit(f"{folder} is not a device folder (no device.sqlite)")
    files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
    manifest = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "kind": "device",
                "files": {p.relative_to(folder).as_posix(): {"sha256": _sha(p), "bytes": p.stat().st_size} for p in files}}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(folder).as_posix())
        z.writestr(MANIFEST, json.dumps(manifest, indent=1))
    return {"files": len(files), "bytes": out.stat().st_size, "archive": str(out)}


def device_restore(archive: str | pathlib.Path, folder: str | pathlib.Path, verify: bool = True) -> dict:
    archive, folder = pathlib.Path(archive), pathlib.Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} is not empty: restore into a new folder")
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read(MANIFEST))
        names = set(z.namelist()) - {MANIFEST}
        if names != set(manifest["files"]):
            raise SystemExit("archive does not match its manifest (files missing or added)")
        for n in names:
            if pathlib.PurePosixPath(n).is_absolute() or ".." in pathlib.PurePosixPath(n).parts:
                raise SystemExit(f"unsafe path in archive: {n}")
        folder.mkdir(parents=True, exist_ok=True)
        z.extractall(folder, members=sorted(names))
    bad = [n for n, m in manifest["files"].items() if _sha(folder / n) != m["sha256"]]
    if bad:
        shutil.rmtree(folder, ignore_errors=True)
        raise SystemExit(f"checksum mismatch in {len(bad)} file(s), e.g. {bad[0]}: restore aborted")
    out = {"files": len(names), "folder": str(folder)}
    if verify:
        out["check"] = _open_and_count(folder)
    return out


def _open_and_count(folder: pathlib.Path) -> dict:
    from edge.store_edge import EdgeStore
    s = EdgeStore(folder / "local", allow_text_model_change=True)
    try:
        return {"points_by_type": s.facet("type")}
    finally:
        s.close()


def cloud_backup(url: str, out: str | pathlib.Path, tenant: str = "acme") -> dict:
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cols = [c["name"] for c in httpx.get(f"{url}/collections", timeout=60).json()["result"]["collections"]
            if c["name"].endswith("_" + tenant)]
    got = {}
    with httpx.Client(timeout=600) as c:
        for col in cols:
            snap = c.post(f"{url}/collections/{col}/snapshots", params={"wait": "true"}).raise_for_status().json()["result"]["name"]
            data = c.get(f"{url}/collections/{col}/snapshots/{snap}").raise_for_status().content
            (out / f"{col}.snapshot").write_bytes(data)
            c.delete(f"{url}/collections/{col}/snapshots/{snap}")
            got[col] = len(data)
    for f in ("tokens.json", "audit.log"):
        src = ROOT / "runtime" / "cloud" / f
        if src.exists():
            shutil.copy2(src, out / f)
    (out / MANIFEST).write_text(json.dumps({"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                            "kind": "cloud", "tenant": tenant, "collections": got}, indent=1))
    return {"collections": got}


def cloud_restore(url: str, folder: str | pathlib.Path) -> dict:
    folder = pathlib.Path(folder)
    man = json.loads((folder / MANIFEST).read_text())
    done = {}
    with httpx.Client(timeout=600) as c:
        for col in man["collections"]:
            with open(folder / f"{col}.snapshot", "rb") as f:
                r = c.post(f"{url}/collections/{col}/snapshots/upload", params={"wait": "true", "priority": "snapshot"},
                           files={"snapshot": (f"{col}.snapshot", f)})
            r.raise_for_status()
            done[col] = httpx.get(f"{url}/collections/{col}", timeout=60).json()["result"]["points_count"]
    return {"restored_points": done}


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    fn = {"device-backup": device_backup, "device-restore": device_restore,
          "cloud-backup": lambda u, o, *r: cloud_backup(u, o, r[1] if len(r) > 1 and r[0] == "--tenant" else "acme"),
          "cloud-restore": cloud_restore}.get(cmd)
    if fn is None:
        print(__doc__)
        sys.exit(2)
    print(json.dumps(fn(*args), indent=1))
