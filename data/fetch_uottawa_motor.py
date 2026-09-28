"""Download the University of Ottawa ELECTRIC MOTOR vibration + acoustic dataset (MATLAB files) into
data/raw/uottawa_motor/ (git-ignored, not redistributed).

UOEMD-VAFCVS: Sehri, M., Dumond, P. et al., Mendeley Data v2, DOI 10.17632/msxs4vj48g.2, licence CC BY 4.0 (verified
via the Mendeley public API, 28 Sep 2026). Paper: Data in Brief 53 (2024) 110144 (PMC11636780).
8 Marathon D396 3-phase motors, each with ONE fault created by SpectraQuest (H-H healthy, R-U rotor unbalance, R-M
rotor misalignment, S-W stator winding, V-U voltage unbalance, B-R bowed rotor, K-A broken rotor bars, F-B faulty
bearing). Speeds 1-4 = constant 15/30/45/60 Hz, 5-8 = ramps; load 0/1. 42 kHz, 10 s. Columns: accelerometer,
MICROPHONE, accelerometer, accelerometer, temperature.
NOTE: each fault is one physical motor, so a classifier TRAINED on this set would learn the motor, not the fault
(leakage). We only use it to TEST fixed physics rules and the gate, never to train.
Run: .venv\Scripts\python.exe -m data.fetch_uottawa_motor
"""
from __future__ import annotations

import pathlib
import sys

import httpx

from data.fetch_hust import sha256

DATASET, VERSION = "msxs4vj48g", 2
OUT = pathlib.Path(__file__).resolve().parent / "raw" / "uottawa_motor"
API = "https://data.mendeley.com/public-api/datasets/{d}/files?folder_id={f}&version={v}"
FOLDERS = {"mat_unloaded": "6f6a2a17-c54d-47dc-bc02-b145233de93d", "mat_loaded": "06fee1a5-1041-4f74-bbf5-4ddb3872876a"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for fid in FOLDERS.values():
        files += httpx.get(API.format(d=DATASET, f=fid, v=VERSION), timeout=60).raise_for_status().json()
    bad = 0
    with httpx.Client(timeout=300, follow_redirects=True) as c:
        for k, f in enumerate(files, 1):
            dest, want = OUT / f["filename"], f["content_details"]["sha256_hash"]
            if dest.exists() and sha256(dest) == want:
                continue
            tmp = dest.with_suffix(".part")
            with c.stream("GET", f["content_details"]["download_url"]) as r:
                r.raise_for_status()
                with open(tmp, "wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        out.write(chunk)
            if sha256(tmp) != want:
                print(f"SHA-256 MISMATCH {f['filename']}", flush=True)
                tmp.unlink()
                bad += 1
                continue
            tmp.replace(dest)
            print(f"[{k}/{len(files)}] {f['filename']}", flush=True)
    print(f"done: {len(files) - bad}/{len(files)} verified files in {OUT}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
