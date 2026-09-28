"""Download the University of Ottawa bearing vibration + ACOUSTIC dataset into data/raw/uottawa/ (git-ignored, not
redistributed).

UODS-VAFDC: Sehri, M., Dumond, P., Bouchard, M. (2023), Mendeley Data v1, DOI 10.17632/y2px5tg92h.1, licence CC BY 4.0
(verified via the Mendeley public API, 28 Sep 2026). Paper: Data in Brief 49 (2023) 109327,
doi 10.1016/j.dib.2023.109327 (PMC10331275).
Setup (from the paper): NSK 6203ZZ / FAFNIR 203KD, 8 balls, ball 6.77 mm, pitch 28.50 mm, 1,750 rpm constant, 42 kHz,
10 s per file. Seals removed and bearings degreased so the faults DEVELOP NATURALLY (accelerated wear, not machined).
Columns: accelerometer (PCB 623C01 on the bearing), microphone (PCB 130F20, 2 cm from the bearing, not touching the
motor), speed, load, temperature. Files <H|I|O|B|C>_<bearing 1-20>_<0 healthy | 1 developing | 2 faulty>.mat.
Each file is checked against the SHA-256 the repository publishes.
Run: .venv\\Scripts\\python.exe data\\fetch_uottawa.py
"""
from __future__ import annotations

import pathlib
import sys

import httpx

from data.fetch_hust import sha256

DATASET, VERSION = "y2px5tg92h", 1
OUT = pathlib.Path(__file__).resolve().parent / "raw" / "uottawa"
API = "https://data.mendeley.com/public-api/datasets/{d}/files?folder_id={f}&version={v}"
FOLDERS = {
    "1_Healthy": "cd68faae-6dbd-4399-b1f2-c2a42ef78037",
    "2_Inner_Race_Faults": "2266c863-1c7d-453a-96d7-0a79b0558232",
    "3_Outer_Race_Faults": "e4ab0fd7-23f6-4615-8cf1-b49be7507486",
    "4_Ball_Faults": "f225e5d8-356e-417a-8af1-32f09fa0f8ee",
    "5_Cage_Faults": "f399e00c-f92c-43b6-a9f2-ca98282c20cb",
}


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
