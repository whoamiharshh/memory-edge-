"""Download the HUST bearing dataset (the held-out "second machine") into data/raw/hust/ (git-ignored, not
redistributed).

HUST bearing: a practical dataset for ball bearing fault diagnosis. Hong, H.S. & Thuan, N. (2023), Mendeley Data,
v3, DOI 10.17632/cbv7jyx4p9.3, licence CC BY 4.0 (verified via the Mendeley public API, 28 Sep 2026). Paper: BMC
Research Notes 16:138 (2023), doi 10.1186/s13104-023-06400-4.
99 .mat files, 51.2 kHz, 10 s each, accelerometer PCB 325C33 vertical; 5 bearing types (6204-6208) x 3 loads
(0/200/400 W) x {N normal, I inner, O outer, B ball, IB, IO, OB}. File name: <fault><bearing digit><load digit>,
e.g. I402 = inner race, 6204, 200 W. Variables: data (vibration), fs (SHAFT frequency, Hz), rpm, ru, ru_raw.
Each file is checked against the SHA-256 the repository publishes.
Run: .venv\\Scripts\\python.exe data\\fetch_hust.py
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import httpx

DATASET, VERSION, FOLDER = "cbv7jyx4p9", 3, "72793f17-cac2-4cc6-ac18-2b047e0d45f8"
OUT = pathlib.Path(__file__).resolve().parent / "raw" / "hust"
API = f"https://data.mendeley.com/public-api/datasets/{DATASET}/files?folder_id={FOLDER}&version={VERSION}"


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = httpx.get(API, timeout=60).raise_for_status().json()
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
