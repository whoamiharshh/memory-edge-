"""Download Loghub HDFS_v1 (real system logs with per-session anomaly labels) into data/raw/loghub/ (git-ignored, not
redistributed). Used to measure the `events` profile (kiosks / apps / vehicles: codes per bucket) on REAL logs.

Loghub: He, S., Zhu, J., He, P., Lyu, M.R., "Loghub: A Large Collection of System Log Datasets for AI-driven Log
Analytics", ISSRE 2023. Zenodo record 8196385, DOI 10.5281/zenodo.8196385, licence CC BY 4.0 (Zenodo metadata, checked
28 Sep 2026). HDFS_v1: 11,175,629 log lines from a 203-node Hadoop cluster, grouped into 575,061 block sessions, each
labelled Normal/Anomaly by the Hadoop domain experts of the original study (Xu et al., SOSP 2009).
Run: .venv\\Scripts\\python.exe -m data.fetch_loghub
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import zipfile

import httpx

OUT = pathlib.Path(__file__).resolve().parent / "raw" / "loghub"
RECORD = "https://zenodo.org/api/records/8196385"
FILE = "HDFS_v1.zip"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = next(f for f in httpx.get(RECORD, timeout=60).json()["files"] if f["key"] == FILE)
    dest = OUT / FILE
    algo, want = meta["checksum"].split(":", 1)
    if not dest.exists() or hashlib.new(algo, dest.read_bytes()).hexdigest() != want:
        with httpx.Client(timeout=600, follow_redirects=True) as c, c.stream("GET", meta["links"]["self"]) as r:
            r.raise_for_status()
            with open(dest.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        dest.with_suffix(".part").replace(dest)
        if hashlib.new(algo, dest.read_bytes()).hexdigest() != want:
            print("CHECKSUM MISMATCH")
            return 1
    with zipfile.ZipFile(dest) as z:
        for n in z.namelist():
            if n.endswith((".csv",)) or "preprocessed" in n.lower():
                z.extract(n, OUT)
        print("\n".join(f"{i.filename} {i.file_size}" for i in z.infolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
