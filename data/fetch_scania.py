"""Download the SCANIA Component X validation split into data/raw/scania/ (git-ignored, not redistributed).

SCANIA Component X dataset, Scania CV AB, Swedish National Data Service, DOI 10.5878/jvb5-d390, version 3,
licence CC BY 4.0 (checked 28 Sep 2026). Paper: Scientific Data (2025), doi 10.1038/s41597-025-04802-6.
Real trucks: operational readouts over time (anonymised counters and histograms of one engine component), truck
specifications, and labels from workshop repair records. The training split (23,550 trucks, ~1.2 GB) carries
time-to-event labels (train_tte.csv: study length and whether Component X was repaired); validation and test carry a
class for each truck's last readout. bench/vehicle_scania.py trains on train (+ validation for model choice) and tests
on the test trucks. Pass --no-train to skip the 1.2 GB training readouts.
Run: .venv\\Scripts\\python.exe data\\fetch_scania.py
"""
import pathlib

import httpx

OUT = pathlib.Path(__file__).resolve().parent / "raw" / "scania"
API = "https://api.researchdata.se/dataset/2024-34/3/file/data?filePath="
FILES = ["validation_labels.csv", "validation_specifications.csv", "validation_operational_readouts.csv",
         "test_labels.csv", "test_specifications.csv", "test_operational_readouts.csv"]
TRAIN = ["train_tte.csv", "train_specifications.csv", "train_operational_readouts.csv"]


def fetch(c: httpx.Client, name: str, tries: int = 12) -> None:
    """Resumable download (HTTP Range): the 1.2 GB training file is often cut off by the server."""
    import time
    dest, tmp = OUT / name, OUT / (name + ".part")
    for attempt in range(tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with c.stream("GET", API + name, headers=headers) as r:
                if r.status_code == 416:                   # already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            break
        except httpx.HTTPError as e:
            print(f"{name}: {type(e).__name__} after {tmp.stat().st_size if tmp.exists() else 0:,} bytes; "
                  f"resuming ({attempt + 1}/{tries})", flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit(f"{name}: gave up after {tries} attempts")
    tmp.replace(dest)
    print(f"{name}: {dest.stat().st_size:,} bytes", flush=True)


def main() -> None:
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(120, read=300), follow_redirects=True) as c:
        for name in FILES + ([] if "--no-train" in sys.argv else TRAIN):
            dest = OUT / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            fetch(c, name)


if __name__ == "__main__":
    main()
