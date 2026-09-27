"""Download the SCANIA Component X validation split into data/raw/scania/ (git-ignored, not redistributed).

SCANIA Component X dataset, Scania CV AB, Swedish National Data Service, DOI 10.5878/jvb5-d390, version 3,
licence CC BY 4.0 (checked 28 Sep 2026). Paper: Scientific Data (2025), doi 10.1038/s41597-025-04802-6.
Real trucks: operational readouts over time (anonymised counters and histograms of one engine component), truck
specifications, and labels from workshop repair records. We train on the validation split and test on the test split (different trucks, ~216 MB each); the 1.2 GB
training readouts are not needed.
Run: .venv\\Scripts\\python.exe data\\fetch_scania.py
"""
import pathlib

import httpx

OUT = pathlib.Path(__file__).resolve().parent / "raw" / "scania"
API = "https://api.researchdata.se/dataset/2024-34/3/file/data?filePath="
FILES = ["validation_labels.csv", "validation_specifications.csv", "validation_operational_readouts.csv",
         "test_labels.csv", "test_specifications.csv", "test_operational_readouts.csv"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=600, follow_redirects=True) as c:
        for name in FILES:
            dest = OUT / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            tmp = dest.with_suffix(".part")
            with c.stream("GET", API + name) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            print(f"{name}: {dest.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
