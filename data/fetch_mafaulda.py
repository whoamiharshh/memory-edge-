"""Download MaFaulDa (Machinery Fault Database, UFRJ) parts into data/raw/mafaulda/ (git-ignored, not redistributed).

ONE machine - a SpectraQuest Machinery Fault Simulator - recorded in normal operation, with rotor imbalance (6-35 g)
and with horizontal (0.5-2.0 mm) and vertical (0.51-1.90 mm) shaft misalignment, at 49 speeds (737-3686 rpm).
50 kHz, 5 s, 8 channels: tachometer, underhang accelerometer (axial, radial, tangential), overhang accelerometer
(axial, radial, tangential), microphone. Source: https://www02.smt.ufrj.br/~offshore/mfs/page_01.html (checked
28 Sep 2026). No licence text is published there: used for evaluation only, not redistributed, credited to the
authors (Ribeiro et al., UFRJ SMT). Because every fault is on the SAME machine, it can test how imbalance and
misalignment are NAMED without the one-motor-per-fault confound of the University of Ottawa motor set.
Run: .venv\\Scripts\\python.exe -m data.fetch_mafaulda [normal imbalance horizontal-misalignment vertical-misalignment]
"""
from __future__ import annotations

import pathlib
import sys
import tarfile
import time

import httpx

OUT = pathlib.Path(__file__).resolve().parent / "raw" / "mafaulda"
BASE = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/{}.tgz"
PARTS = ["normal", "imbalance", "horizontal-misalignment", "vertical-misalignment"]


def fetch(part: str, tries: int = 20) -> pathlib.Path:
    dest, tmp = OUT / f"{part}.tgz", OUT / f"{part}.tgz.part"
    if dest.exists():
        return dest
    with httpx.Client(timeout=httpx.Timeout(60, read=300), follow_redirects=True) as c:
        for attempt in range(tries):
            have = tmp.stat().st_size if tmp.exists() else 0
            try:
                with c.stream("GET", BASE.format(part), headers={"Range": f"bytes={have}-"} if have else {}) as r:
                    if r.status_code == 416:
                        break
                    r.raise_for_status()
                    with open(tmp, "ab" if have and r.status_code == 206 else "wb") as f:
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                break
            except httpx.HTTPError as e:
                print(f"{part}: {type(e).__name__} at {tmp.stat().st_size if tmp.exists() else 0:,} bytes, resuming "
                      f"({attempt + 1}/{tries})", flush=True)
                time.sleep(min(60, 5 * (attempt + 1)))
        else:
            raise SystemExit(f"{part}: gave up")
    tmp.replace(dest)
    print(f"{part}: {dest.stat().st_size:,} bytes", flush=True)
    return dest


def extract(tgz: pathlib.Path) -> None:
    mark = tgz.with_suffix(".extracted")
    if mark.exists():
        return
    with tarfile.open(tgz) as t:
        t.extractall(OUT, filter="data")
    mark.write_text("ok")


def main(parts: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in parts or PARTS:
        extract(fetch(p))
        print(f"{p}: extracted", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
