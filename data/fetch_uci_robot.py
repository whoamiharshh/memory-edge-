"""Download the UCI Robot Execution Failures dataset into data/raw/uci_robot/ (git-ignored, not redistributed).

Lopes, L. & Camarinha-Matos, L. (1998). Robot Execution Failures. UCI Machine Learning Repository,
DOI 10.24432/C5M89N, licence CC BY 4.0 (stated on the UCI page, checked 28 Sep 2026).
463 instances in 5 learning problems (lp1-lp5); each instance = a class label and 15 samples x 6 channels
(Fx Fy Fz Tx Ty Tz) taken right after a failure was detected (315 ms).
Run: .venv\\Scripts\\python.exe data\\fetch_uci_robot.py
"""
import io
import pathlib
import zipfile

import httpx

URL = "https://archive.ics.uci.edu/static/public/138/robot+execution+failures.zip"
OUT = pathlib.Path(__file__).resolve().parent / "raw" / "uci_robot"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = httpx.get(URL, timeout=60, follow_redirects=True).raise_for_status().content
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(OUT)
    print("extracted:", sorted(p.name for p in OUT.rglob("*") if p.is_file()))


def load(lp: str) -> list[tuple[str, list[list[float]]]]:
    """[(label, 15x6 samples), ...] for one learning problem file (e.g. 'lp1')."""
    f = next(OUT.rglob(f"{lp}.data"))
    out, label, rows = [], None, []
    for line in f.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 1 and not parts[0].lstrip("-").isdigit():
            if label is not None:
                out.append((label, rows))
            label, rows = parts[0], []
        else:
            rows.append([float(v) for v in parts])
    if label is not None:
        out.append((label, rows))
    return out


if __name__ == "__main__":
    main()
    for lp in ("lp1", "lp2", "lp3", "lp4", "lp5"):
        inst = load(lp)
        from collections import Counter
        print(lp, len(inst), Counter(lbl for lbl, _ in inst), "rows/instance", Counter(len(r) for _, r in inst))
