"""Download the real datasets used by this project (not redistributed in the repo).

CWRU Bearing Data Center (https://engineering.case.edu/bearingdatacenter): 12k drive-end faults + normal baseline.
Annotated Maintenance Logbook (Zenodo record 17903357, CC BY 4.0): real problem/action maintenance text.
Files already present are skipped, so the script is safe to re-run.
"""
import json, pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent / "raw"
CWRU_BASE = "https://engineering.case.edu/sites/default/files/{}.mat"

# file id -> (fault_class, fault_size_in_mils, load_hp). Each (class, size) is one physically distinct seeded bearing.
CWRU = {
    97: ("normal", 0, 0), 98: ("normal", 0, 1), 99: ("normal", 0, 2), 100: ("normal", 0, 3),
    105: ("inner_race", 7, 0), 106: ("inner_race", 7, 1), 107: ("inner_race", 7, 2), 108: ("inner_race", 7, 3),
    169: ("inner_race", 14, 0), 170: ("inner_race", 14, 1), 171: ("inner_race", 14, 2), 172: ("inner_race", 14, 3),
    209: ("inner_race", 21, 0), 210: ("inner_race", 21, 1), 211: ("inner_race", 21, 2), 212: ("inner_race", 21, 3),
    118: ("ball", 7, 0), 119: ("ball", 7, 1), 120: ("ball", 7, 2), 121: ("ball", 7, 3),
    185: ("ball", 14, 0), 186: ("ball", 14, 1), 187: ("ball", 14, 2), 188: ("ball", 14, 3),
    222: ("ball", 21, 0), 223: ("ball", 21, 1), 224: ("ball", 21, 2), 225: ("ball", 21, 3),
    130: ("outer_race", 7, 0), 131: ("outer_race", 7, 1), 132: ("outer_race", 7, 2), 133: ("outer_race", 7, 3),
    197: ("outer_race", 14, 0), 198: ("outer_race", 14, 1), 199: ("outer_race", 14, 2), 200: ("outer_race", 14, 3),
    234: ("outer_race", 21, 0), 235: ("outer_race", 21, 1), 236: ("outer_race", 21, 2), 237: ("outer_race", 21, 3),
}

def fetch(url: str, dest: pathlib.Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)  # atomic: a crash never leaves a half file that looks complete
    print("downloaded", dest.name, dest.stat().st_size)

LOGBOOK_RECORD = "https://zenodo.org/api/records/17903357"

def fetch_logbook() -> None:
    meta = json.loads(urllib.request.urlopen(LOGBOOK_RECORD, timeout=60).read())
    for f in meta.get("files", []):
        name = f["key"]
        if name.lower().endswith(".csv"):
            fetch(f["links"]["self"], ROOT / "logbook" / name)
    (ROOT / "logbook" / "SOURCE.txt").write_text(
        "Annotated Maintenance Logbook, Zenodo record 17903357, license CC BY 4.0 "
        f"({meta.get('metadata', {}).get('license', {})}). Derived from MaintNet (Akhbardeh et al.).\n")

def main() -> None:
    for fid in CWRU:
        fetch(CWRU_BASE.format(fid), ROOT / "cwru" / f"{fid}.mat")
    (ROOT / "cwru" / "SOURCE.txt").write_text(
        "Case Western Reserve University Bearing Data Center, https://engineering.case.edu/bearingdatacenter . "
        "Not redistributed; downloaded by data/fetch_data.py.\n")
    fetch_logbook()
    print("done")

if __name__ == "__main__":
    main()
