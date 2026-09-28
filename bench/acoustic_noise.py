"""Phone microphone with a LOUDER NEIGHBOURING MACHINE in the room: false alarms and detection, measured.

Signals: the University of Ottawa microphone recordings of bench/acoustic_uottawa.py (20 bearings, healthy /
developing / faulty, 42 kHz) PLUS the microphone of a different, running machine: MaFaulDa's SpectraQuest simulator
in normal operation (UFRJ; data/fetch_mafaulda.py), resampled 50 -> 42 kHz. Both are REAL recordings; they are added
sample by sample (sound pressures add), which is how two machines reach one microphone. Nothing is trained or tuned:
shipped profile `acoustic`, shipped thresholds.
Loudness: the neighbour is scaled so that its RMS is `snr_db` below (+) or above (-) the RMS of the bearing's own
healthy recording: +10 dB (quieter neighbour), 0 dB (as loud), -10 dB (3.2x louder).
Scenarios per bearing and loudness:
  quiet       no neighbour (reference, = bench/acoustic_uottawa.py)
  steady      the neighbour always runs at one speed, also while the device learns "healthy"
  starts      the device learned "healthy" in a quiet room; the neighbour starts afterwards
  changes     the neighbour ran at one speed while the device learned; later it runs at another speed
  starts+teach  as "starts", then the technician's ONE "normal operation" confirmation on the first 10 windows with the
              neighbour running (the product's mark_normal, as in bench/hust_holdout.py); measured on the windows after
Measured: false-alarm share on the unseen half of the healthy recording; share of the first 15 developing / faulty
windows flagged (the neighbour present in the same way as in the healthy test).
Run: .venv\\Scripts\\python.exe -m bench.acoustic_noise
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import numpy as np
from scipy.signal import resample_poly

from bench.acoustic_uottawa import FS, N, bearings, load
from bench.hust_holdout import TEACH, teach
from edge import profiles
from edge.device import Device, DeviceConfig
from shared.embed import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAF = ROOT / "data" / "raw" / "mafaulda" / "normal"
OUT = ROOT / "bench" / "results"
MIC_COL = 7                 # MaFaulDa channel 8: microphone
SNRS = (10, 0, -10)


def neighbour(i: int, n: int) -> np.ndarray:
    """n samples at 42 kHz of the neighbour running at speed index i (two adjacent-speed recordings joined)."""
    import pandas as pd
    files = sorted(MAF.glob("*.csv"), key=lambda p: float(p.stem))
    parts = []
    for f in (files[i % len(files)], files[(i + 1) % len(files)]):
        x = pd.read_csv(f, header=None, usecols=[MIC_COL], dtype=np.float64, engine="c").to_numpy()[:, 0]
        parts.append(resample_poly(x - x.mean(), 21, 25))          # 50 kHz -> 42 kHz
    return np.resize(np.concatenate(parts), n)


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def windows(prof, x) -> np.ndarray:
    return np.stack([prof.features(w, prof.analysis_fs) for w in prof.windows(x, FS)])


def flagged_share(base: np.ndarray, test: np.ndarray, limit: int | None = None,
                  taught: np.ndarray | None = None) -> float:
    root = pathlib.Path(tempfile.mkdtemp(prefix="noise_"))
    d = Device(DeviceConfig("bench", "s", "m", root, profile="acoustic", profile_params={"bearing": "6203-UO"}),
               HashEmbedder())
    try:
        d.fit_baseline(base)
        if taught is not None:
            teach(d, taught)
        W = test[:limit] if limit else test
        return sum(d.ingest_window(w)["state"] != "normal" for w in W) / len(W)
    finally:
        d.close()
        shutil.rmtree(root, ignore_errors=True)


def main() -> dict:
    prof = profiles.make("acoustic", bearing="6203-UO")
    B = bearings()
    speeds = len(list(MAF.glob("*.csv")))
    res = {"method": __doc__.strip().splitlines()[0], "protocol": __doc__.split("Signals:")[1].split("Run:")[0].strip(),
           "bearings": len(B), "neighbour_speeds_available": speeds, "by_snr_db": {}}
    for snr in SNRS:
        acc = {s: {"false_alarm": [], "developing": [], "faulty": []} for s in ("quiet", "steady", "starts", "changes",
                                                                        "starts+teach")}
        for k, (b, kind) in enumerate(B.items()):
            H, _ = load(f"H_{b}_0", 1)
            D1, _ = load(f"{kind}_{b}_1", 1)
            D2, _ = load(f"{kind}_{b}_2", 1)
            half = len(H) // 2
            g = rms(H) / 10 ** (snr / 20)
            i1, i2 = (3 * k) % speeds, (3 * k + speeds // 2) % speeds      # two different neighbour speeds
            n1 = neighbour(i1, len(H))
            n1 = n1 / rms(n1) * g
            n2 = neighbour(i2, len(H))
            n2 = n2 / rms(n2) * g

            def add(x, n, off):                                             # another stretch of the same run
                return x + np.roll(n, off)[:len(x)]

            clean = {"base": windows(prof, H[:half]), "test": windows(prof, H[half:]),
                     "developing": windows(prof, D1), "faulty": windows(prof, D2)}
            steady = {"base": windows(prof, H[:half] + n1[:half]), "test": windows(prof, H[half:] + n1[half:]),
                      "developing": windows(prof, add(D1, n1, len(H) // 3)),
                      "faulty": windows(prof, add(D2, n1, 2 * len(H) // 3))}
            changed = {"test": windows(prof, H[half:] + n2[half:]),
                       "developing": windows(prof, add(D1, n2, len(H) // 3)),
                       "faulty": windows(prof, add(D2, n2, 2 * len(H) // 3))}
            cases = {"quiet": (clean["base"], clean), "steady": (steady["base"], steady),
                     "starts": (clean["base"], steady), "changes": (steady["base"], changed)}
            for name, (base, sig) in cases.items():
                if name == "quiet" and snr != SNRS[0]:
                    continue                                                # the quiet room does not depend on SNR
                acc[name]["false_alarm"].append(flagged_share(base, sig["test"]))
                acc[name]["developing"].append(flagged_share(base, sig["developing"], N))
                acc[name]["faulty"].append(flagged_share(base, sig["faulty"], N))
            t = steady["test"]                                              # the neighbour runs; taught once
            acc["starts+teach"]["false_alarm"].append(flagged_share(clean["base"], t[TEACH + 1:], taught=t))
            acc["starts+teach"]["developing"].append(flagged_share(clean["base"], steady["developing"], N, taught=t))
            acc["starts+teach"]["faulty"].append(flagged_share(clean["base"], steady["faulty"], N, taught=t))
            print(f"snr {snr:+d} dB bearing {b}: " + ", ".join(
                f"{s} FA {acc[s]['false_alarm'][-1]:.2f}" for s in acc if acc[s]["false_alarm"]), flush=True)
        res["by_snr_db"][f"{snr:+d}"] = {
            s: {"false_alarm_window_share": round(float(np.mean(v["false_alarm"])), 3),
                "bearings_with_any_false_alarm": int(sum(x > 0 for x in v["false_alarm"])),
                "developing_flagged": round(float(np.mean(v["developing"])), 3),
                "faulty_flagged": round(float(np.mean(v["faulty"])), 3)}
            for s, v in acc.items() if v["false_alarm"]}
        print(json.dumps(res["by_snr_db"][f"{snr:+d}"]), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "acoustic_noise.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
