"""Storage: what the novelty gate saves (docs/RESEARCH.md Part O, "Storage").

A simulated shift on one machine, built from real CWRU fingerprints: mostly healthy running (unseen-load healthy
files 99/100), with a fault recording every CYCLE (rotating through all 36), each followed by healthy running again.
Windows are 4096 samples with 50 % overlap at 12 kHz, so one window = 2048/12000 s of machine time.

  gate ON   the real Device: healthy windows are counted only; abnormal windows open/merge episodes
  gate OFF  the naive alternative: every window's fingerprint is stored as a point (batched upserts)
Reported: points stored, and shard bytes on disk. Caveat: Qdrant Edge pre-allocates storage pages (32 MB chunks,
see spike/spike_snapshot.py), so on-disk bytes are dominated by fixed allocation until the point count is large;
points stored is the honest growth measure.
Run: .venv\\Scripts\\python.exe -m bench.storage [hours]
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import time

from data.fetch_data import CWRU
from edge.device import Device, DeviceConfig
from edge.fingerprint import FS, HOP, Baseline
from edge.replay import HEALTHY_REPLAY_FILES, Recordings
from edge.store_edge import EdgeStore, StorePoint
from shared import ids
from shared.embed import HashEmbedder

OUT = pathlib.Path(__file__).resolve().parent / "results"
HEALTHY_BETWEEN = 6            # healthy recordings between two faults (~1 min of running)
FAULTS = sorted(f for f, v in CWRU.items() if v[0] != "normal")


def dir_bytes(p: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def stream(n_windows: int):
    """(fid, window) pairs: healthy x HEALTHY_BETWEEN, one fault recording, repeat."""
    out, k = 0, 0
    while True:
        for i in range(HEALTHY_BETWEEN):
            for w in Recordings.windows(HEALTHY_REPLAY_FILES[i % 2]):
                yield 0, w
                out += 1
                if out >= n_windows:
                    return
        for w in Recordings.windows(FAULTS[k % len(FAULTS)]):
            yield 1, w
            out += 1
            if out >= n_windows:
                return
        k += 1


def main(hours: float = 1.0) -> dict:
    n = int(hours * 3600 * FS / HOP)
    base = Recordings.baseline()
    root = pathlib.Path(tempfile.mkdtemp(prefix="storage_"))
    try:
        dev = Device(DeviceConfig("bench", "s", "m", root / "on"), HashEmbedder())
        dev.fit_baseline(base)
        after_baseline = dev.store.count()
        t = time.perf_counter()
        states = {"normal": 0, "merge": 0, "new": 0}
        faults = 0
        for is_fault, w in stream(n):
            states[dev.ingest_window(w)["state"]] += 1
            faults += is_fault
        on_s = time.perf_counter() - t
        on = {"points": dev.store.count(), "points_after_baseline": after_baseline,
              "by_type": dev.store.facet("type"), "episodes": len(dev.episodes()), "states": states,
              "shard_bytes": dir_bytes(root / "on" / "local"), "ingest_seconds": round(on_s, 1)}
        dev.close()

        off_store = EdgeStore(root / "off")
        zfit = Baseline.fit(base)                          # same z-scoring as the device, so the same 27-d vector
        t = time.perf_counter()
        batch, i = [], 0
        for _, w in stream(n):
            batch.append(StorePoint(ids.make_id("win", i), {"type": "window", "machine_id": "m"},
                                    vib=zfit.z(w).tolist()))
            i += 1
            if len(batch) == 256:
                off_store.upsert(batch)
                batch = []
        if batch:
            off_store.upsert(batch)
        off = {"points": off_store.count(), "shard_bytes": dir_bytes(root / "off"),
               "ingest_seconds": round(time.perf_counter() - t, 1)}
        off_store.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    out = {"method": __doc__.strip().splitlines()[0], "hours_simulated": hours, "windows": n,
           "fault_windows": faults, "gate_on": on, "gate_off": off,
           "points_ratio_off_vs_on": round(off["points"] / on["points"], 1)}
    OUT.mkdir(exist_ok=True)
    (OUT / "storage.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
