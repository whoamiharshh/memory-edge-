"""Resources of one edge device on THIS laptop (docs/RESEARCH.md Part O, "Resources").

  memory   resident set size (psutil RSS) after each stage: imports -> bge-small loaded -> device + baseline ->
           local LLM loaded (Qwen2.5-1.5B Q4 GGUF, optional)
  cpu      the real per-window path in REAL TIME: raw CWRU signal (healthy 99, then fault 105) -> DSP fingerprint
           -> novelty gate (-> episode writes), one window every HOP/FS = 170.7 ms, for DURATION_S seconds.
           CPU % = process CPU time / wall time (100 % = one full core of the 12 logical CPUs).
           Also: per-window processing time p50/p95 and the real-time headroom (window period / p95).
  disk     device folder after the run; model files in models_cache/
Numbers are for this laptop only (i5-1335U, 15.7 GB RAM, no GPU, Windows 11).
Run: .venv\\Scripts\\python.exe -m bench.resources
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import time

import numpy as np
import psutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
RAW = ROOT / "data" / "raw" / "cwru"
DURATION_S = 60.0
MB = 1024 * 1024


def rss() -> float:
    return round(psutil.Process().memory_info().rss / MB, 1)


def dir_mb(p: pathlib.Path) -> float:
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / MB, 1) if p.exists() else 0.0


def main() -> dict:
    proc = psutil.Process()
    mem = {"python_start": rss()}
    from edge.device import Device, DeviceConfig
    from edge.fingerprint import FS, HOP, features, load_cwru, windows
    from edge.replay import Recordings
    mem["after_imports"] = rss()
    from shared.embed import BgeEmbedder
    emb = BgeEmbedder()
    emb.embed_query("warm up")
    mem["after_bge_small"] = rss()
    root = pathlib.Path(tempfile.mkdtemp(prefix="res_"))
    out: dict = {"method": __doc__.strip().splitlines()[0], "machine": "Intel i5-1335U, 15.7 GB RAM, no GPU, Windows 11"}
    try:
        dev = Device(DeviceConfig("bench", "s", "m", root / "dev"), emb)
        dev.fit_baseline(Recordings.baseline())
        mem["after_device_and_baseline"] = rss()

        stream = []
        from data.fetch_data import CWRU
        from data.splits import NOMINAL_RPM
        for fid in (99, 105):
            x, rpm = load_cwru(RAW / f"{fid}.mat")
            if not np.isfinite(rpm) or rpm <= 0:              # same fallback as the feature cache (data/splits.py)
                rpm = NOMINAL_RPM[CWRU[fid][2]]
            stream += [(w, rpm) for w in windows(x)]
        period = HOP / FS
        n_target = int(DURATION_S / period)
        per = []
        cpu0, wall0 = proc.cpu_times(), time.perf_counter()
        next_t = wall0
        for i in range(n_target):
            w, rpm = stream[i % len(stream)]
            t = time.perf_counter()
            dev.ingest_window(features(w, rpm=rpm))
            per.append((time.perf_counter() - t) * 1000)
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
        wall = time.perf_counter() - wall0
        cpu1 = proc.cpu_times()
        cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
        p95 = float(np.percentile(per, 95))
        out["cpu_realtime"] = {
            "windows": n_target, "wall_s": round(wall, 1), "window_period_ms": round(period * 1000, 1),
            "cpu_percent_of_one_core": round(100 * cpu_s / wall, 1),
            "per_window_ms_p50": round(float(np.percentile(per, 50)), 2), "per_window_ms_p95": round(p95, 2),
            "realtime_headroom_x": round(period * 1000 / p95, 1), "episodes": len(dev.episodes()),
            "states": {k: v for k, v in dev.counters.items() if k != "windows"}}
        mem["after_realtime_run"] = rss()
        dev.close()
        out["disk_mb"] = {"device_folder_after_run": dir_mb(root / "dev"),
                          "local_shard": dir_mb(root / "dev" / "local"), "mirror_shard_empty": dir_mb(root / "dev" / "mirror")}
    finally:
        shutil.rmtree(root, ignore_errors=True)

    from edge import rag
    llm = rag.LocalLLM()
    if llm.available:
        res = {"fleet": [], "local": [{"id": "e", "episode": {"seq": 1, "component": "bearing", "fault_class": "inner_race",
                                                              "occurrences": 3, "action_code": "lubricate",
                                                              "outcome": "failed", "verify": {"verdict": "symptom_persists"}}}]}
        rag.brief("What has been tried?", res, llm)
        mem["after_local_llm"] = rss()
    out["memory_rss_mb"] = mem
    mc = ROOT / "models_cache"
    out["disk_mb"]["models_cache_total"] = dir_mb(mc)
    OUT.mkdir(exist_ok=True)
    (OUT / "resources.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
