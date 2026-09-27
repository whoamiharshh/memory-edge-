"""Device footprint: disk, RAM and search quality for the Edge storage options (phone / laptop / Pi feasibility).

For each layout and N points (episode-like: 27-d fingerprint, 384-d bge-small note vector of real logbook text,
Edge BM25 of that text, small payload):
  apparent bytes   the sum of file sizes (what Explorer shows as "Size")
  allocated bytes  what the files really occupy on disk. On Windows via GetCompressedFileSizeW (sparse/compressed
                   files report less); on Linux/macOS/Android st_blocks * 512. Qdrant Edge pre-allocates storage
                   pages; if the OS keeps them sparse they cost address space but not disk.
  rss_mb           process memory after opening the shard and running queries (fresh subprocess per cell)
  hybrid p50 ms    vib + note + BM25 RRF query, 100 queries
  top10 overlap    agreement of the hybrid top-10 with the full-precision default layout (quality kept?)
Layouts: default (as shipped) | default in an OS-compressed folder (edge/storage_os.py) | 1 segment | 1 segment + float16 note | 1 segment + int8-quantized note (originals
on disk) + payload on disk.
Run: .venv\\Scripts\\python.exe -m bench.footprint
"""
from __future__ import annotations

import csv
import ctypes
import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
LAYOUTS = {"default": {}, "default_os_compressed": {"os_compress": True}, "one_segment": {"segments": 1},
           "one_segment_f16": {"segments": 1, "note_datatype": "float16"},
           "one_segment_int8_ondisk": {"segments": 1, "quantize_note": True, "payload_on_disk": True}}
SIZES = (0, 1000, 10000, 50000)
ONLY = None      # e.g. ["default", "default_os_compressed"] to re-run a subset


def allocated(path: pathlib.Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        if os.name == "nt":
            high = ctypes.c_ulong(0)
            low = ctypes.windll.kernel32.GetCompressedFileSizeW(str(f), ctypes.byref(high))
            total += (high.value << 32) + low
        else:
            total += f.stat().st_blocks * 512
    return total


def apparent(path: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def build(layout: str, n: int, root: pathlib.Path, vec_file: str) -> None:
    """Runs in a fresh process: create, fill, measure, query; prints one JSON line."""
    import time

    import psutil

    from edge.store_edge import EdgeStore, StorePoint
    from shared import ids
    data = np.load(vec_file, allow_pickle=True)
    texts, vecs, qv, qt = data["texts"], data["vecs"], data["qv"], data["qt"]
    rng = np.random.default_rng(0)
    layout_cfg = {k: v for k, v in LAYOUTS[layout].items() if k != "os_compress"}
    s = EdgeStore(root, storage=layout_cfg)
    for a in range(0, n, 1000):
        s.upsert([StorePoint(ids.make_id("p", i), {"type": "episode", "status": "closed", "component": "bearing"},
                             vib=rng.normal(size=27).tolist(), note=vecs[i % len(vecs)].tolist(),
                             bm25_text=str(texts[i % len(texts)])) for i in range(a, min(n, a + 1000))])
    s.optimize()
    s.close()
    if LAYOUTS[layout].get("os_compress"):          # as the device does it (start-up + hourly): compress existing files
        from edge.storage_os import enable_compression
        enable_compression(root)
    s = EdgeStore(root, storage=layout_cfg)
    lat, tops = [], []
    for k in range(len(qv)):
        t = time.perf_counter()
        hits = s.search(vib=rng.normal(size=27).tolist(), note=qv[k].tolist(), text=str(qt[k]), limit=10) if n else []
        lat.append((time.perf_counter() - t) * 1000)
        tops.append([h.id for h in hits])
    rss = psutil.Process().memory_info().rss / 2 ** 20
    s.close()
    print(json.dumps({"layout": layout, "n": n, "apparent_mb": round(apparent(root) / 2 ** 20, 1),
                      "allocated_mb": round(allocated(root) / 2 ** 20, 1), "rss_mb": round(rss, 1),
                      "hybrid_p50_ms": round(float(np.percentile(lat, 50)), 3), "tops": tops}))


def main() -> dict:
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        build(sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4]), sys.argv[5])
        return {}
    from shared.embed import BgeEmbedder
    f = glob.glob(str(ROOT / "data" / "raw" / "logbook" / "*.csv"))[0]
    texts = [r[1] for r in list(csv.reader(open(f, encoding="utf-8", errors="replace")))[1:] if r[1].strip()]
    emb = BgeEmbedder()
    vecs = np.asarray(emb.embed_documents(texts[:3000]), dtype=np.float32)
    rng = np.random.default_rng(1)
    qt = [texts[i] for i in rng.choice(len(texts), 100, replace=False)]
    qv = np.asarray([emb.embed_query(t) for t in qt], dtype=np.float32)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="footprint_"))
    vec_file = str(tmp / "vecs.npz")
    np.savez(vec_file, texts=np.array(texts[:3000], dtype=object), vecs=vecs, qv=qv, qt=np.array(qt, dtype=object))
    rows, ref = [], {}
    for n in SIZES:
        for layout in (ONLY or LAYOUTS):
            r = subprocess.run([sys.executable, "-m", "bench.footprint", "--one", layout, str(n), str(tmp / f"{layout}_{n}"),
                                vec_file], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                rows.append({"layout": layout, "n": n, "error": r.stderr.strip().splitlines()[-1][:300]})
                print(rows[-1], flush=True)
                continue
            row = json.loads(r.stdout.strip().splitlines()[-1])
            tops = row.pop("tops")
            if layout == "default":
                ref[n] = tops
            row["top10_overlap_vs_default"] = (round(float(np.mean([len(set(a) & set(b)) / max(1, len(a))
                                                                    for a, b in zip(tops, ref[n])])), 3) if n else None)
            rows.append(row)
            print(row, flush=True)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    out = {"method": __doc__.strip().splitlines()[0], "os": os.name, "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "footprint.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
