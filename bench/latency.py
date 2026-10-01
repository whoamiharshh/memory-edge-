"""Latency + bandwidth on THIS laptop (i5-1335U, 16 GB, no GPU, Windows 11). Method: wall-clock time.perf_counter
around each operation, warm-up excluded, p50/p95 over N repetitions. Numbers are for this machine only.

  fingerprint      edge.fingerprint.features() on one 4096-sample window of a real CWRU file (with rpm)
  embed_note       bge-small (FastEmbed, CPU) on one maintenance-log problem text
  durable_write    EdgeStore.upsert of one point incl. flush() (the K4 rule)
  gate             Device.ingest_window() state decision query against 86 baseline points (no episode write)
  hybrid_{n}       EdgeStore.search(vib+note+text, RRF) with n points in the shard (n = 1k, 10k, 50k)
  qt_{type}_{n}    the four device query types (novelty_gate, same_machine_recall, technician_hybrid, fleet_evidence)
                   at n = 1k, 10k, 50k points; "qdrant" = the Edge call with the query vectors ready, "e2e" = the
                   fingerprint (vib types) or the bge-small query embedding (text types) plus the same call
  server_{n}       the same dense note query against the local Qdrant Server over HTTP (if it is running)
  llm_brief        edge.rag.brief() with the local LLM on a 2-item evidence set
  bandwidth        JSON bytes of one shared event vs the raw float32 signal it summarises
Run: .venv\\Scripts\\python.exe -m bench.latency
"""
from __future__ import annotations

import json
import pathlib
import random
import shutil
import tempfile
import time

import numpy as np

from edge import rag
from edge.fingerprint import DIM, HOP, features, load_cwru, windows
from edge.store_edge import EdgeStore, StorePoint
from edge.verifier import REQUIRED_WINDOWS
from shared import ids
from shared.embed import BgeEmbedder

OUT = pathlib.Path(__file__).resolve().parent / "results"
RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "cwru"


def timeit(fn, n: int, warm: int = 3) -> dict:
    for _ in range(warm):
        fn()
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(xs, 50)), 3), "p95_ms": round(float(np.percentile(xs, 95)), 3), "n": n}


def query_types(emb: BgeEmbedder, texts: list[str], vec_cache: list, W, rpm) -> dict:
    """p50/p95 for each of the four EdgeStore query types, Qdrant-only and end-to-end, at 1k/10k/50k points."""
    rng = np.random.default_rng(3)
    res: dict = {}
    for n in (1000, 10000, 50000):
        root = pathlib.Path(tempfile.mkdtemp(prefix="qt_"))
        try:
            dev, mir = EdgeStore(root / "dev"), EdgeStore(root / "mirror")
            comps, fcs = ("bearing", "gear", "fan"), ("inner", "outer", "ball", "imbalance")
            pts = [StorePoint(ids.make_id("b", i), {"type": "baseline", "machine_id": "m1"}, vib=rng.normal(size=DIM).tolist())
                   for i in range(80)]
            pts += [StorePoint(ids.make_id("x", i), {"type": "exemplar", "machine_id": "m1", "episode_id": str(i % 50)},
                               vib=rng.normal(size=DIM).tolist()) for i in range(n // 2)]
            pts += [StorePoint(ids.make_id("e", i), {"type": "episode", "machine_id": f"m{i % 5}", "component": comps[i % 3],
                                                    "fault_class": fcs[i % 4], "note": texts[i % len(texts)]},
                               vib=rng.normal(size=DIM).tolist(), note=vec_cache[i % len(vec_cache)],
                               bm25_text=texts[i % len(texts)]) for i in range(n // 2)]
            for a in range(0, len(pts), 1000):
                dev.upsert(pts[a:a + 1000])
            for a in range(0, n, 1000):
                mir.upsert([StorePoint(ids.make_id("s", i), {"type": "shared", "component": comps[i % 3],
                                                             "fault_class": fcs[i % 4], "status": "active"},
                                       note=vec_cache[i % len(vec_cache)], bm25_text=texts[i % len(texts)])
                            for i in range(a, min(n, a + 1000))])
            dev.optimize(); mir.optimize()
            qs = random.Random(1).sample(texts, 20)
            qv = [emb.embed_query(t) for t in qs]
            vv = [rng.normal(size=DIM).tolist() for _ in qs]
            k = iter(range(10 ** 9))

            def pick():
                i = next(k) % len(qs)
                return qs[i], qv[i], vv[i]

            def e2e_vib():
                return [float(v) for v in features(W[next(k) % len(W)], rpm=rpm)]

            cases = {
                "novelty_gate": (lambda: dev.novelty_gate(pick()[2], "m1"), lambda: dev.novelty_gate(e2e_vib(), "m1")),
                "same_machine_recall": (lambda: dev.same_machine_recall(pick()[2], "m1"),
                                        lambda: dev.same_machine_recall(e2e_vib(), "m1")),
                "technician_hybrid": (lambda: dev.technician_hybrid(note=pick()[1], text=pick()[0], component="bearing", fault_class="inner"),
                                      lambda: (t := pick()[0], dev.technician_hybrid(note=emb.embed_query(t), text=t, component="bearing", fault_class="inner"))),
                "fleet_evidence": (lambda: mir.fleet_evidence(note=pick()[1], text=pick()[0], component="bearing", fault_class="inner"),
                                   lambda: (t := pick()[0], mir.fleet_evidence(note=emb.embed_query(t), text=t, component="bearing", fault_class="inner"))),
            }
            for name, (qdrant_only, end_to_end) in cases.items():
                res[f"{name}_{n}"] = {"qdrant": timeit(qdrant_only, 200), "e2e": timeit(end_to_end, 100)}
            dev.close(); mir.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return res


def main() -> dict:
    out: dict = {"machine": "Intel i5-1335U, 15.7 GB RAM, no GPU, Windows 11, Python 3.12, qdrant-edge-py 0.8.0"}
    x, rpm = load_cwru(RAW / "105.mat")
    W = windows(x)
    it = iter(range(10 ** 9))
    out["fingerprint"] = timeit(lambda: features(W[next(it) % len(W)], rpm=rpm), 200)

    emb = BgeEmbedder()
    notes = ["inner race spall found, bearing replaced", "grease leaking from seal, cleaned and regreased",
             "coupling misaligned 0.3 mm, realigned", "high vibration at 1x, rebalanced fan"]
    out["embed_note"] = timeit(lambda: emb.embed_query(random.choice(notes)), 100)

    root = pathlib.Path(tempfile.mkdtemp(prefix="lat_"))
    rng = np.random.default_rng(0)
    try:
        s = EdgeStore(root / "w")
        k = iter(range(10 ** 9))
        out["durable_write"] = timeit(lambda: s.upsert([StorePoint(ids.make_id("w", next(k)), {"type": "x"},
                                                                     vib=rng.normal(size=DIM).tolist())]), 100)
        s.close()

        from edge.device import Device, DeviceConfig
        from edge.replay import Recordings
        from shared.embed import HashEmbedder
        dev = Device(DeviceConfig("lat", "s", "m", root / "dev"), HashEmbedder())
        dev.fit_baseline(Recordings.baseline())
        hz = Recordings.windows(99)
        j = iter(range(10 ** 9))
        out["gate_normal_window"] = timeit(lambda: dev.ingest_window(hz[next(j) % len(hz)]), 200)
        dev.close()

        import csv, glob
        f = glob.glob(str(pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "logbook" / "*.csv"))[0]
        texts = [r[1] for r in list(csv.reader(open(f, encoding="utf-8", errors="replace")))[1:] if r[1].strip()]
        vec_cache = emb.embed_documents(texts[:2000])
        server = None
        try:
            from qdrant_client import QdrantClient, models as m
            server = QdrantClient(url="http://127.0.0.1:6333", timeout=5)
            server.get_collections()
        except Exception:
            server = None
        out["query_types"] = query_types(emb, texts, vec_cache, W, rpm)
        for n in (1000, 10000, 50000):
            st = EdgeStore(root / f"h{n}")
            pts = [StorePoint(ids.make_id("h", i), {"type": "episode"}, vib=rng.normal(size=DIM).tolist(),
                              note=vec_cache[i % len(vec_cache)], bm25_text=texts[i % len(texts)]) for i in range(n)]
            for a in range(0, n, 1000):
                st.upsert(pts[a:a + 1000])
            st.optimize()
            qs = [(rng.normal(size=DIM).tolist(), emb.embed_query(t), t) for t in random.Random(1).sample(texts, 20)]
            q = iter(range(10 ** 9))

            def one():
                v, nv, t = qs[next(q) % len(qs)]
                st.search(vib=v, note=nv, text=t, limit=10)
            out[f"hybrid_edge_{n}"] = timeit(one, 200)
            st.close()
            if server is not None:
                name = f"latency_bench_{n}"
                if server.collection_exists(name):
                    server.delete_collection(name)
                server.create_collection(name, vectors_config={"note": m.VectorParams(size=384, distance=m.Distance.COSINE)})
                for a in range(0, n, 1000):
                    server.upsert(name, [m.PointStruct(id=i, vector={"note": vec_cache[i % len(vec_cache)]}) for i in range(a, min(n, a + 1000))])
                qq = iter(range(10 ** 9))
                out[f"dense_server_http_{n}"] = timeit(lambda: server.query_points(name, query=qs[next(qq) % len(qs)][1], using="note", limit=10), 100)
                qe = iter(range(10 ** 9))
                st2 = EdgeStore(root / f"h{n}")
                out[f"dense_edge_{n}"] = timeit(lambda: st2.search(note=qs[next(qe) % len(qs)][1], limit=10), 100)
                st2.close()
                server.delete_collection(name)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    llm = rag.LocalLLM()
    if llm.available:
        case = {"component": "bearing", "fault_class": "inner_race", "n_sites": 2, "n_events": 3, "flags": [], "notes": [],
                "actions": [{"action_code": "replace_bearing", "sites_worked": ["s1", "s2"], "sites_failed": [], "machine_verified": 2}]}
        ep = {"seq": 1, "component": "bearing", "fault_class": "inner_race", "occurrences": 30, "action_code": "lubricate",
              "outcome": "failed", "verify": {"verdict": "symptom_persists"}}
        res = {"fleet": [{"id": "c", "case": case}], "local": [{"id": "e", "episode": ep}]}
        out["llm_brief"] = timeit(lambda: rag.brief("What has been tried?", res, llm), 5, warm=1) | {"model": rag.MODEL_NAME}

    event = {"event_id": ids.make_id("e"), "episode_id": ids.make_id("p"), "machine_class": "2hp-induction-motor/SKF6205-DE",
             "component": "bearing", "fault_class": "inner_race", "fault_class_source": "technician",
             "action_code": "replace_bearing", "outcome": "worked", "root_cause_claim": "fatigue_wear", "machine_verified": True,
             "verify_windows_ok": 20, "verify_windows_required": 20, "technician_confirmed": True,
             "fingerprint": [round(float(v), 6) for v in rng.normal(size=DIM)], "fp_version": "fp-v2",
             "note_redacted": None, "occurred_at": "2026-09-27T10:00:00.000+00:00", "schema_version": 1}
    ev_bytes = len(json.dumps(event).encode())
    windows_covered = 40 + REQUIRED_WINDOWS            # a typical demo episode (40 fault windows) + verification
    raw_bytes = (windows_covered * HOP + (4096 - HOP)) * 4
    out["bandwidth"] = {"shared_event_json_bytes": ev_bytes, "raw_float32_signal_bytes_summarised": raw_bytes,
                        "ratio": round(raw_bytes / ev_bytes, 1),
                        "note": f"one event summarises {windows_covered} windows (~{windows_covered * HOP / 12000:.1f} s at 12 kHz); "
                                "the raw signal never leaves the device"}
    OUT.mkdir(exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
