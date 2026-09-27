# Benchmarks: every number, its script, its method and its caveat

All runs on one laptop: Intel i5-1335U (12 logical CPUs), 15.7 GB RAM, **no GPU**, Windows 11, Python 3.12,
`qdrant-edge-py` 0.8.0, Qdrant Server 1.19.1 (official Windows binary). Raw JSON for every table is in
`bench/results/`. Re-run any of them with `.venv\Scripts\python.exe -m bench.<name>`.
**Data caveats that apply throughout:** CWRU faults are artificially seeded (EDM) on a lab rig with **one healthy
bearing**; the maintenance logbook is aviation text (a proxy domain). Numbers are for this laptop only.

---

## 1. Can the sensor verify a fix? K3 (`bench/gate_verifier.py`)
Real Device + Qdrant Edge, all 36 CWRU fault recordings, fresh device each, baseline on healthy loads 0-1.

| Metric | Result |
|---|---|
| Fixed replays (fault → unseen healthy recording) verified "symptom resolved" | **36 / 36** |
| Still-faulty replays verified "symptom persists" | **36 / 36** |
| **False promotions** (still-faulty verified as resolved) | **0** |
| Gate false alarms on unseen-load healthy windows | **0 / 116** |
| Fault windows flagged abnormal | 100 % |
| Episodes opened per fault recording | 1.0 |
| Two *different* faults separated by healthy running → separate episodes | **20 / 20** (was 11/20 before D11) |
| A fault that returns *immediately* after a verified repair is recognised as a recurrence | 35 / 36 (was 36/36) |
| Windows to a verdict after a repair (median) | 20 (= N) |

## 2. Novelty-gate threshold sweep (`bench/gate_sweep.py`)
The gate's two prototype parameters, each swept with the other at its shipped value.

**tau_normal = factor × q99 of healthy-to-healthy distances** (q99 = 5.89 z-units; unseen healthy max 8.8, first-20-windows fault min 28.7)

| factor | 0.5 | 0.75 | 1.0 | **1.5** | **2.0 (shipped)** | **3.0** | **4.0** | 6.0 |
|---|---|---|---|---|---|---|---|---|
| false-alarm rate (116 healthy windows) | 0.50 | 0.50 | 0.46 | **0** | **0** | **0** | **0** | 0 |
| detection rate (720 fault windows) | 1.0 | 1.0 | 1.0 | **1.0** | **1.0** | **1.0** | **1.0** | 0.89 |

**tau_merge = factor × tau_normal** (36 recordings; separation = 20 pairs of different faults with healthy between)

| merge factor | 1.0 | **1.5 (shipped, D11)** | 2.0 | 3.0 (before) | 5.0 | 8.0 |
|---|---|---|---|---|---|---|
| exactly one episode per recording | 36/36 | 36/36 | 36/36 | 36/36 | 36/36 | 36/36 |
| intermittent same fault after a healthy gap stays one episode | 34/36 | **35/36** | 36/36 | 36/36 | 36/36 | 36/36 |
| closed episode recognised when its fault returns | 33/36 | **36/36** | 36/36 | 36/36 | 36/36 | 36/36 |
| different faults separated | 20/20 | **20/20** | 19/20 | 11/20 | 5/20 | 0/20 |

Why 1.5 and not 2.0: each makes one error, but 2.0's error merges two *different* faults into one episode (a fix
could be credited to the wrong fault); 1.5's error splits one intermittent fault in two (a harmless duplicate).
**Caveat:** chosen and measured on the same CWRU data; no second machine exists to hold the choice out.

## 3. Does vibration similarity transfer to an unseen bearing? K2 (`bench/retrieval_vib.py`)
| Split | Method | Fault-class P@3 |
|---|---|---|
| random windows (**leaky**: same bearings on both sides) | fingerprint kNN | 0.997 |
| **bearing-level** (query bearings never indexed) | fingerprint kNN (best variant) | **0.429** (chance 0.277) |
| bearing-level | physics rule: dominant envelope defect frequency | **0.644** accuracy (inner 1.00, ball 0.69, outer 0.24) |

This is why the fleet groups by **technician-confirmed fault class** (DECISIONS D3).

## 4. Text retrieval on real maintenance text (`bench/retrieval_text.py`)
Annotated Maintenance Logbook (6,169 records, CC BY 4.0), 500 queries, relevant = same annotated (problem, part).
| Method (all through Qdrant Edge) | P@3 | MRR@10 | R@10 | query p50 |
|---|---|---|---|---|
| dense (bge-small) | 0.794 | 0.860 | 0.185 | 2.4 ms |
| BM25 (Edge built-in, measured avg_len 6.86) | 0.843 | 0.926 | 0.201 | 1.0 ms |
| **hybrid RRF (one prefetch + fusion request)** | **0.881** | **0.944** | 0.199 | 2.9 ms |

## 5. Sync under partitions and crashes (`bench/sync_partition.py`)
10 seeded runs × 5 devices × 20 events, created *during* the chaos; requests dropped before the cloud (p = 0.3), acks
lost after the cloud applied them (p = 0.3), random partitions, hard device restarts (≥ 1 forced per device).
| Events | Lost | Duplicates stored | Tally error | Hard restarts | Requests dropped | Acks lost |
|---|---|---|---|---|---|---|
| 1,000 | **0** | **0** | **0** | 209 | 207 | 139 |

Plus `tests/failure/test_sync_partition.py` (3 seeds) in the normal test run. In-process cloud; the per-device
rate limit is lifted in this bench only (it is tested separately).

## 6. Offline: every device function with the network blocked (`bench/offline_check.py`)
A socket guard blocks **and records** every connection attempt of the process (loopback too) while the device:
loads bge-small and the local LLM from disk, fits a baseline, gates healthy + fault windows, opens an episode, saves
a note, confirms the fault class, records an action, verifies it with the sensor, decides SHARE, queues it, searches
local + fleet mirror, writes an evidence brief, and ticks the sync worker.
| Checks passed | Connection attempts | Guard self-test (a deliberate request is caught) |
|---|---|---|
| **15 / 15** | **0** | yes |

Wi-Fi-off is a manual rehearsal item in [DEMO.md](DEMO.md); this is the stricter scripted form.

## 7. Fleet mirror: Qdrant snapshots vs scroll rows (`bench/mirror_sync.py`)
Real Qdrant Server + Sync API; bytes on the wire = what the device downloads (gzip for snapshots).
| Cases | Method | First pull | After ONE case changed | Nothing changed |
|---|---|---|---|---|
| 10 | full / partial snapshot | 188 kB | 189 kB (partial) | 49 B (head check only) |
| 10 | scroll rows | **26 kB** | **2.7 kB** | 74 B |
| 100 | full / partial snapshot | **210 kB** | 211 kB (partial) | 50 B |
| 100 | scroll rows | 255 kB | **2.7 kB** | 75 B |
| 1,000 | full / partial snapshot | **398 kB** | 398 kB (partial) | 51 B |
| 1,000 | scroll rows | 2.55 MB | **2.7 kB** | 76 B |

The raw shard snapshot is ~178 MB of mostly pre-allocated zero pages, hence the gzip. With one segment, a partial
snapshot re-ships that segment, so it costs as much as a full one. Hence the default `auto` mode (DECISIONS D10):
full snapshot to bootstrap, then scroll deltas, switching back to a full snapshot whenever rows would cost more.
Snapshot pulls took 7-11 s each on this laptop *while other benchmarks were running* (tar + gzip + unpack of
~178 MB); the timings are not representative, the bytes are. Disk: ~245 MB per mirror copy.

## 8. Storage: what the gate saves (`bench/storage.py`)
One simulated hour (21,093 windows at 4096/2048 hop, 12 kHz): mostly healthy running with a fault recording every ~1
min, rotating through all 36.
| | Points in the shard | Shard bytes on disk |
|---|---|---|
| gate ON (the device) | **164** (86 baseline, 73 exemplars, 5 episodes) | 280 MB |
| gate OFF (store every window) | 21,093 (**129×**) | 281 MB |

Disk bytes are equal because Qdrant Edge pre-allocates its storage pages; at this scale the point count is the
honest growth measure (21k points/hour would keep growing; 78 points/hour of episodes would not).

## 9. Latency (`bench/latency.py`)
Wall clock (`time.perf_counter`), warm-up excluded, p50 / p95 over n repetitions, machine otherwise idle (the demo's
Qdrant Server running for the comparison). Points: random fingerprints + real logbook text (bge + BM25).
| Operation | p50 | p95 |
|---|---|---|
| fingerprint one window (4096 samples, DSP) | 2.3 ms | 4.6 ms |
| embed one note (bge-small, CPU) | 6.1 ms | 7.1 ms |
| gate decision per window (86 baseline points) | 0.21 ms | 0.27 ms |
| durable write (upsert + `flush()`) | 32 ms | 36 ms |
| hybrid query (vib + note + BM25, RRF) on Edge: 1k / 10k / **50k** points | 0.50 / 1.06 / **1.97** ms | 0.63 / 1.53 / 2.99 ms |
| dense query on Edge: 1k / 10k / 50k | 0.30 / 0.35 / 0.24 ms | 0.48 / 0.53 / 0.40 ms |
| same dense query on the **local** Qdrant Server over HTTP: 1k / 10k / 50k | 14.4 / 18.2 / 9.4 ms | 27.6 / 32.6 / 31.9 ms |
| LLM evidence brief (Qwen2.5-1.5B Q4, CPU, 2 evidence items) | 2.6 s | 2.7 s |

Bandwidth: one shared fix is **855 bytes** of JSON; the raw float32 signal it summarises (60 windows, ~10 s at
12 kHz) is 499,712 bytes and never leaves the device (584×).
The Edge-vs-Server gap is mostly HTTP + serialisation on localhost, not search; the reason for the edge is that it
keeps working with no network, not these milliseconds.

## 10. Resources of one device (`bench/resources.py`)
The real per-window path in **real time**: raw CWRU signal (healthy 99, then fault 105) → DSP fingerprint → gate →
episode writes, one window every 170.7 ms for 60 s. CPU % = process CPU time / wall time (100 % = one core of 12).
| Measure | Result |
|---|---|
| CPU while monitoring in real time | **16.5 % of one core** |
| per-window processing time p50 / p95 (incl. durable episode writes on fault windows) | 16 ms / 43 ms |
| real-time headroom (window period / p95) | **4.0×** |
| RAM: Python + imports / + bge-small / + device & baseline | 105 / 270 / **304 MB** |
| RAM with the local LLM loaded (optional evidence brief) | 1,973 MB |
| disk: device folder (local shard 267 MB + empty mirror 203 MB, mostly Edge pre-allocation) | 471 MB |
| disk: models the device uses (bge-small 64 MB + Qwen2.5-1.5B Q4 GGUF 1,066 MB) | 1,130 MB |

Without the optional LLM, one device fits in ~0.3 GB of RAM and a small fraction of one laptop core.

## 11. Laya experiment (`bench/laya_experiment.py`, isolated `.venv-laya`)
Laya = convaiinnovations/laya 0.3.20 (Apache-2.0, ModernBERT-large 421M, ~808 MB, PyTorch CPU). Proposed roles:
pre-fill the action picker from the technician's own description of what they did; second personal-data flag.
Logbook ACTION text → 8 action families (mapping in the script), 1,592 distinct texts, split by distinct text,
319 held out; every method scored on the same 319.

| Method | Accuracy | Macro-F1 | CPU per note |
|---|---|---|---|
| majority class | 0.411 | 0.073 | — |
| bge-small zero-shot (cosine to family descriptions) | 0.740 | 0.690 | 14 ms |
| **Laya zero-shot** (`choice`, same descriptions) | 0.718 | 0.693 | **2,027 ms** p50 (2,609 p95); 19 s load |
| **bge-small + logistic regression** (trained on 1,273 notes) | **0.981** | **0.970** | 14 ms + 0.4 ms |

Laya confidence: mean 0.73; 27 % of answers ≥ 0.9, and those are 91 % correct. The checkpoint warns that some of its
shipped temperatures are invalid ("treat confidence ... as uncalibrated").

**Personal-data probe (synthetic):** 120 held-out notes, 60 with a person's name appended in 4 templates.
| Detector | Recall (names) | Precision | False positives |
|---|---|---|---|
| Laya `noul` "does the text mention a person by name?" | **0.067** | 0.80 | 1 |
| project regex redactor, empty denylist | 0.0 | — | 0 |

**Decision (DECISIONS D12).** Laya is **not** kept: zero-shot it only ties bge-small, at ~140× the CPU time and
~1 GB of extra dependencies, and as a name detector it found 4 of 60. bge-small + logistic regression wins the
action-family task but is **not wired into the device** either: it was trained on aviation action families, which
do not map onto our motor action codes; a real pre-fill needs labelled notes from the target domain. The probe
supports the existing privacy design: free-text notes stay local by default, and names must be on the denylist.
