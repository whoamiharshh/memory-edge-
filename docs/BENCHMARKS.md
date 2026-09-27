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

## 12. Held-out second machine: HUST bearing (`bench/hust_holdout.py`)
The data-realism test. **Nothing was tuned on this data**: thresholds as shipped (chosen on CWRU). HUST bearing
(Hong & Thuan 2023, Mendeley Data, CC BY 4.0, DOI 10.17632/cbv7jyx4p9.3): another lab rig, another sensor
(PCB 325C33), 51.2 kHz, **five bearing types (6204-6208)**, 0/200/400 W; shaft speed from each file. Profile
`rotating-hf`: defect frequencies computed from each bearing's geometry (physics, not memorised). Each bearing type is
its own machine: baseline = healthy at 0 + 200 W; "unseen healthy" = healthy at 400 W.

| Metric | First run | After the 2 bug fixes (D15) | After ONE "normal operation" confirmation (D16) |
|---|---|---|---|
| False alarms on healthy data at the untaught 400 W | 305 / 305 | 92 / 305 | **4 / 255 (1.6 %)** |
| Fault windows detected (single / combined faults) | 98.5 % / 100 % | 98.1 % / 100 % | same |
| Fix verified "symptom resolved" (fault → repair → healthy 400 W) | 0 / 42 | 15 / 42 | **42 / 42** |
| Still-faulty verified "symptom persists" | 38 / 39 | 31 / 32 (+7 not evaluable: the fault merged into the first episode) | same |
| **False promotions** | **0** | **0** | **0** |
| Physics hint (defect frequency from geometry) | 97.6 % | **97.6 %** (ball 12/12, inner 15/15, outer 14/15) | same |

What it taught us (the reason to run it): (1) a low cage frequency could fall between FFT bins and read as "no
energy", making one feature a 13-million-sigma outlier; (2) a baseline spread floor of 1e-6 amplifies any
near-constant feature; (3) even without bugs, **a healthy machine at an operating point it was never taught looks
"abnormal"** (diagnosis: same-session windows 0 % alarms, another session or load often > 90 %). Real condition
monitoring handles (3) by teaching new healthy states; the device now has that step, and one confirmation of 10
windows fixed it. Caveat: the "teach" window count (10) and the 400 W state are this dataset's; a real plant has more
operating states and each needs one confirmation.

Diagnosis script: `spike/diagnose_hust_load.py`.

## 13. Robots: UCI Robot Execution Failures (`bench/robot_failures.py`)
CC BY 4.0, DOI 10.24432/C5M89N. Wrist force/torque, 15 samples × 6 channels per instance, profile `force-torque`,
shipped gate thresholds, baseline = 60 % of the normal instances, 5 seeds averaged.
| Task | False alarms (held-out normal) | Failures detected | Per failure type |
|---|---|---|---|
| lp1 approach to grasp | 0 % | **96 %** | collision 88 %, front collision 96 %, obstruction 100 % |
| lp2 part transfer | 5 % | 55 % | back 63 %, front 70 %, left 36 %, right 60 % |
| lp3 part position after transfer | 5 % | 55 % | lost 80 %, moved 61 %, slightly moved 36 % |
| lp4 approach to ungrasp | 0 % | **98 %** | collision 97 %, obstruction 100 % |
| lp5 motion with part | 0 % | 71 % | bottom collision 88 %, bottom obstruction 100 %, in part 59 %, in tool 54 % |

Honest reading: gross collisions/obstructions are caught; subtle ones (a part slightly moved) are not with these
thresholds. The data set is small (8-18 held-out normals per task).

## 14. Which text model? (`bench/embed_models.py`)
Same protocol as §4 (500 queries). Latency measured while a download ran; relative order is what matters.
| Model (FastEmbed, 384-d) | Hybrid P@3 | Dense P@3 | Embed p50 | Files |
|---|---|---|---|---|
| **BAAI/bge-small-en-v1.5 (shipped)** | **0.885** | 0.794 | 33 ms | 64 MB |
| sentence-transformers/all-MiniLM-L6-v2 (the plan's fallback) | 0.815 | 0.788 | 19 ms | 87 MB |
| snowflake/snowflake-arctic-embed-xs | 0.868 | 0.784 | **7 ms** | 87 MB |
bge-small stays. arctic-embed-xs is the option for weak devices (1.7 points lower, ~4× faster).

## 15. Signal profiles (`tests/integration/test_profiles.py`)
One engine, five input kinds, the same 27-number fingerprint slot. Each test runs the real device end to end.
| Profile | Input | Tested end-to-end |
|---|---|---|
| bearing-12k | CWRU vibration 12 kHz | all of §1-§3 |
| rotating-hf | any accelerometer ≥ 2 kHz, any bearing geometry | synthetic 6206 inner-race defect → hint + geometry-derived defect frequencies; HUST §12 |
| lowrate-accel | phone / IMU ~60 Hz, 3 axes | coin-imbalance on a fan: baseline capture → NEW → hint imbalance (shaft 17 Hz) → rebalance → resolved → SHARE; Nyquist honesty (misalignment "not assessable" when 2x > 30 Hz) |
| force-torque | robot wrist 6 channels | collision opens one episode; UCI §13 |
| events | kiosk / vehicle / app error codes per bucket | printer jam → clear jam → codes stay away 20 buckets → resolved → SHARE |
No licensed public data set with kiosk/vehicle error codes AND the fixes applied was found, so the events profile is
tested with synthetic buckets only.

## 16. Disk and RAM per shard (`bench/footprint.py`, Windows 11 NTFS)
Episode-like points (27-d fingerprint, 384-d bge-small note of real logbook text, BM25). "Allocated" = what the files
really occupy (GetCompressedFileSizeW). Fresh process per cell.

| Points | Default layout | + float16 note | + int8 note (originals on disk) | **Default, folder OS-compressed** |
|---|---|---|---|---|
| 0 | 203 MB | 203 MB | 203 MB | **~0 MB** |
| 1,000 | 267 MB | 267 MB | 267 MB | **3.8 MB** |
| 10,000 | 310 MB | 268 MB | 313 MB | **39 MB** |
| 50,000 | 482 MB (RAM 431 MB) | 446 MB (RAM 398) | 501 MB (RAM **376**) | **196 MB** |

- ~203 MB per shard is **fixed pre-allocation** (32 MB page chunks, mostly zeros); real data adds ~5.6 kB/point.
- Vector tricks help little on disk (float16 ~8 %); int8 quantization lowers RAM ~13 %.
- **OS compression of the device folder is the big lever** (`--compress-storage`): the shard kept answering queries at
  the same speed within noise (0.7-5 ms p50 while other benchmarks ran). Files Edge creates later are not compressed
  until the next pass (the device re-compresses hourly).
- Linux / macOS / Android (sparse pre-allocated files) were **not measured**: no such machine was available.
- The top-10 agreement column in the JSON is **not a quality measure**: even two shards with the identical default
  layout differ, because the approximate (HNSW) index is built non-deterministically.

## 17. Laya experiment (`bench/laya_experiment.py`, isolated `.venv-laya`)
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
