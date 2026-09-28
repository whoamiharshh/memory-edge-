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
> See §29 for capped recall (nR@10) and tune/test halves.

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
> **Updated 28 Sep (§22):** with the replacement-aware verifier, fix verified at the untaught 400 W load is
> **42/42 without any teaching** (was 15/42), still-faulty caught 38/39, 0 false promotions; physics hint unchanged
> at 97.6 %. The table below is the earlier run, kept for the record.

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

## 18. Automatic operating-point check (`bench/operating_point.py`, HUST held out)
The device remembers the speeds (and loads, if given) its healthy baseline covers. An episode at an untaught operating
point is flagged, and physics words the suggestion: the median defect-envelope score of its first 10 windows against
a threshold chosen on CWRU (1.63; `spike/signature_threshold.py`).
| Case | Result |
|---|---|
| Healthy episodes at the untaught 400 W speed flagged "untaught" | 3 / 3 |
| Faults at taught speeds wrongly flagged "untaught" | **0 / 28** |
| Faults at the untaught speed flagged "untaught" | 14 / 14 |
| Suggestion right: healthy → "probably a new normal operating point" | 2 / 3 (the 6205's healthy recordings carry a tone at a defect frequency) |
| Suggestion right: fault → "a fault signature is present" | 11 / 14 |
Per recording on HUST the CWRU threshold separates 12/15 healthy (below) and 36/42 faulty (above).

## 19. Robots: a failure model TRAINED ON REAL DATA (`bench/robot_model.py`)
> **Superseded by §27:** the threshold below came from overfit training scores (9-15 % false alarms); the
> shipped device detector alarms at p > 0.5: 96-99 % detection, 0-10 % false alarms.

UCI Robot Execution Failures (real, CC BY 4.0), 5-fold stratified cross-validation per task (each trace scored by a
model that never saw it); alert threshold fixed on the training folds (≤ 5 % of *training* normals alarm).
| Task | Unsupervised gate (§13): detected / false alarms | Logistic regression: detected / false alarms |
|---|---|---|
| lp1 approach to grasp | 96 % / 0 % | 100 % / 14 % |
| lp2 part transfer | 55 % / 5 % | **100 %** / 10 % |
| lp3 part after transfer | 55 % / 5 % | **100 %** (incl. "slightly moved") / 15 % |
| lp4 approach to ungrasp | 98 % / 0 % | 99 % / 13 % |
| lp5 motion with part | 71 % / 0 % | **98 %** / 9 % |
The trained model catches the subtle failures the gate misses, at 9-15 % false alarms (only ~20 normal traces per
task, so the held-out false-alarm rate is above the 5 % aimed for). Gradient boosting was unstable on data this small
(0 % on lp2/lp3) and is not used. Also tried and rejected by the same honest split: a tighter gate radius (tuned on
lp1+lp2 → default kept) and temporal gate features (`spike/robot_features.py`: no gain).

## 20. Vehicles on REAL data: SCANIA Component X (`bench/vehicle_scania.py`)
Real trucks (Scania CV AB, CC BY 4.0, DOI 10.5878/jvb5-d390): readouts of one anonymised engine component and labels
from workshop repair records. Profile `telemetry` (increments between readouts).
**A. The memory engine per truck (unsupervised):** 4,483 test trucks; last readout flagged abnormal for 20.4 % of trucks
far from a repair vs 20.6 % of trucks within 48 steps of one: **no signal**. These readouts describe how a truck is
used, not this component's health, so a truck's own history does not reveal the coming repair. (The real Device made
the same decision as the fast numpy path in 20/20 spot checks.)
**B. A trained early-warning model, trained on the validation split, tested on the test split (different trucks):**
| Model | ROC-AUC | Alerting the riskiest ~10 % of trucks catches … of those repaired within 48 steps | Precision (base 2.8 %) |
|---|---|---|---|
| age only (reference) | 0.566 | 13 % | 3.7 % |
| **logistic regression (shipped)** | **0.750** | **32 %** (32 % within 6 steps) | **9.4 %** |
| gradient boosting | 0.677 | 24 % | 7.4 % |
The shipped model is plain coefficients (`knowledge/vehicle_risk_model.json`, no pickle); the device computes the
same probability as the evaluation (`tests/integration/test_vehicle_risk.py`, 3 real trucks). It is a hint shown with
these numbers.

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

## 21. Phone microphone instead of the 60 Hz motion sensor (`bench/acoustic_uottawa.py`)
Real data: University of Ottawa UODS-VAFDC (CC BY 4.0, DOI 10.17632/y2px5tg92h.1). 20 bearings whose faults
**developed naturally** (seals removed, degreased), each recorded healthy / developing / faulty by a microphone
(PCB 130F20, 2 cm from the bearing) and an accelerometer, 42 kHz, 10 s. Never used for tuning before this test;
shipped thresholds. One device per bearing; baseline = first half of its healthy recording.

| Metric | Microphone (profile `acoustic`, analysed at 16 kHz = what a phone's 44.1/48 kHz audio becomes) | Accelerometer (`rotating-hf`) |
|---|---|---|
| False alarms, unseen half of the healthy recording | **0 / 380 windows** | 0 / 620 |
| Developing-fault windows flagged | **92.7 %** | 95.0 % |
| Faulty windows flagged | **95.7 %** | 97.0 % |
| Fix verified: fault -> the same bearing's unseen healthy | **20 / 20** | 20 / 20 |
| Fix verified: fault -> **a different, new bearing** (replacement, §22) | **19 / 20** | 18 / 20 |
| Still-faulty caught (fault continues after the action) | **19 / 20** | 19 / 20 |

N = 15 windows for verification (the recordings are only 10 s). Why a microphone: browsers cap motion sensors at
~60 Hz (W3C Generic Sensor / DeviceMotion), too slow for bearing-defect frequencies; audio arrives at 44.1/48 kHz with
echo cancellation, noise suppression and auto-gain switched off (`edge/ui/sensor.js`). A microphone is not a
calibrated vibration sensor, so no ISO velocity zone is given from it.

## 22. Replacement-aware verification (`edge/verifier.py`, spike `spike/replacement_verify.py`)
A **new** bearing has a different healthy signature than the old baseline: in §21 the old rule verified only
**3/20** (microphone) and **0/20** (accelerometer) replacements. For `replace_bearing` / `replace_part` a window now
also counts as healthy when it is **nearer to the machine's healthy state than to the fault episode's own
fingerprints** (parameter-free). Result: new bearing verified **19/20** and **18/20**; a continuing fault was never
judged fixed (spike: 20/20; bench still-faulty 19/20). The verified new part's windows are added to the baseline.
Side effect on HUST (§12): fix verified at an untaught load went from **15/42 to 42/42** without the one-time
teaching; still-faulty 38/39 caught; **0 false promotions**. CWRU K3 unchanged (36/36, 36/36, 0).

## 23. Fault TYPE: physics, fleet-learned model, and "confident only when both agree" (`bench/fault_hint.py`)
All real data, ten 1 s segments per recording. The fleet model (`cloud/hint_model.py`: order-domain envelope features
`physics.order_features`, centred, logistic regression) is always tested on bearings it never saw: UOttawa
leave-one-bearing-out (20), CWRU leave-one-fault-size-out, HUST leave-one-bearing-type-out.

| Data | Physics rule | Fleet model | Shown as "confident" (both agree) | Correct when confident |
|---|---|---|---|---|
| HUST (5 bearing types) | 97.6 % | 95.2 % | 93 % of recordings | **39/39 = 100 %** |
| UOttawa accelerometer (natural wear) | 35.0 % | **72.5 %** | 33 % | **12/13 = 92 %** |
| CWRU (seeded) | 69.4 % | 66.7 % | 75 % | 23/27 = 85 % |
| UOttawa microphone | 32.5 % | 70.0 % | 35 % | 10/14 = 71 % |

Learning curve (UOttawa, accuracy on one unseen bearing vs how many OTHER confirmed bearings the fleet has, 200 random
draws): accelerometer 2 -> 0.33, 4 -> 0.40, 8 -> 0.51, 12 -> 0.68, 16 -> 0.71, 19 -> **0.72**; microphone 0.32 ->
**0.76**. The fleet gets better as technicians confirm cases. Standardised features were compared and lost on every
dataset (small-sample failure shown in `tests/unit/test_fleet_hint.py`). Also tried and **rejected**: a
collision-aware v2 envelope rule (`spike/physics_v2.py`: HUST dropped to 71 %) and a cross-dataset prior model
(`spike/cross_dataset.py`: 53-83 %). Where the physics is weak (natural wear), CWRU's own benchmark study found many
records "not diagnosable with any of the applied methods" (Smith & Randall, MSSP 64-65, 2015) - which is why the
fleet groups by the TECHNICIAN-CONFIRMED class (ISO 15243 damage mode on the removed part) and the hint says "inspect"
when unsure.

## 24. Imbalance / misalignment rules on real motors (`bench/motor_rules.py`)
University of Ottawa motor set UOEMD-VAFCVS (CC BY 4.0), drive-fed motors at 15/30/45/60 Hz, loaded/unloaded.
The absolute textbook order rules called **every** motor "looseness", healthy ones included (the drive's electrical
harmonics sit on shaft orders), so the device now uses a **relative** rule (which order grew vs this machine's own
healthy state, > 3 sigma). Only windows the gate flags get a hint, as in the product.

| | Accelerometer | Microphone |
|---|---|---|
| Healthy motor: no alarm / no hint | **8/8** | **8/8** |
| Unbalanced and misaligned motors detected (flagged abnormal) | **16/16** | **16/16** |
| ... and NAMED imbalance / misalignment | 0/8, 0/8 | 0/8, 0/8 |

Each fault is a DIFFERENT physical motor here, so motor-to-motor differences are mixed into every comparison; paired
by condition the unbalanced motor's 1x was higher than the healthy motor's in 7/8 and the misaligned motor's 2x/1x
in 5/8, but the hint names a bearing class. Naming shaft faults is therefore **not validated on real data**; the
coin-on-a-fan phone test (docs/FIELD_TEST.md) is the planned check. Nothing was tuned on this confounded set.

## 25. Kiosks / apps: the `events` profile on REAL logs (`bench/events_hdfs.py`)
Loghub HDFS_v1 (CC BY 4.0, DOI 10.5281/zenodo.8196385; labels from Xu et al., SOSP 2009): each block session = one
bucket of 29 event-template counts. Baseline 5,000 normal sessions; radius set from HEALTHY data at a 1 % false-alarm
target (event buckets repeat exactly, so 2 x q99 would be 0 - a real bug this test found and fixed in
`edge/gate.calibrate`).

| Normal sessions flagged | Failed sessions detected | F1 (50,000 normal : 16,838 failed) |
|---|---|---|
| **193 / 50,000 = 0.39 %** | **16,835 / 16,838 = 99.98 %** | **0.994** |

At HDFS's natural ~3 % failure rate, about 89 % of alarms would be real. For scale only (different splits): the Loghub
benchmark's unsupervised methods reach F1 0.79 (PCA) - 0.91 (Invariant Mining) (He et al., ISSRE 2016). Server logs
are a proxy for kiosk/app error codes; no public kiosk data set with error codes and applied fixes was found.

## 26. Names in notes the site never listed (`bench/redaction.py`)
Real logbook notes (1,000 held out; the maintenance vocabulary rebuilt without them) with a real given name inserted
in 8 templates (SYNTHETIC insertion: no labelled public set exists). 20 % of the 10,562 Wikidata names (CC0) are held
out of the redactor's list AND of the name model it trains.

| Style | Names on the list | Names NOT on the list | Clean notes kept local by mistake |
|---|---|---|---|
| normal typing (sentence case) | **100 %** | **96.2 %** | 0.6 % |
| ALL CAPS (the logbook's style) | **100 %** | 65.8 % | 2.4 % |
| all lower case | **100 %** | 63.6 % | 0.6 % |

Before: regex + denylist found 0 of 60 unlisted names (Laya probe, §17). Name-likeness model: character n-grams,
logistic regression, trained on real names vs real maintenance words, threshold chosen on a validation split (<= 2 %
word false positives -> 0.9). A flagged note stays on the device, so a false alarm costs sharing, never privacy.

## 27. Robots: the device's own learned detector (`bench/robot_model.py`, `edge/local_detector.py`)
Exactly what a device stores (27-number fingerprint z-scored against the healthy traces), class-balanced logistic
regression, alarm at p > 0.5 (no tuning), 10 x stratified 5-fold CV on real UCI traces.

| Task | Failures detected (range) | False alarms (range) | Normal / failure traces |
|---|---|---|---|
| lp1 | **98.8 %** (98.5-100) | **0 %** | 21 / 67 |
| lp2 | **96.7 %** (88.9-100) | 10 % | 20 / 27 |
| lp3 | **95.6 %** (88.9-100) | 9 % (5-10) | 20 / 27 |
| lp4 | **98.8 %** (97.8-98.9) | **0 %** | 24 / 93 |
| lp5 | **96.3 %** (95-97.5) | 4.5 % | 44 / 120 |

The earlier method (threshold from overfit TRAINING scores) had 9-15 % false alarms. With ~20 normal traces per task,
one false alarm is 5 points. End to end on a device (`tests/integration/test_robot_detector.py`): the technician
confirms the failures the gate caught and TEACHES those it missed ("teach a failure"), the robot trains its detector
and then catches failures the gate alone missed.

## 28. Vehicles: more training trucks did NOT help (`bench/vehicle_scania_train.py`) - rejected
The SCANIA training split (21,637 usable trucks, 67,781 real cut points, labels from repair records; censored cuts
dropped) gave cross-validated AUC 0.79 inside training, but on the untouched test trucks only **0.60 (LR) / 0.66
(gradient boosting)** - worse than the shipped model trained on the validation split (**0.75**). The likely reason: my
cut points do not match how Scania chose each test truck's labelled readout; the validation split is labelled like the
test split. Kept the shipped model; reported as a negative result. (The download stopped at 1,182,793,728 bytes when
the server began refusing requests; the first 22,982 of 23,550 trucks are complete.)

## 29. Text retrieval, recall measured properly (`bench/retrieval_text.py`)
R@10 is capped by large relevant sets (100 relevant records -> at most 0.1), so nR@10 = hits in the top 10 /
min(10, relevant) is added. Tune half / test half of the 500 queries:

| Method | P@3 (test half) | MRR@10 | nR@10 |
|---|---|---|---|
| dense | 0.801 | 0.859 | 0.814 |
| BM25 | 0.848 | 0.921 | 0.862 |
| **hybrid RRF (shipped, equal weights)** | **0.885** | **0.930** | **0.883** |
| hybrid, BM25 x2 / dense x2 | 0.883 / 0.881 | 0.934 / 0.923 | 0.872 / 0.884 |

Hybrid is best or tied on every metric; neither weighting beat equal weights consistently on the tune half.
