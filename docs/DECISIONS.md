# Decision log (what we chose, the evidence, and what it cost)

Newest first. Each entry says what changed, the measurement behind it, and the trade-off. The research-time
decisions (before code existed) are in [RESEARCH.md Appendix 3](RESEARCH.md#appendix-3--decision-log).

---

### D42 · Device clocks are measured and corrected by the cloud (28 Sep 2026)
A device clock more than a day fast would have had its evidence REJECTED as "future" (found by bench/network_faults.py
design review). Every push now carries the device time; the cloud computes the offset and shifts that batch's times
onto its own clock (original kept); devices show "your clock is X h off". Measured: +72 h / -48 h clocks corrected to
within 1.0 s (BENCHMARKS §30). Hybrid logical clocks stay cut: ordering is never needed, only honest dates.

### D41 · Network faults injected per request, not per connection (28 Sep 2026)
`tools/netem_proxy.py`. The first version decided faults per TCP connection; HTTP keep-alive sends many requests over
one connection, so a "40 % loss" run cut 1 connection of 8. Per-chunk faults cut 111 of 113 in the same scenario.
Result under every scenario: 0 lost, 0 counted twice, follow-ups once, mirrors identical (BENCHMARKS §30).

### D40 · Cloud scaling: per-thread Qdrant clients, lock stripes, batched ingest, coalesced recompute (28 Sep 2026)
`bench/scale_fleet.py` (20 devices at once, real Qdrant Server, real HTTP) first **froze the whole cloud**: one Qdrant
call hung while holding the store's single global lock. Fixed: one Qdrant client per thread, 64 lock stripes, 20 s
timeouts, one lookup + one insert per pushed batch (was 2 calls per event), case tallies recomputed once per case by a
background worker (reads flush first, so never stale), the full mirror snapshot built once per version, a small
fleet's first mirror pull by rows (bytes decide), device push timeout 5 -> 30 s. Correctness held in every run:
1,000 events, 0 lost, 0 double-counted, identical mirrors.

### D39 · Vehicle model: the 21,637 training trucks were tried and rejected (28 Sep 2026)
CV AUC 0.79 inside training, 0.60-0.66 on the test trucks vs 0.75 for the shipped validation-trained model
(BENCHMARKS §28). Label construction mismatch; shipped model unchanged.

### D38 · Robots learn from their own confirmed failures (28 Sep 2026)
`edge/local_detector.py` + "teach a failure": 96-99 % detection, 0-10 % false alarms (was 9-15 % with a threshold from
overfit training scores). Enabled for the force-torque profile only, where it is measured; it only adds alarms.

### D37 · Event profile radius from healthy data at a 1 % false-alarm target (28 Sep 2026)
Event buckets repeat exactly, so split-half q99 = 0 and tau_normal collapsed to 0 on real HDFS logs. Now: 99.98 %
detection, 0.39 % false alarms (BENCHMARKS §25). Vibration profiles unchanged (continuous signals never repeat).

### D36 · Name detection without a denylist (28 Sep 2026)
10,562 Wikidata given names (CC0) + maintenance vocabulary + a name-likeness n-gram model: unlisted names found 96 %
in normal typing (was 0/60), 64-66 % in ALL CAPS / lower case (BENCHMARKS §26).

### D35 · Security hardening (28 Sep 2026)
Certificate revocation list for mutual TLS (TLS >= 1.2), two-admin retraction, device quarantine, plausibility checks,
hash-chained audit log, code-integrity status, CSP and security headers, body limits, weak operator tokens refused on a
network address. Threat model updated row by row with the test that proves each.

### D34 · Relative order rule for imbalance / misalignment / looseness (28 Sep 2026)
Absolute textbook rules called every drive-fed motor "looseness", healthy ones included. The hint now names the shaft
order that GREW versus this machine's healthy state (> 3 sigma). Naming shaft faults remains unvalidated on real data
(the only public set has one motor per fault; BENCHMARKS §24). Also tried: requiring bearing defect lines to have grown
- it cut the HUST hint 97.6 -> 78.6 % and was reverted.

### D33 · Fleet-learned fault hint + "confident only when physics and fleet agree" (28 Sep 2026)
Order-domain envelope features travel with CONFIRMED evidence; the cloud trains a logistic regression and measures it
on devices it never saw; devices pull the coefficients as JSON. Confident hints: HUST 100 %, UOttawa 92 %, CWRU 85 %
(BENCHMARKS §23). Rejected: a collision-aware physics v2 (worse on HUST) and a cross-dataset prior (53-83 %).

### D32 · Replacement-aware fix verification (28 Sep 2026)
Nearest-state rule for replace actions: new bearing verified 19/20 (mic) and 18/20 (accel) instead of 3/20 and 0/20;
HUST 42/42 without teaching (was 15/42); no false promotion anywhere (BENCHMARKS §22).

### D31 · Manufacturer data: machine card, manuals, ISO 15243, follow-ups (28 Sep 2026)
ISO 10816-3 tables A.1-A.4 read from the standard's text (it says acceptance limits belong to manufacturer and
customer - so a machine card can override them); "fixed" = back to healthy AND below the machine's limit; manuals
indexed offline with page citations; technicians record the ISO 15243 damage mode they saw; devices report weeks
later whether a fix HELD or the fault RECURRED, and the cloud flags RECURRED.

### D30 · Phone = microphone, not only the 60 Hz motion sensor (28 Sep 2026)
Real natural-wear bearings: 0/380 false alarms, 95.7 % of faulty windows flagged, from sound (BENCHMARKS §21).

### D29 · Licence stays Apache-2.0 (28 Sep 2026)
Chosen by Claude at the user's request: permissive, with a patent grant, the same licence as Qdrant. Nothing is
published; publishing stays the user's decision.


### D28 · Trained models only on real data (28 Sep 2026)
The user's rule: every trained model is trained on real data; made-up data only as a last resort, and labelled.
Trained: the vehicle early-warning model (SCANIA validation → test), the robot failure models (UCI, cross-validated),
the text action-family classifier of the Laya experiment (real logbook; not shipped). Nothing is trained on synthetic
data. Synthetic data is used only in TESTS (e.g. the events profile's printer jam, the phone fan), never to train.

### D27 · Vehicle risk hint: logistic regression, not boosting (28 Sep 2026)
On 5,045 held-out real trucks it beat gradient boosting (ROC-AUC 0.75 vs 0.68) and ships as plain coefficients that
cannot execute code on load. The unsupervised per-truck gate showed no signal on these readouts, and we say so.

### D26 · Security completed: expiring tokens, mutual TLS, notes encrypted at rest (28 Sep 2026)
Tokens expire after 30 days and devices renew them automatically (old token: 10-minute grace). `--mtls` makes the cloud
require a device certificate from our CA. Notes (and the BM25 text that contains them) are AES-256-GCM encrypted
before they reach the shard or the journal; the key is DPAPI-protected on Windows. The encryption test found the
BM25 text in the journal in plain text first; that was fixed.

### D25 · Installable web app instead of a native phone app (28 Sep 2026)
No Qdrant Edge package exists for Android/iOS, and a browser-only device would drop Qdrant Edge. The device UI became
an installable web app (manifest + service worker caching only the app shell, never `/api` data) used next to a
laptop/Pi device that works offline. A native Android device remains possible later via the Rust crate.

### D24 · Vehicle fault-code dictionary shipped (28 Sep 2026)
OBDex (CC0): 9,533 standard codes with meanings, causes by likelihood, symptoms and sources, offline in
`knowledge/vehicle_codes.json`; shown next to event episodes. Sites add their own SOPs through the "Add procedure" form.

### D23 · Automatic operating-point check (28 Sep 2026)
Taught speed/load ranges are kept with the baseline; untaught points are flagged with a physics suggestion (threshold
chosen on CWRU, measured on HUST: flag 0 false on taught points, suggestion right 13/17).

### D22 · Storage: compress the device folder instead of shrinking vectors (28 Sep 2026)
**Evidence.** `bench/footprint.py`: ~203 MB per shard is fixed pre-allocation (mostly zero pages), data adds ~5.6 kB
per point; float16 saves ~8 % disk, int8 quantization ~13 % RAM (and a little extra disk). NTFS compression of the
folder: 267 MB → 1.9 MB with the shard open, writes/search/reopen still working. Files created later do not inherit
compression, so the device compresses at start-up and hourly (`--compress-storage`). Linux/macOS/Android sparse-file
behaviour: not measured (no such machine here).

### D21 · HTTPS enforced off-localhost (28 Sep 2026)
Private CA + server certificate (`tools/make_certs.py`); devices verify the cloud against `ca.pem`; the launchers
refuse plain HTTP on any non-localhost address unless `--insecure-lan` is given. Phones need HTTPS anyway: browsers
only expose motion sensors in a secure context. Tested (`tests/security/test_tls.py`).

### D20 · "Not a fault: normal operation" (28 Sep 2026)
**Evidence.** HUST held-out: a healthy machine at an untaught load / session looked abnormal. The technician can now
teach a new healthy state: the episode's fingerprints become extra baseline points (no refit, so stored vectors stay
comparable); the episode closes as dismissed and is never shared. One confirmation: false alarms 92/305 → 4/255,
fixes verified 15/42 → 42/42.

### D19 · Two bugs found by the held-out test (28 Sep 2026)
(1) `physics.peak_near` never searches narrower than half an FFT bin (a ~10 Hz cage rate fell between 3 Hz bins).
(2) Baseline spread floor 0.05 (log10 units) for the new profiles; bearing-12k keeps 1e-6 because every CWRU number
was measured with it. Effect on HUST: false alarms 305 → 92 of 305 before any teaching.

### D18 · Geometry physics instead of one bearing's constants (28 Sep 2026)
Defect frequencies from the standard kinematic equations (checked against the CWRU table to 4 decimals), velocity
severity with ISO 10816-3 group-2 rigid boundaries (indicative outside that group), order-spectrum rules for
imbalance/misalignment/looseness that say "not assessable" when 2x is above Nyquist. On HUST's five bearing types the
geometry hint reached 97.6 %.

### D17 · Signal profiles (28 Sep 2026)
PS3 names robots, kiosks, vehicles and mobile devices. Every profile maps its input to the same 27-float fingerprint
slot, so gate, verifier, policy, sync and cloud are unchanged; `fp_version` keeps profiles apart. Five profiles:
bearing-12k, rotating-hf, lowrate-accel, force-torque, events.

### D16 · Cited procedure library, never generated (28 Sep 2026)
The user asked for help to actually fix faults. Documented reference checklists (SKF 14219 for bearings, field
balancing, shaft alignment/soft foot, looseness) are shown per fault class with their source; sites add SOPs. The
LLM never writes procedures, and the policy still decides sharing only from verified outcomes.

### D15 · LLM checker: flags verbatim, numbers must match (28 Sep 2026)
The live 3-site demo produced "flagged as DISPUTED because different root causes" (that is COMPETING). Sentences that
name a flag are now dropped and the cited cases' flags appended verbatim; every number must occur in the cited
evidence.

### D14 · Text model: bge-small stays (28 Sep 2026)
`bench/embed_models.py`: hybrid P@3 bge-small 0.885, arctic-embed-xs 0.868 (4× faster), MiniLM 0.815. The plan's
MiniLM fallback was worse than assumed.

### D13 · Plan details completed (28 Sep 2026)
All three disagreement flags tested; live demo shows DISPUTED + COMPETING across 3 sites; repair content hash stops
one repair from counting twice; "last confirmed" per action; text-model migration (new named vector, re-embed,
switch; the fleet dense leg is skipped while the device and cloud models differ); SQLite reads fetched under the lock
(a live race); flush retry on transient Windows file locks; replay errors never hang.

### D12 · Laya: measured, kept out of the device (28 Sep 2026)
**Question.** Should Laya (convaiinnovations/laya, Apache-2.0, 421M, ~808 MB + PyTorch) pre-fill the action-code picker
from the technician's own note, and/or act as a second personal-data flag?
**Evidence.** `bench/laya_experiment.py` → `bench/results/laya_experiment.json` ([BENCHMARKS.md §11](BENCHMARKS.md)).
Action family from the note (319 held-out logbook notes): Laya zero-shot 0.718 accuracy vs bge-small zero-shot 0.740
vs bge-small + logistic regression 0.981; Laya ~2 s per note on CPU vs ~15 ms. Name detection (synthetic probe):
Laya recall 4/60.
**Decision.** Laya is not kept for either role. The supervised bge-small classifier wins the action task but is not
shipped, because its labels are aviation action families that do not map onto our motor action codes. The probe
backs the existing privacy default (notes stay local; names go on the denylist). Laya stays in an isolated
`.venv-laya` for reproducibility only.

### D11 · Novelty-gate merge radius 3.0 → 1.5 × tau_normal (28 Sep 2026)
**Evidence.** `bench/gate_sweep.py` on all 36 CWRU fault recordings. At 3.0, only 11/20 pairs of *different* faults
separated by healthy running got separate episodes (the README's known limitation). At 1.5: 20/20 separated, a
returning closed fault is still recognised as a recurrence 36/36, one episode per recording 36/36, and an
intermittent *same* fault after a healthy gap stays one episode 35/36.
**Trade-off.** 1.5 and 2.0 each make one error, of different kinds: 1.5 splits one intermittent fault into two
episodes (a harmless duplicate); 2.0 merges two different faults into one episode (could attribute a fix to the
wrong fault). We prefer the harmless error.
**Honest limit.** Chosen on CWRU, the same data it is measured on; no second machine exists to hold it out.

### D10 · Fleet mirror: `auto` fill mode by measured cost (28 Sep 2026)
**Evidence.** `bench/mirror_sync.py` (real Qdrant Server). With our collection layout a partial snapshot re-ships the
one mutable segment, so after ONE changed case it costs about the same as a full snapshot (189-398 kB on the wire,
seconds of work on ~178 MB of mostly zero pages), while scroll rows cost ~2.7 kB. For bootstrapping 1,000 cases a
full snapshot is 6x smaller than scroll rows (398 kB vs 2.5 MB).
**Decision.** Default `auto`: full shard snapshot to bootstrap/rebuild (`EdgeShard.unpack_snapshot`, swapped in only
after a probe), then the cheaper of a scroll delta or a new full snapshot, using this device's own measured costs.
Pure partial-snapshot mode (Qdrant's pattern verbatim) stays implemented, tested and selectable
(`--mirror-mode snapshot`). Partial snapshots are never applied on top of scroll writes, because local writes change
the segment versions that the server's manifest comparison relies on.

### D9 · Mirror collection: one shard, one segment, 1 MB WAL (28 Sep 2026)
**Evidence.** `spike/spike_snapshot.py`: default config → a 580 MB shard snapshot for 20 points (pre-allocated pages of
many segments); one segment + 1 MB WAL → 148 MB raw; gzip on the Sync API → ~150 kB on the wire.
**Cost.** ~200 MB of disk per mirror copy on the device (Edge pre-allocates too), briefly twice during a swap.

### D8 · Partial snapshot K5 test passed (28 Sep 2026)
Qdrant Server 1.19.1 + `qdrant-edge-py` 0.8.0: full snapshot → `unpack_snapshot` → hybrid query; server change →
`snapshot_manifest` → `POST /collections/{c}/shards/0/snapshot/partial/create` → `update_from_snapshot`; unchanged
→ HTTP 304. The server's BM25 sparse vectors are made with Edge's own tokenizer (`edge.store_edge.bm25_sparse`), so a
device's BM25 query ranks mirrored text correctly (asserted: BM25 leg rank #1 in the snapshot integration test).

### D7 · Retention ARCHIVE keeps knowledge, drops redundant fingerprints (28 Sep 2026)
Closed, decided, uploaded episodes older than 90 days keep the episode point and exemplar 0 (so recurrence still
works, tested) and drop the other exemplar fingerprints. One journal op, so a crash cannot lose the kept exemplar.

### D6 · Optimistic concurrency for episode edits (28 Sep 2026)
Every UI edit carries `expected_version`; a stale edit gets **409 CONFLICT** and writes nothing (RESEARCH.md G.4
"updated memory"). The note editor pins the version it started from while it has unsaved typing.

### D5 · Usefulness feedback is a counter only (28 Sep 2026)
"Helped / didn't help" per result, stored on the device, shown next to the result, never changes ranking, never
shared (RESEARCH.md G.1: no learned usefulness model).

### D4 · Local LLM evidence brief, outside the decision path (27 Sep 2026)
Qwen2.5-1.5B-Instruct (GGUF, llama.cpp, CPU). Every sentence must cite `[E#]`, advice is removed, template fallback.

### D3 · K2: fleet grouping by technician-confirmed fault class, not vibration similarity (27 Sep 2026)
Bearing-level P@3 0.429 (chance 0.277) vs leaky 0.997. See README "K2".

### D2 · K3 passed: the sensor can verify a fix (27 Sep 2026)
36/36 fixed verified, 36/36 still-faulty persist, 0 false promotions, 0/116 false alarms.

### D1 · K4: flush before acknowledging (27 Sep 2026)
0/200 acknowledged writes survive a hard kill without `flush()`; 200/200 with it.
