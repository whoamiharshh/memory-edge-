# Decision log (what we chose, the evidence, and what it cost)

Newest first. Each entry says what changed, the measurement behind it, and the trade-off. The research-time
decisions (before code existed) are in [RESEARCH.md Appendix 3](RESEARCH.md#appendix-3--decision-log).

---

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
