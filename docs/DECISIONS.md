# Decision log (what we chose, the evidence, and what it cost)

Newest first. Each entry says what changed, the measurement behind it, and the trade-off. The research-time
decisions (before code existed) are in [RESEARCH.md Appendix 3](RESEARCH.md#appendix-3--decision-log).

---

### D12 · Laya: measured, kept out of the device (28 Sep 2026)
**Question.** Should Laya (convaiinnovations/laya, Apache-2.0, 421M, ~808 MB + PyTorch) pre-fill the action-code picker
from the technician's own note, and/or act as a second personal-data flag?
**Evidence.** `bench/laya_experiment.py` → `bench/results/laya_experiment.json`, summarised in
[BENCHMARKS.md](BENCHMARKS.md#laya-experiment). **Decision and reasons:** see BENCHMARKS.md; the device ships the
winner for each role and nothing that would enter the share decision.

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
