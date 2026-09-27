# FEATURES.md — What We're Building, What's From Scratch, and Why (v2)

## Part 1 — Take as-is (existing, proven building blocks)

| Component | Why we don't build it ourselves |
|---|---|
| Qdrant Edge shard (storage, HNSW, hybrid, filters, snapshots) | It's the sponsor's product and the point of the challenge |
| BM25 sparse embedding for notes | Edge has BM25 support built in |
| Small pretrained text embedder (MiniLM-class) | Training our own has no payoff; Qdrant's own Edge packages use MiniLM-class models |
| Qdrant server/cloud collection as sync target | Documented target for Edge sync |
| Hand-built sensor feature extraction (RMS, kurtosis, FFT bands) | Standard, well-understood signal-processing technique, not something to reinvent |

## Part 2 — Build ourselves, and precisely why nothing else solves it

### 2.1 Sensor-state encoder
No off-the-shelf embedder exists for machine signals (confirmed real gap — text/image embedders don't
apply). **Built:** hand-built feature vector (RMS, kurtosis, FFT band energies) as the primary "sensor"
named vector; determines retrieval quality directly.

### 2.2 Novelty gate
Not something Qdrant provides — continuous ingestion would otherwise flood the shard with near-identical
states. **Built:** query nearest neighbor before insert; update existing point's counter/last-seen instead
of inserting a duplicate above a tuned similarity threshold.

### 2.3 Versioned record schema + compare-and-set conflict handling
Qdrant's own docs show how to write a point, not how to handle two devices disagreeing about the same
one — confirmed directly, conflict resolution is explicitly left to the developer. **Built:** append-only
observations with globally unique IDs; editable records with version+device_id+timestamp written via the
real `update_mode`+`condition` primitive; rejected writes become visible sibling revisions in a conflict
log — never last-write-wins, never silently dropped.

### 2.4 Per-device reliability scoring
Not found as an existing product feature anywhere checked (Qdrant's docs, IBM Maximo, Augury). **Built:**
Bayesian-smoothed score `R=(successes+1)/(successes+failures+2)`, shown as context only, never used to
auto-resolve.

### 2.5 Durable outbox + retry
Qdrant's own documented example outbox is in-memory only, and the docs say to consider persisting it —
confirmed real gap. **Built:** SQLite/append-only-file outbox written in the same step as the shard write;
idempotent replay; exponential-backoff retries.

### 2.6 Fleet Mirror + retraction propagation
Not found anywhere as an existing pattern. **Built:** a second read-only local Edge Shard per device,
refreshed via server snapshot (new device) or incremental pull (existing device); retracted incidents are
tombstoned, never hard-deleted, so a withdrawn fix is visibly flagged.

### 2.7 Multi-device partition test harness
The only way to *prove* the sync engine works rather than just claim it. **Built:** N simulated devices,
random offline periods, assert zero lost writes and convergence to the same state.

### 2.8 Inspection UI
The problem statement explicitly asks for a screen to inspect the system — also our main proof surface in
the live demo. **Built:** live shard stats, search box, nearest past incidents, sync queue, conflict log,
latency panel.

### 2.9 Evaluation harness
Turns claims into numbers a judge can check. **Built:** Precision@3/Recall@3 against a hand-built labeled
query set; measured p50/p99 search latency vs. a real (not assumed) cloud round trip; shard size vs. point
count; the partition test above.

### 2.10 Verifying Qdrant Edge's real crash-durability behavior
Not a "build" item — a **test** item. WAL/crash-safe logic is confirmed to exist in the real
configuration, but the exact guarantee must be tested directly before any claim is made on stage.

## Part 3 — Explicitly NOT building (and why)

- **Not an autonomous maintenance recommender** — avoids a real liability question.
- **Not claiming domain novelty in industrial edge AI** — Qdrant's own "Edge Anomaly Triage" pattern
  already names unusual-machine-state use; we claim depth (tested sync + measured quality), not discovery.
- **Not a perception pipeline** (no camera/object-detection core) — avoids reading as a clone of Qdrant's
  own robot/glasses demos, which judges from the sponsor will already know.
- **Not custom end-to-end encryption** — standard OS-level disk encryption only, stated plainly.
- **Not claiming a guaranteed win** — if another team also builds real conflict-aware sync, we tie on that
  axis and must win on proof and clarity, not on being first to think of it.
