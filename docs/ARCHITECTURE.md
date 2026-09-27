# ARCHITECTURE.md — System Design (v2, merged with second research pass)

## 1. High-level pipeline (updated)

```
Sensor (real replayed dataset: CWRU / NASA IMS)
        |
        v
Feature extraction: RMS, kurtosis, FFT band energies (hand-built — no off-the-shelf embedder exists
        |            for raw machine signals; this is a confirmed real gap, not assumed)
        v
Novelty Gate: query nearest neighbor first; if similar, bump counter/last-seen instead of inserting
        |     (prevents duplicate flooding from continuous ingestion — was missing before this update)
        v
Structured Event -> one point with THREE NAMED VECTORS + payload:
        - "sensor"  : hand-built feature vector (dense)
        - "text"    : MiniLM-class embedding of technician note (dense)
        - "sparse"  : BM25 over notes/fault codes
        - payload   : raw numbers, machine_id, fault_code, version, device_id, timestamp
        |
        v
Qdrant Edge shard (in-process, local)
        |
        v
Local Retrieval (offline, hybrid dense+sparse+RRF)
        |
        v
Decision Support UI (never auto-recommends)
        |
        v
Technician records outcome
        |
        v
Promotion Policy (multi-factor gate)
        |
        v
Durable Outbox (SQLite table / append-only file, written in the SAME step as the shard write)
        |
        v
Dual-write to Qdrant Server collection (Qdrant Cloud free tier)
        |
        v
Conflict Detector (real compare-and-set primitive: update_mode + condition filter)
+ Reliability Scoring
        |
        v
Other devices: bootstrap via server snapshot (new device) or incremental sync (existing device),
via their own local read-only "Fleet Mirror" shard
```

Qdrant Edge is the **memory/retrieval layer only** — not the anomaly detector, not the decision-maker.

## 2. Qdrant Edge — confirmed technical facts (merged from two independent research passes)

**Package:** `qdrant-edge-py` (Python), `qdrant-edge` (Rust crate). Install footprint ~11MB. Free,
currently **Beta**. Qdrant describes it as "SQLite for vector search."

**Core objects:**
```python
from pathlib import Path
from qdrant_edge import Distance, EdgeConfig, EdgeVectorParams, EdgeShard, Point, UpdateOperation

SHARD_DIRECTORY = "./machine17_shard"
Path(SHARD_DIRECTORY).mkdir(parents=True, exist_ok=True)

config = EdgeConfig(
    vectors={
        "sensor": EdgeVectorParams(size=32, distance=Distance.Cosine),   # hand-built features
        "text":   EdgeVectorParams(size=384, distance=Distance.Cosine),  # MiniLM-class
    }
)
edge_shard = EdgeShard.create(SHARD_DIRECTORY, config)
```

**Confirmed real constraints:**
- Each `EdgeShard` instance manages **one shard** (one directory) — our two-shard design (personal +
  "Fleet Mirror") uses two separate EdgeShard instances/directories, which is consistent with this, not a
  violation of it.
- Vector **size and distance are immutable** once a shard exists — freeze dimensions at creation time.
  Named vectors CAN be added later (Qdrant's own recommended path for migrating to a new embedding model).
- No setter for quantization or payload storage after creation — decide at creation.
- **Point IDs must be unsigned 64-bit integers or UUIDs** — map our own IDs via `uuid5(device_id, counter)`.
- No background optimizer — call `optimize()` explicitly on a schedule/idle window (it's synchronous and
  can stall the app if run on the main thread; run it on its own thread with a lock).
- **The real conflict-handling primitive:** update operations take a `condition` filter AND an
  `update_mode` (`Upsert`, `InsertOnly`, `UpdateOnly`). The condition applies only to points that already
  exist; new points insert regardless. This is what our compare-and-set conflict logic is built on.

## 3. Sensor encoding (corrected/upgraded from v1)

No off-the-shelf embedder exists for raw machine signals — confirmed real gap. Three real options, in
order to try: **(A) hand-built feature vector** (RMS, kurtosis, FFT band energies) — start here, simplest
and most defensible; (B) render a spectrogram as an image through a pretrained vision embedder; (C) a
small autoencoder trained on the dataset. **(A) is adopted as the primary "sensor" named vector**,
replacing the earlier, weaker v1 approach of embedding a plain-English description of the numbers.
Technician notes get their own separate "text" (MiniLM-class) and "sparse" (BM25) named vectors on the
same point.

## 4. Novelty gate (new)

Continuous ingestion produces near-identical sensor states every tick. Before inserting, query the nearest
neighbor in the shard; if similarity exceeds a tuned threshold, update a counter + last-seen timestamp on
the existing point instead of inserting a duplicate. Tune the threshold on labeled data via a
precision/recall curve.

## 5. Hybrid search (confirmed real capability)

```python
client.query_points(
    collection_name="incidents",
    prefetch=[
        Prefetch(query=sensor_vector, using="sensor", limit=20),
        Prefetch(query=text_vector,   using="text",   limit=20),
        Prefetch(query=sparse_vector, using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
)
```
**Real implementation notes (from Qdrant's own docs):**
- RRF weights have **no universal default** — tune against a hand-built labeled query set (30-50 queries
  with known correct incidents).
- Set `Modifier.Idf` explicitly on the BM25 sparse vector config.
- Measure actual average token length after stemming/stopword removal (Qdrant found real text 15-43%
  shorter than raw counts) — don't leave `average_length` at the default (256).
- **Do not use `score_threshold` on fusion (RRF) queries** — Qdrant's own docs flag this as unreliable.

## 6. Conflict handling (corrected — now using the real Qdrant primitive, not a vague rule)

Observations are **append-only** points with globally unique IDs (device_id + counter -> uuid5) — devices
never collide on these. Editable records carry `version`, `device_id`, `timestamp` and are written with a
**condition (compare-and-set, via the real `update_mode`+`condition` primitive above)**. A rejected write
becomes a **sibling revision**, shown in a visible conflict log — never silently dropped, never
last-write-wins.

**Clock-skew fix:** order conflicting writes by a **logical/hybrid logical clock**, not wall-clock
timestamps (wall time is display-only) — wall clocks can disagree between devices.

Per-device reliability score (Bayesian-smoothed): `R = (successes+1)/(successes+failures+2)` — shown as
context only, never auto-picks a winner. Ranking order for related (non-conflicting) results:
Applicability -> Evidence quality -> Device reliability -> Recency.

## 7. Sync design — durable outbox (upgraded from "queue")

Qdrant's own documented sync example uses an **in-memory** outbox, and the docs themselves say to consider
persisting it. Fix: a **durable outbox** (SQLite table or append-only file) written in the same step as
the shard write, so a crash never loses a pending sync. Replay is idempotent (deterministic IDs,
`InsertOnly` for observations). Retries use exponential backoff.

A new device **bootstraps from a full server snapshot**, then syncs increments — confirmed real Qdrant
pattern, never a full re-upload every time. Each device also keeps a read-only "Fleet Mirror" shard,
refreshed from the server whenever online, queried alongside personal memory when offline.

**Retraction/deletion:** the server marks a bad promoted incident `retracted` (tombstone), never
hard-deletes it; devices update their Fleet Mirror on next pull.

## 8. Promotion policy (unchanged)

Promote local incident to fleet ONLY if ALL: (1) technician marks intervention complete, AND (2) machine
completes a defined stable operating period with no recurrence, AND (3) outcome confirmed successful.
Confidence describes evidence — it is NOT the promotion gate.

## 9. Storage & eviction

Priority when full: (1) old low-value RESOLVED records, (2) duplicates, (3) raw data whose summary is
kept, (4) low-confidence records. **Never** auto-delete unresolved/high-severity incidents.

## 10. Security (prototype-scoped, no overclaiming)

OS-level disk encryption + app login + device identity per record. Do not claim encryption-at-rest unless
actually implemented and verified.

## 11. Real numbers to use (Qdrant's own — cite as theirs, then measure our own)

Qdrant in-house: ~0.1ms on-device (iPhone 16 Pro) vs ~52ms round trip to Qdrant Cloud over cellular for the
same 10,000-vector query (explicitly in-house, not an official benchmark). Robot demo: 0.51ms hybrid query
inside the shard, 3x/sec ingest. **Rule for us: measure our own latency on our actual demo laptop against
a real cloud endpoint, disclose the method, show both numbers — never present Qdrant's numbers as ours.**

## 12. Known open engineering risk

Qdrant Edge's exact crash-durability guarantee is not confirmed from documentation alone — test directly
(kill the process mid-write) before claiming any specific behavior on stage.

## 13. Real datasets

CWRU Bearing Dataset + NASA IMS Bearing Dataset — real, public, widely used in research. License/exact
terms not formally confirmed by either research pass; standard citation-based academic use is the norm.
