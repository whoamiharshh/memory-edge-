# RESEARCH.md — Machine Memory at the Edge (Qdrant PS3)

Research → challenge → architecture document. Written 27 Sep 2026. No code has been written yet.

**Labels used throughout**
- **Verified Fact**: read directly from an official doc, source page, paper, or package registry during this pass (source in §Sources).
- **Prior-pass fact**: verified in an earlier session (see CONTEXT.md). I did not re-verify it today.
- **Proposed Design**: our choice. It is not a property of any tool.
- **Assumption**: something we believe but have not checked.
- **Inference**: a conclusion drawn from verified facts.
- **I could not verify this.**: stated whenever it applies.

---

## 0. TL;DR (read this first)

**Verdict: MODIFY. Do not PIVOT, do not ABANDON.**

What survives:
- The domain (industrial rotating-equipment maintenance).
- The persona (the technician at the machine).
- Hand-built vibration fingerprints.
- Hybrid local retrieval.
- A durable outbox.
- Tombstones for retractions.
- "The system shows evidence and never prescribes."

What has to change:
1. **The core claim changes.** It is no longer "we built a conflict-aware sync layer". That is commodity technology (CouchDB, Couchbase Lite and ObjectBox all do it; see Part D). The new claim is **outcome-verified fleet learning**:
   - **The machine's own sensor stream verifies a fix.** A fix leaves the device only after the vibration fingerprint has returned to that machine's healthy baseline and stayed there for a stability window, and the technician has confirmed it.
   - **The cloud keeps disagreement as evidence.** Fixes are grouped by fault signature. Every outcome is kept as a count: fixes that worked, fixes that failed, and which sites reported them. Nothing is overwritten. Contradictions are flagged rather than silently resolved.
   - **Device B benefits offline** from Device A's verified experience.
2. **Several components are cut** because they add complexity without adding evidence: hybrid logical clocks, the per-device Bayesian reliability score, any local LLM, and Jev AI.
3. **Several earlier claims are corrected** (see §1): the "Fleet Mirror" is Qdrant's own documented pattern, and the ARCHITECTURE.md code used the server client API instead of the Edge API.
4. **The evaluation is made leakage-free.** CWRU is split by bearing, not by segment or load. Hybrid text retrieval is tested on a real, openly licensed maintenance-log dataset.

**Biggest risk: the Round 1 submission deadline (30 Sep 2026), which is 3 days away.** It outranks every architecture risk. See Part M for a 3-day vertical-slice plan.

**Biggest technical risk:** fingerprint retrieval may transfer poorly from one physical bearing to another. That is exactly the case Device B faces when it learns from Device A. We will measure it with a leakage-free split before building the story on it (Part Q, kill test K2).

---

## 1. Corrections to our existing files (Correction Protocol)

**Correction 1: Fleet Mirror originality**
- *Previously said* (FEATURES.md §2.6): "Fleet Mirror … Not found anywhere as an existing pattern."
- *Why wrong:* Qdrant's own "Synchronize with a Server" guide prescribes exactly this: "A mutable Edge Shard that handles local data updates. An immutable Edge Shard that mirrors a shard from a collection on a server using partial snapshots." You query both shards and deduplicate by point ID. **Verified Fact.**
- *Consequence:* we implement this pattern and credit it to Qdrant. It is not something we can claim as our work.

**Correction 2: Hybrid query code**
- *Previously said* (ARCHITECTURE.md §5): hybrid search via `client.query_points(... prefetch=..., FusionQuery ...)`.
- *Why wrong:* that is the **Qdrant server client** API. On Edge the call is `EdgeShard.query(QueryRequest(prefetches=[...], query=<Fusion>, ...))`.
- Edge does support prefetch + fusion in a single request, according to the current Edge "Reading Data" reference. **Verified Fact.**
- Note that the Qdrant agent-skill bundle I loaded says the opposite ("does not fuse dense and sparse at query time"). The official docs are more recent and win. We still confirm it in code on day 1: if `prefetches` + Fusion fails on 0.8.0, we fuse in application code.

**Correction 3: The "conflict-aware sync is our gap" claim**
- *Previously said* (PROBLEM.md §3): the gap is "an explicit, human-visible conflict state … with a tested, durable, conflict-aware sync layer."
- *Why wrong:* this confuses a gap in *Qdrant's docs* with a gap in the *market*.
  - CouchDB keeps losing revisions in `_conflicts` and tells applications to show them to the user. **Verified Fact.**
  - Couchbase Lite ships custom `ConflictResolver`s and on-device vector search with sync. **Verified Fact.**
  - ObjectBox sells on-device vector search plus Sync with "conflict resolution automatically". **Verified Fact.**
- *Consequence:* sync is required engineering. It is not the differentiator. See Part E for the new gap.

**Correction 4: Hybrid logical clocks**
- *Previously said:* order conflicting writes with hybrid logical clocks.
- *Why wrong:* HLC adds nothing once the design is changed.
  - Shared data becomes append-only evidence events with deterministic IDs.
  - The few mutable records use server-side compare-and-set on a `version` field.
  - Neither mechanism needs cross-device ordering.
- *Consequence:* remove HLC. Wall-clock time is kept for display only.

**Correction 5: Per-device reliability score**
- *Previously said:* `R=(s+1)/(s+f+2)` per device.
- *Why wrong:*
  - A "device" is not the thing that is right or wrong. The fix, the technician and the machine context are.
  - The score gives false precision from tiny counts.
  - With one to three sites in the demo it is meaningless.
- *Consequence:* replace it with raw, transparent evidence counts ("worked at 2 sites, failed at 1"). This also follows the brief's §21 instruction: no fake numerical precision.

**Correction 6: Latency as a reason for edge**
- *Previously implied:* the ~0.1 ms vs ~52 ms figure supports edge for our persona.
- *Why wrong for our persona:* a technician's decision takes minutes, so 50 ms makes no difference to them. **Inference.**
- Latency *does* matter in one place: the per-window novelty gate, which runs continuously on sensor data with no human in the loop.
- *Consequence:* in the pitch, lead with segmentation/connectivity, bandwidth, and data locality. Keep latency for the automated triage path only.

**Correction 7: Unverified tuning claims**
- *Previously stated as fact:*
  - "Qdrant found real text 15–43% shorter"
  - "don't use `score_threshold` on fusion queries (per Qdrant docs)"
- *Status:* **I could not verify either of these** in this pass.
- For the second, there is a separate **Inference**: RRF scores are rank-based, so a fixed threshold on them has no stable meaning.
- *Consequence:* both are downgraded to "to be measured / inference".

**Correction 8: Install size**
- "~11 MB" now matches PyPI: `qdrant-edge-py` 0.8.0 wheels are 10.0–12.4 MB. **Verified Fact.** It stays, with that citation.

---

## PART A — Executive understanding

**What the official problem asks for.** An offline-first product built on Qdrant Edge that does all of the following:
- local semantic memory;
- vector + hybrid search without network;
- a *dynamic* decision about what stays local and what syncs;
- sync to Qdrant Server when connectivity returns;
- handling of updates and conflicting information;
- a UI to inspect memory, search, sync status and activity;
- a meaningful edge-to-cloud workflow, not "just a local vector DB".

**What we believed we were building.** Industrial fault memory, whose core claim was a conflict-aware sync layer.

**What we are actually building after research.** Industrial fault memory whose core claim is **outcome-verified fleet learning**:
- a fix becomes fleet knowledge only when the machine's own sensor data shows it held;
- the cloud aggregates such fixes into evidence tallies grouped by fault signature, keeping disagreements rather than resolving them away;
- other devices use that knowledge fully offline.

**Alignment check against the brief.**

| Requirement | How we satisfy it | Status |
|---|---|---|
| Searchable semantic memory on device | Qdrant Edge shard with named vectors: `vib` (fingerprint), `note` (dense), `note_bm25` (sparse) | Proposed Design |
| Decide local vs sync dynamically | Deterministic policy engine that returns KEEP_LOCAL / SHARE / MERGE / ARCHIVE / REJECT with reasons | Proposed Design |
| Sync with Qdrant Server | Durable outbox push + fleet mirror pull, using Qdrant's partial-snapshot pattern | Proposed Design built on Verified primitives |
| Inspection UI | Device view, memory explorer, search with explanations, decision log, sync states, fleet evidence | Proposed Design |
| Low-latency vector + hybrid offline | Edge `query` with prefetch + fusion; measured p50/p95 on our laptop | Verified capability; our measurement pending |
| Intermittent connectivity | Outbox with retries and idempotent replay; all reads local | Proposed Design |
| Evolving memory, updates, conflicts | Novelty-gate merge, versioned case groups with CAS, disagreement surfaced as evidence, tombstones | Proposed Design |
| Meaningful edge-to-cloud workflow | A learns → cloud aggregates → B benefits offline | Proposed Design |

Verdict: aligned. One requirement is only partly served: "hybrid search" is real on the device, but the sensor leg is a hand-built feature vector, not a learned embedding. We say so openly.

### VERIFIED
- Edge supports named dense vectors, sparse vectors with IDF, built-in BM25, filters, facets, prefetch + fusion, conditional upserts with `update_mode`, and partial-snapshot helpers.

### UNVERIFIED
- That all of the above work together in `qdrant-edge-py` 0.8.0 exactly as documented. The sandbox cannot reach PyPI, so it has not been tested yet.

### ASSUMPTIONS
- Judges weight a working, measured system over feature count.

### OPEN QUESTIONS
- The full judging rubric. Only "Innovation & Creativity" and "Technical Implementation" are known, from the prior pass.

---

## PART B — Exact user and problem

**Primary persona:** a **field service / reliability technician responsible for rotating equipment** (motors, pumps, fans) across several sites.

| Question | Answer | Label |
|---|---|---|
| Device | A rugged laptop or tablet at the machine, or an edge gateway wired to the machine's vibration sensor. In the demo, one laptop process stands in for each device. | Proposed Design |
| Environment | Plant-floor OT network, often segmented from the internet. Some sites are remote (e.g. offshore). | Prior-pass fact (NIST SP 800-82r3 segmentation; Maritime Executive on rig connectivity) |
| Information generated | High-rate vibration signals; fault events; free-text notes; actions taken; outcomes | Inference |
| Information needed | "Have we, or any site, seen this vibration pattern before? What was done, and did it hold?" | Proposed Design |
| Decision | Which evidence to consider before choosing an intervention. The system never chooses the intervention. | Proposed Design (liability stance) |
| If information is unavailable | Trial-and-error repairs, repeat failures, longer downtime | Assumption, needs practitioner validation |
| If information is wrong | A wrong fix gets repeated across the fleet. **This is why promotion needs outcome verification.** | Inference |
| Why connectivity is unreliable | Deliberate OT segmentation, remote sites, and shielded plant floors | Prior-pass fact (the last is an Assumption) |
| Why local processing helps | Raw vibration is high-rate and site-confidential, so ship features, not waveforms. Also, triage must keep running when the link is down. | Inference |
| Why semantic memory is necessary | "Similar vibration pattern" cannot be expressed as a SQL predicate. It needs nearest-neighbour search over fingerprints. Technician shorthand needs dense + keyword matching. | Inference; the text side is benchmarked on real logs in Part O |

**Why current solutions are insufficient (narrow and honest).** The prior pass verified that IBM Maximo and Augury already provide fault diagnosis, NL access to asset history, and (Augury) edge diagnostics. We do not compete with them on diagnosis.

What we did **not** find in the products we checked is a mechanism where:
1. a fix is only shared after the machine's own post-repair data confirms it, and
2. the fleet view keeps worked/failed evidence per fault signature across sites instead of one "answer".

That is a finding from a limited search, not a claim that nobody does it.

**Secondary personas:** none. Adding them would dilute the demo.

### VERIFIED
- Leakage issues in CWRU (Part O).
- MaintNet-derived logbook dataset: 6,169 problem–action records, CC BY 4.0.

### UNVERIFIED
- That technicians in practice consult prior-case libraries. We have no practitioner interview.

### ASSUMPTIONS
- Post-repair vibration returning to baseline is an accepted repair check in condition monitoring. **Needs a practitioner or standards citation (e.g. ISO 20816); I did not verify this.**

### OPEN QUESTIONS
- Can anyone on the team talk to a real maintenance technician before 11 Oct?

---

## PART C — Why edge

The brief itself warns that "offline + low latency = edge" is not enough. Qdrant lists five reasons cloud retrieval breaks at the edge: latency, connectivity, cost, privacy, isolation. **Verified Fact** (Qdrant blog, 16 Jun 2026). For *our persona*, they rank like this:

| Factor | Relevant? | Why |
|---|---|---|
| Connectivity | **Yes, primary** | Segmented OT networks and remote sites. Retrieval and triage must work with no link. |
| Cost / bandwidth | **Yes** | Raw vibration is high-rate. Shipping ~20-float fingerprints plus structured outcomes instead of waveforms is the "most of it is noise" argument. Inference; we will measure bytes/day. |
| Privacy / data locality | **Yes** | Technician notes can contain names and site specifics. Raw signals can be commercially sensitive. The policy engine keeps them local by default. |
| Isolation | Partial | Per-site / per-tenant boundaries on the server (Part K). |
| Latency | **Only for automated triage** | The novelty gate runs a nearest-neighbour query per signal window. Human-facing search does not need sub-ms latency. |

**Compared with alternatives**

| Alternative | Why it's insufficient here | Label |
|---|---|---|
| Cloud app / cloud RAG | Stops when the link is down; ships raw data | Inference |
| Local relational DB (SQLite) | No similarity search over fingerprints or meaning. FTS5 covers keywords only. | Inference |
| Local document DB | Same gap | Inference |
| Local vector DB without the policy/evidence layer | Retrieves similar states, but cannot tell verified fixes from guesses. It shares everything or nothing. | Inference; this is the "Qdrant + CRUD" failure |
| Couchbase Lite / ObjectBox | Real on-device vector search + sync + conflict resolution. **These are the strongest "why not X" challenge.** They sync *records*. They do not provide outcome-gated promotion or evidence aggregation, which remain application logic on top. Qdrant Edge adds a server-compatible data model: the same points and filters, BM25 wire-compatible with the server, and snapshot interop. | Verified (features); Inference (the comparison) |
| Maximo / Augury | Centralized diagnosis products. Prior pass: Augury has edge diagnostics. We are not a diagnosis product. | Prior-pass fact |

**Why Qdrant Edge specifically, and not just "any embedded vector DB"** (all **Verified Fact**):
- It is the same engine as the server: "shares the same internals, storage format, and points API".
- A server snapshot can seed a device shard, and it can then be updated incrementally with partial snapshots.
- The built-in BM25 is wire-compatible with server BM25.

So the edge memory and the fleet collection are one data model. That is the real technical reason. Any comparison with another embedded vector DB that we haven't benchmarked is an **Assumption** and must not be claimed as "better".

### VERIFIED
- The five factors, the shared storage format, partial snapshots, and BM25 wire compatibility.

### UNVERIFIED
- Actual bandwidth saving. We will measure it.

### ASSUMPTIONS
- Target sites have intermittent or no internet from the OT zone.

### OPEN QUESTIONS
- None blocking.

---

## PART D — Competitor landscape (feature-level, not ranked)

| Existing solution | What it solves | Overlap with us | What it doesn't solve (per sources checked) | Evidence |
|---|---|---|---|---|
| **Qdrant Edge docs / demos** | Embedded vector search. Named patterns: Robotic Memory, Edge Anomaly Triage (paraphrase: devices compare observations against local baselines and escalate only unfamiliar cases), Private Device Memory. Describes "a two-way channel [that] carries fresh context down and evidence, shared memories, and fleet learning up". | High. Our novelty gate *is* Edge Anomaly Triage. Our mirror *is* their dual-shard pattern. | Their docs say to "handle errors and retries" and that the example queue is "in-memory… consider persisting". They describe no conflict or promotion policy. | Verified (blog 16 Jun 2026; sync guides) |
| **Couchbase Lite 3.2+** | Embedded NoSQL with on-device vector search (SQL++ `APPROX_VECTOR_DISTANCE`) and sync via Sync Gateway / App Services. Default conflict rule is most-revisions-wins or LWW; custom `ConflictResolver` supported. | High on infrastructure: offline vector search + sync + conflict hooks | It resolves conflicts per *document*. Nothing we found covers outcome-gated promotion or cross-site evidence aggregation. | Verified |
| **ObjectBox (4.x) + ObjectBox Sync** | On-device vector DB. Commercial Sync: "handles conflict resolution automatically", delta sync, developer picks which objects sync. Markets field-service and IIoT use. | High on infrastructure, including selective sync | We found no outcome-verified promotion or evidence tallies. Sync is paid and closed. | Verified (their pages) |
| **CouchDB** | Multi-master replication. Deterministic winner, losing revisions kept in `_conflicts`, application must resolve. | The concept of "preserve conflicts visibly" | Not vector or semantic | Verified |
| **Mem0** | LLM agent memory. Retrieves similar memories, then an LLM chooses ADD / UPDATE / DELETE / NOOP. Paper experiments used GPT-4o-mini. | Memory lifecycle with dedup and contradiction handling | Contradiction handling **deletes** ("DELETE for removal of memories contradicted by new information"). It relies on an LLM and is cloud-centric in the paper setup. | Verified (arXiv 2504.19413) |
| **Zep / Graphiti** | Temporal knowledge-graph agent memory. Edges carry validity intervals. Conflicts are invalidated "but not discard[ed]". | Preserving superseded facts | Agent/chat memory, not device fleets or sensor data. We did not verify edge/offline operation. | Verified (Neo4j blog) |
| **IBM Maximo** | EAM/APM: asset history, AI on sensor + text, offline-capable mobile inspection | Asset history retrieval | Centralized. Prior pass found no outcome-gated fleet sharing. | Prior-pass fact |
| **Augury** | Machine-health diagnostics with edge hardware and offline resilience | Vibration-based diagnosis | It is a diagnosis product. We do not compete on diagnosis. | Prior-pass fact |
| **FieldEdge** (github.com/choksi2212/code-cubicle-qdrant), a public repo for this same hackathon brand | Android + Rust + Qdrant Edge photo capture, offline CLIP search, WAL, FastAPI sync to Qdrant Cloud, "timestamp + vector-checksum" conflict heuristics | **Direct competitor pattern:** offline capture → search → WAL sync with conflict heuristics | By its README, no outcome/validation gate and no evidence aggregation. Its conflict heuristic is essentially timestamp-based. | Verified (repo README, via fetch). I could not verify whether it is from this edition or an earlier one. |
| AWS IoT Greengrass / Azure IoT Edge | Generic edge runtimes | Deployment only | Not memory or retrieval systems | Assumption (not re-checked in detail) |

**What this table tells us.** The things we previously treated as the differentiator are all table stakes: offline vector search, sync, a WAL/outbox, and visible conflicts. A competing team already has offline search + WAL sync on Qdrant Edge. **If we only ship "offline search + durable sync + conflict log", we tie with them at best.**

---

## PART E — Market gap (evidence-supported only)

We found overlap in: on-device vector search (Qdrant Edge, Couchbase Lite, ObjectBox); offline sync with conflict handling (CouchDB, Couchbase, ObjectBox); memory lifecycle with dedup and contradiction handling (Mem0, Graphiti); and machine diagnosis (Augury, Maximo).

**We did not find evidence of** any system in that set that does all three of these:
1. gates what leaves a device on **verified outcomes measured from the device's own sensor data** after an intervention;
2. aggregates such outcomes across devices into **per-signature evidence** (worked / failed / sites) that keeps disagreement instead of resolving it;
3. serves that evidence back to devices for **offline** retrieval from the same engine.

This is a gap in the combination. Each piece has precedent: CBR, truth discovery, CouchDB conflicts, Qdrant's anomaly triage. The claim is "we combined them for a real edge workflow and measured it", not "nobody thought of this".

### VERIFIED
- Every overlap row above marked Verified.

### UNVERIFIED
- That no commercial condition-monitoring vendor does post-repair verification plus fleet evidence. **I could not verify this.** Augury-class vendors may well do something similar internally. Do not say "nobody does this".

### ASSUMPTIONS
- The combination is valuable to technicians.

### OPEN QUESTIONS
- Any practitioner evidence either way.

---

## PART F — Core solution (final proposed architecture)

### F.1 One-sentence product
Each machine's edge device remembers vibration fault episodes and what fixed them. It searches that memory offline. It shares a fix with the fleet only after the machine's own data shows the fix held. Every device can then consult the fleet's accumulated evidence, including disagreements, with no network.

### F.2 The single primary differentiator
**Outcome-verified promotion: the sensor stream itself confirms a fix before it becomes fleet knowledge.**

It meets the brief's seven tests:

| Test | How it is met |
|---|---|
| Technically meaningful | Stops the #1 failure of fleet memory, which is spreading wrong fixes |
| Defensible | Rules and measurements anyone can inspect, not an LLM opinion |
| Implementable in time | Needs fingerprints, nearest-neighbour to a baseline, and a counter |
| Live demo | Replay the fault file, then the healthy file → windows turn green → SHARE |
| Uses Qdrant Edge meaningfully | Baseline proximity is a vector query against this machine's healthy-state points |
| Not marketing | Every decision is logged with its reasons |
| Measurable | Promotion precision on labelled replays, false-promotion rate, time-to-verify |

Evidence aggregation with preserved disagreement is the **supporting** mechanism, not a second headline.

### F.3 Edge data flow (Proposed Design)

```
Vibration source (CWRU/IMS replay; a real sensor in future)
  │  fixed-length windows (e.g. 1 s)
  ▼
Fingerprint extractor (deterministic DSP: RMS, peak, crest factor, kurtosis, skewness,
  │                    band energies; z-scored with this machine's baseline stats)
  ▼
Novelty gate ── Edge query(vib, filter machine_id, limit 1)
  │   ├─ near this machine's HEALTHY baseline → state=normal (counter only, no new point)
  │   ├─ near an existing local EPISODE → MERGE (occurrences++, last_seen)
  │   └─ far from both → open new EPISODE (point written) → alert UI
  ▼
Technician: note + action code (structured form, free text optional)
  ▼
Retrieval (offline) ── Edge query on local shard + fleet-mirror shard:
  │   prefetch vib (dense), note (dense), note_bm25 (sparse) → Fusion(RRF), filters
  │   → merge both shards' results, dedupe by id, show evidence + provenance
  ▼
Outcome verifier ── after the action, count consecutive windows back within the
  │                 healthy-baseline radius; stability window N (prototype parameter)
  ▼
Policy engine (deterministic, Part G) → KEEP_LOCAL / SHARE / MERGE / ARCHIVE / REJECT + reasons
  ▼
Outbox (SQLite, written BEFORE the shard write) → sync worker (retry + backoff) → Sync API
```

### F.4 Cloud data flow (Proposed Design)

```
Device (per-device token over TLS)
  ▼
Sync API (FastAPI): authenticate → tenant from token (never from payload) → schema and size
  │                 validation → reject unknown schema_version
  ▼
Idempotent ingest: upsert into Qdrant Server `fleet_events` with update_mode=insert_only,
  │                deterministic event_id → replays are no-ops
  ▼
Grouping: SUPERSEDED by K2 (27 Sep 2026). Case group key = (tenant, component, technician-confirmed
  │       fault_class); vib similarity across bearings measured P@3 0.429 (chance 0.277), so it is NOT used to
  │       group. Implemented in cloud/ingest.py (shared/ids.case_id).
  │       → attach to it, or create a new group (threshold = prototype parameter)
  ▼
Evidence tally: per (group, action_code): worked / failed / distinct sites / last_seen
  │   written to `fleet_knowledge` with conditional update on `version` (CAS) → retry on mismatch
  ▼
Disagreement flags (deterministic):
  │   same group + same action + both outcomes present          → DISPUTED
  │   same group + different root_cause claims                  → COMPETING HYPOTHESES
  │   same group + different actions that both worked           → ALTERNATIVES (not a conflict)
  ▼
Retraction: a reviewer can set status=retracted (tombstone). It is never hard-deleted.
  ▼
`fleet_knowledge` is kept as a single shard so devices can pull it as a (partial) snapshot
```

### F.5 Responsibility matrix

| Component | Responsibility | Type |
|---|---|---|
| **Qdrant Edge** | Local storage of points (vectors + payload); nearest-neighbour / hybrid / filtered queries; facets and counts for the UI; conditional upserts; applying server snapshots to the mirror shard | Infrastructure (Verified capabilities) |
| **Qdrant Server** | Fleet collections; server-side grouping queries; conditional (CAS) updates; producing (partial) snapshots for devices | Infrastructure (Verified; partial-snapshot endpoint to be tested) |
| **Fingerprint extractor** | Signal → ~20-dim vector | Deterministic software (DSP) |
| **Text embedding model** | Note → 384-d vector, offline | AI/ML (pretrained) |
| **BM25** | Note → sparse vector (Edge built-in) | Deterministic (statistical) |
| **Reranker** | Not used in MVP | — |
| **LLM** | Not used | — |
| **Novelty gate / outcome verifier** | Thresholded distance to baseline / episodes | Deterministic over ML retrieval |
| **Policy engine** | Local-vs-share decision with reasons | Deterministic software |
| **Human** | Notes, action code, outcome confirmation, retractions, dispute review | Human decision |
| **Outbox + sync worker** | Durable queue, retries, idempotent replay | Deterministic software |
| **Sync API** | AuthN/Z, validation, ingest, grouping, tallies | Deterministic software |
| **UI** | Inspection: memory, search, decisions, sync states, fleet evidence | Software |
| **Security layer** | Device tokens, TLS, tenant scoping, redaction, input limits | Deterministic software + infrastructure |

**Qdrant does not provide** any of the following. All are ours:
- sync orchestration;
- conflict or disagreement semantics;
- promotion policy;
- outbox durability;
- grouping logic;
- redaction.

### VERIFIED
- The Edge query/update/snapshot primitives referenced above.
- Server `update_mode` (release notes: "add `update_mode` parameter"); server conditional updates exist (1.16 "Conditional Updates" section heading).

### UNVERIFIED
- Exact Python signatures in 0.8.0.
- The server partial-snapshot endpoint path (`/snapshot/partial/create` as cited by the Edge sync page) against our server version. **We test on day 1. The fallback is a scroll-based pull** (`updated_seq > cursor`) upserted into the mirror shard.

### ASSUMPTIONS
- A single-shard server collection is acceptable for the fleet size we demo.

### OPEN QUESTIONS
- Qdrant Cloud free tier or local Docker for the server? Docker is recommended for demo control; Cloud is optional to show a real WAN round trip.

---

## PART G — Memory intelligence

### G.1 Lifecycle: what stays and what goes (checked against brief §3/§17)

| Brief stage | Keep? | Why |
|---|---|---|
| Capture | Keep | Signal windows + technician input |
| Normalize | Keep | z-score fingerprints against machine baseline; normalize text (lowercase, whitespace) before hashing |
| Classify | **Replace** | No classifier model. The "class" is the nearest case or episode (CBR retrieve), plus the technician's structured action code. |
| Embed | Keep | vib (DSP), note (dense), note_bm25 (sparse) |
| Store locally | Keep | Edge shard |
| Retrieve / Use | Keep | Hybrid query over local + mirror shards |
| Evaluate usefulness | **Keep, simplified** | Technician marks "this result helped / didn't" and we store a counter. No learned usefulness model. |
| Validate | **Core** | Outcome verifier (sensor stability) + technician confirmation |
| Detect duplicates | Keep | Novelty gate + content hash + deterministic IDs |
| Detect contradictions | Keep, **at the cloud** | A single device rarely holds contradicting evidence. Disagreement appears when sites meet. |
| Detect staleness | **Reframe** | Age ≠ wrong. We keep `last_confirmed_at` and show it. Eviction uses it only for resolved, low-value records. |
| Sharing policy | Core | Deterministic policy engine |
| KEEP/SHARE/ARCHIVE/DELETE/MERGE | Keep, with REJECT added | DELETE is only for raw windows after the summary is kept. Knowledge is tombstoned, never hard-deleted. |
| Sync / Resolve conflicts / Update shared | Keep | Part H |

### G.2 Memory types: the smallest useful set
We use four types. The academic taxonomy (episodic, semantic, working…) adds nothing here.

1. **baseline**: healthy-state fingerprints per machine, local only. Used by the novelty gate and the verifier.
2. **episode**: one abnormal-state occurrence on this machine. Fingerprint, occurrence count, first/last seen, optional note, action code, outcome. Local by default.
3. **fleet_case** *(mirror shard, read-only on device)*: server-aggregated evidence per fault signature.
4. **tombstone**: implemented as a `status=retracted` field, not a separate type.

"Private vs shared" is a *policy outcome* (the `share_state` field), not a separate type.

### G.3 Point schema (Edge, local shard). Each field has a stated reason.

| Field | Why it exists |
|---|---|
| point id (UUIDv5 of `device_id:episode_seq`) | Edge requires u64 or UUID (prior pass). Deterministic IDs make replay idempotent. |
| vectors: `vib` (≈20-d), `note` (384-d), `note_bm25` (sparse, IDF) | Sensor similarity, meaning, exact codes/shorthand |
| `type` (baseline/episode) | Filter the gate and search |
| `machine_id`, `machine_class`, `component` | Applicability filters |
| `device_id`, `site_id`, `tenant_id` | Provenance; tenant scoping on the server |
| `first_seen`, `last_seen`, `occurrences` | Novelty-gate merges; recency display |
| `note_text` (local), `note_redacted` (shareable form) | Raw note never leaves; the redacted form is only shared on opt-in |
| `action_code` (enum), `root_cause_claim` (enum, optional) | Structured fields make contradiction detection deterministic |
| `outcome` (pending/worked/failed), `verify_windows_ok`, `verified_at` | Promotion gate evidence |
| `share_state` (local/queued/synced/rejected) + `decision_reasons[]` | Policy transparency; UI |
| `content_hash` (sha256 of normalized fields) | Exact-duplicate detection |
| `schema_version`, `fp_version`, `text_model` | Safe evolution when feature definitions or models change |
| `version` (int) | CAS for the rare mutable edits |

Dropped from the brief's candidate list, and why:
- `confidence` as a float: fake precision. Replaced by counts.
- `freshness` score: derived at read time from timestamps.
- `parent_memory_id`: not needed. Merges bump counters and do not create children.
- `conflict_group_id` on device: conflicts live in server case groups.

### G.4 Duplicate / update / contradiction handling (five separate mechanisms)

| Kind | Mechanism | Where |
|---|---|---|
| Exact duplicate (replay, double submit) | Deterministic point/event IDs + `insert_only` → second write is a no-op | Edge + server |
| Near-duplicate sensor state (every tick looks alike) | Novelty gate: distance to nearest episode < τ₁ → MERGE (`occurrences++`). τ₁ is a **prototype parameter chosen by benchmark**. | Edge |
| Semantic duplicate case (two sites report the same fix for the same signature) | Grouping into a case_group (distance < τ₂, same component) → evidence count increments, no new card | Server |
| Updated memory (technician edits own episode) | `version+1`, conditional upsert requiring the old version. Rejected → reload and re-apply. | Edge (and server for shared) |
| Contradiction | Structured-field rules inside a case group: same action with both outcomes → DISPUTED; different root-cause claims → COMPETING. No NLI model and no LLM. | Server |

Why not rely on vector similarity alone? Two notes can read "bearing replaced, fixed" and "bearing replaced, NOT fixed" and still be very close in embedding space. The *outcome* is a structured field, so contradiction is checked on fields. Similarity is used only to decide which records are *about the same thing*. **Inference**, and we will test it: the adversarial case goes in the test suite (Part O).

### G.5 Trust: no invented score
Trust is shown as **evidence**, never as a number with fake precision:
- worked N, failed M, sites K, most recent confirmation date;
- whether each item was machine-verified (stability windows passed) or technician-only;
- status: active / disputed / retracted.

Ranking of fleet cases:
1. applicability filter (same component/class);
2. fused retrieval rank;
3. ties broken by verified count.

If we ever add a formula (Edge supports `Formula` rescoring; **Verified Fact**), its meaning has to be stated in the UI.

### G.6 Promotion criteria (Proposed Design)
An episode becomes **SHARE** only if **all** of these hold:
1. `action_code` is set (someone intervened);
2. the outcome verifier observed ≥ N consecutive post-action windows inside the healthy-baseline radius (N and the radius are prototype parameters, fixed by benchmark);
3. the technician confirmed "worked";
4. the privacy gate passed (G.7).

Failed outcomes (the fault persisted after the action) **are also shareable**, as "failed" evidence, because negative evidence is what prevents repeat mistakes. A generated or retrieved answer never becomes fleet knowledge by itself.

Honest limit: a fingerprint returning to baseline shows the *symptom* went away for N windows. It does not prove the root cause was correctly identified. The UI must say "symptom resolved for N windows", not "root cause confirmed".

### G.7 Privacy and sharing (decided before anything reaches the outbox)

| Data | Default | Reason |
|---|---|---|
| Raw signal windows | **Never leave** the device; delete after summarization | Bandwidth and site confidentiality |
| Baseline fingerprints | Local | Machine-specific, no fleet value |
| Free-text note (raw) | **Never leaves** | Names, site specifics |
| Note, redacted | Shared only if technician opts in **and** the redactor finds nothing it would have to remove after redaction | Deterministic regex for emails, phones and IDs, plus a configurable denylist (staff names, site names) |
| Text **embeddings** of notes | **Never shipped.** The server re-embeds the shared redacted text itself. | Embedding inversion recovered 92% of 32-token inputs exactly and recovered full names from clinical notes (Morris et al., EMNLP 2023; **Verified Fact**). Shown for specific models, but the risk is general in principle. |
| Fingerprint of an episode | Shared with a SHARE decision | Needed for fleet grouping. It is low-dimensional physical features, not text. **Inference:** low leakage, but still site-identifying metadata. |
| Metadata | `site_id` is shared as an opaque ID. No technician identity is sent (only a role). | Metadata leakage |

What the LLM does in this gate: **nothing.** The decision is fully deterministic. A regex/denylist redactor will miss free-form PII. The mitigation is to default the note to KEEP_LOCAL and make structured fields do the work.

### G.8 Policy engine (deterministic, ordered gates; the first hard failure wins)

```
1 Security/privacy gate  → raw note/raw signal? strip. Redaction residue? note → KEEP_LOCAL
2 Validation gate        → schema, sizes, enums, finite numbers; fail → REJECT (logged)
3 Duplicate gate         → content_hash seen / novelty merge → MERGE
4 Evidence gate          → no action or outcome pending → KEEP_LOCAL (reason: "awaiting verification")
5 Verification gate      → stability windows < N → KEEP_LOCAL (reason shows k/N)
6 Human gate             → technician confirmation missing → KEEP_LOCAL
7 Share                  → SHARE (structured fields + fingerprint [+ redacted note])
Retention (separate)     → resolved + low value + old → ARCHIVE; never auto-evict unresolved/high-severity
```

Every decision stores `decision_reasons[]`. The UI renders them. That is the "show WHY" requirement.

### VERIFIED
- The embedding-inversion result.
- Edge `Formula` and `update_mode` / `condition` semantics ("condition applies only to points that already exist").

### UNVERIFIED
- Whether a ~20-d DSP fingerprint separates fault classes *across different bearings*. This is the K2 kill test.

### ASSUMPTIONS
- Technicians will use a structured action-code picker.

### OPEN QUESTIONS
- The action-code taxonomy. **Proposed:** derive it from the MaintNet-annotated "action type" entities.

---

## PART H — Synchronization

### H.1 Push (device → server)
1. **Outbox-first write ordering.** Insert the outbox row with `status=pending` and the full point JSON into SQLite (WAL mode) *before* upserting to the Edge shard. On boot, re-apply every pending row to the shard. Upserts with deterministic IDs are idempotent. This solves the problem that SQLite and the shard cannot share one transaction.
2. **Worker.** When the link is up, send batches `POST /v1/sync/push {device_id, batch_id, events[]}`.
   - The server writes with `insert_only` on deterministic `event_id`s.
   - The server responds per event: `accepted | duplicate | rejected(reason)`.
   - The device marks rows `synced` or `rejected`, and increments `attempts` on transport failure.
   - Backoff is exponential with jitter, capped.
3. **States shown in the UI:** QUEUED → UPLOADING → SYNCED | FAILED → RETRYING | REJECTED. CONFLICT only appears for CAS on mutable records.

### H.2 Pull (server → device)
- **Qdrant's documented dual-shard pattern** (credited). The device keeps an immutable mirror of `fleet_knowledge` and applies partial snapshots using `snapshot_manifest` → server `/snapshot/partial/create` → `update_from_snapshot`. **Verified primitives, untested by us.**
- **Fallback if the partial-snapshot path misbehaves:** scroll `fleet_knowledge` with a filter on `seq > last_seq` and upsert into the mirror. It is simpler, and we would disclose it.
- Queries hit both shards; results are merged and deduped by ID. **Verified pattern.**

### H.3 Failure matrix

| Failure | Behaviour |
|---|---|
| Crash between outbox insert and shard upsert | Replayed on boot (idempotent) |
| Crash after send, before ack | Resend → server returns `duplicate` → marked synced |
| Server down | Retries with backoff; all local reads and writes continue |
| Partial batch | Per-event results; only failed events are retried |
| Corrupted payload | Validation → `rejected(reason)`; kept locally with the reason visible |
| Expired or invalid token | 401 → worker pauses, UI shows "auth required"; no data lost |
| Concurrent edit of the same shared case (CAS miss) | Server returns `conflict` with the current version → device shows CONFLICT, and a human re-applies or keeps both |
| Mirror refresh fails midway | The previous mirror stays in use. Refresh writes to a new directory and swaps only on success. **Proposed** (the Edge doc says to pause updates during restore). |
| Deletion / retraction | `status=retracted` propagates through the mirror; the UI shows a strikethrough; never silently vanishes |
| Embedding model change | New named vector (`create_dense_vector`) for the new model; lazy re-embed; queries use one model's vector consistently. The server re-embeds shared text itself, so devices on different model versions don't corrupt the fleet. |
| Schema change | `schema_version` in every event; the server accepts known majors, rejects others with a reason |
| Edge beta API drift | Pin `qdrant-edge-py==0.8.0`; all Edge calls go through one adapter module |

### H.4 Where Qdrant ends and our code begins
- **Qdrant:** point storage and queries, `insert_only` / condition semantics, snapshots and partial snapshots, snapshot apply on Edge.
- **Ours:** the outbox, retries, IDs, batch protocol, auth, validation, grouping, tallies, dispute flags, retraction workflow, the mirror swap, and the UI states.

### VERIFIED
- The Edge/server primitives listed.
- The Qdrant doc caveats: the queue is in-memory in their example, and updates are paused during restore.

### UNVERIFIED
- Qdrant Edge's crash-durability behaviour, i.e. what survives `kill -9` without `close()`. **This must be tested** (kill-test K4). Our outbox-first design is meant to be safe either way.

### ASSUMPTIONS
- SQLite in WAL mode is durable enough for the outbox on the demo laptop.

### OPEN QUESTIONS
- None.

---

## PART I — AI/ML (every model, and every model we rejected)

**Honest framing for judges.** The intelligence here is *retrieval and evidence*: similarity-based case retrieval over sensor fingerprints and notes (Case-Based Reasoning), plus verified-outcome aggregation. There is **no generative model in the decision path.** That is a deliberate choice, not a gap.

### I.1 Text embedding: `BAAI/bge-small-en-v1.5` (selected; fallback `sentence-transformers/all-MiniLM-L6-v2`)

| Item | Value | Label |
|---|---|---|
| Provider / runtime | BAAI; runs via Qdrant **FastEmbed** (the Edge docs pair Edge with FastEmbed, which runs fully offline after the model files are provisioned, with `local_files_only=True`) | Verified |
| Dimensions / size / license | 384-d, 0.067 GB, MIT (FastEmbed model table) | Verified |
| Fallback | all-MiniLM-L6-v2: 384-d, 0.090 GB, Apache-2.0 | Verified |
| Hardware | CPU is enough at this size | Inference; the demo-laptop specs are still unknown |
| Latency | **Not yet measured**; we will report p50/p95 ms per note on our laptop | — |
| Why this model | Small, permissive license, supported by FastEmbed, 384-d | Selection is a Proposed Design; **final pick by benchmark** on the MaintNet logbook (I.5) |
| Limits | English only; weak on heavy abbreviations, which is why BM25 is in the hybrid | Inference |
| Alternatives considered | Snowflake arctic-embed-xs/s (384-d, Apache-2.0), nomic-embed-text-v1.5 (768-d), jina-v2-small (512-d). All in FastEmbed. | Verified (listing) |

### I.2 Sparse: Qdrant Edge built-in BM25
- `Bm25(Bm25Config(...))`, with `embed_document` for stored notes and `embed_query` for queries, on a sparse vector with `Modifier.Idf`.
- Defaults: k=1.2, b=0.75, avg_len=256. **Verified Fact.**
- We set `avg_len` to the **measured** mean token count of our notes. This is a Proposed Design, not a borrowed number.

### I.3 Sensor fingerprint: deterministic DSP, no model
- Features:
  - time domain: RMS, peak, crest factor, kurtosis, skewness;
  - frequency domain: energy in N log-spaced bands of the FFT.
- Each feature is z-scored against this machine's baseline.
- **Distance: tested both ways.** Cosine discards overall amplitude, and amplitude (RMS) carries severity, so **Euclid on z-scored features is the default hypothesis** (Inference). The distance metric is immutable per vector once a shard is created, so this is settled on a throwaway shard first.
- Why no learned model: there is no pretrained embedder for raw vibration, as the prior pass noted. An autoencoder would need clean training data, would be poisonable, and would be harder to explain. We revisit it only if K2 fails.

### I.4 Components we evaluated and rejected

| Component | Decision | Reason |
|---|---|---|
| Reranker (e.g. bge-reranker-base, 1.04 GB) | **Not in MVP** | Corpus is small (hundreds to thousands of cases). Adds ~1 GB and latency. Revisit only if hybrid Precision@3 is poor. |
| NLI contradiction model | **Rejected** | Outcomes and claims are structured fields, so rules are exact and testable. NLI would add false positives on short maintenance notes (Inference). |
| Local LLM | **Rejected from core** | No stage needs text generation. It adds hallucination and prompt-injection surface, and hardware is unknown. The only defensible optional use is drafting a readable summary of retrieved evidence, clearly labelled and outside the decision path. Classed "impressive but unnecessary". |
| Custom-trained model | **Rejected** | No labelled fleet data exists. The pretrained + DSP combination has to fail measurably first. |
| LLM memory ops (Mem0-style ADD/UPDATE/DELETE) | **Rejected** | Mem0's DELETE-on-contradiction is the opposite of what a safety-relevant fleet needs, which is to keep disagreement. |

### I.5 Jev AI: **Jev AI should not be used here.**

What it is (**Verified Fact**, via The Register 16 Sep 2026, flaviocopes.com, typesafe.ai):
- TypeSafe AI's "System One" decision model. It returns typed probabilistic answers (Choice / Score / Noul) about a text or JSON state, instead of generated text.
- Advertised latency is ~70–500 ms. Input costs $0.042 per million tokens; output tokens are free. Context is about 64k tokens.
- It is **cloud-only**: `POST https://api.typesafe.ai/v1/systemone`, "Run both on the server".
- It is **proprietary** and in **early access**.
- The Register notes that probabilistic outputs "do[] not preclude the possibility of being incorrect".

The ten questions from the brief:

| # | Question | Answer |
|---|---|---|
| 1–3 | What is it, what can it do, how do you integrate it? | A cloud REST decision classifier (see above) |
| 4 | Resources | None locally; needs network + an API key |
| 5 | **Offline?** | **No** |
| 6 | Cloud dependency | **Yes**, a third-party one |
| 7 | Does it add something we can't build? | For our decisions (share or not, duplicate or not, contradiction or not), **no**. Structured fields make them exact rules. |
| 8 | Does it improve the architecture? | No. It moves the privacy decision off-device, which violates "decide BEFORE sync". |
| 9 | Complexity | Adds an external dependency, an access request (early access) and a key-management burden |
| 10 | Privacy / security | Shared knowledge would go to a fourth party. The key must be held server-side. |

The only role that doesn't break a principle is server-side classification of *already-shared* redacted notes into action codes. Even there, our structured picker makes it unnecessary. Include it only if it is a sponsor requirement, which **I could not verify** (the event pages available didn't list it).

### I.6 Evaluation (every model is measured, not asserted)
- **Text retrieval.** Annotated Maintenance Logbook: 6,169 aircraft-engine problem–action records, annotated problem type / part / location / action type / action part, CC BY 4.0, Zenodo, derived from MaintNet. **Verified Fact.**
  - Query = problem text. Relevant = records with the same (problem type, part).
  - Compare dense-only vs BM25-only vs RRF-hybrid on Precision@3 / Recall@10 / MRR.
  - This is a proxy domain (aviation, not motors), and we say so.
- **Sensor retrieval / novelty.** CWRU with a **bearing-level split** (Part O).
- **Latency.** p50/p95 on our laptop for: embedding one note; each query type at 1k / 10k / 50k points.

### VERIFIED
- The FastEmbed model table, the Edge BM25 API, Jev facts, dataset facts.

### UNVERIFIED
- All latencies and quality numbers. Not yet measured.

### ASSUMPTIONS
- bge-small beats MiniLM on maintenance shorthand. This is to be tested, not claimed.

### OPEN QUESTIONS
- Demo laptop CPU/RAM.

---

## PART J — Qdrant: exactly who does what

**Qdrant Edge (`qdrant-edge-py` 0.8.0; Apache-2.0; CPython ≥3.10; wheels for Linux x86-64/ARM64, macOS, Windows x86-64; beta).** **Verified Fact**
- Embedded, in-process shard: "no background services"; single-node.
- Config: named dense vectors, sparse vectors, quantization (Scalar, Product, Binary, TurboQuant), `on_disk_payload`. The Python `EdgeConfig` always needs vectors or sparse vectors declared.
- Writes: `upsert_points(points, condition, update_mode)` with Upsert / InsertOnly / UpdateOnly. The condition applies only to existing points. Also `set_payload`, `delete_points_by_filter`, `create_field_index`, `create_dense_vector` / `create_sparse_vector` / `delete_vector_name`.
- Reads: `query(QueryRequest(prefetches, query=Nearest|Fusion|OrderBy|Formula|Mmr|Sample, filter, score_threshold, params, limit…))`, plus `scroll`, `retrieve`, `count`, `facet`. `query_groups` and `search_matrix` are **Rust only**.
- BM25 built in. Dense embeddings are *not* built in; they come from FastEmbed.
- No background optimizer: call `optimize()` explicitly. WAL is pre-allocated at 32 MB (shrinkable only from Rust).
- Snapshots: `unpack_snapshot`, `snapshot_manifest`, `update_from_snapshot`.

**Qdrant Server (latest release seen: v1.19.1).** **Verified**, though the fetched release list showed conflicting years, so treat the dates as unverified.
- Hosts `fleet_events` and `fleet_knowledge`.
- Conditional updates (1.16) and `update_mode` (1.17).
- Snapshots, including partial snapshots for Edge.
- Payload-based multitenancy. This is from Qdrant docs I did not re-open in this pass.

**Our application** does everything in F.5 marked "ours".

**"Qdrant demo" test.** Would a Qdrant judge see meaningful Edge use? Yes, if we visibly use all of these:
- named vectors, three on one point;
- hybrid prefetch + fusion on device;
- filters and payload indexes;
- conditional upserts for idempotency and CAS;
- facets for the dashboard;
- `optimize()` scheduling;
- **the documented dual-shard partial-snapshot sync.**

If we only use a single dense vector with `query`, the answer is no.

---

## PART K — Security threat model

Prototype scope. We never claim "no security bugs". Format: **Threat → Attack → Impact → Mitigation → Residual → Test.**

| Threat | Attack | Impact | Mitigation (Proposed) | Residual risk | Test |
|---|---|---|---|---|---|
| Stolen device | Read the shard/SQLite from disk | Local notes exposed | OS full-disk encryption (declared, not built by us); raw notes stay local; token revocable server-side | Local data readable if the disk is unencrypted. **We do not claim app-level encryption at rest.** | Revoke token → next push 401 |
| Malicious / compromised device | Submits fabricated "worked" outcomes | Fleet poisoned | Evidence counts are shown per site; one site can't create "consensus"; disputes flagged; reviewer retraction; per-device rate limits | A determined insider with valid credentials can still add false evidence | Inject fake events from one device → tally shows 1 site, DISPUTED where contradicted |
| Unauthorized sync | Push without or with another's token | Pollution / cross-tenant | Per-device bearer token (hashed server-side); tenant & device derived from token, **payload `tenant_id` ignored** | Token theft (below) | Push with no token / wrong tenant → 401/403 |
| Credential theft | Reuse a stolen token | Impersonation | Short-lived tokens + rotation are *future*; MVP: revocation list | Stolen token valid until revoked | Revoked-token test |
| Replay / duplicate submission | Resend old batch | Double counting | Deterministic `event_id` + `insert_only` → duplicates ignored; tallies count distinct events | None known | Send same batch 3× → counts unchanged |
| Malicious payload | Oversized vectors, NaN, huge strings, extra fields | DoS / corruption | Pydantic schema, fixed vector dims, finite floats, length caps, enum fields, body size limit | Parser bugs in dependencies | Fuzz cases in test suite |
| Poisoned knowledge | Plausible wrong fix | Wrong repairs | Outcome-verified promotion; failed outcomes shared; retraction tombstones | Verifier can be fooled by replayed healthy data from a compromised device | Replay-healthy-data attack documented as residual |
| Prompt injection (direct/indirect) | Malicious text in notes | Would matter if an LLM read notes | **No LLM consumes stored text** → attack surface absent by design; UI escapes all text (XSS) | Returns if an LLM summarizer is added later | XSS payload in note renders inert |
| Embedding leakage | Invert shared embeddings | PII recovery | Text embeddings never shipped; server re-embeds redacted text | Fingerprints may identify a site/machine | Assert outbound payload has no `note` vector |
| Metadata leakage | Infer technician/site from fields | Privacy | Opaque site IDs, role not name | Timing patterns | Payload schema test |
| Tenant isolation | Query another tenant's knowledge | Data breach | Server filters every query by token-derived tenant; separate mirror per tenant | Bug in filter = leak | Cross-tenant query test |
| Compromised server | Attacker controls fleet | Poisons all mirrors | TLS; mirror is read-only and never overwrites local episodes; retraction auditable | Full server compromise remains critical | Out of scope, documented |
| Insecure logs / secrets | Tokens/notes in logs, keys in repo | Leak | No note text or tokens in logs; `.env` + `.gitignore`; secret scan before push | Human error | `gitleaks`-style scan in CI (if time) |
| Insecure transport | MITM | Tampering | HTTPS for any non-localhost server | Demo on localhost is plain HTTP — disclosed | — |
| Supply chain | Malicious/compromised package | Code exec | Pinned versions + hashes in lockfile; minimal deps | Upstream compromise | `pip-audit` run |
| Sync DoS | Flood pushes | Server load | Batch caps, per-device rate limit | Distributed flood | Load script |

(Current OWASP API Security Top 10 categories map onto rows above — broken object/tenant authorization, broken authentication, unrestricted resource consumption; I did not re-open the OWASP page in this pass.)

---

## PART L — Scalability (reasoned, not claimed)

What happens per device count, and where it breaks first:

| Devices | What changes |
|---|---|
| 1 | Everything local; Edge easily holds 10⁴–10⁵ small points (Qdrant's own in-house figure: ~0.1 ms for 10k vectors on iPhone 16 Pro — theirs, not ours) |
| 10 | Server ingest trivial; mirror snapshot small |
| 100 | Per-group CAS contention appears for common faults (many sites hitting the same group) |
| 1,000 | **First bottleneck: mirror distribution** — every device pulling snapshots; partial snapshots and scheduled/jittered pulls required |
| 10,000+ | **Second: case-group write hot-spots** (CAS retry storms on popular groups) → move tallies to an append-only event table + periodic aggregation instead of CAS per event. **Third: tenant partitioning** — per-tenant collections or payload multitenancy with tenant index, and per-tenant mirrors |

Other factors: embedding throughput is per-device and tiny (notes are rare events); vibration fingerprinting is O(window) DSP; bandwidth per episode is ~a few hundred bytes of structured data + ~20 floats (to be measured) versus raw waveform streaming. **We will only demonstrate N simulated devices on one laptop; anything beyond is reasoning, and will be labelled so.**

### VERIFIED (Parts K–L)
- Embedding inversion risk.
- Qdrant's in-house 10k-vector latency figure (theirs, not ours).

### UNVERIFIED (Parts K–L)
- Every mitigation above is a design until its test passes.
- The OWASP mapping (page not re-opened).
- All scale behaviour.

### ASSUMPTIONS (Parts K–L)
- Demo traffic is localhost. Any remote server needs TLS.

### OPEN QUESTIONS (Parts K–L)
- Will judges probe multi-tenant isolation? If so, show the cross-tenant test.

---

## PART M — Implementation

### M.1 Stack (Proposed Design)
- **Language:** Python 3.11.
- **Edge:** `qdrant-edge-py==0.8.0` (pinned), `fastembed`, `numpy`, `scipy`; `sqlite3` from the stdlib.
- **APIs:** FastAPI for the edge-local API (one process per simulated device) and for the cloud Sync API.
- **Server:** Qdrant Server in Docker, pinned; `qdrant-client` on the server side.
- **UI:** one static HTML/JS page per process, served by FastAPI. No build step.
- **Tests:** pytest.

**Build environment constraint (discovered today).** This cloud sandbox **cannot reach PyPI, Hugging Face, crates.io or raw GitHub** (egress allowlist). The build, the model download and the CWRU download have to run on the user's laptop ("y4-max"), which is linked to this session. We still need its specs and permission to use a project folder.

### M.2 Repository layout
```
edge/        fingerprint.py  gate.py  verifier.py  policy.py  store_edge.py (ONLY file importing qdrant_edge)
             outbox.py  sync_worker.py  mirror.py  api.py  ui/
cloud/       api.py  auth.py  ingest.py  grouping.py  tally.py  store_server.py
shared/      schema.py (pydantic, schema_version)  ids.py (uuid5)  redact.py
data/        fetch_cwru.py  fetch_logbook.py  splits.py (bearing-level split)
bench/       retrieval_text.py  retrieval_vib.py  latency.py  sync_partition.py  bandwidth.py
tests/       unit/  failure/  security/  ai/
demo/        run_demo.sh  scenario.py
docs/        RESEARCH.md ARCHITECTURE.md DECISIONS.md THREATS.md BENCHMARKS.md DEMO.md
```

### M.3 Edge-local API (ours, not Qdrant's)
```
POST /ingest/window            (demo replay feeds windows)      → gate result
GET  /episodes?status=         GET /episodes/{id}
POST /episodes/{id}/note       POST /episodes/{id}/action       POST /episodes/{id}/confirm
POST /search {text?, episode_id?, use_fleet: bool}  → results + per-leg ranks + provenance
GET  /decisions/{episode_id}   (policy outcome + reasons)
GET  /sync/status   POST /sync/now   POST /network {online: bool}   (demo partition switch)
GET  /stats   (counts, facets, shard size, outbox states, latency histogram)
```
**Cloud:** `POST /v1/sync/push`, `GET /v1/mirror/…`, `GET /v1/cases`, `POST /v1/cases/{id}/retract`, `GET /v1/disputes`.

### M.4 Offline behaviour by network state

| State | Works | Degrades |
|---|---|---|
| Online | Everything; push immediately; mirror refresh on schedule | — |
| Slow | All reads and writes are local, so the UI is unaffected; the worker times out and retries | Mirror freshness |
| Intermittent | Same; the outbox drains in bursts | Fleet evidence lags |
| Offline | Ingest, gate, search (local + last mirror), notes, verification, policy decisions | No new fleet evidence; SHARE items wait as QUEUED |
| Returns | Queue drains idempotently; mirror refreshes | — |

### M.5 Build order and time budget

**By 30 Sep (Round 1: GitHub + LinkedIn; the date is from the prior pass and I could not re-verify it today). A vertical slice that is real and tested:**
1. Day 1:
   - Spike test on the laptop: create a shard with 3 named vectors, upsert, hybrid prefetch + fusion, conditional upsert, facet, `optimize`, close/load, `kill -9` durability. **This day decides whether the design stands.**
   - Download CWRU; build the fingerprint extractor.
2. Day 2:
   - Novelty gate + episodes + search API.
   - Outbox-first write + sync worker + minimal cloud ingest (`insert_only`, idempotency test).
3. Day 3:
   - Verifier + policy engine with reasons.
   - Minimal UI.
   - README with **honest** claims.
   - Tests green; a first benchmark table, even if small.

**1–10 Oct: finals.**
- Fleet mirror via partial snapshot (with the fallback).
- Grouping, tallies, disputes, retraction.
- Full UI.
- Leakage-free benchmarks, partition test harness, security tests.
- Demo script + rehearsal, including network-off rehearsal and a backup recording.

### M.6 Must-have vs scope

| Category | Items |
|---|---|
| **MUST HAVE** | Edge shard with vib + note + bm25; novelty gate; offline hybrid search; verifier + policy with reasons; outbox + idempotent push; cloud grouping + tallies; mirror pull; Device A→B demo; UI with sync states; tests; measured latency & retrieval numbers |
| **USEFUL** | Dispute flags & retraction UI; partition test with N devices; bandwidth measurement; redaction opt-in; Qdrant Cloud WAN round-trip comparison |
| **IMPRESSIVE BUT UNNECESSARY** | LLM evidence summaries; reranker; spectrogram/vision embeddings; mobile app; real sensor hardware |
| **DANGEROUS SCOPE CREEP** | Custom model training; peer-to-peer sync; CRDTs/HLC; multi-tenant admin console; Rust rewrite; Jev integration; Android port |

---

## PART N — Demo (judge-facing, ~5 minutes)

**Setup.** One laptop runs three processes: **Device A (Site 1), Device B (Site 2), and Cloud (Qdrant Server in Docker + Sync API)**. Each device has its own `/network` switch, so the partition is visible. We also cut Wi-Fi once to prove nothing secretly calls out.

1. **Offline memory (A, offline).** Replay CWRU healthy data, then an inner-race fault. The gate flags an **unfamiliar state**: there is no local case, and the mirror is empty. Search returns "no confident match" rather than inventing one. Show the local query latency.
2. **Experience.** The technician picks action "replace bearing", adds a note containing a person's name, and then we replay healthy data. The verifier counts k/N stable windows up to N/N. The technician confirms.
3. **Decision.** The policy shows **SHARE** (structured fields + fingerprint), with the **note KEEP_LOCAL** because the redactor found a name. It also shows a second, unverified episode **KEEP_LOCAL: "awaiting verification 3/N"**. The outbox shows **QUEUED**.
4. **Connectivity returns.** QUEUED → UPLOADING → SYNCED. We press *resend*: the result is `duplicate`, the count is unchanged, and idempotency is proven.
5. **Cloud.** A case group is created with 1 site, worked=1. A pre-seeded Site 3 event, "lubrication → failed", sits in the same group and is shown as **alternative failed evidence**. A seeded contradiction (same action, opposite outcome) is flagged **DISPUTED**, not overwritten.
6. **Device B.** B refreshes its mirror while online, then goes **offline**. It replays the **same fault class from a different CWRU bearing file**, chosen from the held-out split. B's search shows **fleet evidence: "replace bearing — worked at Site 1 (machine-verified); lubrication — failed at Site 3"**, entirely offline.
7. **Proof panel.** Retrieval P@3 on the leakage-free split, latency p50/p95, bytes sent vs raw-waveform bytes avoided, and the partition-test result (0 lost writes over K runs).

The audience sees: **A learned → the machine confirmed it → the cloud aggregated it without erasing disagreement → B benefited offline.**

**Demo risks and mitigations.**
- Qdrant Server in Docker fails to start → pre-pull the image; keep a recorded backup video.
- Step 6 doesn't retrieve across bearings → use K2 results to pick honestly. If cross-bearing retrieval is weak, the text/action filters carry the match, and **we say that on stage.**
- Live DSP is slow → pre-computed windows, with replay timed.

---

## PART O — Success metrics (defined now, measured later; no invented targets)

| Area | Metric | Method |
|---|---|---|
| Sensor retrieval | Precision@3 / Recall@10 of fault class for a query window, **bearing-level split** (no bearing appears in both index and queries) | CWRU. We also report the leaky segment-level split *next to it* to show the gap. Leakage is **Verified** from two sources:<br>• arXiv 2407.14625: random segment splits let a model "memorize a signature of each specific signal", and splitting by load reuses the same physical bearings.<br>• Hendriks et al., *Mech. Syst. Signal Process.* 2022 ("Towards better benchmarking using the CWRU bearing fault dataset"): with independent bearings in the test set, accuracy fell "on average by approximately 45%". |
| Novelty gate | Precision/recall of "unfamiliar" flags vs ground-truth first-occurrence; merge rate | Replayed sequences; τ₁ swept, curve published |
| Text retrieval | P@3, R@10, MRR: dense vs BM25 vs RRF hybrid | Annotated Maintenance Logbook (6,169 records, CC BY 4.0) |
| Latency | p50/p95 per op: fingerprint, embed, gate query, hybrid query (1k/10k/50k points), local vs server round trip | `bench/latency.py`, method disclosed |
| Offline | Checklist of M.4 functions passing with the network switched off *and* Wi-Fi off | Scripted |
| Sync | Lost writes = 0, duplicates = 0, convergence across N devices with random partitions and crashes | `bench/sync_partition.py`, K runs, seed logged |
| Promotion | False-promotion rate (a SHARE where the fault persisted) on labelled replays; time-to-verify | Scripted scenarios |
| Conflicts | Every seeded contradiction flagged; no evidence lost; full provenance traceable | Tests |
| Storage | Points and shard bytes vs hours of replayed signal (with the gate on vs off) | Bench |
| Resources | CPU %, RSS, disk, bytes on the wire per episode | Bench |

**Honest dataset notes**
- **CWRU:** the Bearing Data Center page describes EDM-seeded faults of 0.007–0.040 in on a 2 hp motor, at 0–3 hp loads. **Verified.** **I could not verify an explicit license** on the pages I could reach, so we cite it as the source and do not redistribute the files.
- **Seeded vs natural faults:** CWRU faults are artificially seeded. NASA IMS is run-to-failure (natural degradation), and could show gradual drift if time allows.
- **Proxy domain:** the logbook dataset is aviation maintenance, a proxy domain for the text side.

### VERIFIED (Parts M–O)
- Sandbox egress limits (tested).
- The dataset facts and leakage literature cited.

### UNVERIFIED (Parts M–O)
- The Round 1 deadline of 30 Sep. It comes from the prior pass; today the event page only showed an Aug 27 – Oct 11 span.
- All metrics. None have been measured yet.

### ASSUMPTIONS (Parts M–O)
- The laptop can run Docker + 3 Python processes.
- The build happens on the user's machine.

### OPEN QUESTIONS (Parts M–O)
- Laptop specs.
- Docker availability.
- A folder on the laptop for the repo.
- Whether teammates will do anything, such as the LinkedIn post or demo video.

---

## PART P — Limitations: what this system cannot prove

- **Sensor-verified is not the same as root cause confirmed.** It only shows the symptom disappeared for N windows.
- **Nothing here validates industrial deployment.** There is no real plant, no real technician, no real sensor hardware, and CWRU faults are artificial.
- **Multi-device behaviour is simulated on one laptop.** There are no real networks, no clock skew, and no field hardware.
- **Redaction misses things.** A regex/denylist redactor misses free-form PII, which is why notes default to local.
- **Security is prototype-grade.** There is no app-level encryption at rest, no mTLS and no token rotation.
- **The Edge crash-durability semantics are ours to test**, not documented guarantees.
- **Scalability past a handful of simulated devices is reasoning, not measurement.**
- **Cross-bearing retrieval quality is unknown until K2 runs.**

---

## PART Q — Kill test (explicit triggers)

| # | Finding | Action |
|---|---|---|
| K1 | Day-1 spike: `qdrant-edge-py` 0.8.0 can't do named dense + sparse vectors or fusion on our platform | **Change architecture:** app-side RRF. If named vectors fail as well, split into two shards. We do not abandon. |
| K2 | Bearing-level split: fingerprint P@3 is near chance (≈1/number of classes) | **Change architecture:** sensor leg becomes triage only; fleet matching relies on component + fault-class filters + text; say so. If the gate can't even separate healthy from faulty → **reduce scope** to text memory + verification via RMS thresholds. |
| K3 | Verifier can't distinguish "fixed" replays from "still faulty" replays | **Abandon the differentiator** (outcome-verified promotion) → fall back to technician-only confirmation and say we lost our main claim; reconsider pivot. |
| K4 | Edge loses acknowledged writes on `kill -9` even with outbox-first ordering | Make SQLite the system of record and treat the shard as a rebuildable index (documented change) |
| K5 | Partial-snapshot pull unworkable on our server version | Scroll-based pull fallback (disclosed) |
| K6 | We cannot have a tested vertical slice by the Round 1 deadline | **Reduce scope** to M.5 day-1/2 items; submit honest README; do not submit untested claims |
| K7 | Evidence shows a mainstream product already does outcome-verified fleet promotion | **Change positioning** to "open, inspectable implementation on Qdrant Edge", drop novelty language |
| K8 | Qdrant Edge is replaceable by a flat numpy search with no loss for our design | Would mean Qdrant isn't central → re-check that hybrid, filters, facets, CAS and snapshot sync are all genuinely used; if not, redesign before submitting |

**Abandon** only if K2 and K3 both fail. That would leave no sensor-grounded story, and the project would collapse into "offline notes + sync", which is where the competing team already is.

---

## PART R — Final verdict

| Dimension | Assessment |
|---|---|
| Technical feasibility | High for all components; every primitive is documented. Two unknowns (fusion in 0.8.0, partial snapshot) have fallbacks. |
| Hackathon feasibility | **The weakest dimension.** A solo builder with 3 days to Round 1 and 13 days to finals. Feasible only with the vertical-slice order in M.5 and strict scope. |
| Differentiation | Moderate. Outcome-verified promotion + disagreement-preserving evidence is defensible. Nothing in it is individually novel, and we must not claim otherwise. |
| Demo strength | Strong if step 6 works across bearings, and credible even if it doesn't, as long as we are honest. |
| Real-world usefulness | Plausible but **unvalidated**; there is no practitioner input yet. |
| Security | Prototype-grade, with an honest threat model; there is no LLM surface. |
| Scalability | Reasoned bottlenecks, not measured beyond a few processes |
| Qdrant integration | Meaningful: named vectors, hybrid fusion, filters, facets, conditional updates, dual-shard snapshot sync |
| AI/ML necessity | Honest and minimal: a pretrained text embedder + BM25 + DSP retrieval. There is no LLM, by design. **Risk: judges who expect "GenAI".** Answer: in a safety-relevant fleet, generated text is a liability, not a feature. |
| Operational complexity | Moderate: three processes, SQLite, Docker |
| **Biggest technical risk** | Cross-bearing fingerprint retrieval (K2) |
| **Biggest product risk** | No practitioner validation of the persona or workflow |
| **Biggest demo risk** | Live multi-process + network toggling on stage; have a recorded fallback |
| **Biggest false assumption (corrected)** | That "conflict-aware sync" was our gap. It is commodity. |

**Decision: MODIFY.**
- Keep the domain, the persona and the Qdrant-centred edge design.
- Change the headline to outcome-verified fleet learning.
- Cut HLC, the reliability score, the LLM and Jev.
- Credit the dual-shard pattern to Qdrant.
- Make every claim either measured or labelled.
- Start the day-1 spike immediately. Its results decide K1/K4, and they come before any UI work.

---

## Appendix 1 — "Qdrant + CRUD" failure test

**What would make us look like "Qdrant Edge + embeddings + CRUD dashboard":**
- one dense vector;
- a search box;
- a sync button that uploads everything;
- a conflict log nobody can trigger;
- claims without numbers.

**Safeguards, each of which must be visible on stage:**
1. The novelty gate running *continuously* on the sensor stream: automated, and not user-initiated.
2. A promotion decision that *refuses* to share something, and says why.
3. Outcome verification driven by the sensor data.
4. Fleet evidence that keeps a disagreement.
5. Idempotent replay shown live.
6. Leakage-free benchmark numbers.

---

## Appendix 2 — Judge questions (prepared answers)

- **Why Qdrant Edge?** The device and the fleet use the same engine and data model. It gives on-device hybrid (dense + BM25 + fusion), filters, facets and conditional upserts, and devices are seeded and refreshed from server snapshots with no transformation layer. We use all of these (Part J).
- **Why not SQLite?** SQLite can't do nearest-neighbour search over vibration fingerprints or meaning-based search over notes. We *do* use SQLite, for the outbox, because that's what it's good at.
- **Why not Couchbase Lite or ObjectBox?** Both are credible and have sync with conflict resolution. We didn't benchmark them and don't claim to be better. What we built is the *policy and evidence layer*, which none of them ships. Qdrant Edge gives snapshot-compatible fleet mirrors with the server.
- **Why not sync everything?** Bandwidth (raw vibration), privacy (notes), and *correctness*: unverified fixes shouldn't propagate.
- **Why does AI decide what to share?** It doesn't. Sharing is a deterministic rule set with logged reasons. The ML is in retrieval.
- **How do you stop wrong knowledge spreading?** Sensor-verified outcomes, failed outcomes shared as evidence, disputes flagged, and retraction tombstones. The residual risk is a compromised device replaying healthy data.
- **Conflicts?**
  - Write conflicts: deterministic IDs, plus CAS on the few mutable records.
  - Knowledge conflicts: kept side by side as evidence and flagged. Never overwritten.
- **What if sync fails?** Outbox-first write ordering, per-event acks, retries, idempotent replay. The partition-test numbers are on screen.
- **At 1,000 devices?** Mirror distribution breaks first, then CAS hot-spots, then tenancy (Part L). We haven't measured it and say so.
- **What's AI here?** Pretrained text embeddings, BM25, and similarity-based case retrieval over sensor fingerprints (CBR). There is no LLM, deliberately.
- **What did you build yourselves?** Everything in F.5 marked ours: the gate, the verifier, the policy engine, the outbox and protocol, grouping and tallies, the dispute rules, the UI, and the benchmarks.
- **What's novel?** The combination and its measurement, not any single piece. We'll name the precedents: CBR, CouchDB conflicts, Qdrant's anomaly triage.
- **Isn't CWRU accuracy famously inflated?** Yes. That's why we report a bearing-level split next to the naive one.

---

## Appendix 3 — Decision log

**D1. Headline differentiator**
- Problem: "Conflict-aware sync" is commodity technology.
- Options: sync-as-differentiator; LLM memory ops; outcome-verified promotion.
- Selected: outcome-verified promotion.
- Why: it is measurable and demoable, it uses sensor vectors meaningfully, and we found no overlap in the products we checked.
- Evidence: Part D.
- Tradeoff: it depends on replay realism.
- Risk: K3.
- Invalidated if: the verifier can't separate fixed from unfixed replays.

**D2. No LLM, no Jev**
- Problem: whether to add generative or decision AI.
- Options: local LLM; Jev; deterministic rules.
- Selected: deterministic rules.
- Why: offline requirement, privacy-before-sync, and testability.
- Evidence: Jev is cloud-only (verified).
- Tradeoff: less "AI-looking".
- Risk: judge perception.
- Invalidated if: the rubric explicitly rewards generative AI, in which case add a labelled, optional summarizer outside the decision path.

**D3. Append-only evidence + CAS; drop HLC**
- Problem: concurrent edits.
- Options: HLC + LWW; CRDTs; append-only + CAS.
- Selected: append-only + CAS.
- Why: the simplest correct option, using verified Qdrant primitives.
- Tradeoff: tallies must be aggregated.
- Risk: CAS hot-spots at scale.
- Invalidated if: devices must edit shared records concurrently and often.

**D4. Dual-shard mirror (Qdrant's pattern)**
- Selected: Qdrant's documented pattern, credited to them.
- Risk: the partial-snapshot endpoint.
- Fallback: scroll pull (K5).

**D5. Evidence counts instead of a reliability score**
- Why: no fake precision.
- Invalidated if: case volumes grow large enough for calibrated statistics.

**D6. Euclid on z-scored fingerprints as the hypothesis**
- Why: cosine discards amplitude.
- Decided by: a benchmark on a throwaway shard, because the distance metric is immutable once a shard is created.

**D7. Text embeddings never leave the device**
- Evidence: embedding inversion (verified).
- Tradeoff: the server must embed text itself.

**D8. Leakage-free evaluation**
- Evidence: CWRU leakage literature (verified).
- Tradeoff: lower, honest numbers.

**D9. Outbox-first write ordering**
- Problem: SQLite and the Edge shard can't share one transaction.
- Invalidated if: K4 shows the shard loses acknowledged writes, in which case SQLite becomes the system of record.

---

## Sources (fetched in this pass unless marked prior-pass)

- Qdrant Edge docs: [overview](https://qdrant.tech/documentation/edge/) · [quickstart](https://qdrant.tech/documentation/edge/edge-quickstart/) · [API: updating data](https://qdrant.tech/documentation/edge/edge-api/updating-data/) · [API: reading data](https://qdrant.tech/documentation/edge/edge-api/reading-data/) · [BM25](https://qdrant.tech/documentation/edge/edge-bm25/) · [FastEmbed on-device](https://qdrant.tech/documentation/edge/edge-fastembed-embeddings/) · [sync patterns](https://qdrant.tech/documentation/edge/edge-data-synchronization-patterns/) · [sync with server](https://qdrant.tech/documentation/edge/edge-synchronization-guide/)
- [Qdrant blog: Memory at the Edge (16 Jun 2026)](https://qdrant.tech/blog/qdrant-edge-on-device-vector-search/) · [Qdrant 1.16](https://qdrant.tech/blog/qdrant-1.16.x/) · [Qdrant 1.17](https://qdrant.tech/blog/qdrant-1.17.x/) · [Qdrant releases](https://github.com/qdrant/qdrant/releases)
- [PyPI qdrant-edge-py 0.8.0](https://pypi.org/project/qdrant-edge-py/) · [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
- Jev: [The Register, 16 Sep 2026](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711) · [flaviocopes.com/jev](https://flaviocopes.com/jev/) · [typesafe.ai](https://typesafe.ai)
- Competitors: [Couchbase vector search at the edge](https://www.couchbase.com/blog/vector-search-at-the-edge-with-couchbase-mobile/) · [Couchbase Lite conflicts](https://docs.couchbase.com/couchbase-lite/current/android/conflict.html) · [ObjectBox Sync](https://objectbox.io/sync/) · [ObjectBox on-device vector DB + sync](https://objectbox.io/dev-how-to/on-device-vector-database-sync/) · [CouchDB conflicts](https://docs.couchdb.org/en/stable/replication/conflicts.html) · [Mem0 paper](https://arxiv.org/abs/2504.19413) · [Graphiti](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/) · [FieldEdge repo](https://github.com/choksi2212/code-cubicle-qdrant)
- Data and evaluation: [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) · [CWRU leakage (arXiv 2407.14625)](https://arxiv.org/html/2407.14625v1) · [Towards better benchmarking using CWRU (MSSP 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0888327021010499) · [Annotated Maintenance Logbook (Zenodo, CC BY 4.0)](https://zenodo.org/records/17903357) · [MaintNet](https://arxiv.org/abs/2005.12443) · [NASA PCoE data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- Security: [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816)
- Prior-pass (not re-fetched today): NIST SP 800-82r3; Maritime Executive offshore connectivity; arXiv 2301.00484; IBM Maximo docs; Augury materials; Dawid & Skene 1979; CBR literature.
