# PROBLEM.md — Machine Memory at the Edge

## 1. What is the problem?

Industrial machines (pumps, motors, compressors) periodically develop faults — abnormal vibration,
temperature, or pressure patterns. A plant technician has to answer, quickly:

> "Have we seen something like this before? What did we try? Did it work?"

Today that knowledge is fragmented — in a technician's memory, in disconnected notes, or trapped in one
machine's history without reaching others that could benefit. Backed by real data:
- Roughly half of the world's offshore oil rigs still lack reliable broadband today (Maritime Executive).
- A peer-reviewed paper on remote Industry 4.0 sites states cloud data centers are not latency- or
  reliability-feasible for such operations (arXiv 2301.00484).
- Industrial control networks are commonly, deliberately segmented from the internet for security — NIST
  Special Publication 800-82 Revision 3 (2023).
- Qdrant's own published reasoning for edge-first retrieval names FIVE factors, in this order: **latency,
  connectivity, cost, privacy, isolation** — connectivity/"no signal" is one of five, not the headline.

## 2. Who is this for? (exact user)

**A plant technician or reliability engineer** standing in front of a misbehaving machine, right now — not
a vague "robot user" or "industrial user." The decision they're making:
> "What evidence from previous incidents should I consider before deciding what to do with this machine?"

## 3. Why can't existing systems already solve this? (verified, corrected)

- **IBM Maximo** already offers natural-language access to asset history/failure patterns, AI fusing
  sensor+text data, and offline-capable mobile inspection.
- **Augury** already runs diagnostics on edge hardware, rated 4.7/5 for offline resilience/sync,
  validated across 1.1B+ hours of real machine data.
- **Qdrant itself already publicly names "Edge Anomaly Triage," explicitly including an unusual-machine-
  state example, as one of its own showcased patterns.** An earlier version of this document claimed the
  industrial/machine-state angle was untouched by Qdrant's own demos — that claim was too strong and is
  corrected here. We do **not** claim domain novelty.
- **Qdrant's own sync documentation** provides snapshot-based bootstrap and a dual-write pattern, but
  **explicitly leaves conflict resolution, durable queuing, and retry/error handling to the developer** —
  confirmed directly from their own docs.

**Our actual, narrow, defensible gap:** not "nobody does industrial edge AI" — but that no source checked
(Qdrant's own sync docs, IBM Maximo, Augury) implements an explicit, human-visible conflict state for when
two independent devices disagree about the same incident, with a tested, durable, conflict-aware sync
layer sitting under it.

## 4. Why does Qdrant Edge matter here specifically?

Qdrant Edge runs in-process (described by Qdrant itself as "SQLite for vector search," ~11MB install, no
separate server) — required because each machine must search its own history instantly, with zero
network dependency, and because it provides real hybrid search (dense + sparse/BM25, fused via Reciprocal
Rank Fusion) so exact fault codes aren't missed by meaning-only search.

## 5. Why isn't this "just another predictive-maintenance system"?

> We do not claim a new domain. We claim depth: a conflict-aware sync layer that is built and tested (not
> just described), measured retrieval quality against real labeled data, and a durable outbox — the exact
> parts Qdrant's own documentation leaves to the developer. We are the production layer the sponsor's own
> pattern list leaves out, not a new discovery.

## 6. Safety and liability stance

The system **never recommends an action.** It shows similar past incidents, evidence, and outcomes — the
technician always decides.

## 7. Honest competitive position (do not overclaim on stage)

We are ahead only if the sync layer is genuinely built and tested, retrieval quality is measured, and the
data is real. If another strong team also builds conflict-aware sync, we tie on that axis and must win on
proof and clarity — not on being the only team to think of it.
