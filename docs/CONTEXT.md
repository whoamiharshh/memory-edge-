# CONTEXT.md — Complete Project History, Start to Present (v3, exhaustive)

This is the full chronological record of how this project was decided, not just its final shape. Every
idea that was tried and rejected is kept here, with the real reason it was rejected — nothing removed.

---

## Phase 1 — Choosing among the three hackathon problem statements

The hackathon (Code Cubicle 6.0, by Geek Room) offered three tracks, all under the theme "AI-Powered
Intelligence for the Real World":
1. **Cloudinary — media intelligence platform.** Rejected: this closely mirrors Cloudinary's own built-in
   demo pattern (upload media -> auto-tag -> dashboard); very likely many teams produce near-identical
   projects.
2. **AI-powered data intelligence platform** (plain-English prompt -> AI agent -> scraper -> dashboard).
   Rejected: judged the most overused hackathon pattern currently in circulation; live scraping demos are
   fragile on stage; "gather from permitted sources" invites legal/ethics scrutiny from judges.
3. **Qdrant: AI-Powered Edge Memory & Intelligence Platform.** **Chosen.**

## Phase 2 — First idea: offline field-notes assistant (rejected)

Initial direction: an offline assistant for health workers/disaster-relief volunteers who log notes
(text+photo) with no connectivity, syncing to a coordinator dashboard when back online. Rejected after
the user pushed back hard, correctly, on several fronts:
- "No signal" is a weak argument in places with Starlink/good 5G.
- The specific example ("which patients have fever") was generic and unconvincing — a real coordinator
  would just ask a nurse directly, not search a vector DB.
- Multimodal photo tagging risked sounding like "will you just be Google Search?"

## Phase 3 — Second idea: visual memory / "where did I put my tool" (rejected)

Pivoted to a personal/worker visual-memory device (continuous camera capture, object recall, "where's my
drill") — modeled loosely on Qdrant's own published concept. This was compared explicitly by the user to
Alexa ("remembers where you kept what") and defended on the grounds that it works from vision automatically
rather than requiring the user to say things aloud. Ultimately abandoned once direct research (Phase 5)
confirmed Qdrant's own official public demo is exactly this pattern (a robot walking a home, remembering
objects via YOLOE+SigLIP2+Florence-2, cutting the network live) — building this would read as copying the
sponsor's own showcase, not an original idea.

## Phase 4 — Third idea: local-first developer codebase assistant (rejected)

Pivoted again to a compliance-driven angle: a tool that indexes a developer's own codebase locally (so
proprietary source code never leaves the machine), syncing only derived, policy-approved insights to a
shared team view. Grounded in real, verified precedents: Samsung's 2023 company-wide ban on generative AI
tools after employees leaked internal source code into ChatGPT, and Xperi's real, existing "Source Code
Handling Annex" governing contractor access to source code. This idea was technically sound and the
compliance argument held up under scrutiny — but was set aside once the actual problem statement text was
checked closely: the brief's named example devices are "robots, industrial systems, kiosks, vehicles,
mobile devices" — a developer's laptop is, at best, the weakest possible reading of "mobile device," a
real stretch against the brief's own intended examples of genuinely constrained edge hardware.

## Phase 5 — The turning point: a "flaws" document, and real verification against Qdrant's own demos

The user brought in a detailed critique document listing structural flaws in the reasoning so far
(no specific user defined, "offline alone" being a weak differentiator since Qdrant already supports local
storage/snapshots, "sync" alone being something Qdrant already documents, "AI memory" being vague, local
memory going stale, multiple devices potentially disagreeing with each other, data volume explosion,
embedding cost, security of a stolen device, and — most importantly — no identified "killer workflow").
This document also worked through each of the brief's named device categories (industrial systems, robots,
kiosks, vehicles, mobile devices, drones, smart glasses, disaster robots, scientific instruments) and
flagged which were strong vs. weak opportunities, correctly predicting that "robot remembers objects" and
"smart glasses find my keys" would turn out to be Qdrant's own official demos before this was independently
verified.

Direct research confirmed this document's warnings were accurate:
- Qdrant's own official home-robot demo (YOLOE + SigLIP2 + Florence-2, single shard, live network-cut
  proof) is real and public.
- Qdrant's own official video-anomaly-detection edge-to-cloud tutorial (nearest-neighbor distance from a
  baseline, two-shard edge/cloud design, full open GitHub repo) is real and public.
- Qdrant Edge's own homepage explicitly names "Industrial IoT: anomaly detection, predictive maintenance"
  as a designed-for use case.
- The offline visual-memory smart-glasses demo is also real and public.

This meant every single-device pattern (robot memory, anomaly detection, visual memory) was already a
sponsor showcase. The genuine, verified gap identified: none of these four official demos are
**multi-device** — none show two independent devices disagreeing about the same fact, or one device's
experience becoming useful to a device that never experienced it.

## Phase 6 — Landing on "Fleet Memory": industrial predictive maintenance with multi-device conflict resolution

The chosen direction: multiple machines/devices, each with local Qdrant Edge memory of fault incidents;
one device's confirmed fix becomes fleet knowledge; when two devices disagree about the same incident, the
system shows an explicit "Unresolved Conflict" instead of silently picking a winner. This became the
project's core engineering claim and has not changed since.

## Phase 7 — Deep refinement pass 1 (design corrections, via user-provided critique documents)

A long series of review documents (the user's own analysis, cross-checked and corrected by me in
alternating passes) progressively fixed:
- **Promotion policy:** from a vague "validated outcome" to an exact rule — technician confirms completion
  AND a stability window passes with no recurrence AND outcome confirmed successful. Confidence was
  initially (wrongly) treated as the gate itself; corrected to be descriptive only.
- **Conflict algorithm:** started as a naive "% successful so far" ratio (flawed — a device with 1/1
  success would wrongly read as 100% reliable); corrected to a Bayesian-smoothed score
  `R=(successes+1)/(successes+failures+2)`, later explicitly grounded in the real academic field of
  **Truth Discovery** (foundational method: Dawid-Skene, 1979).
- **Anomaly detection:** a single global 3-sigma threshold was corrected to a per-operating-mode baseline
  (high-load vs. low-load vs. startup baselines computed separately), explicitly labeled a prototype
  method, with Augury's real production-grade multivariate diagnostics named as the honest comparison.
- **Memory model:** an early conflation of "staleness" and "correctness" (treating old records as
  automatically less trustworthy) was corrected — Confidence, Freshness, and Applicability are tracked as
  three separate fields; a rigid "must match machine model AND operating mode" applicability rule was
  loosened into three explicit tiers (Directly applicable / Related / Weakly related), mapped onto the
  real, established Case-Based-Reasoning "Reuse/adaptation" step.
- **Storage eviction:** an early rule ("delete oldest unresolved records first when full") was identified
  as dangerous — an unresolved rare incident could be the most valuable record in the system — and
  corrected to never auto-delete unresolved/high-severity records.
- **Provenance:** expanded from a short field list to a full audit trail including
  `embedding_model_version` (needed because re-embedding with a new model makes old and new vectors
  incomparable).
- **Security:** corrected from a vague "stolen device only exposes its own history" claim to an explicit,
  prototype-scoped stance (OS-level disk encryption + app login + device identity), with an explicit
  refusal to claim custom end-to-end encryption that wasn't actually built.
- **Liability:** corrected the system to never recommend an action — only show similar past incidents and
  evidence, leaving the decision to the human technician — after identifying a real "who's liable if the
  AI's suggested fix is wrong" risk.
- **Qdrant Edge Beta risk:** identified that Edge's API may change since it's officially labeled beta;
  mitigation is to wrap all Edge calls behind our own interface, never call the raw API scattered through
  the app.

## Phase 8 — The competitive-overlap crisis, and the correction that mattered most

An early claim — "SCADA/CMMS systems can't do semantic incident retrieval" — was flagged, checked, and
found **false and too broad**. Direct research into IBM Maximo and Augury's real product documentation
showed both already provide natural-language access to asset/failure history, AI fusion of sensor and
maintenance text, and (Augury specifically) real edge-AI diagnostics with a 4.7/5-rated offline-resilience
feature validated across 1.1 billion+ hours of real machine data. This was the single largest correction
in the whole project: the positioning shifted from "no one does this" (false) to a narrow, defensible
claim — the one piece not found in Qdrant's own sync docs, Maximo, or Augury is **explicit multi-device
conflict detection with reliability-weighted reconciliation**, not a broad "AI-powered maintenance"
category claim.

## Phase 9 — Grounding in real prior art (not inventing from nothing)

Further research found that the core retrieval mechanic ("find similar past cases, let a human decide,
record the outcome") maps directly onto **Case-Based Reasoning**, a real, decades-old, still-active AI
field. The conflict-resolution mechanic was grounded in **Truth Discovery** (Dawid-Skene, 1979) rather than
an invented formula. Ranking multiple retrieved incidents by several weighted factors was grounded in
**Multi-Criteria Decision Analysis (MCDA)**, an established decision-science framework. This was a
deliberate shift from "we invented something new" (a weaker, harder-to-defend claim) to "we correctly
applied established methods to a genuine gap" (a stronger, more credible claim).

## Phase 10 — Real-world verification of every load-bearing claim

Every claim used to justify "why edge, not cloud" was checked against a real source rather than asserted:
- Roughly half of offshore oil rigs still lack reliable broadband (Maritime Executive).
- A peer-reviewed paper on remote Industry 4.0 sites states cloud dependency is infeasible for such sites
  (arXiv 2301.00484).
- Real industrial network segmentation/air-gapping practices are documented in NIST SP 800-82 Rev.3
  (2023) — with an explicit correction that this does NOT mean "every plant is air-gapped."
- The CWRU Bearing Dataset and NASA IMS Bearing Dataset were confirmed as real, public, standard research
  datasets, chosen specifically to avoid presenting invented sensor numbers as real.

## Phase 11 — Real hackathon logistics discovered

Direct research (including fetching the actual event page) confirmed: Code Cubicle 6.0, by Geek Room, at
Paytm Noida, team size 1-4, prize pool $14,000 (general track: Rs12,000/10,000/8,000), Qdrant as technology
partner. Schedule: registration/team formation through Sep 20, 2026; Round 1 "Project Submission"
(evaluated on GitHub + LinkedIn profiles, not a live demo) Sep 21-30, 2026, ~15-20 teams advance; Final
Round Oct 11, 2026, offline, top 3 win. Full judging rubric only partially confirmed (Innovation &
Creativity, Technical Implementation named; rest unknown).

## Phase 12 — A 100-item problem/flaw audit, resolved category by category

A comprehensive 100-item list (spanning product definition, offline/edge issues, memory correctness,
anomaly detection, embedding/search, freshness/applicability, sync, conflict resolution, promotion,
storage, security, industrial integration, competitive positioning, demo mechanics, and scientific
validation) was compiled and worked through systematically. The "10 most dangerous" were prioritized:
existing-product overlap, Edge's actual necessity, sensor-to-semantic-event representation, bad-memory
propagation risk, applicability being harder than plain vector similarity, trust in the anomaly detector,
real industrial data integration, the need for actual evaluation numbers, the exact fleet-sync policy, and
final product differentiation. Each was resolved with a researched, cited answer rather than a guess —
for example, the "Edge necessity" question was resolved using the real offshore-connectivity statistics
above, and "evaluation" was resolved using standard information-retrieval metrics (Precision@3/Recall@3)
rather than a vague "it works" claim.

## Phase 13 — A plain-language reset

At one point the user stated plainly that the technical depth had outpaced their own understanding of the
project and asked to be re-taught from zero. The whole idea was re-explained as a "smart notebook for
machines" — what stays local vs. what goes to a shared notebook, and why disagreement between two devices
is the one genuinely hard, interesting problem — without jargon. This reset is preserved here because it
is the clearest, simplest statement of what the project actually is.

## Phase 14 — Clarifying the brief is a menu, not a mandate

The user asked why the project focused on "machines" when the actual problem-statement screenshot lists
robots, industrial systems, kiosks, vehicles, and mobile devices. Clarified that this list is a set of
possible example settings to demonstrate the brief's required abilities in — not a requirement to build
for a specific one — and that "industrial machines" was deliberately chosen (Phase 5-6) because it was the
one setting not already claimed by one of Qdrant's own official demos.

## Phase 15 — First set of deliverable files, and a design review that found 2 real gaps

Four files were produced (PROBLEM.md, ARCHITECTURE.md, FEATURES.md, CONTEXT.md). On review, two real
architecture gaps were found and fixed:
1. The design only specified how a device *pushes* validated incidents to the fleet — never how a device
   *pulls back* other devices' knowledge. Fixed by adding a second, read-only local shard per device (the
   "Fleet Mirror"), refreshed whenever online.
2. No plan existed for what happens if a bad promoted incident is later deleted centrally. Fixed with a
   tombstone/retraction design — the server marks a bad incident `retracted` rather than deleting it, so
   devices update rather than silently keep trusting it or silently lose the record.

## Phase 16 — Memory saved for cross-session continuity

Given the scale of this project, the design was saved into persistent memory (outside the normal
per-chat container) specifically so a new chat could continue without the user re-explaining everything.

## Phase 17 — A second, independent research pass (via two uploaded documents), and a major correction

The user brought in two documents from a separate, deeper research pass. Merging them into this project
produced one important correction and several real technical upgrades:
- **Correction:** Qdrant's own published Edge writeup names "Edge Anomaly Triage" — explicitly including
  an unusual-machine-state example — as one of its own showcased patterns. The project's earlier claim
  that "no official Qdrant demo covers industrial/machine-state use" (Phase 5-6) was therefore too strong.
  The positioning was corrected from claiming domain novelty to claiming **depth** (a tested conflict-aware
  sync layer, measured retrieval quality, real data) — explicitly, "the production layer the sponsor's own
  pattern list leaves out," not a new discovery.
- **Upgrade:** sensor encoding moved from the earlier "convert numbers to an English sentence and embed
  with a text model" approach to a proper hand-built feature vector (RMS, kurtosis, FFT band energies) —
  more technically legitimate for genuine signal similarity, stored as its own named vector alongside
  separate text and BM25-sparse vectors for technician notes.
- **Upgrade:** conflict handling moved from an invented "timestamp wins" rule to Qdrant Edge's actual real
  API primitive — an `update_mode` (Upsert/InsertOnly/UpdateOnly) combined with a condition filter — the
  genuine compare-and-set mechanism, confirmed directly from Qdrant's own API documentation.
- **Addition:** a "novelty gate" (querying the nearest neighbor before inserting, to avoid flooding the
  shard with near-duplicate sensor readings) — this had been entirely missing before.
- **Upgrade:** the sync "queue" was upgraded to a durable outbox (SQLite table or append-only file written
  in the same step as the local write) — Qdrant's own documented example queue is in-memory only, a real,
  confirmed gap in their own docs.
- **Addition:** logical/hybrid-logical clocks for ordering conflicting writes, instead of wall-clock
  timestamps, which can disagree between devices (clock skew).
- **Addition:** real, cited numbers from Qdrant's own in-house testing (approximately 0.1ms on-device vs.
  approximately 52ms round trip to Qdrant Cloud, on an iPhone 16 Pro, explicitly labeled by Qdrant as
  in-house and not an official benchmark) plus real BM25/RRF tuning notes (no universal RRF weight
  default; explicitly set the IDF modifier; never use a score threshold on a fusion query).
- **Softened claim:** the competitive position was made explicitly honest and non-overclaiming: "we are
  ahead only if the sync layer is genuinely built and tested... cannot promise a win, cannot see the full
  judging rubric."

## Phase 18 — Present state

All four deliverable files were regenerated to reflect every correction and upgrade above. Persistent
memory was updated to match. One open discrepancy remains unresolved, flagged rather than assumed: the two
research passes disagree on team structure (one assumes named per-component owners across a team of
several people; the other assumes the user is doing all technical work solo with Claude) — this needs a
direct answer, not a guess. No code has been written yet; the project is fully designed and researched,
not yet implemented.

## Full source list (every real source used across every phase)

qdrant.tech official documentation (Edge Quickstart, Edge API configuration reference, Hybrid Queries/RRF
documentation, Data Aggregation/synchronization guide, FastEmbed on-device embeddings, Points/conditional
updates, BM25/IDF tuning article, Qdrant Fundamentals FAQ); Qdrant's blog post "Memory at the Edge: On-
Device Vector Search with Qdrant Edge" (16 June 2026); github.com/Qdrant/qdrant; docs.rs/qdrant-edge; PyPI
`qdrant-edge-py`; pub.dev `qdrant_edge`/`qdrant_edge_flutter`; IBM Maximo product documentation; Augury
product materials and independent review platforms; NIST Special Publication 800-82 Revision 3 (2023);
Maritime Executive (offshore rig connectivity reporting); arXiv 2301.00484 (Federated Fog Computing for
Remote Industry 4.0 Applications); reporting on Samsung's 2023 generative-AI ban; Xperi's Source Code
Handling Annex; Dawid & Skene (1979) and the subsequent Truth Discovery literature; the Case-Based
Reasoning academic literature; the CWRU Bearing Dataset (Case Western Reserve University); the NASA IMS
Bearing Dataset (University of Cincinnati experiments, hosted via NASA).
