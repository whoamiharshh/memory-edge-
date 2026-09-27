# CLAUDE.md — Machine Memory at the Edge (handoff for the local build session)

Read this first, then `docs/RESEARCH.md` (the full research + architecture, Parts A–R). The older files in
`docs/` (PROBLEM, ARCHITECTURE, FEATURES, CONTEXT) are **history**. Where they conflict with RESEARCH.md,
RESEARCH.md wins (see its §1, "Corrections").

## What we're building (one paragraph)
Qdrant hackathon PS3 (Code Cubicle 6.0, Geek Room). Each industrial machine's edge device remembers vibration
fault episodes and what fixed them, searches that memory **offline** with Qdrant Edge, and shares a fix with the
fleet **only after the machine's own sensor data shows the fix held** (outcome-verified promotion = our ONE
headline differentiator). The cloud (Qdrant Server) groups fixes by fault signature and keeps worked/failed
evidence per site, **preserving disagreement** instead of overwriting it. Device B then benefits offline.
The system shows evidence and never recommends an action.

## Deadlines
- Round 1 (GitHub + LinkedIn evaluation): **30 Sep 2026**. This date is from an earlier session and was not
  re-verified. The goal is a tested vertical slice with an honest README.
- Final (offline demo, Paytm Noida): **11 Oct 2026**.

## How to work with the user
- The user is a beginner and the sole builder, with Claude as the builder. They have given approval for all
  build decisions.
- Explain in plain language when asked. Otherwise full technical depth is fine.
- **Never claim anything works without running it.** Label claims: Verified Fact / Inference / Assumption /
  Proposed Design / "I could not verify this".
- Verify external facts against official sources. The user is strict about no invented facts.

## Machine / environment (verified 27 Sep 2026)
- Windows 11, PowerShell 5.1. Intel i5-1335U (12 threads), 15.7 GB RAM, **no GPU, no Docker, no WSL**.
- `uv` 0.12.3. Project venv: `.venv` (CPython 3.12.13). Always run with `.venv\Scripts\python.exe`.
- Install with: `$env:VIRTUAL_ENV=".venv"; uv pip install <pkg>`.
- Installed packages: `qdrant-edge-py==0.8.0` (pinned; beta, the API drifts), `qdrant-client==1.19.1`,
  `fastembed`, `numpy`, `scipy`, `fastapi`, `uvicorn`, `httpx`, `pydantic`, `pytest`.
- **Qdrant Server without Docker:** use the official Windows release binary from github.com/qdrant/qdrant
  releases, matching the client (1.19.x). Check that the asset exists first. Fallback: Qdrant Cloud free tier.

## Verified by our own spike (spike/spike_edge.py, spike/spike_durability.py) — 27 Sep 2026
Full API surface dump: `spike/api_surface.txt`. On qdrant-edge-py 0.8.0 / Windows, the following PASSED:
- Named dense vectors (`vib` Euclid, `note` Cosine) + sparse `note_bm25` (Modifier.Idf) on one point.
- `Bm25(Bm25Config(avg_len=...)).embed_document` / `.embed_query`.
- **Hybrid in one request:** `QueryRequest(prefetches=[Prefetch(limit, query=Query.Nearest(v, using=...))...], query=Fusion.Rrf(60))`.
  Weighted form: `Fusion.Rrf(60, [2.0, 1.0])`.
- Payload filter + keyword field index. Facet.
- **CAS:** `UpdateOperation.upsert_points([p], condition=Filter(version==base))`. A matching condition applies;
  a stale condition is rejected silently. You must re-read to detect the rejection.
- `UpdateMode.InsertOnly` leaves existing points untouched (idempotent replay).
- `optimize()`, `snapshot_manifest()`, `close()` + `EdgeShard.load()`.

**K4 FINDING (important):** writes do not survive a hard kill unless `flush()` is called.
- No flush: 0 of 200 acknowledged writes recovered.
- `flush()` after each write, or every 50: 200 of 200 recovered.

**Rule:** the store adapter calls `shard.flush()` before reporting a write as durable. The SQLite outbox is
written BEFORE the shard write. On boot, re-apply outbox rows not yet marked applied (idempotent upserts with
deterministic uuid5 IDs).

## Data (data/fetch_data.py — written, NOT yet run; the user interrupted the first run to switch to VS Code)
- **CWRU** (12k drive-end), 40 files: normal 97–100, plus IR / Ball / OR@6 × fault sizes 7/14/21 mil × loads 0–3.
  The file mapping was verified against the official CWRU table.
- **Each (class, size) is one physically distinct seeded bearing.**
- **Leakage-free split (mandatory):** index bearings of 7 & 21 mil, query with the 14-mil bearings (all loads).
  Also report the naive leaky split next to it to show the gap.
- **Normal baseline sampling rate:** the CWRU page is silent; a secondary source says 48 kHz. Decimate by 4
  (anti-aliased) to 12 kHz and document this as an assumption. The empirical peak check was inconclusive.
- **Logbook:** Zenodo record 17903357, "Annotated Maintenance Logbook", 6,169 aircraft-engine problem/action
  records, CC BY 4.0. Used for the text-retrieval benchmark: dense vs BM25 vs RRF hybrid.
- Raw data goes in `data/raw/` (gitignored, never redistributed).

## Architecture decisions (details in docs/RESEARCH.md, Parts F–H + Appendix 3)
- Edge point named vectors: `vib` (~20-d z-scored DSP fingerprint, **Euclid**), `note` (bge-small-en-v1.5,
  384-d, Cosine, via fastembed), `note_bm25` (Edge built-in BM25, IDF).
- DSP fingerprint features: RMS, peak, crest factor, kurtosis, skewness, plus log-spaced FFT band energies,
  z-scored against the machine's healthy baseline.
- **Novelty gate:** a nearest-neighbour query per window. Near the healthy baseline → normal. Near an existing
  episode → MERGE (occurrences++). Otherwise → new episode. Thresholds are prototype parameters chosen by benchmark.
- **Outcome verifier:** after an action, count N consecutive windows back within the healthy-baseline radius.
  Say "symptom resolved for N windows", never "root cause confirmed".
- **Policy engine:** deterministic ordered gates (privacy → validation → duplicate → evidence → verification →
  human confirmation → SHARE). Every decision stores `decision_reasons[]`.
- **Privacy:** raw signals and raw notes never leave the device. **Text embeddings are never shipped**
  (embedding-inversion risk); the server re-embeds any shared redacted text itself.
- **Sync:**
  - SQLite outbox first; batch push with deterministic event_ids + `insert_only` on the server.
  - Per-event acks: accepted / duplicate / rejected. Backoff with jitter.
  - Pull uses Qdrant's documented **dual-shard pattern** (mutable local + immutable mirror via partial
    snapshots). **Credit it to Qdrant — it is not our invention.** Fallback: a scroll-based pull.
- **Cloud:** group events into case_groups (vib similarity + same component). Evidence tallies per
  (group, action) use CAS on `version`. Flags:
  - DISPUTED: same action, both outcomes.
  - COMPETING: different root causes.
  - ALTERNATIVES: different actions, both worked.
  - Retraction = tombstone.
- **CUT:** hybrid logical clocks, the per-device reliability score, any LLM in the decision path, Jev AI
  (cloud-only).
- **Laya (Apache-2.0, offline, ~800 MB, needs torch):** a week-2 experiment only. It may suggest an action code
  and act as a second PII flag. It is advisory, and may only make the privacy gate stricter. Benchmark it vs
  bge-small + logistic regression trained on the logbook's action-type labels; keep the winner.

## Repo layout (target)
```
edge/    fingerprint.py gate.py verifier.py policy.py store_edge.py(ONLY importer of qdrant_edge)
         outbox.py sync_worker.py mirror.py api.py ui/
cloud/   api.py auth.py ingest.py grouping.py tally.py store_server.py
shared/  schema.py ids.py redact.py
data/    fetch_data.py splits.py
bench/   retrieval_vib.py retrieval_text.py latency.py sync_partition.py bandwidth.py
tests/   unit/ failure/ security/ ai/
demo/    scenario.py
docs/    RESEARCH.md (+ history files)
spike/   day-1 API proofs (keep; they're evidence)
```

## Build order (tick as done)
- [x] venv + packages; Edge API spike (K1 pass, K4 finding)
- [x] Ran `data/fetch_data.py` (40 CWRU files + logbook CSV "Second version of Annotated Maintenance logbook.csv",
  columns: IDENT, PROBLEM, TYPE, LOCATION, PART, TAGEDPROBLEM, ACTION, TYPE(dup), WITH, INSTALL TYPE, ACTIONPART,
  TAGGEDACTION, CAUSE). Wrote `data/splits.py` and `edge/fingerprint.py` (FP_VERSION fp-v2, 27 dims).
- [x] Unit tests for `edge/fingerprint.py` and `data/splits.py` (`tests/unit/`, 30 pass incl. real-CWRU split check;
  run `.venv\Scripts\python.exe -m pytest`)
- [x] **K2 benchmark run** (`python -m bench.retrieval_vib`, results in `bench/results/k2_vib_retrieval.json`).
  (`str(c)` per_class-key fix is already in the code.)

## K2 RESULTS (27 Sep 2026) — these changed the design, read carefully
Retrieval runs through a real Qdrant Edge shard. Queries are ~0.1 ms p50 on this laptop.
| Split | Variant | P@3 fault class | Notes |
|---|---|---|---|
| bearing-level (honest: unseen bearings) | stats healthy-z / Euclid (best) | **0.429** (chance 0.277) | normal 1.0, ball 1.0, inner 0.0, outer 0.0 |
| bearing-level | full v2 incl. physics order features | 0.409–0.429 | physics features did not fix retrieval |
| bearing-level | physics-relative pattern | 0.239 | worse |
| random windows (leaky) | any | **0.997–1.000** | reproduces the published leakage effect |
| bearing-level | RULE: argmax of envelope energy at BPFO/BPFI/BSF (no retrieval) | **0.644** fault acc | inner 1.00, ball 0.69, outer 0.24 |

Per-file diagnosis (`spike/diagnose_orders.py`):
- Inner race: BPFI dominates at every size.
- Outer race: 7- and 21-mil bearings show BPFO clearly, but the 14-mil bearing (files 197–200) is weak and
  near-normal RMS.
- Ball faults: weak at every size.
Defect orders are from the CWRU bearing page (SKF 6205: BPFI 5.4152, BPFO 3.5848, ball 4.7135 × shaft speed).

**Design decision (kill test K2 → "change architecture", not abandon):**
1. The `vib` vector is used for:
   - the **novelty gate** (healthy vs abnormal: 1.0 on unseen bearings);
   - **same-machine recurrence** memory (same bearing ≈ 0.997).
   Both are honest uses.
2. **Fleet matching across machines must NOT rely on vib similarity.** Cloud case_group key =
   `(component, fault_class)`. `fault_class` is suggested on-device by the physics rule, shown with its measured
   per-class accuracy, and **confirmed by the technician** (e.g. what they saw when the bearing was removed).
3. Device B's fleet search filters by `(component, suggested/confirmed fault_class)` and ranks with hybrid text;
   vib is only a tie-break.
4. README/pitch: show this table. "Same machine: 99.7%; new machine: 43%; physics hint: 64%. That is why the
   fleet layer groups by confirmed fault class." This is a strength (honest measurement), not a weakness.
5. Update RESEARCH.md Part F.4 grouping (it said "nearest case_group by vib"); that is superseded by the above.

## Remaining build order (continue from the first unticked item)
- [x] `edge/store_edge.py` (flush-before-ack; K4 rule proven by `tests/failure/test_store_durability.py`) + outbox
  (`edge/outbox.py`, journal-first + replay on boot) + gate + episodes (`edge/device.py`) + hybrid search + tests
- [x] verifier + policy engine with reasons + tests. **K3 PASSED** (`bench/gate_verifier.py`): 36/36 fixed resolved,
  36/36 still-faulty persists, 0 false promotions, 0/116 false alarms. Continuity rule added (a contiguous
  abnormal run = one episode; 14-mil ball faults fragmented before). Limitation: distinct faults separated by
  healthy running get separate episodes only 11/20.
- [x] cloud: Qdrant Server 1.19.1 binary in `qdrant_server/`, Sync API, token auth, idempotent ingest (+ event-id
  ownership check), per-tenant collections, tallies/flags recomputed under CAS, retraction tombstones + tests
- [x] mirror pull: **scroll-based** (K5 fallback, disclosed). Partial-snapshot path NOT implemented yet.
  Device A → cloud → Device B: `tests/integration/test_fleet_flow.py` + `demo/scenario.py` (live, 9 steps pass)
- [x] UIs (device `edge/ui`, fleet `cloud/ui`, vanilla JS, textContent only) + `tests/ui_check.py` (Edge headless);
  README with benchmark tables; `docs/SETUP.md`
- [x] LLM evidence brief (user request, 27 Sep): Qwen2.5-1.5B-Instruct GGUF via llama-cpp-python 0.3.35, OUTSIDE
  the decision path, citation + no-advice checker (`edge/rag.py`, `tests/ai`)
- [x] partition harness (`tests/failure/test_sync_partition.py`), security tests (`tests/security`),
  text/latency/bandwidth benches (`bench/retrieval_text.py`, `bench/latency.py`)
- [ ] `git init` + first commit (done 27 Sep if ticked in git log); create a public GitHub repo and push
  (`gh` is NOT installed; ask the user for their GitHub account); LinkedIn post draft in `docs/LINKEDIN_DRAFT.md`
  → **Round 1 submission (30 Sep)**
- [x] (28 Sep) local `git init` + commits (NOT pushed; the user said to set the submission aside)
- [x] finals build (28 Sep, see docs/DECISIONS.md D5-D12):
  - fleet mirror via Qdrant shard snapshots: `edge/mirror.py`, cloud `mirror_<tenant>` collection + snapshot
    endpoints; default mode `auto` (full snapshot bootstrap, then cheaper of scroll delta / full snapshot, by
    measured bytes); pure partial-snapshot mode selectable. Tests: `tests/integration/test_mirror_snapshot.py`
    (starts the real Qdrant binary per session on free ports).
  - retention/ARCHIVE, versioned edits (409 CONFLICT), usefulness feedback; UI for all
  - benches: `gate_sweep`, `storage`, `resources`, `sync_partition`, `offline_check`, `mirror_sync`, `laya_experiment`
  - docs: THREATS.md, DECISIONS.md, DEMO.md, BENCHMARKS.md; `demo/record_backup.py` (video of the live UIs)
  - Laya: measured and rejected; its venv and model were deleted on the user's request (28 Sep)
- [x] (28 Sep, second pass) plan details 1-7 (flags tests, 3-site live demo, repair content hash, last-confirmed,
  text-model migration, embed model bench); signal profiles (bearing-12k, rotating-hf, lowrate-accel, force-torque,
  events) + physics (edge/physics.py) + cited procedures (knowledge/); HUST held-out second machine (97.6 % physics,
  42/42 fixes verified after one 'normal operation' confirmation); robots (UCI); HTTPS (tools/make_certs.py); phone
  /sensor page; OS storage compression; LICENSE Apache-2.0 (chosen by Claude at the user's request, local only).
  205 tests pass; live demo 9/9; backup video re-recorded.
- [x] (28 Sep, third pass) automatic operating-point check (taught ranges + signature score, threshold 1.63 on
  CWRU; flags 0/28 false, suggestion 13/17); robot threshold tuning + trained robot LR (UCI, cross-validated);
  vehicles: SCANIA risk LR (real trucks; AUC 0.750, `knowledge/vehicle_risk_model.json`, plain JSON) + OBDex CC0
  code dictionary (`knowledge/vehicle_codes.json`); telemetry profile fp-tm1; kurtogram band selection; mTLS;
  30-day tokens with auto-renew; notes AES-GCM at rest (DPAPI key; verified 0 plaintext hits in a live device
  folder); PWA (manifest + service worker, never caches /api); site SOP form; SKF lubrication procedure.
  224 tests pass (358 s, 28 Sep).
  **Training data rule (user):** everything trained uses REAL data only; synthetic signals exist only in tests.
- [ ] USER: real-world field test + phone tests (docs/FIELD_TEST.md) - remind them (they will do it after the web part)
- [ ] USER: the invalid `permissions.allow` rule in `C:\Users\Sir\.claude\settings.json` - needs their yes (global config)
- [ ] USER ONLY: public GitHub repo + push, LinkedIn post. NEVER post/publish anything without their manual yes.

## How to run (current)
- Tests: `.venv\Scripts\python.exe -m pytest` (count and time: see README "Tests"; some tests start
  `qdrant_server\qdrant.exe` themselves)
- Backup video: after `run_demo.ps1 -Reset`, `.venv\Scripts\python.exe -m demo.record_backup` →
  `runtime\recording\backup_demo.webm`
- Laya experiment (venv + model deleted 28 Sep; recreate per docs/SETUP.md step 13 only if re-measuring):
  `$env:HF_HOME="models_cache\hf"; .venv-laya\Scripts\python.exe -m bench.laya_experiment`
- Demo: `powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset` then `.venv\Scripts\python.exe -m demo.scenario`.
  Do NOT pipe the launcher's output (`| Out-Null` hangs: children inherit the pipe). Stop: `demo\stop_demo.ps1`.
- Windows PowerShell 5.1 `Get-Content -Raw` reads UTF-8 as ANSI: edit UTF-8 files with Python or the Edit tool.

## Non-negotiable rules
- Pin `qdrant-edge-py==0.8.0`. All Edge calls go through `edge/store_edge.py`.
- Tests for every module. Run them before claiming anything.
- No invented numbers. Every metric in the README comes from a script in `bench/`, with its method stated.
- Don't claim "nobody does this". Claim the combination and its measurements.
