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
- [x] (28 Sep, weak-point pass; details docs/DECISIONS.md D29-D40, docs/BENCHMARKS.md §21-29, plan was in the session
  scratchpad) phone MICROPHONE profile `acoustic` + `/api/ingest/audio` + sensor.js mic mode (UOttawa natural wear:
  0/380 FA, 95.7 % faulty flagged); replacement-aware verifier (new bearing 19/20; HUST 42/42 without teaching);
  fleet-learned fault hint (cloud/hint_model.py, edge/fleet_hint.py, physics.order_features, "confident" when physics +
  fleet agree: HUST 100 %, UOttawa 92 %, CWRU 85 %); machine card (ISO 10816-3 tables from the standard, manufacturer
  limits in the verifier); manuals (pypdf, offline search with pages); ISO 15243 DamageMode; held/recurred FollowUp +
  RECURRED flag; relative order rule; robots' own learned detector + teach_fault (96-99 %, 0-10 % FA); events radius
  at a 1 % FA target (HDFS 99.98 % / 0.39 %); redactor with Wikidata names + name model (unseen 96 % typed);
  security: CRL + TLS>=1.2 (cloud/tls.py), two-admin retraction, quarantine, plausibility checks, audit chain,
  code-integrity hash, CSP headers, weak token refused; scale: per-thread Qdrant clients, lock stripes, batched ingest,
  coalesced recompute, snapshot cache, bytes-based bootstrap (20 devices: 1,000 events in 9.8 s, 0 lost);
  tools/qdrant_local.py (any OS), tools/backup.py, tools/sensor_bridge.py, .github/workflows/tests.yml (unrun).
  REJECTED (kept as evidence): physics v2, cross-dataset prior, bearing-growth gating, SCANIA training-split model.
- [x] (28 Sep, finish pass; D42-D47, BENCHMARKS §30-33) network-fault proxy + clock-offset correction; fleet scale
  1,000 / 5,000 devices + 50 complete (0 lost / 0 double); mTLS cert bound to token + live CRL; names after cue words;
  CI fetches CWRU (simulated clean run 239 pass); MaFaulDa (one real machine): imbalance named 0 -> 258/333 via the
  harmonic-clash rule, HUST hint 97.6 -> 95.2 % (B604, documented); microphone + louder neighbour measured (one
  "normal operation" confirmation: FA 0-4 %, but 30-66 % of faulty windows masked when the neighbour is as loud).
  302 tests pass (9 min 21 s). Android app NOT built (user's choice).
- [x] (1 Oct) launcher + "ask about the whole record" pass:
  - `demo\stop_demo.ps1` / `stop.ps1`: taskkill's stderr is now redirected **inside cmd.exe**. PowerShell 5.1
    wraps a native command's redirected stderr in a terminating NativeCommandError, so one already-dead PID
    aborted `run_demo.ps1 -Reset` before it started anything. Reproduced, then fixed.
  - `start.ps1` now starts `app.unified` on **9000**, waits for `/health`, writes `runtime\unified_pid.txt`
    (which `stop.ps1` already expected) and opens **http://127.0.0.1:9000/** - the URL README.md:240 always
    documented. It had regressed to opening 8101, which is why the redesigned app looked "unchanged".
    Deleted the 0-byte `start-all.ps1`.
  - `Device.ask`: whole-record questions ("what problems have you seen?", "has anything been fixed?",
    "is anything still unresolved?", "what do you know about me?") are answered from stored episode state.
    They name nothing in a record, so word-overlap relevance always returned "nothing relates to that at
    all" - including on a device holding a technician-confirmed, sensor-verified repair. These are the four
    chips the UI itself offers. A question carrying an identifier still goes down the retrieval path.
    `mode: "overview"`, cited [E1..], wording stays "symptom resolved", never "root cause confirmed".
    Tests: `tests/integration/test_overview_questions.py`.
  - `setup.ps1` passed `--index-strategy` (a **uv** flag that pip rejects), so setup failed at the install
    step on every clean machine; it also listed `abeten.github.io` (a typo of `abetlen`, an unowned domain)
    as a package index. Both removed. `setup.sh` now passes the CPU wheel index so llama-cpp-python does not
    try to compile from source.
  - `.gitignore`: `.venv-*/` and `.playwright-mcp/`.
- [x] (1 Oct, user request) **Ask can reach the wider world**, and the app got Settings:
  - `edge/online.py`: the ONE module on the device that touches the network for anything but its own cloud.
    Providers: duckduckgo+wikipedia (default, **no key, no account**), wikipedia, brave (`BRAVE_SEARCH_API_KEY`),
    tavily (`TAVILY_API_KEY`), none. Never raises - an unreachable network is this product's normal state.
  - `Device.retrieval_mode()` (`local` / `auto` / `online`, default `auto`, stored in the outbox kv so it
    survives a restart) + `_may_go_online()`. Rules: never when set to local, never when offline, never when
    the provider is unconfigured, and in `auto` only when the device holds no answer. A question about this
    device's own record (the overview intent) **never** leaves it in any mode.
  - Web hits face the same word-overlap relevance test as local evidence - a search engine always returns
    something, and "Norfolk State University" for a question about France is worse than citing nothing.
  - Web evidence shares the one [E1..] citation sequence rather than a [W1..] one, because `rag.check_output`
    only recognises E-citations; what marks it as web is `source` + the URL. The answer carries `sources`,
    and the UI says "answered on the device" / "answered using the internet" / both.
  - API `GET|POST /api/retrieval`. **Tests must never hit the network**: `tests/conftest.py` forces
    `EDGE_SEARCH_PROVIDER=none` for the whole session; `tests/unit/test_online.py` stubs the providers.
  - UI: System left the primary tabs (now Ask / Memory / Broadcast / Devices + a gear). New Settings page:
    Appearance (Light/Dark/System, **light is the default**, `:root:not([data-theme])` guards the OS query so
    an explicit Light wins), Ask retrieval mode, system rows + a way into the old System page. Dark tokens
    added; `nav` and the toast had hardcoded colours and stayed light - both are tokens now.
  - Memory: search box + a collapsed "Technical details" block. `app/unified.py` proxy timeout 30s -> 180s
    (the first Ask loads a 1.1 GB GGUF and was returning a bare 500).
  - docs: README privacy section now states the exception honestly; `docs/THREATS.md` has the new surface;
    `.env.example` documents every provider variable.
  - `tools/qdrant_local.py`: test Qdrant now uses 1 segment + a 4 MB WAL. It was sized for production, so a
    collection holding THREE events wrote a **513 MB** snapshot and one suite run left **98 GB** behind, then
    died with StorageFull. (157 GB of old pytest scratch was deleted to get the machine working again.)
- [x] (2 Oct) UI fixes + Hindi/English + attachments in chat:
  - **Which UI is which:** `http://127.0.0.1:9000/` is the CURRENT app (`app/`, served by `app/unified.py`, reads
    files from disk each request, no cache headers, no service worker). `8101/8102` still serve the OLD `edge/ui`
    (PWA with a service worker that caches): a user who "sees the old UI" is on one of those. Redirecting them to
    9000 was proposed, NOT done (waiting for the user's yes). No Tailwind/bundler exists, so nothing can be purged.
  - Sidebar: orb is a CSS sphere (turning texture + 3D tilt + glow); chat list is flat rounded cards (no
    Today/Yesterday headings) with a last-answer `preview` (new key from `Device.conversations()`); active class is
    `active` (JS used `on` before, which no CSS matched); nav is Ask/Memory/Broadcast only (Devices tab removed, its
    page code remains but is unreachable), each with its own coloured icon chip; shared press-feedback transition rule.
  - **Dictation = Whisper `small`** (`shared/speech.py`, faster-whisper 1.2.1, auto language, Devanagari for Hindi;
    Vosk English-only kept as fallback). `faster-whisper`'s own file decoder breaks with `av` 19, so audio is passed
    as a numpy array and that decoder is never used. 6-13 s per 5 s clip (fixed 30 s window). Tested on SYNTHETIC
    speech only (SAPI + edge-tts); real-microphone Hindi is UNTESTED.
  - **Answer language** (`shared/translate.py`, `Device.ask`): Hindi (Devanagari) question -> English -> the normal
    grounded pipeline -> answer back to Hindi. Opus-mt hi-en/en-hi converted to CTranslate2 int8 by
    `tools/build_language_models.py` (torch only in a throwaway venv; verified it reproduces the models byte for
    byte). Hand-written Hindi for fixed messages, the 4 whole-record questions and episode summaries (MT garbled
    them); the rest is MT (rough: "bearing" -> "automobile" once), flagged in the UI with the English original.
    Protected tokens ([E1], ids, numbers) are cut out, never masked-and-restored (masking failed: the model
    transliterated the placeholder). Not handled: Hinglish (Latin-letter Hindi), other languages, Hindi evidence
    snippets. `argostranslate` was rejected (pulls torch + spacy + stanza).
  - Attaching a photo now adds the picture + note to the chat as the user's message, with a device confirmation
    (live session only; replaying an old chat from the sidebar does not rebuild photo bubbles).
  - Test debris left in the user's live device by manual checks: one synthetic photo (`motor1-bearing.png`) and a
    few test chat turns. Pictures have no delete endpoint (not added).
  - Launcher gotcha: never pipe `start.ps1` (`| Out-Null` hangs, children inherit the pipe); use background run.
- [x] (2 Oct, evening) **Offline library + no model answers from its own training** (docs/DECISIONS.md D48):
  - Ask answers only from cited evidence. No match -> "needs an internet connection". The first attempt (Qwen answering
    from training) was removed after it answered a nonsense question as grounded; `LocalLLM.chat` no longer exists.
  - Library = `knowledge/general_facts.json` (212 Wikidata/SI facts) + 30,000 Simple English Wikipedia leads with
    precomputed bge-small vectors (`simple_wikipedia_core.jsonl.gz` 4.2 MB + `.vec.npz` 20.3 MB, CC BY-SA 4.0, notice in
    `knowledge/SIMPLE_WIKIPEDIA_LICENSE.md`), loaded in the background by `edge.main` on first start (`edge/seed.py`
    `AUTO` packs; ~30 s with the shipped vectors; `--no-seed` / `EDGE_NO_SEED=1` skips). Rebuild: `tools/extract_simplewiki.py`
    then `tools/build_wikipedia_core.py` (~45 min). Tests set `EDGE_NO_SEED=1` in `tests/conftest.py`.
  - Library entries must cover >= 60 % of the question's words (min 2); library text is quoted verbatim; a model sentence
    over device/fleet/web evidence may only use words found in the evidence it cites (`_supported` in `edge/device.py`).
  - Answers carry `sources` in {device, fleet, offline_kb, web}; a dead network with the switch ON is reported as offline
    (`online.reachable()`); each Qdrant search leg is isolated (`memory_failed`). `rag.check_output` number bug fixed.
  - Verified: 428 tests pass; live scenario 9/9 (run on a backed-up, restored runtime); real dead-proxy network-cut test;
    fresh-device simulation (library in 29 s, 217 texts embedded). Not measured: wrong-but-cited retrieval.
- [ ] LEFT FOR LATER (tried, not solved): naming misalignment (MaFaulDa 2-15 right of 197-301; needs e.g. phase between
  bearing housings); several cloud processes beyond ~0.8 cores (shared token registry + sequence counter); noise
  cancelling for the microphone; hardware attestation (needs TPM hardware).
- [ ] USER: real-world field test + phone tests (docs/FIELD_TEST.md) - remind them (they will do it after the web part)
- [ ] USER: the invalid `permissions.allow` rule in `C:\Users\Sir\.claude\settings.json` - needs their yes (global config)
- [ ] USER ONLY: public GitHub repo + push, LinkedIn post. NEVER post/publish anything without their manual yes.

## How to run (current)
- Tests: `.venv\Scripts\python.exe -m pytest` (302 tests, all passing, ~9.5 min on 28 Sep; some tests start `qdrant_server\qdrant.exe` themselves;
  pytest.ini already adds -q: do not add another -q or the summary line disappears)
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
