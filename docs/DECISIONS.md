# Decision log (what we chose, the evidence, and what it cost)

Newest first. Each entry says what changed, the measurement behind it, and the trade-off. The research-time
decisions (before code existed) are in [RESEARCH.md Appendix 3](RESEARCH.md#appendix-3--decision-log).

---

### D49 · Library answers fixed; the local model may answer simple general questions, labelled unverified (3 Oct 2026)
**What went wrong (user screenshots, 2 Oct night):** "capital of tamilnadu" found nothing (library says "Tamil Nadu");
"what is dna" answered with Genome / RNA (the DNA article's lead was corrupt: the extractor cut image captions at the first
`]]`); "how many states does India have" answered with a web-cached page about the Great Wall of China (the strict 60 %
coverage rule covered only the library, so cached and shared text matched on the single word "states"); library answers
were headed "From what this device holds" and carried an unrelated second item.
**Fixes:** title lookup (`seed.title_ref_ids`, space-insensitive, longest phrase wins); compound- and plural-aware coverage
(`_covered`); the strict coverage rule now applies to library, learned and shared text; library entries chosen as a group
(`_trim_reference`: title match first, a curated fact is the whole answer, duplicates dropped); source-accurate lead (none for
library-only answers); "this/these/those/there" no longer make a full question a follow-up; 37 Indian state / union-territory
capital facts + the count from Wikidata; extractor fixed and `tools/repair_wikipedia_core.py` repaired 1,086 leads (6 dropped)
re-embedding only those; `seed.refresh_if_stale` reloads the auto packs when the shipped files change (a fix now reaches devices
that already loaded the old text).
**Reversal of D48 rule 1, by the user's explicit request:** the local Qwen model may answer ONLY a short, simple
general-knowledge question (`_simple_general`; anything about machines, faults, repairs, this device or an identifier is
excluded), ONLY when no source anywhere had an answer, ONLY after agreeing with itself across three runs
(`rag.general_answer`: greedy + 2 sampled, every new word and number must repeat), and the answer is labelled
`[Unverified]` with `unverified: true`, `sources: []`. Unsure -> "I don't know." This lowers wrong answers; it cannot remove them.
**Measured (scratchpad `qwen_eval.py`, 30 questions, 28 answerable + 2 unanswerable):** 24 right, 0 wrong, 6 "I don't know".
Small sample: evidence the gate works, not proof it never fails.

### D49a · A passage that is merely about the topic is not an answer: extractive reader + technical library (3 Oct 2026)
**Measured problem:** end to end through the real app, general questions were only **66.7 %** accurate: "largest mammal"
returned the article on elephants, "father of computers" returned Alan Turing's biography, "15th president of Mars" a Taiwan
election. The library matched on topic, not on whether the passage answers.
**Fix:** `shared/qa.py` runs deepset/roberta-base-squad2 (ONNX int8, ~80-120 ms per passage, no torch) over up to 8 candidate
leads; it returns the sentence that answers, built only from the passage's words, or declines. `margin` (best span minus "no
answer") must be >= 6.0 (`Device.QA_MARGIN`); a bare "capital of X" is first phrased as "What is the capital of X?" (the
same fact scores ~6 as a fragment, ~11 as a question). A curated fact must cover EVERY question word (filler excepted);
other entries 75 % (60 % when the title is what was asked). A question that IS an entry's name or alias ("what is a PLC",
"how does an encoder work") skips the reading: the lead is the answer.
**Technical library:** `knowledge/tech_terms.txt` (~1,100 terms from the user's question list) -> `tools/build_tech_pack.py`
-> 862 cited English-Wikipedia leads with aliases ("PLC" -> Programmable logic controller). The audit removed 33 doubtful
mappings (e.g. "Encryption at rest" -> "Digital data", "C-rate" -> "Battery charger") and the builder drops any alias that
points at two articles (ISO, PID, SoC). "X vs Y" shows both definitions and says it is not a comparison. Design,
calculation, what-if, diagnosis and "why use" questions are NOT answered with a lead (`_OPEN_ENDED`): they get
"Needs internet connection for this." (user's wording, 3 Oct).
**Also fixed:** library lookups scanned all 30,000 records (`ref_id` is not an indexed field) - now direct reads by the
deterministic point id; the image model loaded for every question on a device holding a photo - now only when a picture's
note shares a word with the question; a lone digit ("atomic number 1") is no longer an identifier; shared fleet notes and
typed notes keep the lenient one-word rule (only the library and web-cached text are strict).
**Measured (bench/ask_qa.py, 58 questions: 45 known + 13 impossible, real app): accuracy 94.6 % (35/37 answers), coverage
80 %, library answers ~310 ms median.** The two misses: the model's recall "Indian elephant" for India's national animal
(labelled [Unverified]) and the device's own taught note about a room. **(bench/tech_qa.py, 123 questions drawn from the
user's list: 101 correct, 0 wrong entries, 20/20 design/what-if/calculation questions declined, 2 ambiguous aliases
declined on purpose; median 321 ms.)** Selection bias: those questions use terms the library was built from. Not measured:
questions outside both lists. "100 % on everything" is not claimed and cannot be.

### D48 · Offline answers come only from cited text; the model never answers from its own training (2 Oct 2026)
Ask with no network and no matching memory used to say "nothing relates to that". A first attempt let the local Qwen
model answer from its training instead; on the live device it answered a nonsense question ("the ceiling of the zxqv
chapel") from two loosely related articles and marked it grounded. Chosen instead:
- **Offline library** loaded into Qdrant Edge as `reference` records: 212 capitals / SI-unit facts (Wikidata CC0, SI
  Brochure) and the 30,000 best-developed Simple English Wikipedia articles (CC BY-SA, cited by URL). Embedding and
  loading the core took 3,097 s on this CPU (~10 docs/s); the other ~197k articles are an optional pack
  (`simple_wikipedia_rest`).
- **Shipped in the repository** (so a second computer, or the finals machine, has it): the 30,000 passages
  (`knowledge/simple_wikipedia_core.jsonl.gz`, 4.5 MB) and their bge-small vectors (`.vec.npz`, float16), with the CC BY-SA
  attribution notice. A device reuses the vectors only when its own text model is the same; it loads them in the
  background on first start (`edge.main`), so nobody waits ~45 minutes for 30,000 passages to embed. The rule "text
  embeddings never leave the device" protects private notes; these are public Wikipedia passages.
- **Relevance:** a library entry must cover >= 60 % of the question's content words (min. 2). One shared word is enough
  for a note somebody typed in, but not for 30k articles.
- **Library text is quoted word for word**; the model is not asked to paraphrase it. For device, fleet and web evidence a
  model sentence may only use words found in the evidence it cites, otherwise the evidence is quoted.
- **No match:** "needs an internet connection" (offline) and nothing else. No model answer, no fake source.
- **Source separation:** answers name `device`, `fleet`, `offline_kb`, `web` separately; the model is never a source.
- **Offline detection:** a dead network with the online switch ON is reported as offline (tested through a dead proxy).
- **Qdrant failure:** each memory-search leg is isolated; the answer says which one failed instead of crashing.
- Also fixed: `rag.check_output` rejected a correct number that ended a sentence in the evidence ("cabinet 7.").
Cost: "Who wrote Hamlet?" is unanswered offline unless an article lead repeats the question's wording. Wrong-but-cited
retrieval (an on-topic passage that does not answer the question) is still possible and is not measured.

### D47 · Microphone next to a louder machine: measured, and the answer is the existing confirmation (28 Sep 2026)
The README limit "background noise from louder machines was not tested" is now measured with two REAL recordings mixed
(BENCHMARKS §33). No new mechanism was added: the existing one-click "normal operation" confirmation removes the false
alarms (60-100 % -> 0-4 %), and the cost is stated rather than hidden - a neighbour as loud or louder masks 30-66 % of
faulty windows afterwards. A noise-cancelling second microphone or source separation would be the next step; not built
(the phone test in docs/FIELD_TEST.md will show whether it matters in a real room).

### D46 · A bearing line that sits on a shaft harmonic is not taken as a bearing fault unless it grew (28 Sep 2026)
MaFaulDa (one real machine, imbalance and misalignment on the same rig; BENCHMARKS §32) showed the device naming
imbalance "outer/inner race" (underhang radial: 0/333 right, 301/333 wrong). Cause, measured: this simulator's bearing
has BPFO 2.998x and BPFI 5.002x shaft speed, i.e. its defect lines sit ON shaft harmonics, and its healthy bearings
already show them strongly, so the absolute envelope rule fires on every window; meanwhile 1x had grown +9.8 sigma.
Three versions were measured, nothing tuned on the data:
1. "shaft order grew and the defect line did not -> shaft fault", everywhere: MaFaulDa 0 -> 264/333, but HUST fell
   97.6 % -> 81 % (real HUST bearing faults also lift 1x while their line grows little). Rejected.
2. The same, only when the line coincides with a whole shaft harmonic (within the rule's own +-3 % peak window, not a
   new parameter): MaFaulDa 258/333; HUST 95.2 % (40/42): B604 flips because bearing 6206's BSF (4.915x) is within
   3 % of 5x - the rule working as written, not a tuning target, so the tolerance was NOT narrowed to rescue it.
3. Kept: (2) plus, on a clash where the line did not grow and no single order grew, say "unknown - inspect" instead of
   a bearing name: misalignment wrong names 164 -> 28 / 197; HUST unchanged at 40/42.
Misalignment is still rarely NAMED (no single order grows > 3 sigma on this rig); it is detected, then "inspect".

### D45 · Fleet scale: measure the limit before optimising further (28 Sep 2026)
`bench/fleet_scale.py` (1,000 / 5,000 devices + 50 complete Edge devices) with a py-spy profile and per-second CPU
sampling. Speed-ups that the profile justified: O(1) token lookup, group-commit event writer, pooled keep-alive Qdrant
clients, page cache per mirror version, mirror reads not forcing a recompute (eventually consistent within ~2 s, like
the device mirror already was), hint model retrained at most every 30 s, recompute throttled per case. 1,000 + 50
devices: 108 s -> 34.7 s; 0 lost / 0 double in every run (BENCHMARKS §31). The bench itself was wrong once: 50
complete devices as threads of one process serialised their mirror restores (87-160 s); now in worker processes
(3.9 s). Stopped here on purpose: the cloud is one Python process at ~0.8 cores; going further means several cloud
processes (shared token registry + sequence counter), recorded as the next step, not built.

### D44 · Names after "by / with / call ..." need only a "more likely a name than not" score (28 Sep 2026)
Unlisted names in ALL CAPS / lower-case notes were caught only ~65 % of the time (BENCHMARKS §26): the name model
needs 0.9 to flag a word anywhere. After a cue word where notes name people, a non-vocabulary word is now flagged at
0.5 (fixed a priori, not tuned). A first version flagged ANY unknown word after a cue; the unit test "ALIGNED WITH
LASER" caught it, so the model gate was added. To avoid grading the rule on phrasings it was written around, the
benchmark gained 8 held-out templates written after the rule. Cost: a few more clean notes stay local (safe direction).

### D43 · mTLS: certificate bound to the token's device; CRL reloaded while running (28 Sep 2026)
Two items of THREATS.md "Not built / residual": (1) with `--mtls` any valid device certificate could carry any token;
now uvicorn's protocol exposes the verified certificate's name per connection (`cloud/tls.py` CertBoundH11) and the
cloud refuses a token whose device id differs (403). (2) A revoked certificate needed a cloud restart; the CRL is now
re-read when the file changes (OpenSSL uses the newest list of the issuer). Tests: `test_mtls_token_must_match_the_
certificate`, `test_mtls_revocation_takes_effect_without_restart` (refused within the 2 s reload interval). Cost: an
admin in a browser needs a certificate named after the admin id; an already open keep-alive connection is not cut.

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
