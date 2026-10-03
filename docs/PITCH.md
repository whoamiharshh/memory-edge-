# PITCH.md - Machine Memory at the Edge (Qdrant PS3, Code Cubicle 6.0 finals, 11 Oct 2026)

Everything below is either (a) a number from a script in `bench/` or a test in `tests/` that you can re-run on the demo laptop, or
(b) labelled as a limit. Do not say anything on stage that is not in this file. Judges punish a caught exaggeration far more than
an admitted weakness - and this project's strongest card is that it **measured its own weaknesses and changed the design because of them**.

---

## 0. The 20-second version (memorise this, say it first)

> "A technician is standing at a pump that is vibrating wrong, with no signal. The question is: *have we seen this before, what did
> we try, did it work?* Our device answers that **offline**, from its own memory, on Qdrant Edge. And a fix only ever leaves the
> machine **after the machine's own sensor shows it held** - so the fleet learns from evidence, not from someone's say-so. When two
> sites disagree, we **keep both** instead of picking a winner."

The one idea to land: **outcome-verified promotion + preserved disagreement.** Everything else supports it.

---

## 1. The 5-minute talk (timed; each beat has its proof)

| Time | Say | Show | Proof you can quote |
|---|---|---|---|
| 0:00-0:30 | The 20-second version above. | Title slide, one line. | - |
| 0:30-1:00 | **Why edge, not cloud.** Plants segment control networks from the internet on purpose (NIST SP 800-82 r3); roughly half of offshore rigs lack reliable broadband; Qdrant's own edge reasoning lists latency, connectivity, cost, privacy, isolation. | One slide, 3 bullets. | Sources are in `docs/PROBLEM.md`. |
| 1:00-1:40 | **The gap we found - honestly.** IBM Maximo and Augury already do edge/offline diagnostics, and Qdrant's own docs show a sync pattern but *leave conflict handling, durable queues and retry to the developer*. We do not claim domain novelty. We claim the **combination**: a tested, durable sync layer where **disagreement between devices is a visible state**. | Slide: "what exists / what we added". | `docs/PROBLEM.md` section 3. |
| 1:40-3:40 | **Live demo (DEMO.md steps 1-8, ~2 min).** Device A OFFLINE: healthy stays in its green band, a real inner-race fault opens **one** episode. Technician records `replace_bearing`. Machine checks itself: **20 healthy windows -> "symptom resolved (not a root-cause proof)"**. Only then SHARE; the note had a name in it, so the **note stays local**. Device C (another site) tried the same fix and it **failed**: cloud shows **DISPUTED + COMPETING**, both kept. Device B, offline, meets a bearing A never saw and shows the fleet evidence including the disagreement. | The three tabs. | Demo asserts itself: `python -m demo.scenario` -> `ALL STEPS PASSED`. |
| 3:40-4:30 | **The numbers** (section 2 below - pick 4, not 10): 36/36 fixes verified with **0 false promotions**; **0 lost / 0 duplicated** over 1,000 events with 139 lost acknowledgements and 209 hard restarts; **0 network connection attempts** from an offline device; 5,000 devices in 91 s with nothing lost. | The proof panel. | Section 2. |
| 4:30-5:00 | **Close on the honest part + the ask.** "We measured that vibration similarity only reaches **43 %** across a bearing the system has never seen - so we did **not** ship a smart-sounding match. The fleet groups by what the technician *confirmed*. That's a design decision driven by a measurement." Then: "It already runs on a phone microphone, a robot wrist sensor, vehicle fault codes and kiosk logs. We'd like to put it on a real line." | Last slide: 3 numbers. | `docs/BENCHMARKS.md` section 3. |

**Do not** spend demo time on the chat/Ask feature unless a judge asks - it is the newest, least-proven part. If asked, say:
"Ask answers only from cited evidence and says 'Needs internet connection for this.' when it doesn't know; model-written answers are labelled **[Unverified]**."

---

## 2. Proof table - every claim, its result, and the command that reproduces it

Run these on the demo laptop *before* the pitch so the numbers are fresh and you can say "I re-ran this this morning".

| Claim | Tested result | Reproduce |
|---|---|---|
| **The sensor can verify a fix** (K3, 36 real CWRU recordings, fresh device each) | Fixed replays verified **36/36**; still-faulty correctly "persists" **36/36**; **0 false promotions**; **0/116** gate false alarms on unseen healthy loads; fault windows flagged **100 %**; two different faults separated by healthy running -> separate episodes **20/20**; median **20 windows** to a verdict | `python -m bench.gate_verifier` |
| **Works on a machine it was never tuned on** (HUST: another lab, sensor, 51.2 kHz, 5 bearing types) | Fix verified **42/42** after one "normal operation" confirmation; still-faulty caught 38/39; **0 false promotions**; fault-type hint **95.2 %** (40/42); healthy false alarms at an untaught load **4/255 (1.6 %)** | `python -m bench.hust_holdout` |
| **Natural wear, not lab-seeded faults** (Univ. of Ottawa, 20 bearings, phone **microphone**) | **0/380** false alarms; **95.7 %** of faulty windows flagged; new bearing verified **19/20** | `python -m bench.acoustic_uottawa` |
| **Honest limit on similarity** (K2) | Same bearing: P@3 **0.997** (leaky split); **unseen bearing: 0.429** (chance 0.277); physics rule on defect frequency: **0.644** (inner 1.00, ball 0.69, outer 0.24). *This is why the fleet groups by technician-confirmed fault class.* | `python -m bench.retrieval_vib` |
| **Sync never loses or doubles evidence** | 10 seeded runs, 5 devices, 1,000 events with requests dropped (207), acknowledgements lost after the cloud applied them (139) and **209 hard restarts**: **0 lost, 0 duplicates, 0 tally error** | `python -m bench.sync_partition` |
| **Bad networks and wrong clocks** | Six scenarios (slow 16 kB/s, 40 % cuts, stalls, flapping, all at once): **0 lost, 0 counted twice**, mirrors identical to the cloud. Clocks +3 d / -2 d / +40 min corrected; worst corrected-time error **1.0 s** | `python -m bench.network_faults` |
| **Truly offline** | Every device function with a socket guard that blocks and records connections: **15/15** pass, **0 connection attempts** (the guard catches a deliberate request: self-test passes) | `python -m bench.offline_check` |
| **Scale** | **5,000 devices + 50 full devices: all pushes in 91.3 s, 0 lost, 0 double-counted**, 5,050/5,050 mirrors equal the cloud; 1,000-device burst: **419 events/s** | `python -m bench.fleet_scale` |
| **Edge speed** (this laptop) | Hybrid query (vector + note + BM25, RRF) on Edge at **50,000 points: p50 2.6 ms**; dense **0.22-0.45 ms**; the same dense query over HTTP to the local Qdrant **Server: 8-18 ms**. Durable write (upsert + `flush`) **60 ms** | `python -m bench.latency` |
| **Small footprint** | Real-time monitoring uses **16.5 % of one core**, **304 MB** RAM (no LLM), **4x** real-time headroom; one shared fix is **855 bytes** vs 499,712 bytes of raw signal (**584x** less) | `python -m bench.resources` |
| **Privacy** | Raw signals and raw notes never leave; text embeddings are never shipped (embedding-inversion risk); notes encrypted at rest (AES-256-GCM); redactor keeps a note local unless it finds nothing: names on its list **100 %**, unseen names **95 %** typed normally, **85-87 %** in ALL CAPS/lower case | `python -m bench.redaction` |
| **Crash safety** | Qdrant Edge loses unflushed writes on a hard kill (our K4 finding: **0/200** survive without `flush()`, **200/200** with). We journal to SQLite first, then write + `flush()`, and replay on boot: hard-kill test loses **0** acknowledged writes | `pytest tests/failure/test_store_durability.py` |
| **Engineering discipline** | Full test suite green (573 tests at the last full run, 3 Oct), live 9-step scenario asserts itself | `pytest` / `python -m demo.scenario` |
| **One engine, many devices** (same gate/verifier/sync, different signal profile) | Robot wrist sensor: **96-99 %** of failures caught, 0-10 % false alarms (UCI, cross-validated). Kiosk/app logs: **99.98 %** of failed sessions at **0.39 %** false alarms (HDFS real logs). Trucks: ROC-AUC **0.75** on 5,045 held-out SCANIA trucks | `bench/robot_model.py`, `bench/events_hdfs.py`, `bench/vehicle_scania.py` |

**Conversation / Ask layer (newest; measured 3 Oct on the build of that day - re-run before you rely on it):** 50/50 scripted turns across 12 conversations on the live app; 41 of 41 answered general questions correct (91 % coverage); 101 correct / 0 wrong on 123 engineering-concept questions, 20 of 20 design/what-if questions declined. Commands: `python -m bench.conversations`, `bench.ask_qa`, `bench.tech_qa`.

---

## 3. How Qdrant is used (judges from Qdrant will probe this - know it cold)

| Piece | What we do | Why it matters |
|---|---|---|
| **Qdrant Edge** (`qdrant-edge-py==0.8.0`, pinned - it is a beta) | One embedded shard **per device**: episodes, technician notes, the offline library, with **named vectors** - `vib` (27-d DSP fingerprint, Euclid), `note` (bge-small, 384-d, Cosine) and a **sparse BM25** vector - queried with **hybrid search in one request** (`Prefetch` + `Fusion.Rrf`) and payload filters/keyword indexes | Works with no network; ms-level queries at 50k points |
| **Qdrant Server** (v1.19.1 official binary) | The fleet cloud: one set of collections **per tenant** (events, case groups, mirror); tallies recomputed under **CAS on a version field** | Conflict-aware group view across sites |
| **Dual-shard pattern** (Qdrant's documented approach - credit it, don't claim it) | Device keeps a **mutable local shard** plus a **read-only fleet-mirror shard**, filled from Qdrant **shard snapshots** or scroll rows, whichever is cheaper in *measured bytes* | Device B uses the fleet's knowledge offline without ever overwriting its own memory |
| **What Qdrant's docs leave to the developer, and we built** | Durable outbox, deterministic event ids + `insert_only` (idempotent replay), per-event acks, retry with backoff + jitter, conflict states (`DISPUTED`, `COMPETING`, `ALTERNATIVES`, `RECURRED`), tombstone retraction needing two admins | This is the contribution: a tested sync/conflict layer on top of Qdrant |

---

## 4. Predicted judge questions - with the answer and the technical follow-up

Format: **Q** - short answer (say this) - *if they push* (the deeper answer).

### A. "Why does this need to exist?" (product / market)

**Q1. IBM Maximo and Augury already do this. What's new?**
Short: "Correct - we don't claim domain novelty. Offline diagnostics exists. What none of the sources we checked do is make **disagreement between two sites a first-class, visible state**, on a **tested, durable sync layer**, and gate sharing on the **machine's own verification** of the outcome."
*If pushed:* name the three mechanisms: (1) outcome-verified promotion (nothing leaves until 20 healthy windows), (2) `DISPUTED`/`COMPETING` flags instead of a trust score or last-write-wins, (3) failed fixes are shared too. Point to `docs/PROBLEM.md` section 3 - we corrected our own earlier over-claim in writing.

**Q2. Who is the user and what decision are they making?**
"A plant technician or reliability engineer at a misbehaving machine, deciding *what evidence from earlier incidents to weigh before acting*. The system **shows evidence and never recommends an action** - that's deliberate for safety."

**Q3. Why would a plant adopt this over their CMMS?**
"It sits next to it: it works with no network, keeps raw signals on the machine, and only shares what was verified. We'd integrate via the structured event (855 bytes), not replace the CMMS." *Don't claim an existing integration - there isn't one.*

### B. "Is the ML real?" (accuracy)

**Q4. How do you know a fix worked? Isn't that just a timer?**
"It's the machine's own data: after the action, **N=20 consecutive windows must fall back inside its healthy radius**. Result on 36 real recordings: 36/36 fixed verified, 36/36 still-faulty caught, **0 false promotions**. The wording is 'symptom resolved for 20 windows' - we **never** claim root cause."
*If pushed:* a replaced part doesn't match the old baseline, so there's a **replacement-aware rule** ('nearer to healthy than to the fault', parameter-free): a new bearing verifies 19/20 (Ottawa), HUST 42/42. The N and the radius were chosen on CWRU and **not retuned** on HUST.

**Q5. Your similarity search is only 43 % on unseen bearings. Isn't that weak?**
"Yes - and we measured it on purpose. Same bearing: 99.7 %; a bearing never indexed: **43 %**. So we **did not** build the fleet on vibration similarity. The fleet groups by the **technician-confirmed fault class**; vibration is used for the novelty gate (healthy vs abnormal, 100 % on unseen bearings) and same-machine recurrence. That's a design change forced by a kill test (K2)."
*If pushed:* a physics rule on bearing defect frequencies gets 64 %; on the held-out HUST machine the device's fault-type hint is **95.2 %**; and a hint is marked **confident only when physics and the fleet-learned model agree** (the fleet model is measured on devices it never saw).

**Q6. Was anything tuned on the test data? Is this overfit?**
"Thresholds were chosen on CWRU and shipped unchanged. We then ran **HUST** (different lab, sensor, sampling rate, five bearing types) and **Ottawa** (natural wear, not seeded faults) with nothing retuned. The first HUST run **failed** (305/305 false alarms) - that exposed two bugs and a real concept gap (an untaught operating point looks abnormal), which we fixed with a 'teach a new healthy state' step. The report keeps all three runs."

**Q7. What about false alarms in the field?**
"Measured: 0/116 (CWRU unseen loads), 0/620 (Ottawa vibration), 0/380 (phone microphone), 1.6 % at an untaught HUST load. A real plant has more operating states; **each needs one confirmation** - that's a stated caveat."

**Q8. Real data or synthetic?**
"Real public data only for everything trained or measured: CWRU, HUST, Univ. of Ottawa, UCI robot failures, SCANIA trucks, Loghub HDFS, MaFaulDa. Synthetic signals exist **only inside unit tests**. That's a rule in `CLAUDE.md`."

### C. "Does the Qdrant part hold up?" (technical)

**Q9. Why Qdrant Edge instead of SQLite + FAISS or a cloud vector DB?**
"Three reasons: **named + sparse vectors with hybrid RRF in one request** (vibration, text semantics and exact codes together), **payload filtering + CAS**, and **the same engine/API on the device and in the cloud** so the mirror is a Qdrant snapshot. Measured: hybrid query at **50k points = 2.6 ms p50** on the device with no network."
*If pushed:* honesty point - Edge vs Server over localhost HTTP is 0.2-0.45 ms vs 8-18 ms, but **that gap is mostly HTTP + serialisation, not search**. 'The reason for the edge is that it keeps working with no network, not the milliseconds.'

**Q10. Edge is a beta. What broke?**
"Pinned `qdrant-edge-py==0.8.0`; all calls go through one adapter (`edge/store_edge.py`). Two findings: (1) **writes are lost on a hard kill unless `flush()` is called** - 0/200 vs 200/200 - so we journal to SQLite first and replay on boot; (2) a very long path gives `os error 3` on Windows. Both are in our notes with tests."

**Q11. How do you resolve conflicts between devices?**
"We don't resolve them - we **record them**. Tallies per (fault group, action): worked / failed / held / recurred / distinct sites. Flags: `DISPUTED` (same action, both outcomes), `COMPETING` (different root causes), `ALTERNATIVES` (different actions both worked). No trust score, nothing overwritten; the technician sees both."
*If pushed:* tallies are **recomputed from stored events** under CAS, so replays can't double-count; retraction is a **tombstone** and needs **two different admins**.

**Q12. Did you use Qdrant's snapshot sync?**
"Yes - Qdrant's documented dual-shard pattern; credit to them. The mirror is bootstrapped from a full shard snapshot, then the device picks the cheaper of scroll delta or full snapshot by **measured bytes**. Pure partial-snapshot mode is selectable but the default is the measured-cheapest."

**Q13. What happens in a network partition or crash mid-sync?**
"Outbox first, deterministic event ids, `insert_only` on the server, per-event acks, backoff + jitter. Tested with dropped requests **and lost acknowledgements** (the nasty case): 1,000 events, 209 hard restarts -> **0 lost, 0 duplicates**. A torn mirror swap is repaired on boot; a broken mirror never breaks local search."

**Q14. Device clocks are wrong in the field.**
"Handled: every push carries device time; the cloud measures the offset (it saw exactly +72 h, -48 h, +0.67 h) and shifts that device's timestamps; worst error after correction **1.0 s**. Dates are displayed, never used to decide anything."

### D. "Does it scale?"

**Q15. How many devices can one cloud take?**
"Measured on one laptop: **5,000 devices + 50 complete devices, 10,100 events in 91 s, 0 lost, 0 double-counted**; 1,000-device burst at **419 events/s**."
*If pushed - own the limit:* "The cloud is **one Python process (~0.8 core, the GIL)**. Past that you need several cloud processes behind a load balancer, which needs a **shared token registry and sequence counter - not built.** We profiled it: about a third of samples wait on Qdrant calls."

**Q16. Memory/CPU on the device?**
"16.5 % of one core, 304 MB RAM monitoring in real time; 4x real-time headroom; with the optional local LLM about 2 GB." Runs on Windows, Linux x86-64/ARM64 (Raspberry Pi 4/5 packages exist for `qdrant-edge-py`), macOS. *Tested on Windows 11 only* - say that.

### E. "Security and privacy"

**Q17. What leaves the device?**
"Structured evidence only (855 bytes): fault class, action, outcome, verification, fingerprint. **Raw signal never leaves. Raw notes leave only on opt-in and only if the redactor finds no personal data. Text embeddings are never shipped** (embedding inversion); the cloud re-embeds the redacted text itself."

**Q18. A compromised or malicious device?**
"Per-token identity (hashed, 30-day expiry, auto-renew); optional **mTLS** with revocation; **plausibility checks** reject impossible reports; **quarantine** pulls all of a device's evidence in one reversible step; **hash-chained audit log**; one site can't manufacture consensus (counts per site). **Residual risk, stated:** an insider with a valid token can still add *plausible* false evidence; the code-integrity check is tamper *evidence*, real attestation needs a TPM." (`docs/THREATS.md` has 25+ rows with the test that proves each.)

**Q19. Prompt injection through technician notes into your LLM?**
"The LLM has no tools, is outside every decision, and each output sentence must cite evidence and contain no advice or it is dropped. Test: `tests/ai/test_rag.py::test_brief_injection_echo_is_removed`."

### F. "The AI / chat part" (newest - be careful)

**Q20. Can your assistant hallucinate?**
"Yes, which is why it is built to say so: answers come from cited evidence (device, fleet, offline library, web) and are **quoted**, not paraphrased; when it doesn't know it says **'Needs internet connection for this.'** If the local model answers a general question from its own training, it is labelled **[Unverified]**, and the model is **never** allowed near faults, repairs or this device. It is **outside the decision path** - it never drives a share or a recommendation."
*If pushed - the honest measurement:* "On a 58-question set we found that matching *topic* is not the same as *answering* - accuracy was 66.7 % until we added an offline extractive reader that must find the answer in the passage; it reached 41/41 correct answers at 91 % coverage. The model-only recall is slower (about 8 s on this CPU) and can still be wrong - that's the label."

**Q21. Why a small local model at all?**
"The evidence brief summarises retrieved text, nothing more - 1.5B on CPU takes about 3 s. The stronger 7B model, when present, is used only for labelled general-knowledge answers."

### G. "Honesty / limits" (volunteer 2 of these before they ask)

**Q22. What doesn't work yet?** Say these calmly, in this order:
1. "**No field test yet** - everything is on public datasets; `docs/FIELD_TEST.md` (fan + phone, second PC, a technician) is the next step."
2. "**Fleet similarity across unseen bearings is 43 %** - by design we rely on confirmed fault class."
3. "**Cloud is single-process** (~0.8 core); multi-process needs shared state we haven't built."
4. "**No hardware attestation** (needs a TPM); **no Android app** (it's an installable web app + phone sensor page)."
5. "Tested on **Windows 11**; Linux/macOS CI is written but has not run."
6. "A real plant has more operating states; each needs one 'normal operation' confirmation."

**Q23. Is any of this a mock-up?** "No. `demo.scenario` runs the real devices and real Qdrant; the backup video is a recording of the live UIs, and if a step fails it ends with STOPPED and the reason."

**Q24. What would you do with another month?** "Field test on a real machine; multi-process cloud; the pure partial-snapshot mirror as default once measured; and a labelled evaluation of retrieval precision on real technician questions."

### H. Hostile / curveball

**Q25. 'Isn't "symptom resolved for 20 windows" just a fancy threshold?'** "Yes - on purpose. It's the cheapest *honest* verifier: it never claims causation, and it was tested against 36/36 and 42/42 recordings with 0 false promotions. The value is the *gate* it creates - nothing is shared unverified."

**Q26. 'Why not let an LLM recommend the fix?'** "Because a confident wrong repair recommendation is the worst failure in this domain. We show evidence, physics, the manufacturer's manual page and **cited** procedures; the technician decides."

**Q27. 'Show me it offline.'** Turn Wi-Fi off, run steps 1-9. (`bench.offline_check` is the scripted proof: 15/15, 0 connection attempts.)

---

## 5. Slide outline (8 slides max)

1. **Title + the one sentence** - "Offline machine memory that shares a fix only after the machine proves it held."
2. **The problem** - technician, no signal, "have we seen this? what did we try? did it work?" + 3 sourced facts.
3. **What exists / the gap** - Maximo, Augury, Qdrant's own sync docs; our narrow, defensible claim.
4. **How it works** - diagram: sensor -> gate -> episode -> action -> **verify (20 windows)** -> policy -> outbox -> cloud (Qdrant Server) -> fleet mirror -> Device B offline.
5. **Live demo** (switch to the three tabs).
6. **Proof** - 4 numbers big: **36/36, 0 false promotions** - **0 lost / 0 duplicates (1,000 events, 209 restarts)** - **0 connection attempts offline** - **5,000 devices in 91 s**.
7. **We measured our weakness** - the 43 % vs 99.7 % graph and what we changed because of it; HUST first-run failure -> fix.
8. **Beyond bearings + what's next** - phone mic, robot, kiosk logs, trucks; field test; ask.

---

## 6. Demo safety net

- Start 10 minutes early: `demo\run_demo.ps1 -Reset` **(only on a backed-up runtime - a reset wipes the loaded offline library)**, wait ~20 s, open the three tabs. Do not pipe the launcher's output.
- Dry run once: `python -m demo.scenario` -> `ALL STEPS PASSED`; then reset again.
- Have `runtime\recording\backup_demo.webm` ready; if anything fails: "Let me show the recorded run of the same live system" - then show the `demo.scenario` output.
- Laptop on power, notifications off, Windows updates paused, browser zoom 90-100 %, **close other heavy apps** (the optional 7B model needs about 7 GB of free RAM; the disk must have free space - check before the session).
- If a judge asks for something you haven't built: "That's not built yet - here's how we'd do it" is a strong answer. Inventing is the only losing move.

## 7. Cheat card (the 12 numbers)

36/36 fixes verified - 0 false promotions - 42/42 on a lab never tuned on - 95.2 % fault-type hint on held-out HUST - 43 % vs 99.7 % (K2) - 0 lost / 0 dup over 1,000 events with 209 restarts - 15/15 offline, 0 connection attempts - 5,000 devices in 91 s - 2.6 ms hybrid query at 50k points - 304 MB / 16.5 % of one core - 855 bytes per shared fix (584x less than raw) - 0/380 false alarms on a phone microphone.

## 8. Before you present - things only you can verify

- Re-run `pytest` and `python -m demo.scenario` on the demo machine and write down today's numbers; quote **those**.
- The newest changes (conversation layer, assistant answers, library repair) were still being finished and are **not committed**; run `python -m bench.conversations` against the running app and only demo chat if it passes.
- Confirm the demo laptop has the 130 MB answer-reader model (`python -m tools.fetch_qa_model`) and, if you want the stronger assistant, the 4.7 GB model - otherwise Ask falls back to the weaker mode.
- Memorise Q5 (43 %), Q15 (single-process limit) and Q22 (limits). Judges remember the team that volunteers its weaknesses.

---

## 9. The basics judges ask first (survival answers - know these cold)

These are the first questions an experienced judge asks to see whether you understand your own system. Each answer is short enough to say in 20 seconds.

### How information moves between Edge and Server
**Push (device -> cloud):** every candidate fix is first written to a **SQLite outbox** on the device (`queued -> uploading -> synced | rejected`). A background worker sends batches of **up to 50 events** to the cloud's Sync API. The cloud answers **per event**: `accepted`, `duplicate` or `rejected`. Only an acknowledged event is marked synced. **Pull (cloud -> device):** the device refreshes a **read-only fleet-mirror shard** (Qdrant's documented dual-shard pattern) from a Qdrant **shard snapshot** or by scrolling rows - whichever is cheaper in *measured bytes*. A new mirror is swapped in only after a probe, so a broken download never breaks local search.
*Proof:* `bench.sync_partition` (0 lost / 0 duplicates, 1,000 events), `bench.mirror_sync`.

### What happens if a sync fails? How does it retry?
Nothing is lost, because the event is already on disk before it is sent. Failed requests, cut connections and **lost acknowledgements** are retried automatically with **exponential backoff plus jitter** (30 s read timeout). A `401/403` returns the rows to `queued` and **pauses the worker with "auth required"** instead of dropping data. In the demo, **Resend** replays an already-synced event to show the cloud answers `duplicate`.
*Proof:* six bad-network scenarios (40 % of requests cut, stalls, flapping link, all at once): **0 lost, 0 counted twice** (`bench.network_faults`).

### How do you detect duplicate experiences?
Two layers. **On the device:** the novelty gate compares each window with the healthy baseline and with existing episodes - a window near an existing episode is a **recurrence** (`occurrences + 1`), and a contiguous abnormal run is **one** episode, not hundreds. **Between device and cloud:** every event gets a **deterministic id**, the cloud stores with `insert_only`, and tallies are **recomputed from stored events**, so replaying a batch cannot double-count. The cloud also recomputes the id from the *authenticated* device id, so one device cannot forge another's event.

### Who can access what?
- **The technician (operator token):** the device's own UI/API, bound to localhost; a network address refuses a short or demo-style token.
- **A device (device token):** can push *its own* events and pull *its tenant's* mirror - nothing else. Tokens are stored only as hashes, **expire after 30 days**, and are revocable.
- **An admin:** issues/revokes tokens, quarantines a device; **retraction needs two different admins**; every admin action goes into a **hash-chained audit log**.
- **Tenants** are separate Qdrant collections, and the tenant comes from the token, never from the payload. Optional **mTLS** ties a certificate to a device id; stolen certificates can be revoked.
*Proof:* `docs/THREATS.md` (each row names the test).

### Which memories are allowed to reach the server?
Only what passes the policy engine's ordered gates: privacy -> validation -> duplicate -> evidence -> **sensor verification** -> human confirmation -> SHARE. What leaves is **structured evidence** (fault class, action, outcome, verification, fingerprint) - **855 bytes** per fix. **Never leaves:** the raw signal (584x larger), text embeddings (embedding-inversion risk), and raw technician notes unless the technician opts in **and** the redactor finds no personal data. Unverified or still-faulty-without-evidence fixes stay local.

### When are old memories deleted? How long is local data kept?
**Nothing is silently deleted.** The only automatic rule is **ARCHIVE**: a *closed, decided* episode not seen for **90 days** keeps its episode record and best exemplar (so recurrence still works) and drops its other fingerprints - one journalled operation, crash-safe. **Open or undecided episodes are never archived automatically**, and episodes still waiting to sync are skipped. The SQLite journal keeps the last 5,000 applied operations. Conversation history stays until the technician clears or deletes it. Retention is deliberately **separate from the share decision**.
*Honest limit:* the sync outbox has no size cap (it is bounded only by disk).

### How can Qdrant run locally?
- **Qdrant Edge** is a Python package (`qdrant-edge-py`, pinned to 0.8.0) that runs **inside the device process** - no server, no Docker, no port. Data is a folder on disk.
- **Qdrant Server** (the fleet cloud) is the **official Windows/Linux/macOS binary**: `python -m tools.qdrant_local download` fetches v1.19.1 into `qdrant_server\`. **No Docker is needed** (this laptop has none).

### What if the internet doesn't come back for 30 days?
**The device keeps working fully:** it senses, gates, remembers, searches, verifies fixes and queues shares - the offline check proves 15/15 functions with zero network attempts. Evidence **waits in the outbox**, so nothing is lost. When the network returns it drains in order, duplicates are impossible, and clock drift is corrected by the cloud (worst error 1.0 s in test).
*The honest catch:* device tokens **expire after 30 days**, and a device only renews its token when it can reach the cloud with **less than 7 days left**. After a very long outage the token may have expired - the cloud then answers 401 and the worker **pauses with "auth required"** (still no data loss) until an **admin issues a new token**. We have not built an automatic re-enrolment path or a longer offline grace period; say so, and say that is the first thing we would add (renew on every successful sync, or a longer TTL for devices that cannot be reached). The fleet mirror on the device is simply **stale** until then; local memory is unaffected.

### What size is the app? Can it go on the Play Store?
**There is no Android app today** - that was a deliberate scope decision. The phone is the **sensor and the screen**: an installable web app (PWA) that streams microphone/motion data to a device over HTTPS. A thin wrapper of that web app for the Play Store would be a few MB (an *estimate*, not measured); I did **not** verify Google Play's current size limits, so check before promising anything.
The heavy parts live on the edge device (a laptop, mini-PC or Raspberry Pi), not in the phone app. **Measured sizes:** text model bge-small **64 MB**; offline library **~25 MB** (4.3 MB text + 20 MB vectors) plus 0.8 MB technical pack; answer-reader model **~125 MB**; optional local LLM **1.07 GB** (1.5B) or **4.7 GB** (7B, needs ~7 GB RAM); one device's Qdrant folder **~471 MB** (mostly pre-allocation); monitoring RAM **304 MB**. A native phone build that ran the whole engine would be a separate project (`qdrant-edge-py` publishes packages for Windows, Linux x86-64/ARM64 and macOS - not Android).

### What did you use instead of "Jev"?
Jev here means **TypeSafe AI's "System One"** decision model - a **proprietary, cloud-only, early-access** classifier that returns probabilistic typed answers over a REST API (`docs/RESEARCH.md` I.5). We rejected it for three reasons: it **cannot run offline** (the whole point), its **probabilistic output "does not preclude being incorrect"** in a decision path, and it is a **hard external dependency**. Instead the decision path is **deterministic and inspectable**: a DSP fingerprint (27 numbers) + a novelty gate against the machine's own healthy baseline + **bearing physics** (defect frequencies from geometry) + a **sensor-based outcome verifier** + a rules-based policy engine that records `decision_reasons[]` for every decision. Text uses **bge-small embeddings + BM25** (Qdrant Edge's built-in sparse). A local LLM exists only to *summarise cited evidence* and is outside every decision. We also tested and **rejected** a heavier model (Laya) - measured, then deleted.

### What is the accuracy of your model - can it answer everything?
**No, and we don't claim it.** Be precise about *which* part:
- **The fault decision path is not a language model.** It is measured: fixes verified **36/36** and **42/42** (held-out lab), **0 false promotions**, fault-type hint **95.2 %** on the held-out machine, **43 %** similarity on an unseen bearing (which is why the fleet uses confirmed fault class).
- **The question-answering assistant** answers *only from cited sources* and otherwise says **"Needs internet connection for this."** Measured on the running app (3 Oct build): general questions **41/41 correct with 91 % coverage** (58-question set); engineering concepts **101 correct / 0 wrong** (123 questions) with **20/20** design/what-if questions correctly declined; **50/50** scripted conversation turns.
- **When the local model answers from its own training it is labelled [Unverified]** and kept away from faults, repairs and this device. It is slower (~8 s on this CPU) and **can be wrong** - we have not measured the 7B model's recall in isolation.
- **Cannot answer:** anything not in the 31,000-entry offline library, current events, the device's own unseen history, and calculations/design advice by design.

### Qdrant in 60 seconds (if a judge asks "explain it to me")
Qdrant stores vectors plus payload and searches by similarity with filters. **Edge** is the embedded version on each device - same API, no server. We keep, per device, **three kinds of vector on one point** (a 27-d vibration fingerprint, a 384-d text embedding, and a sparse BM25 vector) and ask one **hybrid query** that fuses them (RRF). The **Server** collects verified evidence from all devices, groups it by component and confirmed fault class, and hands every device a **read-only mirror** so it can reason about the fleet offline.
