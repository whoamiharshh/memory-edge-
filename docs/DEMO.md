# Demo: judge-facing run (~5 minutes), rehearsal checklist, and backup

Everything runs on one laptop, offline-capable: Qdrant Server (official Windows binary), the fleet cloud (Sync API +
fleet UI), Device A (site 1), Device B (site 2) and Device C (site 3, the disagreeing report). Optional finale: a
phone on a desk fan as a live sensor (docs/FIELD_TEST.md Part 2). There is no Docker and no internet dependency at demo time.

## Before the session (10 minutes)
1. `powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset` (fresh state; do **not** pipe its output).
2. Wait ~20 s (bge-small and the local LLM load), then open three browser tabs:
   | Tab | URL | Sign in with |
   |---|---|---|
   | Device A | http://127.0.0.1:8101 | `operator-devA` |
   | Device B | http://127.0.0.1:8102 | `operator-devB` |
   | Fleet cloud | http://127.0.0.1:8100 | admin token printed by the launcher (also `runtime\cloud\bootstrap.json`) |
3. Optional dry run without the audience: `.venv\Scripts\python.exe -m demo.scenario` (all 9 steps assert), then
   `-Reset` again for a clean state.
4. Have the backup video ready: `runtime\recording\backup_demo.webm` (see "Backup" below).

## The run (each step: what to click, what to say, what the audience sees)

| # | Click | Say | They see |
|---|---|---|---|
| 1 | Device A: switch **OFFLINE**; file `99 · normal · load 2` → Play; then `105 · inner_race 7 mil` → Play | "This machine has no network. Healthy vibration stays inside its own baseline radius. Now a real inner-race fault." | Chart jumps out of the green band; **one** new episode; physics hint *inner_race* shown **with its measured accuracy** |
| 2 | Similar to selected episode | "Memory is empty, so the system says so. It never invents an answer." | "no fleet evidence" |
| 3 | Type a note that names a person, tick share, Save; confirm fault class; Record action `replace_bearing` | "The technician records what they did. The note has a name in it." | Policy gates: KEEP_LOCAL, *awaiting outcome* |
| 4 | Play `100 · normal · load 3` (post-repair) | "Now the machine itself checks the fix: 20 consecutive windows back inside the healthy radius." | Verification bar fills: **symptom resolved for 20 windows (not a root-cause proof)** |
| 5 | Technician says **Worked** | "Only now may it leave the machine, and only the structured evidence: the redactor found the name, so the note stays local." | **SHARE**, note stays local, outbox **QUEUED** |
| 6 | Switch **ONLINE** | "Connectivity returns: the outbox drains." Press **Resend**: "Same event again: the cloud says duplicate, and counts it once." | QUEUED → SYNCED; activity shows `1 duplicate` |
| 6b | Device C tab (site 3, `operator-devC`): OFFLINE; Play `106`; confirm *inner_race*; record *replace_bearing* with root cause *lubrication_starvation*; Play `106` again from window 30; *Failed*; ONLINE | "Another site tried the same fix, and on its machine the fault persisted. That is shared too, as failed evidence." | Verification: symptom persists → SHARE as failed evidence |
| 7 | Fleet cloud tab → the case | "Grouped by component and **technician-confirmed** fault class, because we measured that vibration similarity does not transfer across bearings (43 %). Disagreement is kept, not overwritten." | replace_bearing worked at site1, failed at site3: **DISPUTED + COMPETING** (fatigue wear vs lubrication starvation), both kept |
| 8 | Device B: ONLINE → Sync now → OFFLINE; Play `169 · inner_race 14 mil` (a bearing A never saw); Similar to selected episode | "B pulled the fleet mirror: a full Qdrant shard snapshot the first time (gzip ~190 kB), then small deltas. Now it is offline and meets a new bearing." | **Fleet evidence offline**, including the disagreement; per-leg ranks vib / note / bm25; physics panel (severity, defect frequencies) and the cited **documented procedure** for the fault class |
| 9 | Evidence brief → Generate | "A local 1.5B model summarises only the retrieved evidence. Every sentence must cite it, and advice is removed. It is never used for decisions." | Cited brief, removed sentences listed |
| 10 | (optional) Wi-Fi off, repeat step 8's search | "Nothing here needs the network." | Same result |

**Proof panel to show at the end** (README "Measurements"): K3 0 false promotions; K2 honest table; partition bench
0 lost / 0 duplicates over 1,000 events; offline check 15/15 with 0 connection attempts; mirror bytes; latency.

## Rehearsal checklist
- [ ] `demo\run_demo.ps1 -Reset` starts all four processes (launcher prints the four URLs)
- [ ] `python -m demo.scenario` → `ALL STEPS PASSED`
- [ ] `python tests\ui_check.py` → `console_errors: []`
- [ ] Network switch rehearsal: steps 1-5 with Device A OFFLINE, step 6 ONLINE
- [ ] **Wi-Fi off** rehearsal: turn Wi-Fi off in Windows, run steps 1-9 again (everything is localhost; the scripted
      equivalent is `python -m bench.offline_check`: 15/15 with every connection attempt blocked)
- [ ] Backup video recorded and played back once
- [ ] Laptop on power; Windows updates paused; notifications off; browser zoom 90-100 %
- [ ] `demo\stop_demo.ps1` stops everything

## Backup
`.venv\Scripts\python.exe -m demo.record_backup` (after a fresh `run_demo.ps1 -Reset`) records the real UIs with a
headless Microsoft Edge while the scripted scenario runs, with a caption per step:
`runtime\recording\backup_demo.webm` plus one PNG per step. It is the live system, not a mock-up; if a step's check
fails, the video ends with "STOPPED" and the reason.

## If something goes wrong on stage
| Symptom | Fix |
|---|---|
| A device tab says "auth required" | Wrong operator token; sign out, sign in again |
| Chart does not move | Baseline not fitted: press "Fit healthy baseline" |
| Cloud tab empty after step 6 | Press "Sync now" on Device A; check the activity feed for the push result |
| Device B shows no fleet evidence | B must pull while ONLINE ("Sync now"), then go OFFLINE; check "mirror" line in Sync & Network |
| LLM brief slow (> 10 s) | It is CPU-only; the template fallback appears if the model is missing |
| Anything else | Play the backup video, then show `demo.scenario` output as the proof it runs live |
