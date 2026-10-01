# Project Context — Machine Memory at the Edge
> Comprehensive handoff/context document for future development sessions.
> Generated from the current repository state on 2026-09-30.
> This file describes the project; it intentionally excludes generated runtime data,
> downloaded datasets, model caches, virtual environments, and Qdrant binary storage.

## 1. Project identity
- **Name:** Machine Memory at the Edge
- **Purpose:** Offline-first fault memory for industrial machines and other constrained edge devices.
- **Technology partner / problem track:** Qdrant, Code Cubicle 6.0 / PS3.
- **Primary language:** Python 3.12.
- **Frontend:** Vanilla HTML/CSS/JavaScript; device UI is also a PWA and includes a phone sensor page.
- **Deployment target:** Windows 11 development laptop; designed for Linux, macOS, ARM64/Raspberry Pi, and phones as sensors.
- **License:** Apache-2.0 (`LICENSE`, `NOTICE`).
- **Current state:** Large implemented prototype with edge, cloud, UI, benchmarks, datasets, security controls, and tests. It is not a field deployment or production-certified system.

## 2. One-paragraph product description
Each machine has a local Qdrant Edge memory containing sensor episodes, technician notes, manuals, procedures, and past outcomes. The machine can search this memory while completely offline. A technician records what was observed and what action was taken. The device shares an incident with the fleet only after its own sensor data verifies that the symptom stayed resolved for the required stable period. The cloud aggregates evidence by confirmed fault group and action, keeps worked and failed outcomes visible, and never silently overwrites disagreement. Other devices pull a read-only fleet mirror and can use that evidence offline. The system presents evidence and citations; it does not autonomously recommend repairs or claim that sensor verification proves root cause.

## 3. Core differentiators and product rules
1. **Outcome-verified promotion:** intervention complete + stable sensor window + successful human confirmation are all required before fleet sharing.
2. **Follow-up verification:** weeks later, a repair can be marked held or recurred; failed fixes are useful evidence too.
3. **Disagreement remains visible:** flags include `DISPUTED`, `COMPETING`, `ALTERNATIVES`, and `RECURRED`; records are not reduced to a winner.
4. **Fleet mirror:** Device B can use Device A's confirmed evidence offline through a separate read-only local shard.
5. **Sensor profiles:** vibration, high-frequency rotating-machine signals, microphone audio, low-rate phone IMU, robot force/torque, event logs, and vehicle telemetry share the same application pipeline.
6. **Technician remains responsible for diagnosis:** sensor state is described as “symptom resolved,” not “root cause confirmed.”
7. **Privacy by default:** raw signals, raw notes, manuals, and text embeddings remain on-device. Shared notes are opt-in and pass redaction first.
8. **Evidence-first UI:** physics, measured model confidence, manuals with page numbers, cited procedures, similar cases, and search provenance are shown.
9. **No invented measurements:** README metrics must originate in `bench/` scripts or `tests/`, with method and caveats documented.
10. **No overclaiming novelty:** the defensible claim is the tested combination and conflict-aware multi-device sync, not that offline vector search or maintenance AI was invented here.

## 4. System architecture
```text
Sensor / audio / robot force / event code / vehicle telemetry
        |
        v
Signal profile -> 27-number fingerprint and profile-specific features
        |
        v
Novelty gate -> merge similar state or create a new episode
        |
        v
Local Qdrant Edge shard
  - sensor/vibration vector
  - text embedding vector
  - BM25 sparse vector
  - payload and provenance
        |
        +--> offline hybrid retrieval -> cited evidence UI
        |
        +--> verifier: healthy baseline + machine-card limit + new-part rule
                         |
                         v
                 deterministic policy gates
                         |
                         v
             SQLite durable outbox / journal-first write
                         |
                         v
                 cloud sync API and acknowledgements
                         |
                         v
         Qdrant Server tenant collections and case tallies
                         |
              +----------+----------+
              |                     |
              v                     v
       conflict/flag recompute   fleet mirror snapshots/deltas
                                    |
                                    v
                     read-only local mirror on Device B
```

### Important architecture boundaries
- `edge/store_edge.py` is the only module that should directly import/use the Qdrant Edge API. Qdrant Edge is beta and its API can change.
- Qdrant Edge is storage/retrieval only. It is not the anomaly detector, policy engine, diagnosis engine, or decision-maker.
- Qdrant Server is used for fleet-level ingestion, collections, tallies, snapshots, and cloud-side knowledge.
- The device's mutable local shard and read-only fleet mirror are separate stores.
- Shared event IDs are deterministic and replay-safe; server ingestion is insert-only where appropriate.
- Device journal/outbox is written before the Edge write; Edge is flushed before a write is reported durable.
- A bad promoted record becomes a retraction tombstone, not a hard delete.
- Conflicting edits use version checks / compare-and-set and return conflict information rather than silently applying last-write-wins.

## 5. Main runtime components
### `app/`

- `app/unified.py`: gateway/UI entry point. Proxies device and cloud APIs and manages local demo routing/tokens.
- `app/__main__.py`: module entry point.

### `edge/`

The per-device application.

- `api.py`: FastAPI device routes, authentication, signal/audio ingestion, memory/episode operations, manuals, procedures, sensor routes, and UI serving.
- `main.py`: device server launcher and CLI configuration.
- `device.py`: device lifecycle and episode orchestration.
- `store_edge.py`: Qdrant Edge adapter; named vectors, hybrid query, filters, CAS, snapshots, flush, open/close.
- `fingerprint.py`: DSP feature extraction and 27-dimensional signal fingerprints.
- `profiles.py`: profile registry and profile-specific signal conversion.
- `physics.py`: bearing defect frequencies, order features, envelope/kurtogram features, severity and relative-order logic.
- `gate.py`: novelty/anomaly gate and merge logic.
- `verifier.py`: checks whether a repair's symptom remains resolved; manufacturer limits and replacement-aware rules.
- `policy.py`: ordered privacy/validation/duplicate/evidence/verification/human-confirmation/share gates with reasons.
- `outbox.py`: durable SQLite journal/outbox and replay support.
- `sync_worker.py`: retries, batching, per-event acknowledgements, backoff, and cloud synchronization.
- `mirror.py`: fleet-mirror pull, full snapshot/scroll modes, safe swap and recovery.
- `fleet_hint.py`, `fault_hint.py`: device-side use and interpretation of fleet fault-type hints.
- `local_detector.py`: learned detector for robot-specific confirmed failures.
- `machine_card.py`: manufacturer, bearing geometry, RPM, limits, and machine metadata.
- `manuals.py`: PDF ingestion and offline page-cited manual retrieval.
- `procedures.py`: cited maintenance procedure library.
- `rag.py`: optional Qwen evidence brief; outside the decision path and checked for citations/no advice.
- `replay.py`: replay of recorded sensor data.
- `seed.py`: initial reference data seeding.
- `vehicle_risk.py`: vehicle risk hint support.
- `crypto.py`, `storage_os.py`: note encryption/storage handling and OS storage options.

### `cloud/`

The fleet/server application.

- `api.py`, `main.py`: FastAPI fleet routes and server launcher.
- `auth.py`: token issuance, roles, expiry, tenant/device/site binding, revocation.
- `tls.py`: TLS/mTLS and certificate revocation handling.
- `ingest.py`: authenticated, validated, idempotent event ingestion.
- `store_server.py`: Qdrant Server collections and server persistence.
- `knowledge.py`: fleet knowledge and case grouping.
- `tally.py`: evidence counts and conflict flags.
- `recompute.py`: coalesced, CAS-protected recomputation of derived case data.
- `hint_model.py`: fleet fault-hint model training/export.
- `audit.py`: hash-chained audit events.
- `ui/`: fleet administrator web UI.

### `shared/`

- `schema.py`: shared Pydantic/domain schemas and enums.
- `ids.py`: deterministic IDs and event identity helpers.
- `embed.py`: embedding abstraction.
- `redact.py`: PII/name redaction and sharing policy.
- `name_model.py`: name-likeness model support.
- `speech.py`: local Vosk speech recognition.
- `image_embed.py`: optional CLIP-style image retrieval.
- `integrity.py`: code-integrity evidence reported to the cloud.

## 6. Data and models
### Shipped/reference knowledge
- `knowledge/procedures.json`: cited maintenance procedures.
- `knowledge/vehicle_codes.json`: OBDex vehicle fault-code dictionary.
- `knowledge/vehicle_risk_model.json`: exported SCANIA risk model.
- `knowledge/domain_vocab.json`: maintenance vocabulary.
- `knowledge/given_names.json`, `knowledge/name_model.json`: real-name resources and model.

### Downloaded data (not redistributed)

Raw datasets belong under `data/raw/` and are ignored by Git.

- CWRU bearing data: primary 12 kHz vibration benchmark.
- HUST bearing data: held-out bearing geometry/type benchmark.
- University of Ottawa: natural-wear bearing, motor, and microphone tests.
- MaFaulDa: imbalance/misalignment real-machine evaluation.
- UCI Robot Execution Failures: robot detector.
- SCANIA Component X: vehicle risk hint.
- Loghub HDFS: event-log profile.
- Annotated Maintenance Logbook: text retrieval and name/redaction evaluation.

### Models
- `bge-small-en-v1.5` through FastEmbed: local note embedding.
- Qdrant built-in BM25 sparse representation for text/code retrieval.
- Optional `Qwen2.5-1.5B-Instruct` GGUF through llama.cpp: evidence wording only.
- Logistic regression models for fleet fault hints, robot detector, and vehicle risk hint.
- Vosk: on-device speech recognition.
- No model is allowed to make the repair decision.

## 7. Signal profiles
All profiles feed the shared gate/verifier/policy/sync/cloud path, but their input and feature extraction differ.

| Profile | Input | Intended scope |
|---|---|---|
| `bearing-12k` | 12 kHz accelerometer | CWRU/industrial bearing faults |
| `rotating-hf` | >=2 kHz accelerometer | rotating machinery and varied bearing geometry |
| `acoustic` | phone or machine microphone | natural-wear acoustic fault detection |
| `lowrate-accel` | phone IMU / telematics around 60 Hz | shaft-rate faults within Nyquist limits |
| `force-torque` | robot wrist force/torque | robot execution failures |
| `events` | application/kiosk/vehicle fault codes | discrete event/log anomalies |
| `telemetry` | vehicle/machine counters | risk hint, not diagnosis |

## 8. Security and privacy model
Implemented controls include:

- Per-device bearer tokens stored as hashes, roles, tenant isolation, expiry and revocation.
- Optional HTTPS and mutual TLS; certificate revocation list support.
- Two-admin requirement for retraction.
- Hash-chained audit log.
- Quarantine of suspect devices and their evidence.
- Pydantic validation, finite numeric checks, fixed vector sizes, enum/length limits, body/batch limits, rate limiting.
- Deterministic event IDs and insert-only replay protection.
- Raw sensor signals and manuals remain local.
- Text embeddings are never transmitted; the cloud re-embeds approved shared text.
- Notes are encrypted at rest on the device; other structured fields depend on OS disk encryption.
- Redaction of email, phone, IDs, URLs, denylist terms, names, and name-like terms.
- UI uses `textContent`, strict CSP/security headers, and no `innerHTML`.
- Localhost HTTP is acceptable for the demo; network-address operation should use HTTPS/mTLS and strong random tokens.

Residual risks explicitly acknowledged:

- No hardware-rooted identity / TPM / secure boot attestation.
- Searchable structured fields and vectors are not individually encrypted.
- A fully compromised server can poison fleet mirrors.
- An authenticated insider can submit plausible false evidence.
- Supply-chain vulnerabilities are possible; dependencies are pinned but not hash-locked.
- This is prototype-grade security, not a certification.

## 9. Important measured results
These figures are repository claims only when backed by the named benchmark/test and should be re-run before presenting them as current.

- CWRU repair verification: 36/36 fixed resolved, 36/36 still-faulty caught, 0 false promotions.
- HUST held-out: 42/42 fixes verified; physics hint about 97.6%, fleet hint about 95.2%.
- University of Ottawa acoustic test: 0/380 false alarms, 95.7% faulty windows flagged; new-bearing verification 19/20.
- Hybrid maintenance-text retrieval: P@3 0.885, MRR 0.93, capped recall 0.88.
- Honest cross-bearing vibration retrieval is weak (~0.429 P@3); same-machine/novelty behavior is strong. Therefore fleet grouping uses confirmed fault class/component, not raw vibration similarity.
- Robot detector: approximately 96–99% detection with 0–10% false alarms on the evaluated UCI data.
- SCANIA risk hint: ROC-AUC approximately 0.75 on held-out trucks.
- Large local scale runs reported 0 lost and 0 duplicate events under tested partitions/network faults.
- Offline check reported 15/15 paths with zero connection attempts.
- README and `docs/BENCHMARKS.md` contain the authoritative method/caveat tables.

## 10. Test suite
Run from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest
```

Test categories:

- `tests/unit/`: DSP, physics, schemas, store, profiles, redaction, names, manuals, policy, procedures, models, and utilities.
- `tests/integration/`: device lifecycle, cloud/fleet flow, mirror snapshots, profiles, robot/vehicle/fleet hints, follow-ups, and recomputation.
- `tests/failure/`: hard-kill durability, offline behavior, network partitions, backup/restore.
- `tests/security/`: tokens, tenant isolation, TLS/mTLS, revocation, audit, quarantine, validation, notes at rest, UI headers, secrets.
- `tests/ai/`: citation and no-advice checks for the optional RAG brief.
- `tests/ui_check.py`: headless UI smoke check using Playwright/system Microsoft Edge.

The repository documentation reports 302 passing tests on the build laptop, but any future work must run the current suite rather than relying on that historical number.

## 11. Setup and run commands
### Requirements
- Windows 11 / PowerShell 5.1+ for the documented local workflow.
- Python 3.12.
- `uv` recommended.
- No Docker, GPU, WSL, or external cloud required for the local demo.
- Qdrant Server 1.19.1 binary is downloaded into ignored `qdrant_server/`.

### Initial setup
```powershell
uv venv .venv --python 3.12
$env:VIRTUAL_ENV=".venv"
uv pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --index-strategy unsafe-best-match
.venv\Scripts\python.exe -m tools.qdrant_local download
.venv\Scripts\python.exe data\fetch_data.py
```

Or use the portable installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Useful setup flags: `-SkipQdrant`, `-SkipModels`, and `-Dev`.

### Run demo
```powershell
powershell -ExecutionPolicy Bypass -File .\demo\run_demo.ps1 -Reset
.venv\Scripts\python.exe -m demo.scenario
```

Normal launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

- **The app: `http://127.0.0.1:9000/`** - one page, no sign-in. `start.ps1` opens this. The unified
  gateway (`app/unified.py`) proxies to the device and the cloud and injects their tokens itself.
- Device UI (engineering detail): `http://127.0.0.1:8101/`
- Fleet UI: `http://127.0.0.1:8100/`
- Qdrant Server: normally `6333`
- Additional demo devices: `8102`, `8103`
- Stop demo: `demo\stop_demo.ps1` or `stop.ps1` as appropriate.
- Logs: ignored `runtime\logs\`.

### Optional checks
```powershell
.venv\Scripts\python.exe -m demo.scenario
.venv\Scripts\python.exe tests\ui_check.py
.venv\Scripts\python.exe -m bench.retrieval_vib
.venv\Scripts\python.exe -m bench.retrieval_text
```

### Phone / external sensor
Use HTTPS for a second computer or phone. Generate certificates with `tools\make_certs.py`; start a device bound to a network interface with a profile such as `acoustic` or `lowrate-accel`; open `/sensor` on the phone. `tools\sensor_bridge.py` accepts serial accelerometer input. See `docs/FIELD_TEST.md` and `docs/SETUP.md` for the full procedure.

## 12. Repository map
```text
app/        unified gateway and application entry points
bench/      reproducible benchmark scripts and result files
cloud/      fleet API, storage, auth, models, audit, and fleet UI
data/       download/build scripts and split logic
demo/       three-device scenario, launcher, recording, stop script
docs/       architecture, research, setup, threats, benchmarks, decisions, demo, field test
edge/       device runtime, Qdrant Edge adapter, profiles, verification, UI
knowledge/  shipped derived reference/procedure/model data
shared/     schemas, IDs, embeddings, redaction, speech, image support
spike/      exploratory API and algorithm experiments retained as evidence
storage/    local generated storage; ignored runtime state
snapshots/  generated snapshot artifacts; do not treat as source
tests/      unit, integration, failure, security, AI, and UI tests
tools/      Qdrant downloader, backup, TLS certificates, network proxy, sensor bridge
```

Root files:

- `README.md`: current user-facing overview, metrics, run instructions, honest limits, credits.
- `CLAUDE.md`: detailed developer handoff, decisions, build order, constraints, and historical implementation state.
- `requirements.txt`: pinned Python dependencies.
- `pytest.ini`: pytest configuration.
- `setup.ps1`, `setup.sh`: environment setup.
- `start.ps1`, `run.bat`, `stop.ps1`, `stop.bat`: launch/stop helpers.
- `.env.example`: environment variable template.
- `.gitignore`: excludes virtual environments, runtime data, models, raw datasets, Qdrant storage, and local secrets.
- `docs/CONTEXT.md`: chronological project history and discarded ideas.
- `docs/RESEARCH.md`: original research and architecture history; consult current README/docs when it conflicts with older history.

## 13. Benchmark and utility map
- Retrieval: `bench/retrieval_vib.py`, `bench/retrieval_text.py`, `bench/latency.py`, `bench/embed_models.py`.
- Gate/verifier/physics: `bench/gate_verifier.py`, `bench/gate_sweep.py`, `bench/hust_holdout.py`, `bench/acoustic_uottawa.py`, `bench/operating_point.py`, `bench/motor_rules.py`, `bench/mafaulda_rules.py`.
- Learned hints: `bench/fault_hint.py`, `bench/robot_model.py`, `bench/robot_failures.py`, `bench/vehicle_scania.py`, `bench/vehicle_scania_train.py`.
- Profiles: `bench/events_hdfs.py`, `bench/acoustic_noise.py`, `bench/scale_fleet.py`.
- Sync/scale: `bench/network_faults.py`, `bench/sync_partition.py`, `bench/mirror_sync.py`, `bench/fleet_scale.py`, `bench/resources.py`, `bench/storage.py`, `bench/footprint.py`.
- Privacy/security: `bench/redaction.py`, `bench/offline_check.py`.
- Data retrieval/building: scripts in `data/`.
- Operations: `tools/backup.py`, `tools/make_certs.py`, `tools/qdrant_local.py`, `tools/sensor_bridge.py`, `tools/netem_proxy.py`.

## 14. Current known limitations / unfinished items
- No real industrial field deployment yet; public datasets and one-laptop fleet simulations are not field validation.
- Phone and two-computer field test plan exists but is not necessarily executed.
- Misalignment is detected more reliably than it is named; do not present naming as solved.
- Microphone results degrade with neighboring machines and have no calibrated severity.
- Vehicle result is a modest risk hint, not an explanatory diagnosis.
- Unknown names and some writing styles can evade redaction; safe default is to keep uncertain notes local.
- Optional LLM is small and must remain outside decisions.
- Video and OCR are not wired into the UI; photo search retrieves images but does not describe them.
- Device kind is fixed at launch; UI lists profiles but does not dynamically switch a running device.
- The application is prototype-grade and has no hardware attestation.
- Android native app is intentionally not built; the phone uses the HTTPS PWA/sensor page.
- `git` was not available in the inspected PowerShell session; do not infer repository status from that failed command.
- Public GitHub publishing and LinkedIn posting require explicit user approval and must never be performed automatically.

## 15. Development rules
1. Read `CLAUDE.md` and relevant current docs before changing architecture.
2. Keep all direct Qdrant Edge calls behind `edge/store_edge.py`.
3. Pin `qdrant-edge-py==0.8.0` unless deliberately revalidating the entire adapter.
4. Add or update tests for every behavior change.
5. Run pytest before claiming a fix works.
6. Do not fabricate benchmark results, sources, or field validation.
7. Keep synthetic signals inside tests only; trained/evaluated claims use real data.
8. Preserve privacy defaults and never ship raw signals, raw notes, or text embeddings.
9. Do not turn the optional LLM into a decision-maker or advice generator.
10. Preserve visible conflicts, evidence provenance, tombstones, version checks, and auditability.
11. Do not commit runtime state, downloaded data, models, secrets, certificates, or local machine configuration.
12. Keep changes targeted and update README/docs when externally visible behavior changes.

## 16. Recommended reading order for a new coding session
1. `PROJECT_CONTEXT.md` — this compact complete handoff.
2. `CLAUDE.md` — detailed implementation handoff and historical build checklist.
3. `README.md` — current user-facing behavior and measured results.
4. `docs/SETUP.md` — installation, operation, HTTPS, phone, backup, and field setup.
5. `docs/ARCHITECTURE.md` — architecture decisions and Qdrant Edge design.
6. `docs/THREATS.md` — security claims and the test proving each one.
7. `docs/BENCHMARKS.md` — exact benchmark methods and caveats.
8. The specific module and its corresponding tests before editing code.

## 17. Truth status convention
When documenting or discussing the project, distinguish:

- **Verified:** directly supported by a current test, benchmark, or source file.
- **Historical:** reported in project history or an earlier measured run; rerun before repeating as current.
- **Proposed design:** intended behavior not yet verified.
- **Known limitation:** explicitly measured or documented weakness.
- **Not verified:** no evidence in the repository; do not present it as fact.
