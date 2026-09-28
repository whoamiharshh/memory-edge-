# Setup (Windows 11, no Docker, no GPU)

Everything below was run on the build laptop on 27-28 Sep 2026.

1. **Python env**
   ```powershell
   uv venv .venv --python 3.12
   $env:VIRTUAL_ENV=".venv"; uv pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --index-strategy unsafe-best-match
   ```
2. **Data** (not redistributed): `.venv\Scripts\python.exe data\fetch_data.py`
3. **Fingerprint cache + K2**: `.venv\Scripts\python.exe -m bench.retrieval_vib`
4. **Text model** (about 67 MB, into `models_cache\`):
   ```powershell
   .venv\Scripts\python.exe -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='models_cache')"
   ```
   At runtime the devices set `HF_HUB_OFFLINE=1`, so no network is touched.
5. **Local LLM** (optional, 1.1 GB; Apache-2.0 according to its model card):
   ```powershell
   .venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF','qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='models_cache/llm')"
   ```
   Without it, the evidence brief falls back to a deterministic template.
6. **Qdrant Server**: `.venv\Scripts\python.exe -m tools.qdrant_local download` fetches the v1.19.1 release binary
   for this OS (Windows zip, Linux x86-64 / ARM64, macOS Intel / Apple Silicon) into `qdrant_server\`.
7. **Run**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset
   ```
   Stop with `demo\stop_demo.ps1`. Logs are in `runtime\logs\`.
8. **Checks**:
   ```powershell
   .venv\Scripts\python.exe -m pytest
   .venv\Scripts\python.exe -m demo.scenario
   .venv\Scripts\python.exe tests\ui_check.py
   ```
9. **Extra data sets** (optional, for the held-out and robot benchmarks; both CC BY 4.0, not redistributed):
   ```powershell
   .venv\Scripts\python.exe data\fetch_hust.py        # 640 MB, 99 files, SHA-256 checked
   .venv\Scripts\python.exe data\fetch_uci_robot.py   # 58 kB
   .venv\Scripts\python.exe -m bench.hust_holdout
   .venv\Scripts\python.exe -m bench.robot_failures
   ```
10. **HTTPS** (needed for a second computer or a phone; see docs/FIELD_TEST.md):
    ```powershell
    .venv\Scripts\python.exe tools\make_certs.py      # runtime\tls\ca.pem, ca.key, server.pem, server.key
    ```
    Add `--tls` to `cloud.main` / `edge.main`; devices verify the cloud with `--ca ca.pem`. Both refuse plain HTTP on a
    network address. Never copy `ca.key` off the laptop.
11. **Phone as a sensor**: start a device with `--host 0.0.0.0 --tls --profile acoustic --component fan` (microphone,
    recommended) or `--profile lowrate-accel` (motion sensor, ~60 Hz), open `https://<laptop IP>:8101/sensor` on the
    phone and pick Microphone or Motion. Enter the machine's rpm from its nameplate if known. Other profiles:
    `rotating-hf` (with `--profile-params "{\"bearing\": \"6206\", \"shaft_hz\": 29}"`), `force-torque`, `events`.
    On a network address the operator token must be >= 16 random characters (omit `--operator-token` to get one).
12. **Smaller disk on Windows**: add `--compress-storage` to `edge.main` (NTFS-compresses the device folder at start and
    hourly; a shard went from 267 MB to 1.9 MB in our test).
13. **Laya experiment** (optional, isolated, ~1 GB): `uv venv .venv-laya --python 3.12`, then in it `torch` (CPU wheel
    index), `laya`, `fastembed==0.8.1`, `scikit-learn`; run with `$env:HF_HOME="models_cache\hf";
    .venv-laya\Scripts\python.exe -m bench.laya_experiment`.
14. **Vehicles (real data)**: `data\fetch_scania.py` (SCANIA validation + test splits, ~430 MB, CC BY 4.0), then
    `python -m bench.vehicle_scania` (trains the risk model on the validation trucks, tests on the test trucks, writes
    `knowledge/vehicle_risk_model.json`). `data\fetch_obdex.py` rebuilds `knowledge/vehicle_codes.json` (already
    shipped, CC0). Robots: `python -m bench.robot_model` (trained on real UCI traces, cross-validated).
15. **Mutual TLS**: `tools\make_certs.py device devA` → `runtime\tls\devices\devA.pem/.key`; start the cloud with
    `--mtls` and the device with `--client-cert ... --client-key ...`. Tokens expire after 30 days and renew by
    themselves; notes are encrypted at rest automatically (key: `runtime\<device>\note_key.dpapi`, useless on another
    Windows account or PC).
16. **Phone app**: open the device UI over HTTPS on the phone and use the browser's "Add to Home screen"; it behaves
    like an app. The phone sensor page is `/sensor`.

17. **Machine card and manuals** (device UI, bottom): enter the manufacturer, power, foundation, bearing (from the list or
    the catalogue geometry), nominal rpm, mains frequency and the manufacturer's vibration limits with their source
    (manual, page). Upload the machine's PDF manual: it is indexed on the device and searched offline next to each
    episode, with page numbers. Changing the bearing geometry asks you to re-capture the healthy baseline.
18. **Extra real data sets** (optional, for the benchmarks; all CC BY 4.0 or CC0, not redistributed):
    `python -m data.fetch_uottawa` (bearings, natural wear, microphone), `python -m data.fetch_uottawa_motor` (motors),
    `python -m data.fetch_loghub` (HDFS logs), `python -m data.fetch_names` + `python -m data.build_vocab` +
    `python -m data.build_name_model` (rebuild the shipped name lists in `knowledge/`), `python -m data.fetch_scania`
    (add `--no-train` to skip the 1.2 GB training split). Then `python -m bench.<name>`.
19. **External sensor** (instead of a phone): a microcontroller accelerometer printing `ax,ay,az` lines over USB:
    `python tools\sensor_bridge.py --device https://127.0.0.1:8101 --token <operator> --serial COM5 --fs 1600`
    (needs `uv pip install pyserial`; the bridge is unit-tested, not yet run on hardware).
20. **Backups**: stop the device, then `python -m tools.backup device-backup runtime\devA devA.zip`; restore with
    `device-restore devA.zip <new folder>` (checksums verified, the store is reopened and counted). Cloud:
    `python -m tools.backup cloud-backup http://127.0.0.1:6333 backups\cloud` / `cloud-restore`.
21. **Robots**: after a few failures are confirmed (or taught with "teach a failure"), `POST /api/detector/train`
    trains the robot's own detector; its cross-validated accuracy is shown in the device stats.

## Production checklist (beyond the localhost demo)
- HTTPS everywhere: `tools\make_certs.py`, cloud `--tls` or `--mtls`, devices `--ca` (+ `--client-cert/--client-key`).
- A lost device: revoke its token (fleet UI or `POST /v1/admin/devices/<id>/revoke`) AND its certificate
  (`tools\make_certs.py revoke <id>`, then restart the cloud); if its reports are suspect, **Quarantine** it.
- Strong random operator tokens (the launcher refuses weak ones on a network address); keep `ca.key` offline.
- Two admin tokens: retraction needs both. Keep a copy of the audit log's head hash (fleet UI, Audit log card).
- OS disk encryption (BitLocker / LUKS / FileVault) on devices: notes are encrypted by the app, other fields are not.
- Scheduled backups (step 20) stored off the machine; re-sign the CRL yearly (`tools\make_certs.py crl`).
- Linux / macOS / Raspberry Pi: the same commands with `.venv/bin/python`; `.github/workflows/tests.yml` runs the tests
  there once the repository is on GitHub.

## Demo walkthrough in the UI (about 5 min)

1. **Device A:** sign in with `operator-devA` and switch the network **OFFLINE**.
2. Replay file 99 (healthy): the chart stays inside the green healthy radius.
3. Replay file 105 (inner race). A red dot marks a **new episode**, and the physics hint says inner_race with its
   measured accuracy.
4. Search shows no fleet evidence, and no answer is invented.
5. In the episode:
   - write a note that contains a person's name (e.g. "Ravi" - found even without a denylist) and tick "share";
   - confirm the fault class;
   - record `replace_bearing`.
6. Replay file 100 (healthy) and watch verification count to 20/20. Press **Worked**. The decision becomes
   **SHARE**, with "note kept local: redactor found 1 denylist", and the outbox shows **queued**.
7. Switch the network **ONLINE**. The event goes from queued to **synced**. Press **Resend**: the cloud answers
   `duplicate` and the counts are unchanged.
8. **Fleet cloud:** sign in with the admin token. The case `bearing / inner_race` shows the tallies and the
   reports (with Retract).
9. **Device B:** sync once, then switch **OFFLINE**. Replay file 169 (a 14-mil bearing that A never saw) and use
   "Similar to selected episode": A's evidence appears, offline. Press **Generate** for the cited AI brief.
