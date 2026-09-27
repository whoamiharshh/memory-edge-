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
6. **Qdrant Server**: download `qdrant-x86_64-pc-windows-msvc.zip` from the v1.19.1 GitHub release and unpack
   `qdrant.exe` into `qdrant_server\`.
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
11. **Phone as a sensor**: start a device with `--host 0.0.0.0 --tls --profile lowrate-accel --component fan`, open
    `https://<laptop IP>:8101/sensor` on the phone. Other profiles: `rotating-hf` (with `--profile-params
    "{\"bearing\": \"6206\", \"shaft_hz\": 29}"`), `force-torque`, `events`.
12. **Smaller disk on Windows**: add `--compress-storage` to `edge.main` (NTFS-compresses the device folder at start and
    hourly; a shard went from 267 MB to 1.9 MB in our test).
13. **Laya experiment** (optional, isolated, ~1 GB): `uv venv .venv-laya --python 3.12`, then in it `torch` (CPU wheel
    index), `laya`, `fastembed==0.8.1`, `scikit-learn`; run with `$env:HF_HOME="models_cache\hf";
    .venv-laya\Scripts\python.exe -m bench.laya_experiment`.

## Demo walkthrough in the UI (about 5 min)

1. **Device A:** sign in with `operator-devA` and switch the network **OFFLINE**.
2. Replay file 99 (healthy): the chart stays inside the green healthy radius.
3. Replay file 105 (inner race). A red dot marks a **new episode**, and the physics hint says inner_race with its
   measured accuracy.
4. Search shows no fleet evidence, and no answer is invented.
5. In the episode:
   - write a note that contains a name from the denylist (e.g. "Ravi") and tick "share";
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
