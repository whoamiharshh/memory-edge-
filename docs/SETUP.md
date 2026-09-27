# Setup (Windows 11, no Docker, no GPU)

Everything below was run on the build laptop on 27 Sep 2026.

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
