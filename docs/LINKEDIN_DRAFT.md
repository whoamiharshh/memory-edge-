# LinkedIn post draft (edit freely; every number is from bench/results/)

Built for Code Cubicle 6.0 (Geek Room) × Qdrant, PS3: **Machine Memory at the Edge**.

Industrial machines fail, get fixed, and the fix is often forgotten or repeated wrongly at the next site. I built
an offline fault memory on **Qdrant Edge**. Each machine's device remembers vibration fault episodes and what
fixed them, and it searches that memory with no network.

The core idea: a fix is shared with the fleet **only after the machine's own sensor data shows it held**.
Failed fixes are shared too. The cloud keeps disagreement as evidence ("worked at 2 sites, failed at 1")
instead of overwriting it.

What I measured, honestly:
🔹 Sensor verification on all 36 CWRU fault recordings: 36/36 fixes verified, 36/36 still-faulty cases caught, **0 false promotions**
🔹 Hybrid search (dense + BM25, fused in one Qdrant Edge query) beat either alone on real maintenance logs: P@3 0.881 vs 0.843 vs 0.794
🔹 A shared fix is 863 bytes; the raw signal it summarises (~500 kB) never leaves the machine
🔹 Crash + network-partition harness: 0 lost, 0 double-counted events
🔹 The honest one: vibration similarity reaches 99.7% on the usual (leaky) split but only 43% on bearings it has never seen. So fleet matching uses the technician-confirmed fault class, not vibration similarity.

A small local LLM (Qwen2.5-1.5B, offline) writes a cited summary of the evidence. Uncited or advice-giving
sentences are removed, and it never makes a decision.

Stack: Qdrant Edge + Qdrant Server, FastEmbed (bge-small), FastAPI, SQLite outbox, llama.cpp. It runs on a
laptop with no GPU and no Docker.

Repo: <GitHub link>
#Qdrant #EdgeAI #VectorSearch #PredictiveMaintenance #Hackathon #CodeCubicle
