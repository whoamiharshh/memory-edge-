# LinkedIn post draft (a DRAFT only - nothing is posted without your manual decision; every number is from bench/results/)

Built for Code Cubicle 6.0 (Geek Room) × Qdrant, PS3: **Machine Memory at the Edge**.

Machines fail, get fixed, and the fix is forgotten or repeated wrongly at the next site. I built an offline fault
memory on **Qdrant Edge**: each machine's device remembers fault episodes and what fixed them, searches that memory
with no network, and shares a fix with the fleet **only after its own sensor data shows the fix held** - then reports
weeks later whether it still held. The cloud keeps disagreement as evidence ("worked at 2 sites, failed at 1") instead
of overwriting it.

What I measured, on public real-world data:
🔹 Fix verification: 36/36 fixes verified and 36/36 still-faulty cases caught on CWRU, **0 false promotions**; 42/42 on a second lab (HUST) it was never tuned on
🔹 A **phone microphone** as the sensor: 0 false alarms in 380 healthy windows and 96 % of faulty windows flagged on bearings that wore out naturally (University of Ottawa data)
🔹 The fleet **learns fault types** from technician-confirmed cases: when physics and the fleet model agree the hint was right 39/39 (HUST) and 12/13 (natural wear) on bearings it never saw
🔹 Hybrid search (dense + BM25 fused in one Qdrant Edge query) beat either alone on real maintenance logs: P@3 0.885
🔹 20 devices syncing at once: 1,000 events, 0 lost, 0 double-counted
🔹 The honest one: vibration similarity is 99.7 % on the usual (leaky) split but 43 % on unseen bearings - so the fleet groups by the technician-confirmed fault class, not by similarity

Every model is trained on real data only; made-up data appears only in unit tests.

Stack: Qdrant Edge + Qdrant Server, FastEmbed (bge-small), FastAPI, SQLite outbox, llama.cpp (optional cited summary).
Runs on a laptop with no GPU and no Docker.

Repo: <GitHub link>
#Qdrant #EdgeAI #VectorSearch #PredictiveMaintenance #Hackathon #CodeCubicle
