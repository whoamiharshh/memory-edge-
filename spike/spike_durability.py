"""K4 deep-dive: what makes Qdrant Edge writes survive a hard kill (TerminateProcess)?
Variants: no flush / flush after every write / flush every 50 writes. Recovered count vs acknowledged count.
"""
import os, subprocess, sys, tempfile
from qdrant_edge import EdgeShard, CountRequest

CHILD = r'''
import sys, uuid, random, time
from qdrant_edge import *
path, mode = sys.argv[1], sys.argv[2]
s = EdgeShard.create(path, EdgeConfig(vectors={"vib": EdgeVectorParams(size=8, distance=Distance.Euclid)}))
for i in range(200):
    s.update(UpdateOperation.upsert_points([Point(str(uuid.uuid5(uuid.NAMESPACE_URL, str(i))), {"vib": [random.random() for _ in range(8)]}, {"i": i})]))
    if mode == "each" or (mode == "every50" and (i + 1) % 50 == 0):
        s.flush()
    if mode == "every50" and (i + 1) % 50 != 0:
        continue
    print("ACK", i + 1, flush=True)
time.sleep(60)
'''

for mode in ["none", "each", "every50"]:
    path = os.path.join(tempfile.mkdtemp(), "s"); os.makedirs(path)
    p = subprocess.Popen([sys.executable, "-c", CHILD, path, mode], stdout=subprocess.PIPE, text=True)
    acked = 0
    for line in p.stdout:
        acked = int(line.split()[1])
        if acked == 200: break
    p.kill(); p.wait()
    s = EdgeShard.load(path); n = s.count(CountRequest()); s.close()
    print(f"mode={mode:8s} acked_by_app={acked:3d} recovered_after_kill={n}")
