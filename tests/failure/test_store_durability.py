"""K4 rule applied to OUR adapter: every write EdgeStore acknowledges (method returned) must survive a hard
kill of the process (TerminateProcess on Windows / SIGKILL elsewhere) with no close()."""
import os
import pathlib
import signal
import subprocess
import sys
import textwrap

from edge.store_edge import EdgeStore

N = 60
REPO = pathlib.Path(__file__).resolve().parents[2]

CHILD = textwrap.dedent(f"""
    import os, sys, time, uuid
    from edge.store_edge import EdgeStore, StorePoint
    print("PID", os.getpid(), flush=True)
    s = EdgeStore(sys.argv[1], vib_dim=4, note_dim=3)
    for i in range({N}):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(i)))
        if i % 2:
            s.upsert([StorePoint(pid, {{"i": i}}, vib=[i, 0, 0, 0], bm25_text="note " + str(i))])
        else:
            s.upsert([StorePoint(pid, {{"i": i, "occurrences": 0}}, vib=[i, 0, 0, 0])])
            s.modify(pid, lambda p: {{"occurrences": 1}})
        print("ACK", i + 1, flush=True)
    time.sleep(120)   # parent hard-kills us here: no close()
""")


def test_acknowledged_writes_survive_hard_kill(tmp_path):
    root = tmp_path / "dev"
    p = subprocess.Popen([sys.executable, "-c", CHILD, str(root)], stdout=subprocess.PIPE, text=True, cwd=REPO)
    acked, real_pid = 0, None
    try:
        for line in p.stdout:
            if line.startswith("PID"):
                real_pid = int(line.split()[1])
            elif line.startswith("ACK"):
                acked = int(line.split()[1])
                if acked == N:
                    break
    finally:
        # A venv's python.exe on Windows is a launcher that runs the real interpreter as a child, so p.kill()
        # alone would kill the launcher and leave the real process to die asynchronously (still holding the
        # shard's WAL lock). Hard-kill the real interpreter (TerminateProcess / SIGKILL); the launcher exits once
        # its child has fully terminated, so p.wait() returning means the lock is released.
        if real_pid is not None and real_pid != p.pid:
            os.kill(real_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        else:
            p.kill()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    assert acked == N
    with EdgeStore(root, vib_dim=4, note_dim=3) as s:
        assert s.count() == N
        evens = [r for r in s.scroll() if r.payload["i"] % 2 == 0]
        assert len(evens) == N // 2 and all(r.payload["occurrences"] == 1 for r in evens)   # modify() durable too
