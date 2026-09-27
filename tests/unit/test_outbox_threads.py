"""Regression: the device SQLite connection is shared by the API threads and the sync worker. Reads must fetch under
the lock; before the fix a live demo run saw kv_get() read NULL while /api/stats and the worker raced."""
import threading

from edge.outbox import Outbox


def test_concurrent_reads_and_writes_never_mix_results(tmp_path):
    ob = Outbox(str(tmp_path / "d.sqlite"))
    for i in range(8):
        ob.kv_set(f"k{i}", i)
    errors = []

    def reader(i):
        try:
            for _ in range(400):
                assert ob.kv_get(f"k{i}") == i
                ob.counts()
                ob.activity(20)
        except Exception as e:           # noqa: BLE001 - collect everything from the threads
            errors.append(repr(e))

    def writer():
        try:
            for n in range(400):
                ob.kv_set("online", n % 2 == 0)
                ob.log("t", f"message {n}")
        except Exception as e:           # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(8)] + [threading.Thread(target=writer)
                                                                                for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ob.close()
    assert errors == []
