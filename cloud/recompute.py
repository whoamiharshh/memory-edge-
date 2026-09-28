"""Coalesced case recomputation.

A push only has to make the events DURABLE before it acknowledges them; recomputing the case tallies can wait a
moment. With 20 devices pushing at once, recomputing every touched case inside every push made pushes take ~7 s (the
same cases recomputed again and again under compare-and-set contention; bench/scale_fleet.py). Now a push marks its
cases dirty; one background thread recomputes each dirty case once, and every READ first flushes the dirty cases of
its tenant, so nobody ever sees a stale tally. Tallies are still always rebuilt from ALL events (never incremented).
"""
from __future__ import annotations

import threading

from cloud.ingest import recompute_case


class Recomputer:
    def __init__(self, store, embedder, interval_s: float = 0.5, background: bool = True):
        self.store, self.embedder = store, embedder
        self._dirty: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._stop = threading.Event()
        self.recomputed = 0
        if background:
            threading.Thread(target=self._loop, args=(interval_s,), name="recompute", daemon=True).start()

    def mark(self, tenant: str, cid: str, component: str, fault_class: str) -> None:
        with self._lock:
            self._dirty[(tenant, cid)] = (component, fault_class)

    def flush(self, tenant: str | None = None) -> int:
        with self._run_lock:                     # one recompute pass at a time; later marks are picked up next time
            with self._lock:
                todo = {k: v for k, v in self._dirty.items() if tenant is None or k[0] == tenant}
                for k in todo:
                    del self._dirty[k]
            done = 0
            try:
                for (t, cid), (comp, fc) in todo.items():
                    recompute_case(self.store, self.embedder, t, cid, comp, fc)
                    done += 1
            finally:
                if done < len(todo):             # put back what was not recomputed (retried on the next pass)
                    with self._lock:
                        for k, v in list(todo.items())[done:]:
                            self._dirty.setdefault(k, v)
            self.recomputed += done
            return done

    def pending(self) -> int:
        with self._lock:
            return len(self._dirty)

    def _loop(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            try:
                self.flush()
            except Exception:                    # the unfinished marks were put back; the next pass retries them
                pass

    def stop(self) -> None:
        self._stop.set()
