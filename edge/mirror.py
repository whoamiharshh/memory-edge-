"""Fleet mirror: the device's read-only copy of fleet knowledge.

This is Qdrant's documented dual-shard pattern (docs: "Synchronize with a Server"): a mutable local shard for the
device's own memory plus an immutable mirror shard kept current from the server; queries read both and merge.
The pattern is Qdrant's, not ours. What is ours: the swap, the crash repair and the choice of fill mode.

Fill modes (a mirror is only ever filled by ONE mode; switching modes rebuilds it empty):
  auto      DEFAULT. Full Qdrant shard snapshot to bootstrap or rebuild; small deltas as scroll rows; per pull the
            cheaper one by estimated bytes (bench/mirror_sync.py measured why: at our fleet sizes a partial
            snapshot re-ships the whole mutable segment, ~190-400 kB, while one changed case is ~2.7 kB of rows).
            Partial snapshots are never applied on top of scroll writes (those change the local segment versions
            that the server's manifest comparison relies on), so a rebuild is always a full snapshot.
  snapshot  Qdrant's pattern verbatim: a full shard snapshot once (EdgeShard.unpack_snapshot), then partial
            snapshots (snapshot_manifest -> server -> update_from_snapshot).
  scroll    upsert case rows with seq > cursor only (kill-test K5 fallback; used against an in-process cloud).

Failure handling (docs/RESEARCH.md H.3):
  full refresh     restore into mirror.new/, probe it, THEN swap (mirror -> mirror.old, mirror.new -> mirror).
                   A failure before the swap leaves the old mirror in use; a crash mid-swap is repaired on boot.
  partial refresh  applied in place while reads wait on the lock (Qdrant's docs: pause updates during a restore).
                   An `applying` flag is stored first. If the apply fails, or the process dies with the flag set,
                   the next pull rebuilds the mirror from a full snapshot. A search that meets a broken mirror
                   returns local results and says the fleet mirror is unavailable (edge/device.py).
"""
from __future__ import annotations

import pathlib
import shutil
import threading
from typing import Any, Mapping, Sequence

from edge.outbox import Outbox
from edge.store_edge import EdgeStore, Hit, StorePoint

MODES = ("auto", "snapshot", "scroll")
SNAPSHOT_MODES = ("auto", "snapshot")


class Mirror:
    def __init__(self, device_root: pathlib.Path, outbox: Outbox, **settings):
        self.base = pathlib.Path(device_root)
        self.dir = self.base / "mirror"
        self.settings = settings
        self.kv = outbox
        self.log = outbox.log
        self.lock = threading.RLock()
        self.last_refresh: dict[str, Any] | None = None
        self._repair_interrupted_swap()
        self.store = EdgeStore(self.dir, **settings)

    # ---- state ------------------------------------------------------------------------------------------
    @property
    def mode(self) -> str | None:
        return self.kv.kv_get("mirror_mode", None)

    @property
    def seq(self) -> int:
        return int(self.kv.kv_get("mirror_seq", 0))

    def set_seq(self, seq: int) -> None:
        self.kv.kv_set("mirror_seq", int(seq))

    @property
    def needs_full(self) -> bool:
        """True until a full snapshot has been restored, and again after a failed/interrupted partial apply."""
        return self.mode in SNAPSHOT_MODES and bool(self.kv.kv_get("mirror_needs_full", True)
                                                    or self.kv.kv_get("mirror_applying", False))

    def request_full(self) -> None:
        self.kv.kv_set("mirror_needs_full", True)

    def _repair_interrupted_swap(self) -> None:
        new, old = self.base / "mirror.new", self.base / "mirror.old"
        if new.exists():
            shutil.rmtree(new, ignore_errors=True)
        if old.exists():
            if self.dir.exists():
                shutil.rmtree(old, ignore_errors=True)
            else:                                        # died between the two renames: put the old one back
                old.rename(self.dir)
                self.log("mirror", "repaired an interrupted mirror swap (previous mirror restored)")

    # ---- mode -------------------------------------------------------------------------------------------
    def ensure_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        if self.mode != mode:
            self.reset(mode)

    def reset(self, mode: str) -> None:
        with self.lock:
            self.store.close()
            shutil.rmtree(self.dir, ignore_errors=True)
            self.store = EdgeStore(self.dir, **self.settings)
            self.kv.kv_set("mirror_mode", mode)
            self.kv.kv_set("mirror_seq", 0)
            self.kv.kv_set("mirror_applying", False)
            self.kv.kv_set("mirror_needs_full", mode in SNAPSHOT_MODES)
        self.log("mirror", f"fleet mirror reset (fill mode: {mode})")

    # ---- snapshot fill ----------------------------------------------------------------------------------
    def manifest(self) -> dict:
        with self.lock:
            return self.store.snapshot_manifest()

    def replace_from_snapshot(self, snapshot_path: str | pathlib.Path, seq: int) -> None:
        """Restore a full snapshot next to the live mirror, probe it, then swap it in."""
        new, old = self.base / "mirror.new", self.base / "mirror.old"
        shutil.rmtree(new, ignore_errors=True)
        try:
            EdgeStore.from_snapshot(snapshot_path, new, **self.settings).close()
        except Exception:
            shutil.rmtree(new, ignore_errors=True)
            raise                                        # the old mirror was never touched
        with self.lock:
            self.store.close()
            try:
                self.dir.rename(old)
                new.rename(self.dir)
            except OSError:
                if not self.dir.exists() and old.exists():
                    old.rename(self.dir)
                self.store = EdgeStore(self.dir, **self.settings)
                raise
            self.store = EdgeStore(self.dir, **self.settings)
            shutil.rmtree(old, ignore_errors=True)
            self.kv.kv_set("mirror_needs_full", False)
            self.kv.kv_set("mirror_applying", False)
            self.set_seq(seq)

    def apply_partial(self, snapshot_path: str | pathlib.Path, seq: int) -> None:
        with self.lock:
            self.kv.kv_set("mirror_applying", True)
            try:
                self.store.apply_snapshot(snapshot_path)
            except Exception:
                self.kv.kv_set("mirror_needs_full", True)
                raise
            self.kv.kv_set("mirror_applying", False)
            self.set_seq(seq)

    # ---- scroll fill ------------------------------------------------------------------------------------
    def upsert_rows(self, points: Sequence[StorePoint], seq: int) -> None:
        with self.lock:
            self.store.upsert(points)
            self.set_seq(seq)            # cursor advances only after the durable write

    # ---- reads ------------------------------------------------------------------------------------------
    def count(self, filter: Mapping[str, Any] | None = None) -> int:
        with self.lock:
            return self.store.count(filter)

    def changed_since(self, seq: int) -> int:
        return self.count({"seq": {"gt": seq}})

    def search(self, **kw) -> list[Hit]:
        with self.lock:
            return self.store.search(**kw)

    def scroll(self, **kw) -> list:
        with self.lock:
            return list(self.store.scroll(**kw))

    def info(self) -> dict:
        with self.lock:
            size = sum(f.stat().st_size for f in self.dir.rglob("*") if f.is_file())
            return {"mode": self.mode, "seq": self.seq, "cases": self.store.count(), "needs_full": self.needs_full,
                    "disk_bytes": size, "last_refresh": self.last_refresh}

    def close(self) -> None:
        with self.lock:
            self.store.close()
