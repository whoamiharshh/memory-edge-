"""Device-local SQLite (WAL, synchronous=FULL) holding:

  journal   outbox-first write log for the Edge shard. A point is journaled BEFORE the shard write and marked
            applied after the (flushed) shard write. On boot, unapplied rows are re-applied; upserts with
            deterministic ids are idempotent, so a crash anywhere in between loses nothing.
  outbox    share events waiting for / done with the cloud: queued -> uploading -> synced | rejected, and
            failed (transport error, retried with capped exponential backoff + full jitter).
  kv        small settings and cursors (e.g. the fleet-mirror sequence number).
  activity  human-readable audit trail for the UI.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sqlite3
import threading
import time
from typing import Any, Iterable

BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS journal(op_id TEXT PRIMARY KEY, body TEXT NOT NULL, applied INTEGER NOT NULL DEFAULT 0,
                                   created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS outbox(event_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, body TEXT NOT NULL,
                                  status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
                                  next_attempt_at REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                                  updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS outbox_status ON outbox(status, next_attempt_at);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS activity(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,
                                    episode_id TEXT, message TEXT NOT NULL);
"""


def backoff_delay(attempts: int, rng: random.Random | None = None) -> float:
    """Full jitter: uniform(0, min(cap, base * 2^attempts))."""
    return (rng or random).uniform(0, min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** attempts))


class Outbox:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _exec(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        """Statements whose result is not read. Reads use _all/_one, which FETCH under the lock too: one connection
        is shared by the API threads and the sync worker, and a fetch outside the lock can interleave with another
        thread's statement (seen live: kv_get read a NULL value while the stats endpoint and the worker raced)."""
        with self._lock:
            return self._db.execute(sql, tuple(args))

    def _all(self, sql: str, args: Iterable[Any] = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    def _one(self, sql: str, args: Iterable[Any] = ()) -> tuple | None:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchone()

    # ---- journal ----------------------------------------------------------------------------------------
    def journal_put(self, op_id: str, body: dict) -> None:
        self._exec("INSERT OR REPLACE INTO journal(op_id, body, applied, created_at) VALUES (?,?,0,?)",
                   (op_id, json.dumps(body), time.time()))

    def journal_applied(self, op_id: str) -> None:
        self._exec("UPDATE journal SET applied=1 WHERE op_id=?", (op_id,))

    def journal_pending(self) -> list[tuple[str, dict]]:
        rows = self._all("SELECT op_id, body FROM journal WHERE applied=0 ORDER BY created_at")
        return [(r[0], json.loads(r[1])) for r in rows]

    def journal_prune(self, keep: int = 5000) -> None:
        self._exec("DELETE FROM journal WHERE applied=1 AND op_id NOT IN "
                   "(SELECT op_id FROM journal ORDER BY created_at DESC LIMIT ?)", (keep,))

    # ---- outbox -----------------------------------------------------------------------------------------
    def enqueue(self, event: dict) -> bool:
        """True if newly queued; False if this event id is already known (idempotent)."""
        now = time.time()
        cur = self._exec("INSERT OR IGNORE INTO outbox(event_id, episode_id, body, status, created_at, updated_at) "
                         "VALUES (?,?,?,'queued',?,?)", (event["event_id"], event["episode_id"], json.dumps(event), now, now))
        return cur.rowcount == 1

    def known_event_ids(self) -> set[str]:
        return {r[0] for r in self._all("SELECT event_id FROM outbox WHERE status != 'rejected'")}

    def claim_due(self, limit: int = 50, now: float | None = None) -> list[dict]:
        """Atomically move due queued/failed rows to 'uploading' and return their bodies."""
        now = time.time() if now is None else now
        with self._lock:
            rows = self._db.execute("SELECT event_id, body FROM outbox WHERE status IN ('queued','failed') "
                                    "AND next_attempt_at <= ? ORDER BY created_at LIMIT ?", (now, limit)).fetchall()
            self._db.executemany("UPDATE outbox SET status='uploading', updated_at=? WHERE event_id=?",
                                 [(now, r[0]) for r in rows])
        return [json.loads(r[1]) for r in rows]

    def recover_uploading(self) -> int:
        """On boot: rows stuck in 'uploading' (crash mid-send) go back to 'queued'; the resend is idempotent."""
        with self._lock:
            return self._db.execute("UPDATE outbox SET status='queued' WHERE status='uploading'").rowcount

    def mark(self, event_id: str, status: str, error: str | None = None) -> None:
        self._exec("UPDATE outbox SET status=?, last_error=?, updated_at=? WHERE event_id=?",
                   (status, error, time.time(), event_id))

    def mark_failed(self, event_ids: Iterable[str], error: str, rng: random.Random | None = None) -> None:
        now = time.time()
        with self._lock:
            for eid in event_ids:
                (att,) = self._db.execute("SELECT attempts FROM outbox WHERE event_id=?", (eid,)).fetchone()
                self._db.execute("UPDATE outbox SET status='failed', attempts=?, last_error=?, next_attempt_at=?, "
                                 "updated_at=? WHERE event_id=?", (att + 1, error, now + backoff_delay(att, rng), now, eid))

    def retry_now(self) -> None:
        self._exec("UPDATE outbox SET next_attempt_at=0 WHERE status='failed'")

    def requeue(self, event_id: str) -> None:
        """Demo/admin: send an already-synced event again (the cloud must answer 'duplicate')."""
        self._exec("UPDATE outbox SET status='queued', next_attempt_at=0 WHERE event_id=?", (event_id,))

    def counts(self) -> dict[str, int]:
        return {s: n for s, n in self._all("SELECT status, COUNT(*) FROM outbox GROUP BY status")}

    def rows(self, limit: int = 100) -> list[dict]:
        cols = ["event_id", "episode_id", "status", "attempts", "last_error", "next_attempt_at", "created_at", "updated_at"]
        rs = self._all(f"SELECT {','.join(cols)} FROM outbox ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(zip(cols, r)) for r in rs]

    def status_of(self, event_id: str) -> str | None:
        r = self._one("SELECT status FROM outbox WHERE event_id=?", (event_id,))
        return r[0] if r else None

    # ---- kv / activity ----------------------------------------------------------------------------------
    def kv_get(self, key: str, default: Any = None) -> Any:
        r = self._one("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(r[0]) if r else default

    def kv_set(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO kv(key, value) VALUES (?,?)", (key, json.dumps(value)))

    def log(self, kind: str, message: str, episode_id: str | None = None) -> None:
        self._exec("INSERT INTO activity(ts, kind, episode_id, message) VALUES (?,?,?,?)",
                   (dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), kind, episode_id, message))

    def activity(self, limit: int = 100) -> list[dict]:
        rs = self._all("SELECT ts, kind, episode_id, message FROM activity ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(zip(["ts", "kind", "episode_id", "message"], r)) for r in rs]
