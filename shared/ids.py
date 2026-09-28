"""Deterministic ids (UUIDv5). The same logical object always gets the same id on every device and on every
retry, which is what makes replays idempotent (insert_only on the same id is a no-op)."""
from __future__ import annotations

import hashlib
import json
import uuid

NAMESPACE = uuid.UUID("6f1c2f0e-5d0b-4f7e-9a51-3b8f3c1d9e27")   # fixed project namespace; never change it


def make_id(*parts: object) -> str:
    return str(uuid.uuid5(NAMESPACE, "|".join(str(p) for p in parts)))


def baseline_point_id(machine_id: str, i: int) -> str:
    return make_id("baseline", machine_id, i)


def episode_id(device_id: str, seq: int) -> str:
    return make_id("episode", device_id, seq)


def event_id(device_id: str, episode: str, outcome: str, action_code: str) -> str:
    """One shareable outcome per (episode, action, outcome). Re-deciding SHARE for the same facts yields the
    same event id, so the cloud counts it once."""
    return make_id("event", device_id, episode, action_code, outcome)


def case_id(tenant_id: str, component: str, fault_class: str) -> str:
    return make_id("case", tenant_id, component, fault_class)


def content_hash(obj: object) -> str:
    """sha256 of a canonical JSON rendering (sorted keys) - exact-duplicate detection."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def repair_hash(machine_id: str, fault_class: str, action_code: str, outcome: str, when: str) -> str:
    """The same physical repair, whatever episode it was recorded on: (machine, fault class, action, outcome, day).
    Catches one repair logged twice, e.g. on two episodes the gate split from one intermittent fault
    (docs/RESEARCH.md G.4, exact duplicates). Text fields are normalised (strip, lowercase) before hashing."""
    norm = lambda s: " ".join(str(s or "").split()).lower()
    return content_hash({"machine": norm(machine_id), "fault": norm(fault_class), "action": norm(action_code),
                         "outcome": norm(outcome), "day": str(when or "")[:10]})


def followup_id(device_id: str, refers_to: str) -> str:
    """One hold/recurrence follow-up per shared fix: whichever comes first is final (idempotent on retries)."""
    return make_id("followup", device_id, refers_to)
