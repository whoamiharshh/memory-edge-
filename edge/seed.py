"""Knowledge the device already holds the first time someone opens it.

A device whose memory is empty can only ever say "I do not have anything about that", which makes it
useless until somebody has typed into it for a week. The fix is not a bigger language model: it is to load
the reference material this repository already carries, so the first question has something real to hit.

Everything here comes from a file in knowledge/ that cites its own source. Nothing is invented and nothing
is downloaded. Seeded records are marked `type="reference"` so an answer can say where a fact came from,
and so they are never confused with what this device observed or what a person typed into it.

Packs:
  vehicle_codes   ~9.5k standard OBD-II fault codes with causes and repair notes (knowledge/vehicle_codes.json)
  procedures      documented maintenance checklists, each citing the public document it follows
"""
from __future__ import annotations

import json
import pathlib
from typing import Iterator

from edge.store_edge import StorePoint
from shared import ids

KNOWLEDGE = pathlib.Path(__file__).resolve().parent.parent / "knowledge"
SEED_FLAG = "seeded_packs"
BATCH = 256


def _vehicle_codes() -> Iterator[dict]:
    """One record per fault code. The text leads with the code so an exact lookup ("P0301") matches on
    BM25, and carries the description and causes so a plain-language question ("engine misfires") matches
    on the dense vector."""
    path = KNOWLEDGE / "vehicle_codes.json"
    if not path.exists():
        return
    blob = json.loads(path.read_text(encoding="utf-8"))
    for code, c in (blob.get("codes") or {}).items():
        title = c.get("title") or ""
        causes = [x.get("cause") for x in (c.get("causes") or []) if x.get("cause")]
        parts = [f"{code}: {title}.", c.get("description") or ""]
        if causes:
            parts.append("Common causes: " + "; ".join(causes[:4]) + ".")
        hours = (c.get("repair") or {}).get("estimated_hours") or []
        if len(hours) >= 2:
            parts.append(f"Typically {hours[0]}-{hours[1]} hours to repair.")
        yield {"ref_id": f"dtc:{code}", "title": f"{code} — {title}",
               "text": " ".join(p for p in parts if p).strip(),
               "sources": c.get("sources") or [], "topic": "vehicle fault codes"}


def _procedures() -> Iterator[dict]:
    path = KNOWLEDGE / "procedures.json"
    if not path.exists():
        return
    for p in json.loads(path.read_text(encoding="utf-8")).get("procedures") or []:
        parts = [(p.get("title") or "") + "."]
        if p.get("confirm_first"):
            parts.append("Confirm first: " + " ".join(p["confirm_first"]))
        if p.get("steps"):
            parts.append("Steps: " + " ".join(p["steps"]))
        yield {"ref_id": f"proc:{p.get('id')}", "title": p.get("title", ""),
               "text": " ".join(parts).strip(), "sources": p.get("sources") or [],
               "topic": "maintenance procedures"}


PACKS = {"vehicle_codes": _vehicle_codes, "procedures": _procedures}


def available() -> dict[str, int]:
    """How many records each pack would contribute."""
    return {name: sum(1 for _ in fn()) for name, fn in PACKS.items()}


def seeded(device) -> list[str]:
    return sorted(device.outbox.kv_get(SEED_FLAG, []) or [])


def seed_if_empty(device, packs: list[str] | None = None) -> dict:
    """Load the reference packs once per device. Records are kept local and never synced: every device can
    rebuild them from its own copy of the repository, so shipping them over the network would be waste."""
    done = set(seeded(device))
    want = [p for p in (packs or list(PACKS)) if p in PACKS and p not in done]
    if not want:
        return {"seeded": 0, "packs": [], "already": sorted(done)}

    total = 0
    for name in want:
        batch: list[dict] = []
        for rec in PACKS[name]():
            batch.append(rec | {"pack": name})
            if len(batch) >= BATCH:
                total += _flush(device, batch)
                batch = []
        if batch:
            total += _flush(device, batch)
        done.add(name)

    device.outbox.kv_set(SEED_FLAG, sorted(done))
    device.outbox.log("memory", f"loaded {total} reference record(s) from {', '.join(want)}")
    return {"seeded": total, "packs": want, "already": sorted(done)}


def _flush(device, batch: list[dict]) -> int:
    """Embed and store one batch. The dense vectors are computed for the whole batch at once, which is
    far faster on CPU than one call per record — it is the difference between minutes and hours for the
    ~9.5k code pack."""
    vecs = device.embedder.embed_documents([r["text"] for r in batch])
    points = [
        StorePoint(
            ids.make_id("reference", device.cfg.device_id, r["ref_id"]),
            {"type": "reference", "ref_id": r["ref_id"], "title": r["title"], "note_text": r["text"],
             "topic": r["topic"], "source_pack": r["pack"], "sources": r["sources"][:4],
             "device_id": device.cfg.device_id, "share_state": "local"},
            None, v, r["text"])
        for r, v in zip(batch, vecs)
    ]
    with device._lock:
        device._upsert(points)
    return len(points)
