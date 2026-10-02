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

import gzip
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


def _general_facts() -> Iterator[dict]:
    """Capitals and SI units, built from Wikidata / the SI Brochure by tools/build_general_facts.py."""
    path = KNOWLEDGE / "general_facts.json"
    if not path.exists():
        return
    for f in json.loads(path.read_text(encoding="utf-8")).get("facts") or []:
        yield {"ref_id": f["id"], "title": f["title"], "text": f["text"], "sources": f.get("sources") or [],
               "topic": "general knowledge"}


WIKI_CORE = 30_000          # the leads file is sorted longest article first, so this is the best-developed core


def _wiki(start: int, stop: int | None) -> Iterator[dict]:
    """Opening paragraphs of Simple English Wikipedia articles (CC BY-SA 4.0), cited by URL. Produced by
    tools/extract_simplewiki.py from a dump in data/raw/; absent until someone runs it."""
    path = KNOWLEDGE.parent / "data" / "raw" / "simplewiki_leads.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < start:
                continue
            if stop is not None and i >= stop:
                return
            a = json.loads(line)
            yield {"ref_id": "wiki:" + a["title"], "title": a["title"], "text": f"{a['title']}: {a['text']}",
                   "sources": [a["url"]], "topic": "Simple English Wikipedia"}


def _wiki_core_from_repo(path: pathlib.Path) -> Iterator[dict]:
    """The 30,000-article core that ships in the repository (knowledge/simple_wikipedia_core.jsonl.gz, built by
    tools/build_wikipedia_core.py), with its precomputed vectors when they are present. Each record carries the
    vector and the name of the model that made it; `_flush` uses it only if this device's embedder is that model."""
    vecs, model = None, None
    vec_path = path.with_name("simple_wikipedia_core.vec.npz")
    if vec_path.exists():
        try:
            import numpy as np
            blob = np.load(vec_path, allow_pickle=False)
            vecs, model = blob["vecs"], str(blob["model"])
        except Exception:
            vecs = None                              # unreadable vectors cost speed, never correctness
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            a = json.loads(line)
            rec = {"ref_id": "wiki:" + a["title"], "title": a["title"], "text": f"{a['title']}: {a['text']}",
                   "sources": [a["url"]], "topic": "Simple English Wikipedia"}
            if vecs is not None and i < len(vecs):
                rec |= {"vec": vecs[i], "vec_model": model}
            yield rec


def _simple_wikipedia() -> Iterator[dict]:
    core = KNOWLEDGE / "simple_wikipedia_core.jsonl.gz"
    return _wiki_core_from_repo(core) if core.exists() else _wiki(0, WIKI_CORE)


def _simple_wikipedia_rest() -> Iterator[dict]:
    return _wiki(WIKI_CORE, None)


# Loaded by itself on a device's first start (edge/main.py): the packs that ship with precomputed vectors or are
# tiny, so a first start costs seconds, not minutes. The 9.5k vehicle codes (~16 minutes of CPU to embed) and the
# long tail of Wikipedia stay opt-in, through POST /api/reference/load.
AUTO = ("procedures", "general_facts", "simple_wikipedia")

# quick packs first, so the answers a person is most likely to ask for are searchable within a minute of a first
# start; the 9.5k vehicle codes (about 7 minutes to embed) come last
PACKS = {"procedures": _procedures, "general_facts": _general_facts, "simple_wikipedia": _simple_wikipedia,
         "vehicle_codes": _vehicle_codes, "simple_wikipedia_rest": _simple_wikipedia_rest}


def available() -> dict[str, int]:
    """How many records each pack would contribute."""
    return {name: sum(1 for _ in fn()) for name, fn in PACKS.items()}


def seeded(device) -> list[str]:
    return sorted(device.outbox.kv_get(SEED_FLAG, []) or [])


def seed_if_empty(device, packs: list[str] | None = None) -> dict:
    """Load the reference packs once per device. Records are kept local and never synced: every device can
    rebuild them from its own copy of the repository, so shipping them over the network would be waste."""
    done = set(seeded(device))
    # the long tail of Wikipedia takes hours to embed, so it is only loaded when asked for by name
    default = [p for p in PACKS if not p.endswith("_rest")]
    want = [p for p in (packs or default) if p in PACKS and p not in done]
    if not want:
        return {"seeded": 0, "packs": [], "already": sorted(done)}

    total = 0
    for name in want:
        batch: list[dict] = []
        loaded = 0
        for rec in PACKS[name]():
            batch.append(rec | {"pack": name})
            loaded += 1
            if len(batch) >= BATCH:
                total += _flush(device, batch)
                batch = []
        if batch:
            total += _flush(device, batch)
        if loaded:                                  # a pack whose source file is not there yet is retried later
            done.add(name)
            device.outbox.kv_set(SEED_FLAG, sorted(done))     # per pack: a stop half-way keeps what finished

    device.outbox.log("memory", f"loaded {total} reference record(s) from {', '.join(want)}")
    return {"seeded": total, "packs": want, "already": sorted(done)}


def _flush(device, batch: list[dict]) -> int:
    """Embed and store one batch. The dense vectors are computed for the whole batch at once, which is
    far faster on CPU than one call per record — it is the difference between minutes and hours for the
    ~9.5k code pack."""
    # A record that arrives with a precomputed vector from the SAME model skips embedding. A different model
    # (another embedder, the hash embedder in tests) never reuses it: vectors from two models are not comparable.
    shipped = [r.get("vec") is not None and r.get("vec_model") == device.embedder.name for r in batch]
    fresh = iter(device.embedder.embed_documents([r["text"] for r, ok in zip(batch, shipped) if not ok])
                 if not all(shipped) else [])
    vecs = [[float(x) for x in r["vec"]] if ok else next(fresh) for r, ok in zip(batch, shipped)]
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
