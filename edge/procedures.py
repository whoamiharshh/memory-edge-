"""Documented fix procedures (knowledge/procedures.json + an optional knowledge/site_procedures.json), looked up by
fault class and component and shown next to an episode. Deterministic lookup, no generation: the text shown is the
documented checklist and its source, never model output. Every entry must cite a source (a public document with a
URL, or a 'site SOP'); entries without one are refused at load time."""
from __future__ import annotations

import json
import pathlib

KNOWLEDGE = pathlib.Path(__file__).resolve().parents[1] / "knowledge"
REQUIRED = ("id", "fault_classes", "components", "title", "steps", "sources")


def _valid(p: dict) -> bool:
    if any(k not in p for k in REQUIRED) or not p["steps"]:
        return False
    return all(s.get("url") or s.get("kind") == "site SOP" for s in p["sources"]) and bool(p["sources"])


def load(root: pathlib.Path = KNOWLEDGE) -> list[dict]:
    out = []
    for name, origin in (("procedures.json", "reference"), ("site_procedures.json", "site")):
        f = root / name
        if not f.exists():
            continue
        for p in json.loads(f.read_text(encoding="utf-8")).get("procedures", []):
            if not _valid(p):
                raise ValueError(f"{name}: procedure {p.get('id')!r} has no steps or no source; refusing to load it")
            out.append(p | {"origin": origin})
    return out


_CACHE: list[dict] | None = None


def lookup(fault_class: str | None, component: str | None = None) -> list[dict]:
    """Procedures for a fault class; component-specific ones first, site SOPs before reference documents."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load()
    if not fault_class or fault_class == "unknown":
        return []
    hits = [p for p in _CACHE if fault_class in p["fault_classes"]]
    return sorted(hits, key=lambda p: (p["origin"] != "site", component not in p["components"], p["id"]))
