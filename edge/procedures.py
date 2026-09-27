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


_CODES: dict | None = None


def code_info(code: str) -> dict | None:
    """Meaning, likely causes and sources of a standard vehicle fault code (knowledge/vehicle_codes.json, OBDex,
    CC0). None for unknown codes (e.g. manufacturer-specific or kiosk/app codes: those need site knowledge)."""
    global _CODES
    if _CODES is None:
        f = KNOWLEDGE / "vehicle_codes.json"
        _CODES = json.loads(f.read_text(encoding="utf-8"))["codes"] if f.exists() else {}
    e = _CODES.get(str(code).strip().upper())
    return None if e is None else {"code": str(code).strip().upper(), **e,
                                   "source_note": "OBDex (CC0), generic SAE J2012 meaning; check the service manual"}


_CACHE: list[dict] | None = None


def device_sops(path: pathlib.Path | None) -> list[dict]:
    """Site SOPs a technician added on this device (the 'Add procedure' form), stored next to its memory."""
    if path is None or not path.exists():
        return []
    return [p | {"origin": "site"} for p in json.loads(path.read_text(encoding="utf-8")).get("procedures", []) if _valid(p)]


def add_site_procedure(path: pathlib.Path, proc: dict) -> dict:
    """Validate and append one site SOP. Every SOP must name its source (e.g. 'Pump P-12 manual, section 7')."""
    import re
    import time
    p = {"id": "site-" + re.sub(r"[^a-z0-9]+", "-", proc["title"].lower()).strip("-")[:40] + f"-{int(time.time())}",
         "fault_classes": proc["fault_classes"], "components": proc["components"], "title": proc["title"],
         "confirm_first": proc.get("confirm_first") or [], "steps": proc["steps"],
         "sources": [{"title": proc["source_title"], "kind": "site SOP"}], "added_at": time.strftime("%Y-%m-%d")}
    if not _valid(p) or not proc["source_title"].strip():
        raise ValueError("a procedure needs a title, at least one step and a source (the manual or SOP it follows)")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"procedures": []}
    data["procedures"].append(p)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return p | {"origin": "site"}


def lookup(fault_class: str | None, component: str | None = None, site_file: pathlib.Path | None = None) -> list[dict]:
    """Procedures for a fault class; component-specific ones first, site SOPs before reference documents."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load()
    if not fault_class or fault_class == "unknown":
        return []
    hits = [p for p in _CACHE + device_sops(site_file) if fault_class in p["fault_classes"]]
    return sorted(hits, key=lambda p: (p["origin"] != "site", component not in p["components"], p["id"]))
