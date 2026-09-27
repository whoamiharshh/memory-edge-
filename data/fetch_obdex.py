"""Build knowledge/vehicle_codes.json: the vehicle fault-code dictionary used offline by the device.

Source: OBDex (github.com/foerbsnavi/OBDex), all 9,533 generic SAE J2012 / ISO 15031-6 codes with meaning, affected
components, common causes with likelihood, symptoms, repair difficulty and public sources per code.
Data licence: CC0 1.0 (public domain; LICENSE-DATA checked 28 Sep 2026), so the compact English extract may ship
with this repository. We still credit it. The raw YAML goes to data/raw/obdex/ (git-ignored).
Run: .venv\\Scripts\\python.exe data\\fetch_obdex.py
"""
from __future__ import annotations

import json
import pathlib

import httpx
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "obdex"
OUT = ROOT / "knowledge" / "vehicle_codes.json"
BASE = "https://raw.githubusercontent.com/foerbsnavi/OBDex/main/data/generic/"
FAMILIES = ["P0xxx", "P2xxx", "P3xxx", "U0xxx", "U3xxx", "B0xxx", "C0xxx"]


def compact(e: dict) -> dict:
    en = lambda v: (v or {}).get("en", "") if isinstance(v, dict) else (v or "")
    rep = e.get("repair") or {}
    return {"title": en(e.get("title")), "description": en(e.get("description")), "category": e.get("category"),
            "components": e.get("affected_components") or [],
            "causes": [{"cause": en(c.get("label")), "likelihood": c.get("likelihood")} for c in e.get("common_causes") or []],
            "symptoms": [en(s) for s in e.get("symptoms") or []],
            "repair": {"difficulty": rep.get("difficulty"), "diy_possible": rep.get("diy_possible"),
                       "estimated_hours": rep.get("estimated_hours")},
            "related": e.get("related_codes") or [], "sources": e.get("sources") or []}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    codes = {}
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        for fam in FAMILIES:
            f = RAW / f"{fam}_enriched.yaml"
            if not f.exists():
                f.write_bytes(c.get(BASE + f.name).raise_for_status().content)
            for e in yaml.safe_load(f.read_text(encoding="utf-8")):
                codes[e["code"]] = compact(e)
    OUT.write_text(json.dumps({
        "about": "Vehicle fault codes (generic SAE J2012 / ISO 15031-6) from OBDex, CC0 1.0, "
                 "github.com/foerbsnavi/OBDex. Reference only: meanings are generic; manufacturer codes and repair "
                 "steps come from the vehicle's service manual.",
        "count": len(codes), "codes": codes}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} with {len(codes)} codes, {OUT.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
