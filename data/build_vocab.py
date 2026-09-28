"""Build knowledge/domain_vocab.json: words that occur in real maintenance / technical text, so the redactor does not
mistake them for people's names (e.g. JACK, MARK, WILL in a logbook are tools and verbs).

Sources (all real text): the Annotated Maintenance Logbook (Zenodo 17903357, CC BY 4.0: a word list with counts is
derived from it, credited), the OBDex fault-code descriptions (CC0), the cited procedures (knowledge/procedures.json)
and the project's own enum names. Words kept: 3-20 letters, seen at least MIN_COUNT times.
`exclude` lets the benchmark build the list WITHOUT its held-out test notes (no leakage).
Run: .venv\\Scripts\\python.exe -m data.build_vocab
"""
from __future__ import annotations

import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "knowledge" / "domain_vocab.json"
LOGBOOK = ROOT / "data" / "raw" / "logbook" / "Second version of Annotated Maintenance logbook.csv"
MIN_COUNT = 2
WORD = re.compile(r"[A-Za-z]{3,20}")


def logbook_notes() -> list[str]:
    import pandas as pd
    d = pd.read_csv(LOGBOOK, encoding="latin-1")
    return [f"{p} {a}" for p, a in zip(d["PROBLEM"].fillna(""), d["ACTION"].fillna(""))]


def other_texts() -> list[str]:
    out = []
    codes = json.loads((ROOT / "knowledge" / "vehicle_codes.json").read_text(encoding="utf-8"))
    for v in (codes.get("codes") or codes).values() if isinstance(codes, dict) else []:
        if isinstance(v, dict):
            out.append(" ".join(str(x) for x in v.values() if isinstance(x, str)))
        elif isinstance(v, str):
            out.append(v)
    out.append((ROOT / "knowledge" / "procedures.json").read_text(encoding="utf-8"))
    from shared import schema
    for e in (schema.Component, schema.FaultClass, schema.ActionCode, schema.RootCause, schema.DamageMode):
        out.append(" ".join(m.value.replace("_", " ") for m in e))
    return out


def build(exclude: set[int] | None = None) -> set[str]:
    c = collections.Counter()
    for i, t in enumerate(logbook_notes()):
        if exclude and i in exclude:
            continue
        c.update(w.lower() for w in WORD.findall(t))
    for t in other_texts():
        c.update(w.lower() for w in WORD.findall(t))
    return {w for w, n in c.items() if n >= MIN_COUNT}


def main() -> None:
    v = build()
    OUT.write_text(json.dumps({"about": "Words seen >= 2 times in real maintenance/technical text; the redactor never "
                                        "treats them as names. Derived from: Annotated Maintenance Logbook (Zenodo "
                                        "17903357, CC BY 4.0, derived from MaintNet); OBDex (CC0); knowledge/"
                                        "procedures.json; project enums.",
                               "words": sorted(v)}, separators=(",", ":")))
    print(f"wrote {OUT}: {len(v)} words")


if __name__ == "__main__":
    main()
