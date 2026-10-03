"""Repair the corrupted leads in the shipped Wikipedia core without re-embedding all 30,000 of them.

The first extraction cut image captions at their first ']]', so about 2 % of leads began mid-sentence ("of DNA. The
phosphate groups are yellow ...") and, worst, the article on DNA itself answered nothing. tools/extract_simplewiki.py
now strips media by counting brackets. This tool applies that fix to the shipped core:

  1. find the leads that look broken (start mid-sentence, or still carry wiki markup)
  2. re-extract exactly those titles from the dump (data/raw/simplewiki-latest-pages-articles-multistream.xml.bz2)
  3. re-embed only the repaired texts (about a minute), keep every other vector as it was
  4. a lead that still looks broken after the fix is dropped rather than shipped

  .venv\\Scripts\\python.exe -m tools.repair_wikipedia_core
"""
from __future__ import annotations

import bz2
import gzip
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

from tools.extract_simplewiki import DUMP, _title_words, clean

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT = ROOT / "knowledge" / "simple_wikipedia_core.jsonl.gz"
VEC = ROOT / "knowledge" / "simple_wikipedia_core.vec.npz"
_MARKUP = re.compile(r"\]\]|\[\[|\{\{|\|\s*\w+\s*=|thumb\||upright|<\w+|\|\|")


def has_markup(text: str) -> bool:
    return bool(_MARKUP.search((text or "")[:400]))


def looks_broken(text: str) -> bool:
    """Worth re-extracting: markup left in, too short, or it opens mid-sentence / with punctuation."""
    t = (text or "").strip()
    if len(t) < 40 or "&nbsp;" in t[:400] or has_markup(t):
        return True
    return not (t[0].isupper() or t[0].isdigit() or t[0] in "\"'“‘(")


def main() -> None:
    rows = [json.loads(line) for line in gzip.open(TEXT, "rt", encoding="utf-8")]
    blob = np.load(VEC, allow_pickle=False)
    vecs, model = blob["vecs"], str(blob["model"])
    assert len(vecs) == len(rows), "vectors and texts are out of step"
    # A lead is re-checked when it looks broken OR does not name its own subject near the start (the cricket article opened
    # with a photo caption, the apple article with its second paragraph).
    def off_subject(r):
        tw = _title_words(r["title"])
        return bool(tw) and not any(w in r["text"][:160].lower() for w in tw)
    # --all: re-extract EVERY lead with the current rule. The first extraction skipped one-line openings ("An apple is a
    # fruit.") in favour of a later paragraph that still named the subject, so those leads were not flagged by any check.
    every = "--all" in sys.argv
    bad = {r["title"]: i for i, r in enumerate(rows) if every or looks_broken(r["text"]) or off_subject(r)}
    print(f"{len(bad)} of {len(rows)} leads look broken; re-extracting them from the dump", flush=True)

    fixed: dict[int, str] = {}
    with bz2.open(DUMP, "rb") as src:
        for _, el in ET.iterparse(src):
            if el.tag.rsplit("}", 1)[-1] != "page":
                continue
            ns = {c.tag.rsplit("}", 1)[-1]: c for c in el}
            title = ns["title"].text or ""
            if title in bad and (ns["ns"].text or "") == "0":
                rev = next((c for c in ns["revision"] if c.tag.rsplit("}", 1)[-1] == "text"), None)
                new = clean(rev.text or "", title) if rev is not None else ""
                if new and not has_markup(new) and len(new) >= 40 and new != rows[bad[title]]["text"]:
                    fixed[bad[title]] = new
            el.clear()
    # What the fix could not improve stays if it is harmless (a lead that merely begins with a lowercase "is a ..."
    # because a name template was removed is still a true sentence, shown after its title). Only text that still
    # carries wiki markup is dropped.
    for i in set(bad.values()) - set(fixed):
        old = rows[i]["text"].replace("&nbsp;", " ")
        if len(old.strip()) >= 40 and not has_markup(old):
            fixed[i] = old
    dropped = sorted(set(bad.values()) - set(fixed))
    print(f"repaired {len(fixed)}, cannot repair {len(dropped)} (dropped)", flush=True)

    from shared.embed import load_embedder
    emb = load_embedder()
    if emb.name != model:
        sys.exit(f"embedder is {emb.name} but the shipped vectors are {model}; refusing to mix them")
    idx = sorted(fixed)
    new_vecs = np.asarray(emb.embed_documents([f"{rows[i]['title']}: {fixed[i]}" for i in idx]), dtype=np.float32)
    for i, v in zip(idx, new_vecs):
        rows[i]["text"], vecs[i] = fixed[i], v.astype(np.float16)
    gone = set(dropped)
    keep = [i for i in range(len(rows)) if i not in gone]
    with gzip.open(TEXT, "wt", encoding="utf-8", compresslevel=9) as g:
        for i in keep:
            g.write(json.dumps({k: rows[i][k] for k in ("title", "text", "url")}, ensure_ascii=False) + "\n")
    np.savez_compressed(VEC, vecs=vecs[keep], model=np.array(model))
    print(f"wrote {len(keep)} articles; {len(idx)} re-embedded", flush=True)


if __name__ == "__main__":
    main()
