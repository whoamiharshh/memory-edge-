"""Turn the Simple English Wikipedia dump into one lead paragraph per article (data/raw/simplewiki_leads.jsonl).

Download first (CC BY-SA 4.0, about 370 MB):
  https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles-multistream.xml.bz2
into data/raw/. Only main-namespace, non-redirect articles are kept, and only the opening paragraph, with
wiki markup removed. edge/seed.py embeds these into the device's reference memory with a link back to the
article, so every offline answer cites the page it came from.
"""
from __future__ import annotations

import bz2
import json
import pathlib
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
DUMP = RAW / "simplewiki-latest-pages-articles-multistream.xml.bz2"
OUT = RAW / "simplewiki_leads.jsonl"
MIN_CHARS, MAX_CHARS = 120, 700

_SKIP = re.compile(r"(list of|lists of|deaths in|births in|opinion polling|results of|timeline of|"
                   r"years in|\d{4} in |index of|outline of)", re.I)       # bookkeeping pages, not facts
_TEMPLATE =re.compile(r"\{\{[^{}]*\}\}")
_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.S)
_TAG = re.compile(r"<[^>]+>")
_FILE = re.compile(r"\[\[(?:File|Image|Category):[^\]]*\]\]", re.I)
_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_EXT = re.compile(r"\[https?://\S+(?: ([^\]]*))?\]")


def clean(wikitext: str) -> str:
    """The first real paragraph of an article, as plain text ('' if there is none)."""
    t = _REF.sub("", wikitext)
    for _ in range(4):                                # templates nest
        t = _TEMPLATE.sub("", t)
    t = _FILE.sub("", t)
    t = _LINK.sub(r"\1", t)
    t = _EXT.sub(lambda m: m.group(1) or "", t)
    t = _TAG.sub("", t).replace("'''", "").replace("''", "")
    for para in t.split("\n"):
        p = para.strip()
        if len(p) >= MIN_CHARS and not p.startswith(("{", "|", "!", "*", "#", "=", ":", ";", "[")):
            if len(p) > MAX_CHARS:
                cut = p.rfind(". ", 0, MAX_CHARS)
                p = p[:cut + 1] if cut > MIN_CHARS else p[:MAX_CHARS]
            p = re.sub(r"\(\s*[,;]?\s*\)", "", p)         # brackets emptied by a removed pronunciation template
            return re.sub(r"\s+", " ", p).replace(" ,", ",").strip()
    return ""


def extract(dump: pathlib.Path = DUMP, out: pathlib.Path = OUT) -> int:
    """Writes the leads sorted by full article length, longest first: a long article is a well-developed one,
    so the first N lines are the best core subset when loading everything would take hours."""
    rows: list[dict] = []
    with bz2.open(dump, "rb") as src:
        for _, el in ET.iterparse(src):
            if el.tag.rsplit("}", 1)[-1] != "page":
                continue
            ns = {c.tag.rsplit("}", 1)[-1]: c for c in el}
            if (ns["ns"].text or "") == "0" and "redirect" not in ns:
                title = ns["title"].text or ""
                rev = next((c for c in ns["revision"] if c.tag.rsplit("}", 1)[-1] == "text"), None)
                text = clean(rev.text or "") if rev is not None else ""
                if text and not _SKIP.match(title):
                    url = "https://simple.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                    rows.append({"title": title, "text": text, "url": url, "size": len(rev.text or "")})
            el.clear()
    rows.sort(key=lambda r: -r["size"])
    with open(out, "w", encoding="utf-8") as dst:
        for r in rows:
            dst.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    print(f"wrote {extract()} article leads to {OUT}", file=sys.stderr)
