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


_MEDIA_OPEN = re.compile(r"\[\[\s*(?:File|Image|Category|Media)\s*:", re.I)


def _strip_media(t: str) -> str:
    """Remove [[File:...]] / [[Image:...]] with their captions. A caption can hold its own [[links]], so the end of
    the tag is found by counting brackets: stopping at the first ']]' left the rest of the caption behind, and
    for the article on DNA that fragment ("of DNA. The phosphate groups are yellow ...") became its lead."""
    while True:
        m = _MEDIA_OPEN.search(t)
        if not m:
            return t
        depth, i = 0, m.start()
        while i < len(t):
            if t.startswith("[[", i):
                depth, i = depth + 1, i + 2
            elif t.startswith("]]", i):
                depth, i = depth - 1, i + 2
                if depth == 0:
                    break
            else:
                i += 1
        t = t[:m.start()] + t[i:]


_GALLERY = re.compile(r"<gallery.*?</gallery>", re.S | re.I)
_TABLE = re.compile(r"\{\|.*?\|\}", re.S)


def _title_words(title: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", re.sub(r"\(.*?\)", "", title or "").lower())
            if len(w) > 2 and w not in ("the", "and", "for", "list")]


def clean(wikitext: str, title: str = "") -> str:
    """The opening paragraph of an article, as plain text ('' if there is none).

    The lead is the first paragraph that is about the article's SUBJECT: it names the title in its first words. Taking merely
    the first paragraph longer than 120 characters skipped real one-line openings ("An apple is a fruit.") in favour of
    a later paragraph or an image caption, so the article on cricket opened with a photo caption about a bowler. A short
    opening is joined to the paragraph after it."""
    t = _REF.sub("", wikitext)
    t = _GALLERY.sub("", _TABLE.sub("", t))
    for _ in range(4):                                # templates nest
        t = _TEMPLATE.sub("", t)
    t = _strip_media(t)
    t = _FILE.sub("", t)
    t = _LINK.sub(r"\1", t)
    t = _EXT.sub(lambda m: m.group(1) or "", t)
    t = _TAG.sub("", t).replace("'''", "").replace("''", "").replace("&nbsp;", " ")
    paras = []
    for para in t.split("\n"):
        p = para.strip()
        if (len(p) >= 12 and not p.startswith(("{", "|", "!", "*", "#", "=", ":", ";", "[", ".", ")", "]"))
                and not re.search(r"\]\]|\[\[|\|\s*\w+\s*=|thumb\||upright", p[:300])):
            paras.append(p)
        if len(paras) >= 6:
            break
    if not paras:
        return ""
    tw = _title_words(title)
    pick = next((i for i, p in enumerate(paras) if tw and any(w in p[:160].lower() for w in tw)), None)
    if pick is None:                                  # no paragraph names the subject: the first long one, as before
        pick = next((i for i, p in enumerate(paras) if len(p) >= MIN_CHARS), None)
        if pick is None:
            return ""
    p = paras[pick]
    if len(p) < MIN_CHARS and pick + 1 < len(paras):
        p = p + " " + paras[pick + 1]
    if len(p) < 40:
        return ""
    if len(p) > MAX_CHARS:
        cut = p.rfind(". ", 0, MAX_CHARS)
        p = p[:cut + 1] if cut > MIN_CHARS else p[:MAX_CHARS]
    p = re.sub(r"\(\s*[,;]?\s*\)", "", p)             # brackets emptied by a removed pronunciation template
    return re.sub(r"\s+", " ", p).replace(" ,", ",").strip()


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
                text = clean(rev.text or "", title) if rev is not None else ""
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
