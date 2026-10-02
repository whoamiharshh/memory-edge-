"""Build the offline TECHNICAL library: cited English-Wikipedia leads for the engineering concepts in knowledge/tech_terms.txt.

  .venv\\Scripts\\python.exe -m tools.build_tech_pack            (needs the internet once; ~2 minutes)

For every term it fetches the opening paragraph of the named Wikipedia article through the MediaWiki API, refuses
disambiguation pages and missing articles (they are listed, not shipped), and writes

  knowledge/technical_wikipedia.jsonl.gz   title, text (opening paragraph, verbatim), url, aliases (the terms people type)
  knowledge/technical_wikipedia.vec.npz    the bge-small-en-v1.5 vector of each text, float16

Text is quoted from Wikipedia unchanged apart from whitespace and cutting at a sentence boundary, and every record keeps its
URL: the library never writes a definition itself. English Wikipedia is CC BY-SA 4.0 (see knowledge/SIMPLE_WIKIPEDIA_LICENSE.md,
which covers both libraries). The audit printed at the end lists every term whose article title differs from the term, so a
wrong mapping can be seen and fixed in tech_terms.txt.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
TERMS = ROOT / "knowledge" / "tech_terms.txt"
OUT_TEXT = ROOT / "knowledge" / "technical_wikipedia.jsonl.gz"
OUT_VEC = ROOT / "knowledge" / "technical_wikipedia.vec.npz"
API = "https://en.wikipedia.org/w/api.php"
UA = "machine-memory-edge/0.1 (offline technical library build; contact: project repository)"
MAX_CHARS, MIN_CHARS = 900, 160


def read_terms() -> list[tuple[str, str]]:
    out = []
    for line in TERMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term, _, title = line.partition("=>")
        out.append((term.strip(), (title.strip() or term.strip())))
    return out


CACHE = ROOT / "data" / "raw" / "tech_pack_cache.json"      # raw API answers, so a re-run (or a 429) never refetches


def fetch(titles: list[str]) -> dict:
    key = "|".join(titles)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if key in cache:
        return cache[key]
    q = {"action": "query", "format": "json", "formatversion": "2", "prop": "extracts|pageprops", "exintro": "1",
         "explaintext": "1", "exlimit": str(len(titles)), "redirects": "1", "ppprop": "disambiguation",
         "titles": key}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": UA})
    wait = 20
    for attempt in range(10):
        try:
            data = json.load(urllib.request.urlopen(req, timeout=60))["query"]
            cache[key] = data
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            wait = int(e.headers.get("Retry-After") or wait)          # the API says how long it wants us to wait
        except Exception:
            pass
        print(f"  rate limited / failed, waiting {wait}s", flush=True)
        time.sleep(wait)
        wait = min(wait * 2, 240)
    raise RuntimeError(f"could not fetch {titles[:3]}...")


def lead(extract: str) -> str:
    """The opening paragraph(s), whitespace-collapsed, cut at a sentence end within MAX_CHARS."""
    t = re.sub(r"[ \t]+", " ", (extract or "").strip())
    paras = [p.strip() for p in t.split("\n") if p.strip()]
    text = paras[0] if paras else ""
    if len(text) < MIN_CHARS and len(paras) > 1:
        text += " " + paras[1]
    text = re.sub(r"\s+", " ", text).replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    text = re.sub(r"\(\s*[;,]?\s*\)", "", text)
    text = re.sub(r"\(\s*(?:(?:UK|US|AU|CA)\s*:\s*,?\s*)+\)", "", text)       # pronunciation left empty: "(UK:, US:)"
    text = re.sub(r"\s{2,}", " ", text).replace(" ,", ",")
    if len(text) > MAX_CHARS:
        cut = text.rfind(". ", 0, MAX_CHARS)
        text = text[:cut + 1] if cut >= MIN_CHARS else text[:MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return text.strip()


def main() -> None:
    terms = read_terms()
    wanted = sorted({title for _, title in terms})
    resolved: dict[str, dict] = {}          # requested title -> page
    for i in range(0, len(wanted), 20):
        batch = wanted[i:i + 20]
        q = fetch(batch)
        hop = {}
        for n in q.get("normalized", []):
            hop[n["from"]] = n["to"]
        for r in q.get("redirects", []):
            hop[r["from"]] = r["to"]
        pages = {p["title"]: p for p in q["pages"]}
        for t in batch:
            cur, seen = t, 0
            while cur in hop and seen < 5:
                cur, seen = hop[cur], seen + 1
            if cur in pages:
                resolved[t] = pages[cur]
        print(f"  fetched {min(i + 20, len(wanted))}/{len(wanted)}", flush=True)
        time.sleep(2.5)

    records: dict[str, dict] = {}
    problems, remapped = [], []
    for term, title in terms:
        p = resolved.get(title)
        if not p or p.get("missing"):
            problems.append((term, title, "article not found"))
            continue
        if (p.get("pageprops") or {}).get("disambiguation") is not None:
            problems.append((term, title, "disambiguation page"))
            continue
        text = lead(p.get("extract") or "")
        if len(text) < 60:
            problems.append((term, title, "lead too short"))
            continue
        final = p["title"]
        if final.lower() != term.lower():
            remapped.append((term, final))
        rec = records.setdefault(final, {"title": final, "text": text,
                                         "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(final.replace(" ", "_")),
                                         "aliases": []})
        for a in (term, final):
            if a not in rec["aliases"]:
                rec["aliases"].append(a)
    # An alias that points at two different articles ("SoC": System on a chip, State of charge) cannot be resolved by a
    # lookup, and guessing is how a wrong answer gets quoted with a citation. It is dropped from both; the articles stay
    # reachable by their own titles, and the question falls through to "needs internet" instead of the wrong entry.
    owners: dict[str, set[str]] = {}
    for r in records.values():
        for a in r["aliases"]:
            owners.setdefault(re.sub(r"[^a-z0-9]", "", a.lower()), set()).add(r["title"])
    ambiguous = {k for k, v in owners.items() if len(v) > 1}
    for r in records.values():
        r["aliases"] = [a for a in r["aliases"] if re.sub(r"[^a-z0-9]", "", a.lower()) not in ambiguous
                        or a == r["title"]]
    for k in sorted(ambiguous):
        print(f"  ambiguous alias {k!r} dropped: {sorted(owners[k])}")
    rows = sorted(records.values(), key=lambda r: r["title"].lower())
    with gzip.open(OUT_TEXT, "wt", encoding="utf-8", compresslevel=9) as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} articles for {len(terms)} terms to {OUT_TEXT} ({OUT_TEXT.stat().st_size / 1e6:.2f} MB)")

    from shared.embed import load_embedder
    emb = load_embedder()
    vecs = np.asarray(emb.embed_documents([f"{r['title']}: {r['text']}" for r in rows]), dtype=np.float32)
    np.savez_compressed(OUT_VEC, vecs=vecs.astype(np.float16), model=np.array(emb.name))
    print(f"wrote {OUT_VEC} ({OUT_VEC.stat().st_size / 1e6:.2f} MB), model {emb.name}")

    print(f"\n--- {len(problems)} terms NOT shipped (fix the title in tech_terms.txt) ---")
    for term, title, why in problems:
        print(f"  {term!r} -> {title!r}: {why}")
    print(f"\n--- {len(remapped)} terms whose article title differs from the term (audit these) ---")
    for term, final in remapped:
        print(f"  {term} -> {final}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
