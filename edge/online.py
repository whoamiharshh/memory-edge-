"""Online retrieval: the wider world, for the questions this device cannot answer from what it holds.

Everything else in this project is deliberately offline. This module is the one place that reaches the
internet, so the boundary is easy to audit:

  * It is only ever called from `Device.ask`, and only when the device is online AND the retrieval mode
    allows it AND the question is not about this machine (see `Device._may_go_online`).
  * It sends the question text and nothing else. No episode, note, fingerprint, embedding, device id,
    site id or machine id is ever part of a request.
  * It returns snippets with their source URL, so an answer built on them can cite where each line came
    from, exactly as a local answer cites [E1].
  * Every failure is swallowed into an empty result. A device with no internet, a blocked DNS, a provider
    that changed its HTML - none of them may turn into an error the technician sees.

Providers. The default pair needs no account and no key, which is what makes the feature work on a fresh
clone. A keyed provider is a drop-in: set EDGE_SEARCH_PROVIDER and the matching key.

  EDGE_SEARCH_PROVIDER = duckduckgo   (default) DuckDuckGo's HTML endpoint + Wikipedia. No key.
                       = wikipedia    Wikipedia only. No key. Factual lookups, no current events.
                       = brave        needs BRAVE_SEARCH_API_KEY   (brave.com/search/api)
                       = tavily       needs TAVILY_API_KEY         (tavily.com)
                       = none         online retrieval disabled outright

DuckDuckGo's HTML endpoint is not a documented API: it is scraped, rate-limited, and it can change
without notice. That is the price of working without a signup, and it is why `available()` reports which
provider is active and whether its key is present, so the UI can say so plainly rather than silently
answering from nothing.
"""
from __future__ import annotations

import html
import os
import re
import urllib.parse

import httpx

TIMEOUT_S = 6.0
MAX_RESULTS = 5
MAX_SNIPPET = 400
# A browser UA: the HTML endpoint serves a consent interstitial to clients it does not recognise.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/124.0 Safari/537.36")

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def provider() -> str:
    return (os.environ.get("EDGE_SEARCH_PROVIDER") or "duckduckgo").strip().lower()


def available() -> dict:
    """What online retrieval can do right now - for the UI, so it never has to guess."""
    p = provider()
    keys = {"brave": "BRAVE_SEARCH_API_KEY", "tavily": "TAVILY_API_KEY"}
    needed = keys.get(p)
    return {"provider": p, "enabled": p != "none",
            "needs_key": needed, "key_present": bool(needed and os.environ.get(needed)),
            "ready": p != "none" and (not needed or bool(os.environ.get(needed)))}


def _clean(s: str, limit: int = MAX_SNIPPET) -> str:
    text = _WS.sub(" ", html.unescape(_TAG.sub(" ", s or ""))).strip()
    return text[:limit].rstrip() + ("…" if len(text) > limit else "")


def _result(title: str, url: str, snippet: str) -> dict | None:
    title, snippet = _clean(title, 160), _clean(snippet)
    if not (title and url.startswith(("http://", "https://"))):
        return None
    return {"title": title, "url": url, "snippet": snippet}


# ---- providers -------------------------------------------------------------------------------------------
def _duckduckgo(q: str, limit: int, client: httpx.Client) -> list[dict]:
    r = client.post("https://html.duckduckgo.com/html/", data={"q": q})
    r.raise_for_status()
    out: list[dict] = []
    # One result block: <a class="result__a" href="URL">TITLE</a> ... <a class="result__snippet">TEXT</a>
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
                         r'(?P<rest>.*?)(?=<a[^>]+class="result__a"|\Z)', r.text, re.S):
        href = html.unescape(m.group("href"))
        # results are wrapped in a redirect: /l/?uddg=<percent-encoded real url>
        if "uddg=" in href:
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                                        .get("uddg", [""])[0])
        snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', m.group("rest"), re.S)
        hit = _result(m.group("title"), href, snip.group(1) if snip else "")
        if hit:
            out.append(hit)
        if len(out) >= limit:
            break
    return out


def _wikipedia(q: str, limit: int, client: httpx.Client) -> list[dict]:
    api = "https://en.wikipedia.org/w/api.php"
    r = client.get(api, params={"action": "query", "list": "search", "srsearch": q,
                                "srlimit": min(limit, 3), "format": "json"})
    r.raise_for_status()
    hits = (r.json().get("query") or {}).get("search") or []
    if not hits:
        return []
    titles = [h["title"] for h in hits]
    # one extract call for every title, rather than one call each
    e = client.get(api, params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                "titles": "|".join(titles), "format": "json"})
    e.raise_for_status()
    pages = ((e.json().get("query") or {}).get("pages") or {}).values()
    extracts = {p.get("title"): p.get("extract", "") for p in pages}
    out = []
    for t in titles:
        hit = _result(t, "https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")),
                      extracts.get(t) or next((h["snippet"] for h in hits if h["title"] == t), ""))
        if hit:
            out.append(hit)
    return out


def _brave(q: str, limit: int, client: httpx.Client) -> list[dict]:
    key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not key:
        return []
    r = client.get("https://api.search.brave.com/res/v1/web/search", params={"q": q, "count": limit},
                   headers={"X-Subscription-Token": key, "Accept": "application/json"})
    r.raise_for_status()
    out = []
    for item in ((r.json().get("web") or {}).get("results") or [])[:limit]:
        hit = _result(item.get("title", ""), item.get("url", ""), item.get("description", ""))
        if hit:
            out.append(hit)
    return out


def _tavily(q: str, limit: int, client: httpx.Client) -> list[dict]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    r = client.post("https://api.tavily.com/search",
                    json={"api_key": key, "query": q, "max_results": limit})
    r.raise_for_status()
    out = []
    for item in (r.json().get("results") or [])[:limit]:
        hit = _result(item.get("title", ""), item.get("url", ""), item.get("content", ""))
        if hit:
            out.append(hit)
    return out


_PROVIDERS = {"brave": [_brave], "tavily": [_tavily], "wikipedia": [_wikipedia],
              "duckduckgo": [_duckduckgo, _wikipedia]}


def search(question: str, limit: int = MAX_RESULTS) -> list[dict]:
    """Snippets from the web for this question, or [] if the internet cannot be reached.

    Never raises. A caller that had to wrap this in try/except would eventually forget to, and an
    unreachable network is the normal condition for this product, not an exceptional one.
    """
    q = (question or "").strip()
    p = provider()
    if not q or p == "none":
        return []
    out: list[dict] = []
    seen: set[str] = set()
    try:
        with httpx.Client(timeout=TIMEOUT_S, follow_redirects=True,
                          headers={"User-Agent": _UA, "Accept-Language": "en"}) as client:
            for fn in _PROVIDERS.get(p, _PROVIDERS["duckduckgo"]):
                try:
                    found = fn(q, limit, client)
                except Exception:
                    continue                       # one provider failing must not lose the other's hits
                for hit in found:
                    if hit["url"] not in seen:
                        seen.add(hit["url"])
                        out.append(hit)
                if len(out) >= limit:
                    break
    except Exception:
        return []
    return out[:limit]


def as_evidence(hits: list[dict]) -> list[dict]:
    """Web hits in the shape `ask` uses for every other source.

    They are numbered in the same [E1..] sequence as local evidence rather than in a separate [W1..] one:
    the grounding check in edge/rag.py only recognises E-citations, and a second prefix would have meant
    loosening it. What marks a line as coming from the internet is `source` and the URL carried with it,
    which is also what the UI needs in order to link to it.
    """
    return [{"source": "web", "kind": "web", "id": h["url"], "url": h["url"], "title": h["title"],
             "text": f"{h['title']}: {h['snippet']}" if h["snippet"] else h["title"]}
            for h in hits]
