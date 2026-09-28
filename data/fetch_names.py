"""Build knowledge/given_names.json: real given names from Wikidata (CC0 1.0 - may be redistributed), used by the
note redactor (shared/redact.py) to find people's names that are not on a site's denylist.

For each country: given names (P735) of humans (Q5) with that citizenship (P27), ranked by how many people carry
them. India first (the deployment context), then other countries whose names appear in Indian and global workforces.
Only alphabetic names of >= 3 letters are kept; initials ("K.") are dropped.
Run: .venv\\Scripts\\python.exe -m data.fetch_names
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import time

import httpx

OUT = pathlib.Path(__file__).resolve().parents[1] / "knowledge" / "given_names.json"
RAW = pathlib.Path(__file__).resolve().parent / "raw" / "names"
COUNTRIES = {"India": ("Q668", 6000), "Nepal": ("Q837", 800), "Bangladesh": ("Q902", 1000), "Pakistan": ("Q843", 1500),
             "Sri Lanka": ("Q854", 800), "United States": ("Q30", 5000), "United Kingdom": ("Q145", 3000),
             "Germany": ("Q183", 2000), "China": ("Q148", 1000), "Philippines": ("Q928", 1000),
             "United Arab Emirates": ("Q878", 500), "Nigeria": ("Q1033", 1000)}
QUERY = """SELECT ?nameLabel (COUNT(DISTINCT ?p) AS ?n) WHERE {{
  ?p wdt:P31 wd:Q5; wdt:P27 wd:{q}; wdt:P735 ?name.{extra}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}
GROUP BY ?nameLabel ORDER BY DESC(?n) LIMIT {k}"""
UA = {"User-Agent": "machine-memory-edge/1.0 (hackathon research project; offline name redaction)"}


# very large countries time out on the public endpoint: restrict them to people born since 1960 (today's workforce)
RECENT = " ?p wdt:P569 ?born. FILTER(YEAR(?born) >= 1960)"
BIG = {"Q30", "Q145", "Q183"}


def fetch(q: str, k: int) -> list[tuple[str, int]] | None:
    cached = RAW / f"{q}.json"
    if cached.exists():
        return [tuple(r) for r in json.loads(cached.read_text())]
    for attempt in range(3):
        try:
            query = QUERY.format(q=q, k=k, extra=RECENT if q in BIG else "")
            r = httpx.get("https://query.wikidata.org/sparql", params={"query": query, "format": "json"},
                          headers=UA, timeout=180)
            r.raise_for_status()
            return [(b["nameLabel"]["value"], int(b["n"]["value"])) for b in r.json()["results"]["bindings"]]
        except (httpx.HTTPError, ValueError) as e:
            print("retry", q, type(e).__name__, flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    names: dict[str, dict] = {}
    for country, (q, k) in COUNTRIES.items():
        rows = fetch(q, k)
        if rows is None:
            print(f"{country}: SKIPPED (Wikidata timed out; re-run later to add it)", flush=True)
            continue
        (RAW / f"{q}.json").write_text(json.dumps(rows))
        kept = 0
        for label, n in rows:
            for part in label.split():                       # "Mohan Lal" counts as two given names
                if re.fullmatch(r"[A-Za-z]{3,20}", part):
                    rec = names.setdefault(part.lower(), {"people": 0, "countries": []})
                    rec["people"] += n
                    if country not in rec["countries"]:
                        rec["countries"].append(country)
                    kept += 1
        print(f"{country}: {len(rows)} names fetched, {kept} kept", flush=True)
        time.sleep(2)
    OUT.write_text(json.dumps({
        "about": "Given names of people with these citizenships, from Wikidata (P735 of Q5 humans by P27), for the "
                 "note redactor. Data: Wikidata, CC0 1.0 (https://www.wikidata.org/wiki/Wikidata:Licensing).",
        "fetched": dt.date.today().isoformat(), "countries": list(COUNTRIES),
        "names": dict(sorted(names.items()))}, separators=(",", ":")))
    print(f"wrote {OUT}: {len(names)} distinct given names")


if __name__ == "__main__":
    main()
