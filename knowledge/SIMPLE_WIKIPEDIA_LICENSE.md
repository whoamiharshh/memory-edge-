# Simple English Wikipedia core - attribution and licence

`simple_wikipedia_core.jsonl.gz` contains the opening paragraph of 30,000 articles of **Simple English Wikipedia**
(https://simple.wikipedia.org), and `simple_wikipedia_core.vec.npz` contains a vector for each paragraph.

- **Source:** the Wikimedia database dump `simplewiki-latest-pages-articles-multistream.xml.bz2`, downloaded from
  https://dumps.wikimedia.org/simplewiki/latest/ on 2 October 2026 (dump dated 1 October 2026).
- **Licence:** Creative Commons Attribution-ShareAlike 4.0 International
  (https://creativecommons.org/licenses/by-sa/4.0/). The text is by the Wikipedia contributors of each article.
  Every record carries the URL of its article, whose page history lists the authors; the device shows that URL next to
  every answer it quotes.
- **Changes made:** only the opening paragraph of each article is kept; wiki markup, templates, references and
  files are removed; empty brackets left by removed pronunciation templates are dropped; "list of", "deaths in",
  "opinion polling", "timeline of" and similar bookkeeping pages are left out. The 30,000 longest remaining articles
  are kept (`tools/extract_simplewiki.py`, `tools/build_wikipedia_core.py`).
- **ShareAlike:** this file and the vectors derived from it stay under CC BY-SA 4.0. The rest of this repository is
  Apache-2.0 (see `LICENSE`); the two licences apply to their own files and do not change each other.
- `general_facts.json` (capitals, SI units) is built from Wikidata (CC0) and the BIPM SI Brochure.
