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

## Technical library (`technical_wikipedia.jsonl.gz`, `technical_wikipedia.vec.npz`)

About 860 opening paragraphs of **English Wikipedia** (https://en.wikipedia.org) for engineering concepts (robots, PLCs,
vehicles, phones, mathematics, electronics ...), fetched on 3 October 2026 through the MediaWiki API by
`tools/build_tech_pack.py` from the terms in `tech_terms.txt`. Licence, authorship and the one-URL-per-record rule are
exactly as above: Creative Commons Attribution-ShareAlike 4.0; every record carries the URL of its article. Changes: only
the opening paragraph is kept, whitespace is collapsed, it is cut at a sentence end, and empty pronunciation brackets are
removed. A term that mapped to a disambiguation page, to a wrong article, or to an article shared by two aliases was left
out rather than guessed.

## Answer-reader model (not in this repository)

`tools/fetch_qa_model.py` downloads **deepset/roberta-base-squad2** (Creative Commons Attribution 4.0, trained on SQuAD 2.0)
as an ONNX int8 build published by `onnx-community`, into `models_cache/` (git-ignored). It only reads text the device
already holds and returns a span of that text; it adds no facts of its own.
