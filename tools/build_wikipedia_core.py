"""Build the offline Wikipedia core that ships inside the repository.

Takes the first 30,000 articles of data/raw/simplewiki_leads.jsonl (written by tools/extract_simplewiki.py; the
file is sorted longest article first, so these are the best-developed ones) and writes, into knowledge/:

  simple_wikipedia_core.jsonl.gz   title, text (opening paragraph), url   (~4.5 MB)
  simple_wikipedia_core.vec.npz    the bge-small-en-v1.5 vector of each text, float16, plus the model name

The vectors are shipped so a new machine does not spend ~50 minutes embedding 30,000 passages on a CPU. They are
public Wikipedia text and nothing private; edge/seed.py uses them only when the device's own embedder is the same
model, and embeds the text itself otherwise.

Simple English Wikipedia is CC BY-SA 4.0: see knowledge/SIMPLE_WIKIPEDIA_LICENSE.md.
  .venv\\Scripts\\python.exe -m tools.build_wikipedia_core            (about 25 minutes on a laptop CPU)
  .venv\\Scripts\\python.exe -m tools.build_wikipedia_core --no-vectors
"""
from __future__ import annotations

import gzip
import itertools
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEADS = ROOT / "data" / "raw" / "simplewiki_leads.jsonl"
OUT_TEXT = ROOT / "knowledge" / "simple_wikipedia_core.jsonl.gz"
OUT_VEC = ROOT / "knowledge" / "simple_wikipedia_core.vec.npz"
CORE = 30_000
BATCH = 256


def main() -> None:
    rows = []
    with LEADS.open(encoding="utf-8") as fh:
        for line in itertools.islice(fh, CORE):
            a = json.loads(line)
            rows.append({"title": a["title"], "text": a["text"], "url": a["url"]})
    with gzip.open(OUT_TEXT, "wt", encoding="utf-8", compresslevel=9) as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} articles to {OUT_TEXT} ({OUT_TEXT.stat().st_size / 1e6:.1f} MB)", flush=True)
    if "--no-vectors" in sys.argv:
        return

    from shared.embed import load_embedder
    emb = load_embedder()
    texts = [f"{r['title']}: {r['text']}" for r in rows]          # exactly what edge/seed.py embeds
    out = np.zeros((len(texts), emb.dim), dtype=np.float16)
    t = time.time()
    for i in range(0, len(texts), BATCH):
        out[i:i + BATCH] = np.asarray(emb.embed_documents(texts[i:i + BATCH]), dtype=np.float32)
        if (i // BATCH) % 10 == 0:
            done = i + BATCH
            print(f"  {min(done, len(texts))}/{len(texts)}  {done / max(time.time() - t, 1e-9):.1f} docs/s", flush=True)
    np.savez_compressed(OUT_VEC, vecs=out, model=np.array(emb.name))
    print(f"wrote {OUT_VEC} ({OUT_VEC.stat().st_size / 1e6:.1f} MB), model {emb.name}", flush=True)


if __name__ == "__main__":
    main()
