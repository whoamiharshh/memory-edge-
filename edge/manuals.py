"""The machine's own manuals, searchable OFFLINE next to every episode (the user's "take data from the manufacturer's
manual" request, answered with retrieval, not with invented rules).

A PDF uploaded by an operator is split into page-numbered text chunks, embedded with the device's text model and
indexed in the device's own Qdrant Edge shard (dense + BM25, fused in one query, like episodes). Search returns the
manual title and PAGE of each passage, so the technician reads the manufacturer's own words, cited - the device never
paraphrases a manual into an instruction.

Limits (untrusted files): <= 20 MB, <= 400 pages, <= 2,000 chunks; text only (no scripts, no images). Manuals are the
manufacturer's documents: they stay on this device and are never shared with the fleet.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
from typing import Any

from edge.store_edge import StorePoint
from shared import ids

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 400
MAX_CHUNKS = 2000
CHUNK = 700                 # characters per passage (about a paragraph or two)
OVERLAP = 120
KV = "manuals"


def extract_pages(pdf: bytes) -> list[str]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    if len(pdf) > MAX_BYTES:
        raise ValueError(f"manual too large (max {MAX_BYTES // 1024 // 1024} MB)")
    if not pdf.startswith(b"%PDF"):
        raise ValueError("not a PDF file")
    try:
        r = PdfReader(io.BytesIO(pdf))
        if r.is_encrypted:
            raise ValueError("encrypted PDF: export an unprotected copy of the manual")
        if len(r.pages) > MAX_PAGES:
            raise ValueError(f"manual has {len(r.pages)} pages (max {MAX_PAGES}): upload the relevant chapters")
        return [(p.extract_text() or "") for p in r.pages]
    except PdfReadError as e:
        raise ValueError(f"unreadable PDF ({e})"[:200])


def chunks(pages: list[str]) -> list[tuple[int, str]]:
    out = []
    for no, text in enumerate(pages, 1):
        t = " ".join(text.split())
        i = 0
        while i < len(t):
            piece = t[i:i + CHUNK]
            if len(piece.strip()) >= 40:
                out.append((no, piece))
            i += CHUNK - OVERLAP
    if len(out) > MAX_CHUNKS:
        raise ValueError(f"manual too long ({len(out)} passages, max {MAX_CHUNKS}): upload the relevant chapters")
    return out


def add(device, title: str, pdf: bytes, source: str = "") -> dict[str, Any]:
    title, source = title.strip()[:160], source.strip()[:300]
    if len(title) < 3:
        raise ValueError("give the manual a title (e.g. 'XYZ motor operating manual rev 3')")
    sha = hashlib.sha256(pdf).hexdigest()
    docs = device.outbox.kv_get(KV, [])
    if any(d["sha256"] == sha for d in docs):
        raise ValueError("this manual is already indexed")
    pages = extract_pages(pdf)
    parts = chunks(pages)
    if not parts:
        raise ValueError("no text found (a scanned manual needs OCR first)")
    doc_id = ids.make_id("manual", device.cfg.device_id, sha)
    for s in range(0, len(parts), 64):
        batch = parts[s:s + 64]
        vecs = device.embedder.embed_documents([t for _, t in batch])
        device._upsert([StorePoint(ids.make_id("manual-chunk", doc_id, s + k),
                                   {"type": "manual", "machine_id": device.cfg.machine_id, "doc_id": doc_id,
                                    "title": title, "page": page, "text": text},
                                   note=vec, bm25_text=f"{title} {text}")
                        for k, ((page, text), vec) in enumerate(zip(batch, vecs))])
    doc = {"doc_id": doc_id, "title": title, "source": source, "sha256": sha, "pages": len(pages),
           "passages": len(parts), "added_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    device.outbox.kv_set(KV, [*docs, doc])
    device.outbox.log("manual", f"manual indexed offline: '{title}' ({len(pages)} pages, {len(parts)} passages)")
    return doc


def listing(device) -> list[dict]:
    return device.outbox.kv_get(KV, [])


def remove(device, doc_id: str) -> bool:
    docs = device.outbox.kv_get(KV, [])
    keep = [d for d in docs if d["doc_id"] != doc_id]
    if len(keep) == len(docs):
        return False
    with device._lock:
        device.store.delete_where({"type": "manual", "doc_id": doc_id})
    device.outbox.kv_set(KV, keep)
    device.outbox.log("manual", f"manual removed ({doc_id[:8]})")
    return True


def search(device, text: str, limit: int = 5) -> list[dict]:
    if not listing(device):
        return []
    hits = device.store.search(note=device.embedder.embed_query(text), text=text, limit=limit,
                               filter={"type": "manual", "machine_id": device.cfg.machine_id})
    return [{"doc_id": h.payload["doc_id"], "title": h.payload["title"], "page": h.payload["page"],
             "text": h.payload["text"], "rrf": round(h.score, 4)} for h in hits]


def query_for(episode: dict) -> str:
    """What to look up in the manual for an episode: component + fault class + action words."""
    fc = episode.get("fault_class") or (episode.get("fault_hint") or {}).get("fault_class") or ""
    words = {"inner_race": "bearing inner race damage replacement", "outer_race": "bearing outer race damage replacement",
             "ball": "bearing rolling element damage replacement", "cage": "bearing cage damage",
             "imbalance": "balancing unbalance vibration", "misalignment": "shaft alignment coupling",
             "looseness": "mounting bolts torque foundation", "overheating": "lubrication grease temperature",
             "wear": "lubrication wear inspection"}.get(fc, fc.replace("_", " "))
    return f"{episode.get('component', '')} {words} maintenance".strip()
