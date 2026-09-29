"""Shared free-text knowledge: one device publishes, chosen devices pull it and then search it offline.

This is a second, separate channel from the outcome-verified evidence in cloud/ingest.py. Evidence is about a
machine and is only allowed out after a sensor confirmed a fix held. Knowledge is text a person deliberately
publishes for other people — a village register, a site procedure, an opening-hours notice — and it carries no
sensor claim, so it must never be mixed into the case groups or their tallies.

Addressing is enforced on the server, not on the device:

    audience="everyone"  every device in the tenant may pull it
    audience="device"    only the devices named in `recipients` may pull it
    audience="site"      only devices whose token carries one of the named site ids may pull it

The filter is applied to the query, so a device is never sent a record it may not read. Doing this on the
device instead would be theatre: a local copy can be read by anyone holding the device.
"""
from __future__ import annotations

import time
import uuid

from qdrant_client import models as m

from edge.store_edge import BM25_AVG_LEN, bm25_sparse
from cloud.store_server import CloudStore, NOTE_DIM, _cname

MAX_TEXT = 4000
MAX_RECIPIENTS = 200
AUDIENCES = ("everyone", "device", "site")


def ensure(store: CloudStore, tenant: str) -> str:
    """Create the tenant's knowledge collection on first use. Safe to call on every request."""
    name = _cname("knowledge", tenant)
    if not store.client.collection_exists(name):
        store.client.create_collection(
            name,
            vectors_config={"note": m.VectorParams(size=NOTE_DIM, distance=m.Distance.COSINE)},
            sparse_vectors_config={"note_bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)})
        for k in ("audience", "topic", "status", "author_device", "recipients"):
            store.client.create_payload_index(name, k, m.PayloadSchemaType.KEYWORD)
        store.client.create_payload_index(name, "seq", m.PayloadSchemaType.INTEGER)
    return name


def _next_seq(store: CloudStore, name: str) -> int:
    pts, _ = store.client.scroll(name, limit=1, with_payload=["seq"],
                                 order_by=m.OrderBy(key="seq", direction=m.Direction.DESC))
    return (int(pts[0].payload["seq"]) if pts else 0) + 1


def publish(store: CloudStore, embedder, tenant: str, author_device: str, *, text: str,
            topic: str = "general", audience: str = "everyone",
            recipients: list[str] | None = None, record_id: str | None = None) -> dict:
    """Publish one text record. Re-publishing the same record_id replaces it (an edit), which is why the
    id is supplied by the caller rather than generated here."""
    text = (text or "").strip()
    if not 1 <= len(text) <= MAX_TEXT:
        raise ValueError(f"text must be 1..{MAX_TEXT} characters")
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}")
    recipients = [r for r in (recipients or []) if r][:MAX_RECIPIENTS]
    if audience in ("device", "site") and not recipients:
        raise ValueError(f"audience={audience!r} needs at least one recipient")
    if audience == "everyone":
        recipients = []

    name = ensure(store, tenant)
    rid = record_id or str(uuid.uuid4())
    payload = {"text": text, "topic": (topic or "general")[:80], "audience": audience,
               "recipients": recipients, "author_device": author_device, "status": "active",
               "seq": _next_seq(store, name),
               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    idx, val = bm25_sparse(text, BM25_AVG_LEN)
    store.client.upsert(name, [m.PointStruct(id=rid, payload=payload, vector={
        "note": embedder.embed_documents([text])[0],
        "note_bm25": m.SparseVector(indices=idx, values=val)})], wait=True)
    return {"id": rid} | payload


def withdraw(store: CloudStore, tenant: str, record_id: str, author_device: str) -> bool:
    """Mark a record withdrawn so it stops being handed out. Only its author may withdraw it.
    Devices that already pulled it keep their copy: a publish cannot be un-rung, and promising
    otherwise would be false."""
    name = ensure(store, tenant)
    got = store.client.retrieve(name, [record_id], with_payload=True)
    if not got or got[0].payload.get("author_device") != author_device:
        return False
    # the text is blanked but the record is kept, and kept visible to its original recipients: it is the
    # tombstone that tells their next pull to delete their copy. Dropping the row instead would leave every
    # device that already pulled it answering from it forever.
    store.client.set_payload(name, {"status": "withdrawn", "text": "", "seq": _next_seq(store, name)},
                             points=[record_id], wait=True)
    return True


def _visible_to(device_id: str, site_id: str) -> m.Filter:
    """Records this caller is allowed to read, including withdrawn ones — a withdrawn row carries no text
    and exists only so the recipient's next pull deletes its local copy. Applied to every query, never
    client-side."""
    return m.Filter(
        should=[
            m.FieldCondition(key="audience", match=m.MatchValue(value="everyone")),
            m.Filter(must=[m.FieldCondition(key="audience", match=m.MatchValue(value="device")),
                           m.FieldCondition(key="recipients", match=m.MatchValue(value=device_id))]),
            m.Filter(must=[m.FieldCondition(key="audience", match=m.MatchValue(value="site")),
                           m.FieldCondition(key="recipients", match=m.MatchValue(value=site_id))]),
            m.FieldCondition(key="author_device", match=m.MatchValue(value=device_id)),
        ])


def fetch(store: CloudStore, tenant: str, device_id: str, site_id: str,
          since: int = 0, limit: int = 200) -> dict:
    """Everything this caller may read with seq > since, oldest first, for an incremental device pull."""
    name = ensure(store, tenant)
    flt = _visible_to(device_id, site_id)
    flt.must = [m.FieldCondition(key="seq", range=m.Range(gt=since))]
    limit = max(1, min(limit, 500))
    pts, _ = store.client.scroll(name, scroll_filter=flt, limit=limit + 1, with_payload=True,
                                 order_by=m.OrderBy(key="seq", direction=m.Direction.ASC))
    rows = [{"id": str(p.id)} | dict(p.payload) for p in pts[:limit]]
    return {"items": rows, "more": len(pts) > limit,
            "seq": max([r["seq"] for r in rows], default=since)}


def mine(store: CloudStore, tenant: str, author_device: str, limit: int = 200) -> list[dict]:
    """What this device has published, newest first, so an author can review and withdraw."""
    name = ensure(store, tenant)
    pts, _ = store.client.scroll(
        name, limit=max(1, min(limit, 500)), with_payload=True,
        scroll_filter=m.Filter(must=[m.FieldCondition(key="author_device",
                                                      match=m.MatchValue(value=author_device))]),
        order_by=m.OrderBy(key="seq", direction=m.Direction.DESC))
    return [{"id": str(p.id)} | dict(p.payload) for p in pts]
