"""Cloud ingest: validate each event on its own, store it idempotently, then recompute every touched case group
from all of its events under compare-and-set (retry on version conflict)."""
from __future__ import annotations

import datetime as dt
import logging

from pydantic import ValidationError

from cloud.auth import AuthContext
from cloud.store_server import CloudStore
from cloud.tally import aggregate, summary_text
from edge.fingerprint import DIM as FP_DIM
from shared import ids
from shared.embed import Embedder
from shared.schema import EventResult, PushRequest, PushResponse, ShareEvent

log = logging.getLogger("cloud.ingest")
CAS_RETRIES = 8


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def push(store: CloudStore, embedder: Embedder, ctx: AuthContext, req: PushRequest) -> PushResponse:
    results: list[EventResult] = []
    touched: dict[str, tuple[str, str]] = {}
    for raw in req.events:
        eid = str(raw.get("event_id", "?"))[:36] if isinstance(raw, dict) else "?"
        try:
            ev = ShareEvent(**raw)
        except (ValidationError, TypeError) as e:
            msg = e.errors()[0] if isinstance(e, ValidationError) else {"loc": (), "msg": str(e)}
            results.append(EventResult(event_id=eid, status="rejected",
                                       reason=f"schema: {'.'.join(map(str, msg['loc']))} {msg['msg']}"[:200]))
            continue
        expected = ids.event_id(ctx.device_id, ev.episode_id, ev.outcome, ev.action_code.value)
        if ev.event_id != expected:
            results.append(EventResult(event_id=eid, status="rejected",
                                       reason="event id does not belong to the authenticated device"))
            continue
        cid = ids.case_id(ctx.tenant_id, ev.component.value, ev.fault_class.value)
        payload = ev.model_dump(mode="json", exclude={"fingerprint"}) | {
            "case_id": cid, "site_id": ctx.site_id, "device_id": ctx.device_id, "status": "active",
            "received_at": _now()}
        status = store.insert_event(ctx.tenant_id, ev.event_id, payload, ev.fingerprint)
        results.append(EventResult(event_id=ev.event_id, status=status))
        if status == "accepted":
            touched[cid] = (ev.component.value, ev.fault_class.value)
    for cid, (component, fault_class) in touched.items():
        recompute_case(store, embedder, ctx.tenant_id, cid, component, fault_class)
    return PushResponse(batch_id=req.batch_id, results=results)


def recompute_case(store: CloudStore, embedder: Embedder, tenant: str, cid: str, component: str,
                   fault_class: str) -> bool:
    for _ in range(CAS_RETRIES):
        cur = store.get_case(tenant, cid)
        events = store.events(tenant, cid, with_vectors=True)
        agg = aggregate(events)
        text = summary_text(component, fault_class, agg)
        vib = agg.pop("centroid") or [0.0] * FP_DIM
        payload = {"type": "fleet_case", "case_id": cid, "component": component, "fault_class": fault_class,
                   "text": text, "updated_at": _now(), **agg}
        note = embedder.embed_documents([text])[0]
        if store.put_case(tenant, cid, payload, vib, note, cur["version"] if cur else None):
            return True
        log.info("case %s: version conflict, retrying", cid)
    log.error("case %s: gave up after %d CAS conflicts", cid, CAS_RETRIES)
    return False


def retract(store: CloudStore, embedder: Embedder, ctx: AuthContext, event_id: str, reason: str) -> dict:
    ev = store.get_event(ctx.tenant_id, event_id)
    if ev is None:
        raise KeyError(event_id)
    if ev.get("status") != "retracted":
        store.set_event_fields(ctx.tenant_id, event_id, {"status": "retracted", "retracted_by": ctx.device_id,
                                                         "retracted_at": _now(), "retract_reason": reason[:200]})
        recompute_case(store, embedder, ctx.tenant_id, ev["case_id"], ev["component"], ev["fault_class"])
    return store.get_event(ctx.tenant_id, event_id)
