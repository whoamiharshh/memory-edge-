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
from shared.schema import EventResult, FollowUp, PushRequest, PushResponse, ShareEvent

log = logging.getLogger("cloud.ingest")
CAS_RETRIES = 8


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


CLOCK_TOLERANCE_S = 120.0     # device clocks within 2 minutes of the cloud are trusted as they are


def clock_offset(device_time: str | None) -> float | None:
    """Device clock minus cloud clock, in seconds (from the X-Device-Time header of this request), or None."""
    if not device_time:
        return None
    try:
        t = dt.datetime.fromisoformat(device_time)
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return (t - dt.datetime.now(dt.timezone.utc)).total_seconds()


def corrected(ts: str, offset: float | None) -> str:
    """A device timestamp on the cloud's clock (unchanged when the offset is within tolerance or unknown)."""
    if offset is None or abs(offset) <= CLOCK_TOLERANCE_S:
        return ts
    t = dt.datetime.fromisoformat(ts)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return (t - dt.timedelta(seconds=offset)).isoformat(timespec="seconds")


def push(store: CloudStore, embedder: Embedder, ctx: AuthContext, req: PushRequest, defer=None,
         device_time: str | None = None) -> PushResponse:
    """`defer(tenant, case_id, component, fault_class)`: hand the case recomputation to cloud/recompute.py instead of
    doing it inside the request (the events are already durable when acknowledged).
    `device_time`: the device's clock at sending time; if it is off by more than CLOCK_TOLERANCE_S, every timestamp
    of this batch is shifted onto the cloud's clock (the device's own value is kept as occurred_at_device)."""
    offset = clock_offset(device_time)
    results: list[EventResult] = []
    touched: dict[str, tuple[str, str]] = {}
    batch: list[tuple[str, dict, list[float]]] = []
    where: dict[str, tuple[int, str, str, str]] = {}             # event id -> (result slot, case id, component, fault)
    for raw in req.events:
        eid = str(raw.get("event_id", "?"))[:36] if isinstance(raw, dict) else "?"
        if isinstance(raw, dict) and raw.get("kind") == "followup":
            res, case = _followup(store, ctx, raw, eid, offset)
            results.append(res)
            if case:
                touched[case[0]] = case[1:]
            continue
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
        occurred = corrected(ev.occurred_at, offset)
        why = implausible(ev, occurred)
        if why:
            results.append(EventResult(event_id=eid, status="rejected", reason=f"implausible: {why}"))
            continue
        cid = ids.case_id(ctx.tenant_id, ev.component.value, ev.fault_class.value)
        payload = ev.model_dump(mode="json", exclude={"fingerprint"}) | {
            "case_id": cid, "site_id": ctx.site_id, "device_id": ctx.device_id, "status": "active",
            "received_at": _now(), "occurred_at": occurred}
        if occurred != ev.occurred_at:
            payload |= {"occurred_at_device": ev.occurred_at, "clock_offset_s": round(offset, 1)}
        where.setdefault(ev.event_id, (len(results), cid, ev.component.value, ev.fault_class.value))
        results.append(EventResult(event_id=ev.event_id, status="duplicate"))   # set below
        batch.append((ev.event_id, payload, ev.fingerprint))
    statuses = store.insert_events(ctx.tenant_id, batch)
    for eid, (slot, cid, comp, fc) in where.items():
        results[slot] = EventResult(event_id=eid, status=statuses[eid])
        if statuses[eid] == "accepted":
            touched[cid] = (comp, fc)
    for cid, (component, fault_class) in touched.items():
        if defer is not None:
            defer(ctx.tenant_id, cid, component, fault_class)
        else:
            recompute_case(store, embedder, ctx.tenant_id, cid, component, fault_class)
    return PushResponse(batch_id=req.batch_id, results=results)


def implausible(ev: ShareEvent, occurred_at: str | None = None) -> str | None:
    """Cheap sanity checks a lying or broken device fails: the claimed verification must be complete, the time must
    be real and not in the future, a limit claim must be consistent."""
    if not ev.machine_verified or not ev.technician_confirmed:
        return "evidence must be sensor-verified and technician-confirmed"
    if ev.verify_windows_ok < ev.verify_windows_required:
        return f"verification shorter than required ({ev.verify_windows_ok} < {ev.verify_windows_required} windows)"
    try:
        t = dt.datetime.fromisoformat(occurred_at or ev.occurred_at)
    except ValueError:
        return "occurred_at is not an ISO timestamp"
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    if t > now + dt.timedelta(days=1) or t.year < 2000:
        return "occurred_at is in the future or before 2000"
    if ev.outcome == "worked" and ev.limit_mm_s and ev.severity_mm_s is not None and ev.severity_mm_s > ev.limit_mm_s:
        return "a fix reported as worked is above its own vibration limit"
    return None


def _followup(store: CloudStore, ctx: AuthContext, raw: dict, eid: str, offset: float | None = None):
    """A device reports whether ITS OWN shared fix held for the hold period or the fault recurred first. The first
    report is final; the original event keeps it and the case tallies count held / recurred per action."""
    try:
        fu = FollowUp(**raw)
    except (ValidationError, TypeError) as e:
        msg = e.errors()[0] if isinstance(e, ValidationError) else {"loc": (), "msg": str(e)}
        return EventResult(event_id=eid, status="rejected",
                           reason=f"schema: {'.'.join(map(str, msg['loc']))} {msg['msg']}"[:200]), None
    if fu.event_id != ids.followup_id(ctx.device_id, fu.refers_to):
        return EventResult(event_id=eid, status="rejected", reason="follow-up id does not belong to the authenticated device"), None
    orig = store.get_event(ctx.tenant_id, fu.refers_to)
    if orig is None or orig.get("device_id") != ctx.device_id:
        return EventResult(event_id=eid, status="rejected", reason="refers to no event of this device"), None
    if orig.get("outcome") != "worked":
        return EventResult(event_id=eid, status="rejected", reason="follow-ups apply to fixes reported as worked"), None
    if orig.get("followup"):
        return EventResult(event_id=eid, status="duplicate"), None
    store.set_event_fields(ctx.tenant_id, fu.refers_to, {"followup": {
        "status": fu.status, "days_after_fix": round(fu.days_after_fix, 2), "hold_days": fu.hold_days,
        "reported_at": corrected(fu.occurred_at, offset), "received_at": _now()}})
    return EventResult(event_id=eid, status="accepted"), (orig["case_id"], orig["component"], orig["fault_class"])


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


class SameApprover(PermissionError):
    """The admin who requested a retraction tried to approve it too."""


def retract(store: CloudStore, embedder: Embedder, ctx: AuthContext, event_id: str, reason: str,
            approvals: int = 2) -> dict:
    """Two-person rule: the first admin's call RECORDS a retraction request; a DIFFERENT admin's call carries it out
    (tombstone, tallies recomputed). approvals=1 keeps the single-admin behaviour for one-person deployments."""
    ev = store.get_event(ctx.tenant_id, event_id)
    if ev is None:
        raise KeyError(event_id)
    if ev.get("status") == "retracted":
        return ev
    req = ev.get("retraction_request")
    if approvals >= 2 and req is None:
        store.set_event_fields(ctx.tenant_id, event_id, {"retraction_request": {
            "by": ctx.device_id, "reason": reason[:200], "at": _now()}})
        return store.get_event(ctx.tenant_id, event_id)
    if approvals >= 2 and req["by"] == ctx.device_id:
        raise SameApprover("a second, different admin must approve this retraction (two-person rule)")
    by = [req["by"], ctx.device_id] if req else [ctx.device_id]
    store.set_event_fields(ctx.tenant_id, event_id, {"status": "retracted", "retracted_by": by, "retracted_at": _now(),
                                                     "retract_reason": (req or {}).get("reason", reason)[:200],
                                                     "retraction_request": None})
    recompute_case(store, embedder, ctx.tenant_id, ev["case_id"], ev["component"], ev["fault_class"])
    return store.get_event(ctx.tenant_id, event_id)


def quarantine(store: CloudStore, embedder: Embedder, tenant: str, device_id: str, on: bool, reason: str = "") -> dict:
    """Pull (on=True) or restore (on=False) ALL evidence of one device at once - the answer to a lying or compromised
    device. Reversible; retracted evidence stays retracted. Tallies of every touched case are recomputed."""
    touched, n = {}, 0
    for e in store.events(tenant):
        if e.get("device_id") != device_id:
            continue
        if on and e.get("status", "active") == "active":
            store.set_event_fields(tenant, e["event_id"], {"status": "quarantined", "quarantine_reason": reason[:200],
                                                           "quarantined_at": _now()})
        elif not on and e.get("status") == "quarantined":
            store.set_event_fields(tenant, e["event_id"], {"status": "active", "quarantine_reason": None})
        else:
            continue
        n += 1
        touched[e["case_id"]] = (e["component"], e["fault_class"])
    for cid, (component, fc) in touched.items():
        recompute_case(store, embedder, tenant, cid, component, fc)
    return {"device_id": device_id, "quarantined" if on else "restored": n, "cases_recomputed": len(touched)}
