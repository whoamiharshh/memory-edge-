"""Policy engine: deterministic, ordered gates that decide what happens to an episode (docs/RESEARCH.md G.8).
No model and no LLM takes part in this decision. Every gate reports a reason, pass or fail, and the UI shows
all of them.

  1 privacy       raw note / raw signal never leave; note shared only if opted in AND the redactor found nothing
  2 validation    fingerprint finite, right size, right fp_version; enums valid           -> REJECT
  3 duplicate     the same evidence (same event id), or the same repair logged on another
                  episode (same machine/fault/action/outcome/day), already queued or shared -> MERGE
  4 evidence      no action recorded / outcome still pending                              -> KEEP_LOCAL
  5 verification  sensor verdict must agree with the technician's outcome                 -> KEEP_LOCAL
  6 human         technician confirmed the outcome and the fault class                    -> KEEP_LOCAL
  7 share         structured fields + fingerprint (+ redacted note)                       -> SHARE
Retention (separate): closed, decided and older than ARCHIVE_AFTER_DAYS                   -> ARCHIVE
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import ValidationError

from edge.fingerprint import DIM as FP_DIM, FP_VERSION
from shared import ids
from shared.redact import Redaction
from shared.schema import ActionCode, FaultClass, ShareEvent

ARCHIVE_AFTER_DAYS = 90


@dataclass
class Reason:
    gate: str
    ok: bool
    detail: str


@dataclass
class Decision:
    action: str                                   # SHARE | KEEP_LOCAL | MERGE | REJECT
    reasons: list[Reason] = field(default_factory=list)
    event: dict[str, Any] | None = None
    note_shared: bool = False

    def to_dict(self) -> dict:
        return {"action": self.action, "reasons": [asdict(r) for r in self.reasons],
                "event_id": self.event["event_id"] if self.event else None, "note_shared": self.note_shared}


def decide(ep: dict[str, Any], *, fingerprint: list[float], redaction: Redaction | None, device_id: str,
           machine_class: str, already_queued: set[str], already_repairs: dict[str, str] | None = None,
           fp_version: str = FP_VERSION) -> Decision:
    """already_queued: event ids this device already queued/shared (other episodes' included).
    already_repairs: repair_hash -> episode id, for events already queued/shared from OTHER episodes."""
    d = Decision("KEEP_LOCAL")
    R = d.reasons.append

    # 1 privacy -----------------------------------------------------------------------------------------
    note = (ep.get("note_text") or "").strip()
    share_note = False
    if not note:
        R(Reason("privacy", True, "no note written; raw signal is never shared"))
    elif not ep.get("note_share_opt_in"):
        R(Reason("privacy", True, "note kept local: technician did not opt in to share it"))
    elif redaction is None or not redaction.clean:
        R(Reason("privacy", True, f"note kept local: redactor found {redaction.summary() if redaction else 'unscanned text'}"))
    else:
        share_note = True
        R(Reason("privacy", True, "note may be shared: opted in and redactor found no personal data"))

    # 2 validation --------------------------------------------------------------------------------------
    problems = []
    if len(fingerprint) != FP_DIM or not all(math.isfinite(x) for x in fingerprint):
        problems.append("fingerprint missing, wrong size or not finite")
    if ep.get("fp_version") != fp_version:
        problems.append(f"fingerprint version {ep.get('fp_version')} != this device's {fp_version}")
    if ep.get("action_code") and ep["action_code"] not in ActionCode.__members__:
        problems.append(f"unknown action code {ep['action_code']!r}")
    if ep.get("fault_class") and ep["fault_class"] not in FaultClass.__members__:
        problems.append(f"unknown fault class {ep['fault_class']!r}")
    if problems:
        R(Reason("validation", False, "; ".join(problems)))
        d.action = "REJECT"
        return d
    R(Reason("validation", True, "schema, sizes, enums and numbers valid"))

    # 3 duplicate ---------------------------------------------------------------------------------------
    action, outcome = ep.get("action_code"), ep.get("outcome", "pending")
    eid = ids.event_id(device_id, ep["episode_id"], outcome, action) if action and outcome != "pending" else None
    if eid and eid in already_queued:
        R(Reason("duplicate", False, f"identical evidence already queued/shared as event {eid[:8]}"))
        d.action = "MERGE"
        return d
    rh = (ids.repair_hash(ep.get("machine_id"), ep.get("fault_class"), action, outcome,
                          ep.get("action_at") or ep.get("first_seen")) if eid and ep.get("fault_class") else None)
    if rh and rh in (already_repairs or {}):
        R(Reason("duplicate", False, f"the same repair (machine, fault, action, outcome, day) was already shared from "
                                     f"episode {already_repairs[rh][:8]}; not counted twice"))
        d.action = "MERGE"
        return d
    R(Reason("duplicate", True, "no identical evidence shared before"))

    # 4 evidence ----------------------------------------------------------------------------------------
    if not action:
        R(Reason("evidence", False, "awaiting action: no intervention recorded yet"))
        return d
    if outcome == "pending":
        R(Reason("evidence", False, f"action '{action}' recorded; awaiting outcome"))
        return d
    R(Reason("evidence", True, f"action '{action}' with outcome '{outcome}'"))

    # 5 verification ------------------------------------------------------------------------------------
    v = ep.get("verify") or {}
    verdict, k, n = v.get("verdict", "verifying"), v.get("consecutive_ok", 0), v.get("required", 0)
    if verdict == "verifying":
        R(Reason("verification", False, f"awaiting verification: {k}/{n} consecutive healthy, "
                                        f"{v.get('consecutive_bad', 0)}/{n} consecutive abnormal windows"))
        return d
    expected = "symptom_resolved" if outcome == "worked" else "symptom_persists"
    if verdict != expected:
        R(Reason("verification", False, f"conflict: technician says '{outcome}' but sensor verdict is "
                                        f"'{verdict.replace('_', ' ')}'; kept local for review"))
        return d
    detail = (f"symptom resolved for {v.get('consecutive_ok')} consecutive windows (not a root-cause proof)"
              if outcome == "worked" else f"symptom persisted for {v.get('consecutive_bad')} consecutive windows")
    R(Reason("verification", True, detail))

    # 6 human -------------------------------------------------------------------------------------------
    if not ep.get("technician_confirmed"):
        R(Reason("human", False, "awaiting technician confirmation of the outcome"))
        return d
    if not ep.get("fault_class") or ep.get("fault_class_source") != "technician":
        R(Reason("human", False, "fault class is only a physics hint; technician must confirm it"))
        return d
    R(Reason("human", True, f"technician confirmed outcome and fault class '{ep['fault_class']}'"))

    # 7 share -------------------------------------------------------------------------------------------
    body = dict(event_id=eid, episode_id=ep["episode_id"], machine_class=machine_class, component=ep["component"],
                fault_class=ep["fault_class"], fault_class_source="technician", action_code=action, outcome=outcome,
                root_cause_claim=ep.get("root_cause_claim") or None, machine_verified=True,
                verify_windows_ok=int(v.get("consecutive_ok", 0) if outcome == "worked" else v.get("consecutive_bad", 0)),
                verify_windows_required=int(n), technician_confirmed=True, fingerprint=[float(x) for x in fingerprint],
                note_redacted=(redaction.text if share_note and redaction else None),
                occurred_at=ep.get("action_at") or ep.get("first_seen"), content_hash=rh, fp_version=fp_version)
    try:
        d.event = ShareEvent(**body).model_dump(mode="json")
    except ValidationError as e:
        R(Reason("share", False, f"event failed wire validation: {e.errors()[0]['msg']}"))
        d.action = "REJECT"
        return d
    d.note_shared = share_note
    R(Reason("share", True, "SHARE: structured outcome + fingerprint" + (" + redacted note" if share_note else "")))
    d.action = "SHARE"
    return d


def retention(ep: dict[str, Any], now: dt.datetime) -> str | None:
    """ARCHIVE closed, decided episodes after ARCHIVE_AFTER_DAYS. Never auto-archives open or undecided ones."""
    if ep.get("status") != "closed" or ep.get("outcome", "pending") == "pending":
        return None
    last = dt.datetime.fromisoformat(ep.get("last_seen") or ep["first_seen"])
    return "ARCHIVE" if (now - last).days >= ARCHIVE_AFTER_DAYS else None
