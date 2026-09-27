"""Evidence aggregation for one case group (tenant, component, fault_class). Pure functions, no I/O.

A case is always recomputed from ALL of its events (never incremented in place), so replays, duplicates and
retractions can never double-count. Disagreement is kept and flagged, never resolved away:
  DISPUTED      the same action has both worked and failed reports
  COMPETING     reporters claim different root causes
  ALTERNATIVES  different actions have each worked (not a conflict: several fixes are known to hold)
Trust is shown as counts (worked / failed / distinct sites / machine-verified), never as a score.
"""
from __future__ import annotations

from typing import Any

import numpy as np

MAX_NOTES = 20


def aggregate(events: list[dict[str, Any]]) -> dict[str, Any]:
    live = [e for e in events if e.get("status") != "retracted"]
    retracted = len(events) - len(live)
    actions: dict[str, dict[str, Any]] = {}
    root_causes: dict[str, set] = {}
    sites = set()
    for e in live:
        a = actions.setdefault(e["action_code"], {"action_code": e["action_code"], "worked": 0, "failed": 0,
                                                  "sites_worked": set(), "sites_failed": set(),
                                                  "machine_verified": 0, "last_seen": ""})
        a[e["outcome"]] += 1
        a["sites_" + e["outcome"]].add(e["site_id"])
        a["machine_verified"] += int(bool(e.get("machine_verified")))
        a["last_seen"] = max(a["last_seen"], e.get("occurred_at", ""))
        sites.add(e["site_id"])
        rc = e.get("root_cause_claim")
        if rc and rc != "unknown":
            root_causes.setdefault(rc, set()).add(e["site_id"])
    flags = []
    for a in actions.values():
        if a["worked"] and a["failed"]:
            flags.append({"kind": "DISPUTED", "action_code": a["action_code"],
                          "detail": f"{a['action_code']}: worked at {len(a['sites_worked'])} site(s), "
                                    f"failed at {len(a['sites_failed'])} site(s)"})
    if len(root_causes) > 1:
        flags.append({"kind": "COMPETING", "detail": "different root causes claimed: " + ", ".join(sorted(root_causes))})
    worked = sorted(a["action_code"] for a in actions.values() if a["worked"])
    if len(worked) > 1:
        flags.append({"kind": "ALTERNATIVES", "detail": "several actions each worked: " + ", ".join(worked)})
    table = sorted(({**a, "sites_worked": sorted(a["sites_worked"]), "sites_failed": sorted(a["sites_failed"])}
                    for a in actions.values()), key=lambda a: (-a["worked"], a["failed"], a["action_code"]))
    notes = [{"event_id": e["event_id"], "site_id": e["site_id"], "action_code": e["action_code"],
              "outcome": e["outcome"], "text": e["note_redacted"]}
             for e in sorted(live, key=lambda e: e.get("occurred_at", ""), reverse=True) if e.get("note_redacted")][:MAX_NOTES]
    fps = [e["fingerprint"] for e in live if e.get("fingerprint")]
    return {
        "actions": table, "flags": flags,
        "root_causes": {k: sorted(v) for k, v in sorted(root_causes.items())},
        "n_events": len(live), "n_retracted": retracted, "n_sites": len(sites), "sites": sorted(sites),
        "notes": notes, "status": "active" if live else "retracted",
        "last_seen": max((e.get("occurred_at", "") for e in live), default=""),
        "centroid": np.mean(fps, axis=0).tolist() if fps else None,
    }


def summary_text(component: str, fault_class: str, agg: dict[str, Any]) -> str:
    """Plain-text rendering used for the case's BM25 + dense text vectors (and readable in the UI)."""
    parts = [f"{component} {fault_class.replace('_', ' ')} fault."]
    for a in agg["actions"]:
        parts.append(f"{a['action_code'].replace('_', ' ')} worked at {len(a['sites_worked'])} site(s), "
                     f"failed at {len(a['sites_failed'])} site(s).")
    for f in agg["flags"]:
        parts.append(f"{f['kind'].lower()}: {f['detail']}.")
    for n in agg["notes"][:5]:
        parts.append(n["text"])
    return " ".join(parts)
