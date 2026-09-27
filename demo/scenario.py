"""Scripted end-to-end demo against the RUNNING system (demo/run_demo.ps1): real Qdrant Server, real bge-small
embeddings, real local LLM, real CWRU recordings. Every step asserts what it claims and prints the evidence.

  .venv\\Scripts\\python.exe -m demo.scenario
Exit code 0 = every step passed.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOT = json.loads((ROOT / "runtime" / "cloud" / "bootstrap.json").read_text())
A = httpx.Client(base_url="http://127.0.0.1:8101", headers={"X-Operator-Token": "operator-devA"}, timeout=120)
B = httpx.Client(base_url="http://127.0.0.1:8102", headers={"X-Operator-Token": "operator-devB"}, timeout=120)
CLOUD = httpx.Client(base_url="http://127.0.0.1:8100", headers={"Authorization": f"Bearer {BOOT['admin']}"}, timeout=60)
STEP = [0]


def step(title: str) -> None:
    STEP[0] += 1
    print(f"\n=== {STEP[0]}. {title}")


def ok(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        sys.exit(1)


def call(c: httpx.Client, method: str, path: str, body=None):
    r = c.request(method, path, json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text}")
    return r.json()


def play(c: httpx.Client, fid: int, n: int, start: int = 0) -> None:
    call(c, "POST", "/api/replay", {"fid": fid, "n": n, "interval": 0.0, "start": start})
    while call(c, "GET", "/api/stats")["replay"]["playing"]:
        time.sleep(0.2)


def latest(c: httpx.Client) -> dict:
    return call(c, "GET", "/api/episodes")[0]


def main() -> None:
    t0 = time.time()
    step("Device A goes OFFLINE; healthy data then an inner-race fault (CWRU 105) is replayed")
    call(A, "POST", "/api/network", {"online": False})
    call(B, "POST", "/api/network", {"online": False})
    before = call(A, "GET", "/api/stats")["windows"].get("new", 0)
    play(A, 99, 30)
    s = call(A, "GET", "/api/stats")
    ok(s["windows"].get("normal", 0) >= 30, f"30 unseen healthy windows -> all 'normal' (gate p50 {s['gate_ms']['p50']} ms)")
    play(A, 105, 40)
    ep = latest(A)
    ok(call(A, "GET", "/api/stats")["windows"].get("new", 0) == before + 1 and ep["occurrences"] == 40,
       f"one new episode #{ep['seq']} opened, 40 windows merged into it; physics hint = {ep['fault_hint']['fault_class']}")
    res = call(A, "POST", "/api/search", {"episode_id": ep["episode_id"]})
    ok(res["fleet"] == [], "search: no fleet evidence yet (mirror empty) -> no invented answer")

    step("Technician: note with a person's name, confirms fault class, records the action")
    eid = ep["episode_id"]
    call(A, "POST", f"/api/episodes/{eid}/note", {"text": "Inner race spall found on DE bearing. Replaced bearing with Ravi, regreased.", "share_opt_in": True})
    call(A, "POST", f"/api/episodes/{eid}/fault_class", {"fault_class": "inner_race"})
    d = call(A, "POST", f"/api/episodes/{eid}/action", {"action_code": "replace_bearing", "root_cause": "fatigue_wear", "required_windows": 20})
    ok(d["action"] == "KEEP_LOCAL", "decision KEEP_LOCAL: " + d["reasons"][-1]["detail"])

    step("Post-repair signal (unseen healthy recording, CWRU 100) -> the machine verifies the fix")
    play(A, 100, 25)
    call(A, "POST", f"/api/episodes/{eid}/confirm", {"outcome": "worked"})
    e = call(A, "GET", f"/api/episodes/{eid}")
    ok(e["verify"]["verdict"] == "symptom_resolved", f"sensor: symptom resolved for {e['verify']['consecutive_ok']} consecutive windows")
    ok(e["decision"]["action"] == "SHARE" and not e["decision"]["note_shared"],
       "policy SHARE; note kept local: " + e["decision"]["reasons"][0]["detail"])
    ok(e["share_state"] == "queued", "outbox: QUEUED while offline")

    step("Device C-style second report is simulated on Device A: a lubrication attempt that FAILED (fault persists)")
    play(A, 106, 30)
    e2 = latest(A)
    call(A, "POST", f"/api/episodes/{e2['episode_id']}/fault_class", {"fault_class": "inner_race"})
    call(A, "POST", f"/api/episodes/{e2['episode_id']}/action", {"action_code": "lubricate", "required_windows": 20})
    play(A, 106, 25, start=30)                   # file 106 has ~58 windows; 30.. are not yet replayed
    call(A, "POST", f"/api/episodes/{e2['episode_id']}/confirm", {"outcome": "failed"})
    e2 = call(A, "GET", f"/api/episodes/{e2['episode_id']}")
    ok(e2["verify"]["verdict"] == "symptom_persists" and e2["decision"]["action"] == "SHARE",
       "failed fix, machine-verified (symptom persisted) -> shared as FAILED evidence")

    step("Connectivity returns -> outbox drains; resend proves idempotency")
    call(A, "POST", "/api/network", {"online": True})
    r = call(A, "POST", "/api/sync/now")
    ok(r["push"].get("accepted") == 2, f"push: {r['push']}")
    call(A, "POST", f"/api/outbox/{e['event_id']}/resend")
    r = call(A, "POST", "/api/sync/now")
    ok(r["push"].get("duplicate") == 1, f"resend of the same event -> {r['push']} (counted once)")

    step("Cloud (Qdrant Server): evidence grouped by (component, fault class), disagreement kept")
    cases = call(CLOUD, "GET", "/v1/cases")
    c = next(x for x in cases if x["fault_class"] == "inner_race")
    tallies = {a["action_code"]: (a["worked"], a["failed"]) for a in c["actions"]}
    ok(tallies == {"replace_bearing": (1, 0), "lubricate": (0, 1)}, f"tallies {tallies}; flags {[f['kind'] for f in c['flags']]}")
    ok(all("Ravi" not in n["text"] for n in c["notes"]), "no raw note / name reached the cloud")

    step("Device B pulls the mirror, goes OFFLINE, meets a bearing A never saw (14-mil inner race, CWRU 169)")
    call(B, "POST", "/api/network", {"online": True})
    r = call(B, "POST", "/api/sync/now")
    ok(r["pull"]["pulled"] >= 1, f"mirror pull: {r['pull']}")
    call(B, "POST", "/api/network", {"online": False})
    play(B, 169, 30)
    eb = latest(B)
    res = call(B, "POST", "/api/search", {"episode_id": eb["episode_id"]})
    fleet = res["fleet"][0]["case"] if res["fleet"] else {}
    ok(bool(fleet) and fleet["fault_class"] == "inner_race",
       f"OFFLINE fleet evidence for {res['query']['fleet_filter']}: " +
       "; ".join(f"{a['action_code']} worked {a['worked']} failed {a['failed']}" for a in fleet.get("actions", [])) +
       f"  ({res['latency_ms']} ms)")

    step("Evidence brief on Device B (local LLM, offline, grounded)")
    b = call(B, "POST", "/api/brief", {"episode_id": eb["episode_id"]})
    ok(b["mode"] in ("llm", "template") and b["text"], f"[{b['mode']}, {b.get('latency_ms')} ms] {b['text']}")
    if b["dropped"]:
        print(f"        ({len(b['dropped'])} sentence(s) removed by the grounding/no-advice check)")

    step("Security: missing token / wrong role are refused (revocation: tests/security)")
    ok(httpx.post("http://127.0.0.1:8100/v1/sync/push", json={"batch_id": "x", "events": []}).status_code == 401, "push without token -> 401")
    ok(httpx.get("http://127.0.0.1:8101/api/episodes").status_code == 401, "device API without operator token -> 401")
    devB = httpx.Client(base_url="http://127.0.0.1:8100", headers={"Authorization": f"Bearer {BOOT['devices']['devB']['token']}"})
    ok(devB.post(f"/v1/events/{e['event_id']}/retract", json={"reason": "nope"}).status_code == 403, "device cannot retract evidence -> 403")
    call(B, "POST", "/api/network", {"online": True})
    print(f"\nALL STEPS PASSED in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
