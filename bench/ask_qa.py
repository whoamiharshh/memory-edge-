"""End-to-end accuracy and speed of general-knowledge Ask, through the REAL running app (library -> local model -> "I don't know").

  (start the app first)   .venv\\Scripts\\python.exe -m bench.ask_qa

Uses the same 45 known-answer questions and 13 unanswerable "trap" questions as bench/general_qa.py, but sends each one to
the gateway at http://127.0.0.1:9000, so it measures what a person sees: the shipped library, the title lookup, the
relevance rules, the model fallback and its self-consistency gate, and the latency.

  answered  = a sourced answer or an [Unverified] model answer was given
  correct   = it contains an accepted string for that question
  accuracy  = correct / answered              (the number that must be high: how often a given answer is right)
  coverage  = answered / known questions      (how often it dares to answer)
  traps     = trap questions that got an answer anyway (each one is an invented answer)
"""
from __future__ import annotations

import json
import pathlib
import statistics
import time

import httpx

from bench.general_qa import KNOWN, TRAPS

BASE = "http://127.0.0.1:9000/proxy/device/"
ROOT = pathlib.Path(__file__).resolve().parent.parent
ANSWER_MODES = ("quoted", "llm", "model_unverified")


def main() -> None:
    c = httpx.Client(timeout=300)
    rows = []
    for kind, items in (("known", KNOWN), ("trap", [(q, []) for q in TRAPS])):
        for q, keys in items:
            t = time.perf_counter()
            r = c.post(BASE + "ask", json={"text": q}).json()
            ms = (time.perf_counter() - t) * 1000
            answered = r.get("mode") in ANSWER_MODES
            rows.append({"q": q, "kind": kind, "mode": r.get("mode"), "sources": r.get("sources"), "ms": round(ms),
                         "answer": r["answer"][:200], "answered": answered,
                         "correct": bool(answered and kind == "known" and any(k in r["answer"].lower() for k in keys))})
    known = [r for r in rows if r["kind"] == "known"]
    traps = [r for r in rows if r["kind"] == "trap"]
    answered = [r for r in known if r["answered"]]
    correct = [r for r in answered if r["correct"]]
    invented = [r for r in traps if r["answered"]]
    wrong = [r for r in answered if not r["correct"]]
    # an invented answer to an unanswerable question is a wrong answer too, so it counts against accuracy
    total_answers = len(answered) + len(invented)
    by_mode: dict[str, list[int]] = {}
    for r in rows:
        by_mode.setdefault(str(r["mode"]), []).append(r["ms"])
    summary = {"answered": len(answered), "correct": len(correct), "wrong": len(wrong),
               "traps_answered": len(invented), "traps": len(traps),
               "accuracy_pct": round(100 * len(correct) / total_answers, 1) if total_answers else 0.0,
               "coverage_pct": round(100 * len(answered) / len(known), 1),
               "latency_ms_median_by_mode": {m: int(statistics.median(v)) for m, v in by_mode.items()},
               "latency_ms_max_by_mode": {m: max(v) for m, v in by_mode.items()}}
    for r in wrong + invented:
        print("WRONG/INVENTED |", r["q"], "->", r["answer"][:110].replace("\n", " "), f"[{r['mode']}]")
    print(json.dumps(summary, indent=1))
    (ROOT / "bench" / "results").mkdir(exist_ok=True)
    (ROOT / "bench" / "results" / "ask_qa.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
