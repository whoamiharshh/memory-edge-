"""The multi-turn conversations, run against the REAL running app (UI gateway :9000 -> device -> retrieval -> answer).

  (start the app first)   .venv\\Scripts\\python.exe -m bench.conversations

Each conversation is played exactly as the UI plays it: the first message carries no session id, the device names the
conversation in its reply, and every later message sends that id back. Each turn is checked on what the device understood
(message type, the question it resolved to, the topic) and on how it answered (mode, source), and the whole transcript is
printed so a person can read the answers. Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import httpx

BASE = "http://127.0.0.1:9000/proxy/device/"
ROOT = pathlib.Path(__file__).resolve().parent.parent


class Chat:
    def __init__(self, client: httpx.Client, name: str):
        self.c, self.name, self.session, self.turns = client, name, None, []

    def say(self, message: str, **expect) -> dict:
        if self.session is None:               # what the UI's "New conversation" button does before the first message
            self.c.delete(BASE + "memory/chat")
        t = time.perf_counter()
        r = self.c.post(BASE + "ask", json={"text": message, "use_fleet": True, "limit": 5, "session": self.session}).json()
        ms = round((time.perf_counter() - t) * 1000)
        self.session = r.get("session") or self.session
        tr = r.get("trace") or {}
        problems = []
        if "kind" in expect and tr.get("message_type") != expect["kind"]:
            problems.append(f"message_type {tr.get('message_type')!r} != {expect['kind']!r}")
        if "resolved_has" in expect:
            for w in ([expect["resolved_has"]] if isinstance(expect["resolved_has"], str) else expect["resolved_has"]):
                if w.lower() not in (tr.get("resolved") or "").lower():
                    problems.append(f"resolved question {tr.get('resolved')!r} lacks {w!r}")
        if "resolved_lacks" in expect and expect["resolved_lacks"].lower() in (tr.get("resolved") or "").lower():
            problems.append(f"resolved question {tr.get('resolved')!r} still has {expect['resolved_lacks']!r}")
        if "mode" in expect and r.get("mode") not in ([expect["mode"]] if isinstance(expect["mode"], str) else expect["mode"]):
            problems.append(f"mode {r.get('mode')!r} not in {expect['mode']!r}")
        if expect.get("no_retrieval") and (r.get("used") or r.get("retrieval")):
            problems.append("casual message went through retrieval")
        if expect.get("never_no_evidence") and r.get("mode") == "no_evidence":
            problems.append("answered 'no evidence' to a conversational message")
        if "answer_has" in expect and expect["answer_has"].lower() not in (r.get("answer") or "").lower():
            problems.append(f"answer lacks {expect['answer_has']!r}")
        if "source" in expect and expect["source"] not in (r.get("sources") or []):
            problems.append(f"sources {r.get('sources')} lack {expect['source']!r}")
        if expect.get("no_this_device_holds") and "this device holds" in (r.get("answer") or "").lower() \
                and "offline_kb" in (r.get("sources") or []):
            problems.append("library answer labelled as what the device holds")
        turn = {"message": message, "answer": r.get("answer"), "mode": r.get("mode"), "sources": r.get("sources"),
                "session": self.session, "ms": ms, "trace": {k: tr.get(k) for k in
                ("message_type", "original", "asked_as", "resolved", "topics", "context_turns", "selected")},
                "problems": problems}
        self.turns.append(turn)
        return turn


def main() -> int:
    c = httpx.Client(timeout=600)
    chats: dict[str, Chat] = {}

    def conv(name):
        chats[name] = Chat(c, name)
        return chats[name]

    k = conv("1  hi -> computers -> inputs -> outputs -> thanks")
    k.say("Hi", kind="casual", no_retrieval=True, never_no_evidence=True)
    k.say("Tell me about computers.", kind="knowledge", resolved_has="computer", source="offline_kb", no_this_device_holds=True)
    k.say("What are their inputs?", kind="followup", resolved_has=["computers", "inputs"])
    k.say("What are their outputs?", kind="followup", resolved_has=["computers", "outputs"])
    k.say("Thanks", kind="casual", no_retrieval=True, never_no_evidence=True)

    k = conv("2  hello -> DNA -> tell me about it -> does -> found")
    k.say("Hello", kind="casual", no_retrieval=True, never_no_evidence=True)
    k.say("What is DNA?", kind="knowledge", source="offline_kb", answer_has="genetic", no_this_device_holds=True)
    k.say("Tell me about it.", kind="followup", resolved_has="DNA", source="offline_kb")
    k.say("What does it do?", kind="followup", resolved_has="DNA")
    k.say("Where is it found?", kind="followup", resolved_has="DNA")

    k = conv("3  system -> inputs and outputs -> example")
    k.say("What is a system?", kind="knowledge", source="offline_kb")
    k.say("What are its inputs and outputs?", kind="followup", resolved_has=["system", "inputs", "outputs"])
    k.say("Give me an example.", kind="followup", resolved_has=["example", "system"])

    k = conv("4  CPU -> does -> parts -> example")
    k.say("What is a CPU?", kind="knowledge", source="offline_kb", answer_has="central processing unit")
    k.say("What does it do?", kind="followup", resolved_has="CPU")
    k.say("What are its main parts?", kind="followup", resolved_has=["CPU", "parts"])
    k.say("Give me an example.", kind="followup", resolved_has=["example", "CPU"])

    k = conv("5  topic switch DNA -> CPU -> it")
    k.say("What is DNA?", kind="knowledge", source="offline_kb")
    k.say("What is a CPU?", kind="knowledge", source="offline_kb")
    k.say("What does it do?", kind="followup", resolved_has="CPU", resolved_lacks="DNA")

    k = conv("6  topic switch CPU -> DNA -> it")
    k.say("What is a CPU?", kind="knowledge", source="offline_kb")
    k.say("What is DNA?", kind="knowledge", source="offline_kb")
    k.say("Where is it found?", kind="followup", resolved_has="DNA", resolved_lacks="CPU")

    k = conv("7  casual only")
    k.say("Hi", kind="casual", no_retrieval=True, never_no_evidence=True, answer_has="help")
    k.say("How are you?", kind="casual", no_retrieval=True, never_no_evidence=True)
    k.say("Thanks", kind="casual", no_retrieval=True, never_no_evidence=True)

    k = conv("8  same meaning, different wording (3 unrelated topics, 5 wordings each)")
    for topic in ("photosynthesis", "HTTP", "gravity"):
        titles = set()
        for w in ("What is {x}?", "Tell me about {x}.", "Explain {x}.", "Can you explain {x}?", "Give me an overview of {x}."):
            t = k.say(w.format(x=topic), kind="knowledge", mode=("quoted", "llm"), source="offline_kb")
            titles.add(((t["trace"]["selected"] or [{}])[0].get("title") or "").lower())
        if len(titles) != 1:
            k.turns[-1]["problems"].append(f"different wordings of {topic!r} selected different entries: {titles}")

    k = conv("9  standalone question after a topic does not inherit it")
    k.say("What is DNA?", kind="knowledge", source="offline_kb")
    k.say("What is the capital of Japan?", kind="knowledge", resolved_lacks="DNA", answer_has="Tokyo")

    k = conv("10 ambiguous reference is asked, not guessed")
    k.say("How is a CPU different from a GPU?", kind="knowledge")
    k.say("What does it do?", mode="clarify", answer_has="CPU")

    k = conv("11 isolation: two conversations at once")
    a, b = Chat(c, "A"), Chat(c, "B")
    a.say("What is DNA?", kind="knowledge")
    b.say("What is a CPU?", kind="knowledge")
    t = b.say("What does it do?", kind="followup", resolved_has="CPU", resolved_lacks="DNA")
    a.say("Where is it found?", kind="followup", resolved_has="DNA", resolved_lacks="CPU")
    if a.session == b.session:
        t["problems"].append("two conversations were given the same session id")
    k.turns += a.turns + b.turns

    k = conv("12 a brand-new conversation has nothing to refer to")
    k.say("What does it do?", mode="clarify")

    failed = 0
    for name, chat in chats.items():
        print(f"\n=== {name}")
        for t in chat.turns:
            mark = "FAIL" if t["problems"] else "ok  "
            failed += bool(t["problems"])
            print(f"  [{mark}] {t['message']!r}  ->  {t['trace']['message_type']}/{t['mode']}  resolved={t['trace']['resolved']!r}"
                  f"  {t['ms']} ms\n         {(t['answer'] or '')[:150]!r}")
            for p in t["problems"]:
                print("         !!", p)
    total = sum(len(ch.turns) for ch in chats.values())
    print(f"\n{total - failed}/{total} turns passed their checks")
    (ROOT / "bench" / "results").mkdir(exist_ok=True)
    (ROOT / "bench" / "results" / "conversations.json").write_text(
        json.dumps({n: ch.turns for n, ch in chats.items()}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
