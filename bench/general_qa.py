"""Measure how often a local model gives a RIGHT, WRONG or "I don't know" answer to simple general-knowledge questions,
through the same gate the device uses (edge.rag.general_answer: greedy + 2 sampled runs that must agree).

  .venv\\Scripts\\python.exe -m bench.general_qa models_cache\\llm\\<file>.gguf [more.gguf ...]

45 questions have a known answer (an answer is right if it contains one of the accepted strings). 13 "trap" questions
have no knowable answer (a future event, a private fact, an invented title): any answer to those is a hallucination,
so the only correct reply is "I don't know". The primary number is WRONG (fewer is better), then RIGHT, then speed.
The answer keys are common knowledge written by hand; they are not taken from any model.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

from edge import rag

ROOT = pathlib.Path(__file__).resolve().parent.parent

KNOWN = [
    ("Who wrote Pride and Prejudice?", ["austen"]), ("What is the capital of Canada?", ["ottawa"]),
    ("What is the chemical symbol for sodium?", ["na"]), ("Who painted Starry Night?", ["gogh"]),
    ("What is the largest mammal?", ["blue whale"]), ("How many sides does a hexagon have?", ["6", "six"]),
    ("What is the speed of light in kilometres per second?", ["299", "300,000", "300000"]),
    ("Which planet is closest to the sun?", ["mercury"]), ("Who discovered gravity?", ["newton"]),
    ("What is the capital of Brazil?", ["bras"]), ("Who wrote the Odyssey?", ["homer"]),
    ("What is the largest hot desert in the world?", ["sahara"]), ("What is the capital of Egypt?", ["cairo"]),
    ("Who was the first president of the United States?", ["washington"]),
    ("In which year did World War II end?", ["1945"]), ("What is the chemical formula of water?", ["h2o", "h₂o", "two hydrogen"]),
    ("How many bones are in the adult human body?", ["206"]), ("What is the capital of Kenya?", ["nairobi"]),
    ("Who composed the Moonlight Sonata?", ["beethoven"]), ("What is the tallest animal?", ["giraffe"]),
    ("Who is known as the father of computers?", ["babbage"]), ("What is the capital of Turkey?", ["ankara"]),
    ("Which element has atomic number 1?", ["hydrogen"]), ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("What is the longest river in South America?", ["amazon"]), ("What is the capital of South Korea?", ["seoul"]),
    ("Who invented the light bulb?", ["edison"]), ("What is the currency of the United Kingdom?", ["pound"]),
    ("How many players of one team are on the field in football (soccer)?", ["11", "eleven"]),
    ("What is the smallest country in the world?", ["vatican"]),
    ("Who was the first woman to win a Nobel Prize?", ["curie"]), ("What is the capital of Argentina?", ["buenos aires"]),
    ("What is the most common gas in Earth's atmosphere?", ["nitrogen"]),
    ("Who painted the ceiling of the Sistine Chapel?", ["michelangelo"]), ("What is the capital of Nigeria?", ["abuja"]),
    ("In which country is the Taj Mahal?", ["india"]), ("What is the largest organ of the human body?", ["skin"]),
    ("Who wrote Nineteen Eighty-Four?", ["orwell"]), ("How many minutes are in a day?", ["1440", "1,440"]),
    ("What is the capital of Pakistan?", ["islamabad"]), ("Which metal is liquid at room temperature?", ["mercury"]),
    ("Who was the first Mughal emperor?", ["babur"]), ("Who wrote the Indian national anthem?", ["tagore"]),
    ("What is the capital of Bangladesh?", ["dhaka"]), ("What is the national animal of India?", ["tiger"]),
]
TRAPS = [
    "Who is the mayor of Tromso?", "What did I eat for breakfast yesterday?", "Who won the 2031 World Cup?",
    "Who was the 15th president of Mars?", "What is the phone number of the nearest pharmacy?",
    "How many people are in this room?", "What is the name of Albert Einstein's third cousin?",
    "Who wrote the novel The Silent Zxqv Lantern?", "What is the exact weight of the moon rock on my desk?",
    "Who won the Ballon d'Or in 2041?", "What is the wifi password of the Pentagon?",
    "What was the temperature in Delhi at noon yesterday?", "What colour is the fourth unicorn born in 1800?",
]


def run(path: pathlib.Path) -> dict:
    llm = rag.LocalLLM(path=path)
    right = wrong = idk = 0
    rows, t0 = [], time.time()
    for q, keys in KNOWN:
        a = rag.general_answer(llm, q)
        verdict = "idk" if a is None else "right" if any(k in a.lower() for k in keys) else "wrong"
        right, wrong, idk = right + (verdict == "right"), wrong + (verdict == "wrong"), idk + (verdict == "idk")
        rows.append({"q": q, "a": a, "verdict": verdict})
    trap_wrong = 0
    for q in TRAPS:
        a = rag.general_answer(llm, q)
        trap_wrong += a is not None
        rows.append({"q": q, "a": a, "verdict": "trap-answered (hallucination)" if a else "trap-idk (correct)"})
    return {"model": path.name, "known": len(KNOWN), "right": right, "wrong": wrong, "idk": idk,
            "traps": len(TRAPS), "trap_hallucinated": trap_wrong,
            "seconds_per_question": round((time.time() - t0) / (len(KNOWN) + len(TRAPS)), 1), "rows": rows}


if __name__ == "__main__":
    results = []
    for p in sys.argv[1:]:
        r = run(ROOT / p if not pathlib.Path(p).is_absolute() else pathlib.Path(p))
        results.append(r)
        print(f"{r['model']}: right {r['right']}/{r['known']}, wrong {r['wrong']}, idk {r['idk']}; traps answered "
              f"{r['trap_hallucinated']}/{r['traps']}; {r['seconds_per_question']} s/question", flush=True)
        for row in r["rows"]:
            if row["verdict"] in ("wrong", "trap-answered (hallucination)"):
                print("    ", row["verdict"], "|", row["q"], "->", row["a"], flush=True)
    (ROOT / "bench" / "results").mkdir(exist_ok=True)
    (ROOT / "bench" / "results" / "general_qa.json").write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                                                encoding="utf-8")
