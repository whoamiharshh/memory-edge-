"""How well does the note redactor catch people's names nobody put on the denylist?

Notes: REAL maintenance notes (Annotated Maintenance Logbook, CC BY 4.0: PROBLEM + ACTION). 1,000 notes are held
out (seed 7); the maintenance vocabulary the redactor uses is rebuilt WITHOUT them (data/build_vocab.build(exclude)).
Half of the held-out notes get a person's name inserted with one of 8 templates (SYNTHETIC insertion: no public data
set of maintenance notes with labelled names exists); the other half stay clean (false-alarm test).
Names: real given names (Wikidata, CC0). 20 % of the name list is held out of the redactor's list, so we measure
  in-list names       what the shipped redactor sees for common names
  unseen names        names it has never been given (caught by the capitalisation rule and the name-likeness model,
                      which the benchmark trains itself on the in-list names only)
Three writing styles: the logbook's ALL CAPS, sentence case (as people type) and all lower case.
Recall = inserted-name notes flagged; false alarms = clean notes flagged (a false alarm only keeps a note local).
Run: .venv\\Scripts\\python.exe -m bench.redaction
"""
from __future__ import annotations

import json
import pathlib
import random
import re

from data import build_vocab
from shared.redact import NOT_NAMES, redact

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
TEMPLATES = ["{note} replaced by {n}", "{n} checked it: {note}", "{note} informed {n} supervisor",
             "{note} per {n}", "{n} and {m} did the job. {note}", "{note} call {n} if it recurs",
             "{note} ({n} night shift)", "done with {n}. {note}"]
# written AFTER the cue-word rule (shared/redact.py _CUE), so that rule is also measured on phrasings it was not
# designed around (several use no cue word at all)
TEMPLATES_HELDOUT = ["{note} - {n} on shift", "{n} says {note}", "{note}, {n} to follow up", "{note} handed over to {n}",
                     "spoke to {n}. {note}", "{note} witnessed by {n}", "{n} / {m}: {note}", "{note}. {n} will recheck"]


def style(text: str, how: str) -> str:
    if how == "caps":
        return text.upper()
    if how == "lower":
        return text.lower()
    out = text.lower()
    return ". ".join(s.strip().capitalize() for s in out.split(". "))


def main() -> dict:
    rng = random.Random(7)
    notes = build_vocab.logbook_notes()
    idx = list(range(len(notes)))
    rng.shuffle(idx)
    test = idx[:1000]
    vocab = frozenset(build_vocab.build(exclude=set(test))) | NOT_NAMES
    names_all = sorted(json.loads((ROOT / "knowledge" / "given_names.json").read_text(encoding="utf-8"))["names"])
    rng.shuffle(names_all)
    cut = int(0.8 * len(names_all))
    listed, unseen = names_all[:cut], [n for n in names_all[cut:] if n not in vocab]
    listed_ok = [n for n in listed if n not in vocab]
    lex = (frozenset(listed), vocab)
    from shared import name_model                     # trained here WITHOUT the unseen names and test-note words
    model = name_model.train(listed, sorted(vocab))
    import shared.redact as R
    cue_rule = R._CUE
    R._CUE = re.compile(r"(?!x)x")                    # first WITHOUT the cue-word rule (the state before it), then with
    without = evaluate(notes, test, listed_ok, unseen, lex, model)
    R._CUE = cue_rule
    res = evaluate(notes, test, listed_ok, unseen, lex, model)
    out = {"method": __doc__.strip().splitlines()[0], "protocol": __doc__.split("Notes:")[1].split("Run:")[0].strip(),
           "names_in_list": len(listed_ok), "names_unseen": len(unseen), "results": res,
           "results_without_cue_rule": without,
           "before_this_pass": "regex + denylist only: 0/60 unlisted names found (docs/BENCHMARKS.md, personal-data probe)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "redaction.json").write_text(json.dumps(out, indent=1))
    return out


def evaluate(notes, test, listed_ok, unseen, lex, model) -> dict:
    rng = random.Random(11)                           # the same names in both runs
    res = {}
    for how in ("caps", "sentence", "lower"):
        row = {}
        for tset, templates in (("", TEMPLATES), ("_heldout_templates", TEMPLATES_HELDOUT)):
            for pool_name, pool in (("in_list", listed_ok), ("unseen", unseen)):
                hit = 0
                for k, i in enumerate(test[:500]):
                    base = " ".join(notes[i].split()).rstrip(".").lower()
                    n, m = rng.choice(pool), rng.choice(pool)
                    # the template and note take the writing style; the name is written as a person would in that style
                    case = {"caps": str.upper, "lower": str.lower, "sentence": str.capitalize}[how]
                    t = style(templates[k % len(templates)].format(note=base, n="QQNAMEQQ", m="QQMATEQQ"), how)
                    t = re.sub("qqnameqq", case(n), t, flags=re.I)
                    t = re.sub("qqmateqq", case(m), t, flags=re.I)
                    hit += not redact(t, lexicon=lex, model=model).clean
                row[f"recall_{pool_name}{tset}"] = round(hit / 500, 3)
        fa = sum(not redact(style(" ".join(notes[i].split()).lower(), how), lexicon=lex, model=model).clean
                 for i in test[500:])
        row["false_alarm_clean_notes"] = round(fa / 500, 3)
        res[how] = row
        print(how, row, flush=True)
    return res


if __name__ == "__main__":
    main()
