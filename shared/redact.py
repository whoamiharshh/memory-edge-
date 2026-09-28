"""Deterministic redactor for technician notes (no model). It finds emails, phone numbers, URLs, IP addresses,
long numeric/employee ids, honorific + name patterns, a configurable denylist (staff and site names), and - since the
weak-point pass - people's names that nobody listed:

  given_name           a word that is one of 10,562 real given names (knowledge/given_names.json, Wikidata CC0; India
                       first) and NOT a word of real maintenance text (knowledge/domain_vocab.json)
  capitalised_unknown  in normally-cased text, a Capitalised word in mid-sentence that is not maintenance vocabulary
  name_like            any other unknown word that a character n-gram model trained on real names vs real
                       maintenance words scores as a given name (shared/name_model.py; catches ALL CAPS / lower case)

Measured on real logbook notes with names inserted (bench/redaction.py). Policy use (docs/RESEARCH.md G.7): a note may
leave the device only if the technician opted in AND this redactor finds NOTHING; anything found keeps the whole note
local (a false alarm only costs sharing, never privacy).
"""
from __future__ import annotations

import functools
import json
import pathlib
import re
from dataclasses import dataclass, field

KNOWLEDGE = pathlib.Path(__file__).resolve().parents[1] / "knowledge"
NOT_NAMES = {  # calendar words and very common English words that also occur as given names
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march",
    "april", "june", "july", "august", "september", "october", "november", "december", "the", "and", "for", "not",
    "all", "but", "new", "old", "see", "day", "will", "may", "hope", "grace", "joy", "faith", "summer", "winter",
    "bill", "mark", "pat", "rich", "rob", "sue", "art", "sky", "star", "king", "rose", "lily", "ray", "sun", "moon"}


@functools.lru_cache(maxsize=1)
def _lexicon() -> tuple[frozenset, frozenset]:
    names = json.loads((KNOWLEDGE / "given_names.json").read_text(encoding="utf-8"))["names"]
    vocab = json.loads((KNOWLEDGE / "domain_vocab.json").read_text(encoding="utf-8"))["words"]
    return frozenset(names), frozenset(vocab) | NOT_NAMES


_WORD = re.compile(r"[A-Za-z]{3,20}")
_SENT_END = re.compile(r"[.!?:;\n]\s*$")


def _name_findings(text: str, lexicon: tuple[frozenset, frozenset] | None = None,
                   model: dict | None | bool = None) -> list[tuple[str, str]]:
    from shared import name_model
    names, vocab = lexicon or _lexicon()
    model = name_model.load() if model is None else (model or None)
    all_caps = text == text.upper()
    out, unknown = [], []
    for m in _WORD.finditer(text):
        w, lw = m.group(0), m.group(0).lower()
        if "REDACTED" in w or lw in vocab:
            continue
        if lw in names:
            out.append(("given_name", w))
        elif not all_caps and w[0].isupper() and w[1:].islower():
            before = text[:m.start()]
            if before.strip() and not _SENT_END.search(before):      # mid-sentence capital
                out.append(("capitalised_unknown", w))
            else:
                unknown.append(w)
        else:
            unknown.append(w)
    if model and unknown:
        for w, pr in zip(unknown, name_model.probability(model, [u.lower() for u in unknown])):
            if pr >= model["threshold"]:
                out.append(("name_like", w))
    return out

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "url": re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "phone": re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w.])"),
    "id_number": re.compile(r"\b(?:[A-Z]{2,5}[-_ ]?\d{3,}|\d{6,})\b"),
    "person": re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Sri|Smt|Shri|Er)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
    "signed_by": re.compile(r"\b(?:by|tech|technician|signed|contact|call|ask)\s*[:\-]?\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?"),
}


@dataclass
class Redaction:
    text: str
    findings: list[tuple[str, str]] = field(default_factory=list)   # (kind, matched text) - stays on device

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.clean:
            return "no personal data found"
        kinds: dict[str, int] = {}
        for k, _ in self.findings:
            kinds[k] = kinds.get(k, 0) + 1
        return ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))


def redact(text: str, denylist: list[str] | tuple[str, ...] = (),
           lexicon: tuple[frozenset, frozenset] | None = None, model: dict | None | bool = None) -> Redaction:
    findings: list[tuple[str, str]] = []
    out = text
    for word in sorted({w for w in denylist if w.strip()}, key=len, reverse=True):
        pat = re.compile(rf"\b{re.escape(word)}\b", re.I)
        for m in pat.finditer(out):
            findings.append(("denylist", m.group(0)))
        out = pat.sub("[REDACTED]", out)
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(out):
            if "[REDACTED]" not in m.group(0):
                findings.append((kind, m.group(0)))
        out = pat.sub(lambda m: m.group(0) if "[REDACTED]" in m.group(0) else "[REDACTED]", out)
    for kind, word in _name_findings(out, lexicon, model):
        findings.append((kind, word))
        out = re.sub(r"\b" + re.escape(word) + r"\b", "[REDACTED]", out)
    return Redaction(out, findings)


def normalise(text: str) -> str:
    return " ".join(text.lower().split())
