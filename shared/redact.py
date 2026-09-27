"""Deterministic redactor for technician notes (no model). It finds emails, phone numbers, URLs, IP addresses,
long numeric/employee ids, honorific + name patterns and a configurable denylist (staff and site names).

Policy use (docs/RESEARCH.md G.7): a note may leave the device only if the technician opted in AND this
redactor finds NOTHING. If anything is found, the whole note stays local. Regex redaction misses free-form
PII; that is why notes default to local and structured fields carry the shared evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

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


def redact(text: str, denylist: list[str] | tuple[str, ...] = ()) -> Redaction:
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
    return Redaction(out, findings)


def normalise(text: str) -> str:
    return " ".join(text.lower().split())
