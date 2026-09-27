"""Outcome verifier: after an action, does the machine's own vibration return to its healthy baseline and
stay there?

Each post-action window is "healthy" if its distance to the nearest baseline point is <= tau_normal (the same
radius the novelty gate uses). The verdict:
  symptom_resolved  : REQUIRED consecutive healthy windows
  symptom_persists  : REQUIRED consecutive abnormal windows
  verifying         : neither yet
The UI must say "symptom resolved for N windows", never "root cause confirmed" (docs/RESEARCH.md G.6).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

REQUIRED_WINDOWS = 20          # prototype parameter: 20 windows x 2048-sample hop at 12 kHz ~= 3.4 s of signal


@dataclass
class VerifyState:
    required: int = REQUIRED_WINDOWS
    seen: int = 0
    consecutive_ok: int = 0
    consecutive_bad: int = 0
    ok_total: int = 0
    bad_total: int = 0
    verdict: str = "verifying"          # verifying | symptom_resolved | symptom_persists

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "VerifyState":
        return cls(**d) if d else cls()

    @property
    def done(self) -> bool:
        return self.verdict != "verifying"

    def label(self) -> str:
        if self.verdict == "symptom_resolved":
            return f"symptom resolved for {self.consecutive_ok} consecutive windows"
        if self.verdict == "symptom_persists":
            return f"symptom persists: {self.consecutive_bad} consecutive abnormal windows after the action"
        return f"verifying: {self.consecutive_ok}/{self.required} consecutive healthy windows"


def step(s: VerifyState, healthy: bool) -> VerifyState:
    """Feed one post-action window. A final verdict is frozen (later windows do not change it)."""
    if s.done:
        return s
    s.seen += 1
    if healthy:
        s.ok_total += 1
        s.consecutive_ok += 1
        s.consecutive_bad = 0
    else:
        s.bad_total += 1
        s.consecutive_bad += 1
        s.consecutive_ok = 0
    if s.consecutive_ok >= s.required:
        s.verdict = "symptom_resolved"
    elif s.consecutive_bad >= s.required:
        s.verdict = "symptom_persists"
    return s
