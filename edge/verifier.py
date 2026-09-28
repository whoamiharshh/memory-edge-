"""Outcome verifier: after an action, does the machine's own vibration return to its healthy baseline and
stay there?

Each post-action window is "healthy" if its distance to the nearest baseline point is <= tau_normal (the same
radius the novelty gate uses). The verdict:
  symptom_resolved  : REQUIRED consecutive healthy windows
  symptom_persists  : REQUIRED consecutive abnormal windows
  verifying         : neither yet
The UI must say "symptom resolved for N windows", never "root cause confirmed" (docs/RESEARCH.md G.6).

With a machine card (edge/machine_card.py) a window is healthy only if it is ALSO below the manufacturer's (or the
ISO table's) acceptable vibration velocity: two independent references - this machine's own healthy past and the
maker's limit. A repaired machine that is quieter than before but still above the limit is not "resolved".
Weeks later, the device reports whether the fix HELD or the fault RECURRED (shared.schema.FollowUp).

REPLACEMENT actions (a new bearing / part) change the machine's healthy signature: measured on 20 real bearings
(bench/acoustic_uottawa.py) a new healthy bearing lay outside the old healthy radius 17-20 times out of 20. For them
a window also counts as healthy when it is NEARER to the machine's healthy state than to the fault episode's own
fingerprints (parameter-free nearest-state rule; the same data: new bearing judged fixed 18/20, continuing fault
never judged fixed 20/20). When resolved, those windows extend the healthy baseline: the new part becomes normal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

REPLACEMENT_ACTIONS = ("replace_bearing", "replace_part")

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
    limit_mm_s: float | None = None     # the machine card's acceptable velocity (None = no card: baseline only)
    limit_source: str | None = None
    last_velocity_mm_s: float | None = None
    over_limit_windows: int = 0         # windows inside the healthy radius but above the limit
    mode: str = "same_part"             # same_part | replacement (nearest-state rule, see module docstring)
    new_part_windows: int = 0           # windows accepted by the nearest-state rule (outside the old radius)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "VerifyState":
        return cls(**d) if d else cls()

    @property
    def done(self) -> bool:
        return self.verdict != "verifying"

    def label(self) -> str:
        lim = (f" and vibration {self.last_velocity_mm_s} mm/s is below the limit {self.limit_mm_s} mm/s"
               if self.limit_mm_s and self.last_velocity_mm_s is not None else "")
        if self.verdict == "symptom_resolved":
            if self.mode == "replacement" and self.new_part_windows:
                return (f"symptom resolved for {self.consecutive_ok} consecutive windows{lim}: the new part is nearer to "
                        f"this machine's healthy state than to the fault ({self.new_part_windows} windows outside the "
                        "old healthy radius were adopted as the new part's normal)")
            return f"symptom resolved for {self.consecutive_ok} consecutive windows{lim}"
        if self.verdict == "symptom_persists":
            return f"symptom persists: {self.consecutive_bad} consecutive abnormal windows after the action"
        return f"verifying: {self.consecutive_ok}/{self.required} consecutive healthy windows"


def step(s: VerifyState, healthy: bool, velocity_mm_s: float | None = None) -> VerifyState:
    """Feed one post-action window. A final verdict is frozen (later windows do not change it). With a limit, a
    window inside the healthy radius but above the limit counts as NOT healthy."""
    if s.done:
        return s
    if s.limit_mm_s and velocity_mm_s is not None:
        s.last_velocity_mm_s = round(float(velocity_mm_s), 3)
        if healthy and velocity_mm_s > s.limit_mm_s:
            healthy = False
            s.over_limit_windows += 1
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
