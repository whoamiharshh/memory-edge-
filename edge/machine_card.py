"""Machine card: what the MANUFACTURER says about this machine, entered once from its manual / nameplate / bearing
catalogue. It makes the physics machine-specific instead of generic:

  bearing geometry    -> the defect frequencies the envelope rule and the fleet hint look at
  power, type, mount  -> which ISO 10816 severity table applies (verified values in edge/physics.py)
  manufacturer limits -> override ISO: the standard itself says its zones are guidelines and acceptance limits
                         "shall be subject to agreement between the machine manufacturer and customer"
  mains frequency     -> electrical hum lines to ignore
  source              -> where each value came from (manual title, page), shown next to every limit

"Fixed" then means: back inside this machine's own healthy radius AND below the manufacturer's (or ISO's)
acceptable level - two independent references instead of one (edge/verifier.py).
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any

from edge import physics as P

MACHINE_TYPES = ("motor", "fan", "compressor", "gearbox", "pump_separate_driver", "pump_integrated_driver", "other")
# ISO 10816-1:1995 Annex B (informative), class I = small machines up to about 15 kW. Values as reproduced by
# secondary sources (vibromera.eu glossary, 28 Sep 2026); the standard's own text was NOT read - labelled as such.
ISO10816_1_CLASS_I = (0.71, 1.8, 4.5)
FILE = "machine_card.json"


@dataclass
class MachineCard:
    manufacturer: str = ""
    model: str = ""
    machine_type: str = "motor"
    power_kw: float | None = None
    foundation: str = "rigid"                     # rigid | flexible (ISO 10816-3 clause 4)
    bearing: str | None = None                    # a key of physics.BEARINGS ...
    bearing_geometry: dict | None = None          # ... or the catalogue geometry {n_elements, ball_d, pitch_d, contact_deg}
    nominal_rpm: float | None = None
    mains_hz: float | None = None                 # 50 or 60
    limits_mm_s: dict | None = None               # manufacturer {"acceptable": x, "trip": y} (velocity r.m.s.)
    lubrication: dict | None = None               # {"grease": "...", "interval_h": 2000, "quantity_g": 5}
    source: str = ""                              # e.g. "XYZ motor manual rev 3, section 7.2, p. 41"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    # ---- validation ----------------------------------------------------------------------------------------
    def validate(self) -> "MachineCard":
        for k in ("manufacturer", "model", "source", "notes"):
            if len(getattr(self, k) or "") > 300:
                raise ValueError(f"{k} too long (max 300 characters)")
        if self.machine_type not in MACHINE_TYPES:
            raise ValueError(f"machine_type must be one of {', '.join(MACHINE_TYPES)}")
        if self.foundation not in ("rigid", "flexible"):
            raise ValueError("foundation must be rigid or flexible")
        for k, lo, hi in (("power_kw", 0.001, 100_000), ("nominal_rpm", 1, 200_000), ("mains_hz", 10, 1000)):
            v = getattr(self, k)
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{k} out of range ({lo}-{hi})")
        if self.bearing and self.bearing not in P.BEARINGS:
            raise ValueError(f"unknown bearing {self.bearing!r}; give bearing_geometry from the catalogue instead")
        if self.bearing_geometry:
            P.custom_geometry(self.bearing_geometry)
        if self.limits_mm_s:
            a, t = self.limits_mm_s.get("acceptable"), self.limits_mm_s.get("trip")
            if not (isinstance(a, (int, float)) and 0 < a < 1000):
                raise ValueError("limits_mm_s.acceptable must be a velocity in mm/s")
            if t is not None and not (isinstance(t, (int, float)) and a < t < 1000):
                raise ValueError("limits_mm_s.trip must be above the acceptable limit")
        return self

    # ---- what it means for the physics --------------------------------------------------------------------
    def geometry(self) -> P.BearingGeometry | None:
        if self.bearing_geometry:
            return P.custom_geometry(self.bearing_geometry)
        return P.BEARINGS.get(self.bearing) if self.bearing else None

    def iso(self) -> tuple[tuple[float, float, float], str] | None:
        """(A/B, B/C, C/D boundaries in mm/s, which table) from the type, power and foundation, or None."""
        kw = self.power_kw
        if self.machine_type in ("pump_separate_driver", "pump_integrated_driver"):
            if kw is None or kw <= 15:
                return None
            g = 3 if self.machine_type == "pump_separate_driver" else 4
        elif kw is None:
            return None
        elif kw > 300:
            g = 1
        elif kw > 15:
            g = 2
        else:
            return ISO10816_1_CLASS_I, ("ISO 10816-1 Annex B class I (small machines <= 15 kW): 0.71/1.8/4.5 mm/s "
                                        "(secondary source; informative annex)")
        b = P.ISO10816_3[(g, self.foundation)]
        return b, f"ISO 10816-3 Table A.{g}, group {g}, {self.foundation}: {b[0]}/{b[1]}/{b[2]} mm/s"

    def severity_bounds(self) -> tuple[tuple[float, float, float], str]:
        """Zone boundaries used for this machine. The manufacturer's limits win: 'acceptable' becomes the B/C
        boundary (the verifier's pass level) and 'trip' the C/D boundary."""
        iso = self.iso()
        base, ref = iso if iso else (P.ISO10816_G2_RIGID_MM_S, "ISO 10816-3 group 2 rigid (default: no machine type/"
                                                              "power on the card)")
        if self.limits_mm_s:
            a = float(self.limits_mm_s["acceptable"])
            t = float(self.limits_mm_s.get("trip") or max(base[2], a * 1.6))
            ab = min(base[0], a * 0.5)
            return (ab, a, t), (f"manufacturer limits: acceptable {a} mm/s, trip {t} mm/s"
                                + (f" ({self.source})" if self.source else "") + f"; zone A/B from {ref.split(':')[0]}")
        return base, ref

    def acceptable_mm_s(self) -> float:
        return self.severity_bounds()[0][1]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        bounds, ref = self.severity_bounds()
        g = self.geometry()
        d["derived"] = {"severity_bounds_mm_s": list(bounds), "severity_reference": ref,
                        "acceptable_mm_s": bounds[1],
                        "defect_orders": {k: round(v, 4) for k, v in g.orders().items()} if g else None,
                        "defect_frequencies_hz": ({k: round(v * self.nominal_rpm / 60, 2) for k, v in g.orders().items()}
                                                  if g and self.nominal_rpm else None)}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MachineCard":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known).validate()


def load(root: pathlib.Path) -> MachineCard | None:
    p = pathlib.Path(root) / FILE
    return MachineCard.from_dict(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None


def save(root: pathlib.Path, card: MachineCard) -> None:
    card.validate()
    d = asdict(card)
    (pathlib.Path(root) / FILE).write_text(json.dumps(d, indent=1), encoding="utf-8")
