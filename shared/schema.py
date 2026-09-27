"""Wire schema between devices and the cloud (pydantic). Strict: unknown fields are rejected, numbers must be
finite, strings are length-capped. Tenant, device and site identity are NEVER part of an event body: the
cloud derives them from the device's token."""
from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from edge.fingerprint import DIM as FP_DIM, FP_VERSION

SCHEMA_VERSION = 1
MAX_NOTE_CHARS = 600
MAX_BATCH = 100


class Component(str, Enum):
    bearing = "bearing"
    gearbox = "gearbox"
    coupling = "coupling"
    shaft = "shaft"
    fan = "fan"
    pump = "pump"
    motor = "motor"


class FaultClass(str, Enum):
    """Bearing defect location (CWRU classes) plus generic rotating-equipment faults."""
    inner_race = "inner_race"
    outer_race = "outer_race"
    ball = "ball"
    imbalance = "imbalance"
    misalignment = "misalignment"
    looseness = "looseness"
    unknown = "unknown"


class ActionCode(str, Enum):
    """Structured intervention codes (Proposed Design). Loosely modelled on the action types in the Annotated
    Maintenance Logbook (removed & replaced, tightened, cleaned, installed, checked) mapped to rotating
    equipment."""
    replace_bearing = "replace_bearing"
    lubricate = "lubricate"
    realign = "realign"
    rebalance = "rebalance"
    tighten = "tighten"
    clean = "clean"
    replace_seal = "replace_seal"
    inspect_no_action = "inspect_no_action"
    other = "other"


class RootCause(str, Enum):
    lubrication_starvation = "lubrication_starvation"
    contamination = "contamination"
    misalignment = "misalignment"
    overload = "overload"
    fatigue_wear = "fatigue_wear"
    installation_damage = "installation_damage"
    electrical_erosion = "electrical_erosion"
    unknown = "unknown"


class Outcome(str, Enum):
    pending = "pending"
    worked = "worked"
    failed = "failed"


class ShareEvent(BaseModel):
    """One outcome-verified piece of evidence leaving a device. Contains no raw signal, no raw note and no
    text embedding (embedding inversion risk); the cloud embeds the redacted note itself."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = SCHEMA_VERSION
    event_id: str = Field(min_length=36, max_length=36)
    episode_id: str = Field(min_length=36, max_length=36)
    machine_class: str = Field(min_length=1, max_length=80)
    component: Component
    fault_class: FaultClass
    fault_class_source: Literal["technician", "physics_hint"]
    action_code: ActionCode
    outcome: Literal["worked", "failed"]
    root_cause_claim: RootCause | None = None
    machine_verified: bool
    verify_windows_ok: int = Field(ge=0, le=100_000)
    verify_windows_required: int = Field(ge=1, le=100_000)
    technician_confirmed: bool
    fingerprint: list[float] = Field(min_length=FP_DIM, max_length=FP_DIM)
    fp_version: Literal["fp-v2"] = FP_VERSION
    note_redacted: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    occurred_at: str = Field(min_length=10, max_length=40)

    @field_validator("fingerprint")
    @classmethod
    def _finite(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("fingerprint must be finite")
        return v


class PushRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_id: str = Field(min_length=1, max_length=64)
    events: list[dict] = Field(max_length=MAX_BATCH)     # validated one by one: a bad event rejects only itself


class EventResult(BaseModel):
    event_id: str
    status: Literal["accepted", "duplicate", "rejected"]
    reason: str | None = None


class PushResponse(BaseModel):
    batch_id: str
    results: list[EventResult]
