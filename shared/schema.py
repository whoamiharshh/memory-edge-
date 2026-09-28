"""Wire schema between devices and the cloud (pydantic). Strict: unknown fields are rejected, numbers must be
finite, strings are length-capped. Tenant, device and site identity are NEVER part of an event body: the
cloud derives them from the device's token."""
from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from edge.fingerprint import DIM as FP_DIM, FP_VERSION
from edge.physics import ORDER_FEATURE_NAMES

ORDER_DIM = len(ORDER_FEATURE_NAMES)

SCHEMA_VERSION = 1
MAX_NOTE_CHARS = 600
MAX_BATCH = 100


class Component(str, Enum):
    """What was worked on. Rotating machines first; then robots, kiosks, vehicles and mobile/app devices (PS3)."""
    bearing = "bearing"
    gearbox = "gearbox"
    coupling = "coupling"
    shaft = "shaft"
    fan = "fan"
    pump = "pump"
    motor = "motor"
    compressor = "compressor"
    robot_joint = "robot_joint"
    robot_gripper = "robot_gripper"
    conveyor = "conveyor"
    kiosk = "kiosk"
    printer = "printer"
    card_reader = "card_reader"
    touchscreen = "touchscreen"
    engine = "engine"
    brake = "brake"
    wheel = "wheel"
    battery = "battery"
    network = "network"
    app = "app"
    sensor = "sensor"


class FaultClass(str, Enum):
    """Bearing defect location (CWRU classes), generic rotating-equipment faults, then robot / device faults."""
    inner_race = "inner_race"
    outer_race = "outer_race"
    ball = "ball"
    cage = "cage"
    imbalance = "imbalance"
    misalignment = "misalignment"
    looseness = "looseness"
    collision = "collision"
    obstruction = "obstruction"
    slip = "slip"
    overheating = "overheating"
    jam = "jam"
    power = "power"
    connectivity = "connectivity"
    software_error = "software_error"
    hardware_error = "hardware_error"
    sensor_fault = "sensor_fault"
    wear = "wear"
    unknown = "unknown"


class ActionCode(str, Enum):
    """Structured intervention codes (Proposed Design). Loosely modelled on the action types in the Annotated
    Maintenance Logbook (removed & replaced, tightened, cleaned, installed, checked), extended to robots and devices."""
    replace_bearing = "replace_bearing"
    lubricate = "lubricate"
    realign = "realign"
    rebalance = "rebalance"
    tighten = "tighten"
    clean = "clean"
    replace_seal = "replace_seal"
    replace_part = "replace_part"
    recalibrate = "recalibrate"
    clear_jam = "clear_jam"
    reseat_connector = "reseat_connector"
    restart = "restart"
    update_firmware = "update_firmware"
    adjust_settings = "adjust_settings"
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
    loose_mounting = "loose_mounting"
    software_bug = "software_bug"
    configuration = "configuration"
    environment = "environment"
    operator_error = "operator_error"
    unknown = "unknown"


class DamageMode(str, Enum):
    """What the technician SAW on the removed bearing, per ISO 15243:2017 (terms as summarised by SKF, 'Bearing
    damage analysis: ISO 15243 is here to help you'). This is the physical ground truth the fleet learns from."""
    subsurface_initiated_fatigue = "subsurface_initiated_fatigue"      # 5.1.2
    surface_initiated_fatigue = "surface_initiated_fatigue"            # 5.1.3
    abrasive_wear = "abrasive_wear"                                    # 5.2.2
    adhesive_wear = "adhesive_wear"                                    # 5.2.3
    moisture_corrosion = "moisture_corrosion"                          # 5.3.2
    fretting_corrosion = "fretting_corrosion"                          # 5.3.3.2
    false_brinelling = "false_brinelling"                              # 5.3.3.3
    excessive_current_erosion = "excessive_current_erosion"            # 5.4.2
    current_leakage_erosion = "current_leakage_erosion"                # 5.4.3
    overload_deformation = "overload_deformation"                      # 5.5.2
    indentation_from_particles = "indentation_from_particles"          # 5.5.3
    forced_fracture = "forced_fracture"                                # 5.6.2
    fatigue_fracture = "fatigue_fracture"                              # 5.6.3
    thermal_cracking = "thermal_cracking"                              # 5.6.4
    not_inspected = "not_inspected"


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
    fp_version: Literal["fp-v2", "fp-rh1", "fp-lr1", "fp-ft1", "fp-ev1", "fp-tm1", "fp-ac1"] = FP_VERSION  # edge/profiles.py
    note_redacted: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    occurred_at: str = Field(min_length=10, max_length=40)
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")   # shared.ids.repair_hash
    damage_mode: DamageMode | None = None
    # physics numbers the fleet-learned fault hint trains on (edge/physics.order_features): dimensionless log ratios,
    # not a raw signal. Only with a technician-confirmed fault class.
    order_features: list[float] | None = Field(default=None, min_length=ORDER_DIM, max_length=ORDER_DIM)
    of_version: Literal["of-v1"] | None = None
    bearing: str | None = Field(default=None, max_length=60)
    severity_mm_s: float | None = Field(default=None, ge=0, le=1000)     # after the fix, vs the machine card limit
    limit_mm_s: float | None = Field(default=None, gt=0, le=1000)

    @field_validator("fingerprint")
    @classmethod
    def _finite(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("fingerprint must be finite")
        return v

    @field_validator("order_features")
    @classmethod
    def _bounded(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and not all(math.isfinite(x) and abs(x) <= 50 for x in v):
            raise ValueError("order_features must be finite log ratios (|x| <= 50)")
        return v


class FollowUp(BaseModel):
    """Did a shared fix HOLD? Sent once per shared 'worked' event: 'held' after the hold period with no recurrence of
    the fault on that machine, or 'recurred' when it came back first. Travels in the same outbox and push batches."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["followup"] = "followup"
    event_id: str = Field(min_length=36, max_length=36)        # shared.ids.followup_id(device, refers_to)
    episode_id: str = Field(min_length=36, max_length=36)      # the episode of the original fix
    refers_to: str = Field(min_length=36, max_length=36)       # the original ShareEvent id
    status: Literal["held", "recurred"]
    days_after_fix: float = Field(ge=0, le=3650)
    hold_days: float = Field(gt=0, le=3650)
    occurred_at: str = Field(min_length=10, max_length=40)


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
