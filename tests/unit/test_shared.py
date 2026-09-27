"""Unit tests for shared/ids.py, shared/schema.py, shared/redact.py."""
import math
import uuid

import pytest
from pydantic import ValidationError

from edge.fingerprint import DIM
from shared import ids
from shared.redact import redact
from shared.schema import ShareEvent


# ---- ids ------------------------------------------------------------------------------------------
def test_ids_are_deterministic_uuids():
    a = ids.event_id("devA", "ep1", "worked", "replace_bearing")
    assert a == ids.event_id("devA", "ep1", "worked", "replace_bearing")
    assert a != ids.event_id("devA", "ep1", "failed", "replace_bearing")
    assert uuid.UUID(a).version == 5
    assert ids.case_id("t1", "bearing", "inner_race") != ids.case_id("t2", "bearing", "inner_race")


def test_content_hash_ignores_key_order():
    assert ids.content_hash({"a": 1, "b": [1, 2]}) == ids.content_hash({"b": [1, 2], "a": 1})
    assert ids.content_hash({"a": 1}) != ids.content_hash({"a": 2})


# ---- schema ---------------------------------------------------------------------------------------
def event(**kw):
    base = dict(event_id=ids.make_id("e"), episode_id=ids.make_id("ep"), machine_class="2hp-motor/SKF6205",
                component="bearing", fault_class="inner_race", fault_class_source="technician",
                action_code="replace_bearing", outcome="worked", machine_verified=True, verify_windows_ok=20,
                verify_windows_required=20, technician_confirmed=True, fingerprint=[0.1] * DIM,
                occurred_at="2026-09-27T10:00:00+00:00")
    return base | kw


def test_valid_event():
    e = ShareEvent(**event())
    assert e.schema_version == 1 and e.fp_version == "fp-v2"


@pytest.mark.parametrize("bad", [
    {"fingerprint": [0.1] * (DIM - 1)},
    {"fingerprint": [math.nan] + [0.1] * (DIM - 1)},
    {"fingerprint": [math.inf] + [0.1] * (DIM - 1)},
    {"outcome": "pending"},
    {"component": "spaceship"},
    {"action_code": "rm -rf"},
    {"schema_version": 2},
    {"fp_version": "fp-v1"},
    {"note_redacted": "x" * 10_000},
    {"tenant_id": "someone-else"},         # identity is never accepted from the body
    {"note_text": "raw note"},             # raw note field does not exist on the wire
    {"note_vector": [0.1] * 384},          # text embeddings are never shipped
])
def test_invalid_events_rejected(bad):
    with pytest.raises(ValidationError):
        ShareEvent(**event(**bad))


# ---- redaction ------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,kind", [
    ("mail ravi.k@plant.com for parts", "email"),
    ("call +91 98765 43210 after shift", "phone"),
    ("see http://intranet/wo/5 for photos", "url"),
    ("PLC at 10.0.4.17 rebooted", "ip"),
    ("employee EMP-20931 did the job", "id_number"),
    ("Mr. Sharma replaced it", "person"),
    ("done by Anil Kumar", "signed_by"),
])
def test_redactor_finds_pii(text, kind):
    r = redact(text)
    assert not r.clean
    assert kind in [k for k, _ in r.findings]
    assert "[REDACTED]" in r.text


def test_denylist_case_insensitive():
    r = redact("Pune site bearing swapped, told priya", denylist=["Priya", "Pune"])
    assert {k for k, _ in r.findings} == {"denylist"}
    assert "priya" not in r.text.lower() and "pune" not in r.text.lower()


def test_clean_technical_note_passes():
    r = redact("Inner race spall found on DE bearing 6205, replaced bearing, regreased, vibration back to normal")
    assert r.clean, r.findings
    assert r.summary() == "no personal data found"
