"""The fix-procedure library: every entry cites a source, lookups follow fault class + component, and entries
without a source are refused."""
import json

import pytest

from edge import procedures as PR


def test_every_shipped_procedure_cites_a_public_source():
    ps = PR.load()
    assert len(ps) >= 4
    for p in ps:
        assert p["sources"] and all(s["url"].startswith("http") for s in p["sources"]), p["id"]
        assert p["steps"] and p["title"]


@pytest.mark.parametrize("fault,expected", [("inner_race", "bearing-damage-general"), ("ball", "bearing-damage-general"),
                                            ("imbalance", "imbalance-field-balancing"),
                                            ("misalignment", "misalignment-shaft-alignment"), ("looseness", "looseness-mounting")])
def test_lookup_by_fault_class(fault, expected):
    assert PR.lookup(fault, "motor")[0]["id"] == expected


def test_unknown_or_unmapped_fault_returns_nothing():
    assert PR.lookup("unknown") == [] and PR.lookup(None) == [] and PR.lookup("software_error") == []


def test_site_sops_come_first_and_sourceless_entries_are_refused(tmp_path):
    (tmp_path / "procedures.json").write_text(json.dumps({"procedures": [
        {"id": "ref", "fault_classes": ["jam"], "components": ["printer"], "title": "t", "steps": ["s"],
         "sources": [{"title": "x", "url": "https://example.org"}]}]}))
    (tmp_path / "site_procedures.json").write_text(json.dumps({"procedures": [
        {"id": "sop", "fault_classes": ["jam"], "components": ["printer"], "title": "Kiosk printer jam", "steps": ["open tray"],
         "sources": [{"title": "Kiosk SOP 12", "kind": "site SOP"}]}]}))
    ps = PR.load(tmp_path)
    assert [p["origin"] for p in ps] == ["reference", "site"]
    (tmp_path / "site_procedures.json").write_text(json.dumps({"procedures": [
        {"id": "bad", "fault_classes": ["jam"], "components": [], "title": "t", "steps": ["s"], "sources": []}]}))
    with pytest.raises(ValueError, match="no source"):
        PR.load(tmp_path)


def test_vehicle_code_dictionary_is_offline_and_cited():
    info = PR.code_info(" p0301 ")
    assert info["code"] == "P0301" and "Misfire" in info["title"]
    assert info["causes"] and info["sources"] and "CC0" in info["source_note"]
    assert PR.code_info("PRN-JAM") is None                      # kiosk/app codes need site knowledge


def test_event_episode_shows_its_code_meanings(tmp_path):
    from fastapi.testclient import TestClient
    from edge.api import create_app
    from edge.device import Device, DeviceConfig
    from edge.sync_worker import SyncWorker
    from shared.embed import HashEmbedder
    d = Device(DeviceConfig(device_id="car", site_id="s", machine_id="v1", root=tmp_path / "car", profile="events",
                            component="engine"), HashEmbedder())
    try:
        d.start_baseline_capture(12)
        for _ in range(12):
            d.ingest_signal({"codes": {"HEARTBEAT": 1}}, 1.0)
        d.ingest_signal({"codes": {"P0301": 3, "P0171": 1}, "severity": 2}, 1.0)
        c = TestClient(create_app(d, SyncWorker(d, None, None), "op-1234567"))
        eid = d.episodes()[0]["episode_id"]
        r = c.get(f"/api/procedures?episode_id={eid}", headers={"X-Operator-Token": "op-1234567"}).json()
        assert [x["code"] for x in r["codes"]] == ["P0301", "P0171"] and all(x["title"] for x in r["codes"])
    finally:
        d.close()
