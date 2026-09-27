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
