"""Machine card: the manufacturer's data picks the severity table / limits and the bearing geometry."""
import pytest

from edge import physics as P
from edge import profiles
from edge.machine_card import ISO10816_1_CLASS_I, MachineCard, load, save


def test_iso_group_follows_power_type_and_foundation():
    assert MachineCard(power_kw=75, foundation="rigid").severity_bounds()[0] == (1.4, 2.8, 4.5)       # group 2
    assert MachineCard(power_kw=75, foundation="flexible").severity_bounds()[0] == (2.3, 4.5, 7.1)
    assert MachineCard(power_kw=500, foundation="rigid").severity_bounds()[0] == (2.3, 4.5, 7.1)      # group 1
    assert MachineCard(power_kw=500, foundation="flexible").severity_bounds()[0] == (3.5, 7.1, 11.0)
    assert MachineCard(machine_type="pump_separate_driver", power_kw=40).severity_bounds()[0] == (2.3, 4.5, 7.1)
    assert MachineCard(machine_type="pump_integrated_driver", power_kw=40).severity_bounds()[0] == (1.4, 2.8, 4.5)
    small = MachineCard(power_kw=0.05, machine_type="fan")
    assert small.severity_bounds()[0] == ISO10816_1_CLASS_I and "10816-1" in small.severity_bounds()[1]


def test_manufacturer_limits_override_iso_and_are_cited():
    c = MachineCard(power_kw=75, limits_mm_s={"acceptable": 3.2, "trip": 6.0}, source="Pump manual rev 3, p. 41")
    b, ref = c.severity_bounds()
    assert b[1] == 3.2 and b[2] == 6.0 and "manufacturer" in ref and "p. 41" in ref
    assert c.acceptable_mm_s() == 3.2


@pytest.mark.parametrize("bad", [
    {"foundation": "wobbly"}, {"machine_type": "toaster"}, {"power_kw": -1}, {"bearing": "nope"},
    {"bearing_geometry": {"n_elements": 8, "ball_d": 30, "pitch_d": 20}},        # ball bigger than pitch
    {"limits_mm_s": {"acceptable": 4, "trip": 2}}, {"manufacturer": "x" * 400},
])
def test_invalid_cards_are_refused(bad):
    with pytest.raises(ValueError):
        MachineCard.from_dict(bad)


def test_catalogue_geometry_drives_the_profile_defect_frequencies(tmp_path):
    card = MachineCard.from_dict({"bearing_geometry": {"n_elements": 8, "ball_d": 6.77, "pitch_d": 28.5,
                                                       "name": "NSK 6203"}, "nominal_rpm": 1800})
    prof = profiles.make("rotating-hf")
    prof.apply_card(card)
    assert prof.geometry.orders()["bpfo"] == pytest.approx(P.BEARINGS["6203-UO"].orders()["bpfo"])
    assert prof.shaft_hz == pytest.approx(30.0)
    assert card.to_dict()["derived"]["defect_frequencies_hz"]["bpfo"] == pytest.approx(3.05 * 30, abs=0.5)
    save(tmp_path, card)
    assert load(tmp_path).bearing_geometry["name"] == "NSK 6203"


def test_severity_uses_the_card():
    prof = profiles.make("rotating-hf")
    prof.apply_card(MachineCard(limits_mm_s={"acceptable": 1.0}))
    z = prof._severity(1.5)
    assert z["zone"] == "C" and "manufacturer" in z["reference"]
