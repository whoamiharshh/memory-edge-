"""A bearing whose defect line sits on a whole shaft harmonic (the MaFaulDa simulator: BPFO 2.998x, BPFI 5.002x) must
not turn a shaft-rate fault into a bearing fault (edge/profiles.py RotatingHF.hint, DECISIONS D46). Synthetic feature
vectors: only the positions the rule reads are set (15:18 defect lines, 19:23 shaft orders)."""
import numpy as np

from edge import profiles

MAFAULDA = {"n_elements": 8, "ball_d": 0.7145, "pitch_d": 2.8519, "name": "MaFaulDa simulator bearing"}


def vectors(bpfo_line_z: float, one_x_z: float):
    raw = np.zeros(27)
    raw[15], raw[16], raw[17] = 1.5, 0.3, 0.2           # BPFO envelope 10^1.5 = 32x the median: over the bearing bar
    raw[19:23] = -1.0                                    # shaft orders present (the "no shaft speed" guard)
    z = np.zeros(27)
    z[15], z[20] = bpfo_line_z, one_x_z
    return raw, z


def test_clash_detected_only_where_the_geometry_puts_a_line_on_a_harmonic():
    assert set(profiles.make("rotating-hf", geometry=MAFAULDA).harmonic_clash()) == {"bpfo", "bpfi"}
    assert profiles.make("rotating-hf", bearing="SKF6205-CWRU").harmonic_clash() == {}


def test_grown_1x_with_a_quiet_clashing_line_is_imbalance():
    h = profiles.make("rotating-hf", geometry=MAFAULDA, shaft_hz=30.0).hint(*vectors(bpfo_line_z=0.8, one_x_z=8.0))
    assert h["fault_class"] == "imbalance"
    assert "3x shaft harmonic" in h["why"]


def test_a_quiet_clashing_line_with_no_grown_order_says_inspect():
    # MaFaulDa misalignment: the line on the harmonic is strong but did not grow, no single shaft order grew either
    h = profiles.make("rotating-hf", geometry=MAFAULDA, shaft_hz=30.0).hint(*vectors(bpfo_line_z=0.8, one_x_z=1.0))
    assert h["fault_class"] == "unknown"
    assert "inspect" in h["why"]


def test_a_clashing_line_that_grew_stays_a_bearing_fault():
    h = profiles.make("rotating-hf", geometry=MAFAULDA, shaft_hz=30.0).hint(*vectors(bpfo_line_z=6.0, one_x_z=8.0))
    assert h["fault_class"] == "outer_race"


def test_without_a_clash_the_bearing_rule_is_unchanged():
    # HUST-like: a real bearing fault can lift 1x while its (non-harmonic) line barely grows - keep the bearing name
    h = profiles.make("rotating-hf", bearing="SKF6205-CWRU", shaft_hz=30.0).hint(*vectors(bpfo_line_z=0.8, one_x_z=8.0))
    assert h["fault_class"] == "outer_race"
