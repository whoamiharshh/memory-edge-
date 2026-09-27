"""Physics maths checked against known answers: the CWRU bearing table, analytic sine integration, and synthetic
order spectra for the imbalance / misalignment / looseness rules."""
import math

import numpy as np
import pytest

from edge import physics as P


def test_bearing_orders_reproduce_the_cwru_table():
    o = P.BEARINGS["SKF6205-CWRU"].orders()
    # CWRU Bearing Data Center, drive end SKF 6205: BPFI 5.4152, BPFO 3.5848, FTF 0.39828, BSF 4.7135 (= 2 x ball spin)
    assert o["bpfi"] == pytest.approx(5.4152, abs=2e-3)
    assert o["bpfo"] == pytest.approx(3.5848, abs=2e-3)
    assert o["ftf"] == pytest.approx(0.39828, abs=2e-3)
    assert o["bsf"] == pytest.approx(4.7135, abs=5e-3)


def test_bpfo_plus_bpfi_equals_number_of_balls():
    for g in P.BEARINGS.values():
        o = g.orders()
        assert o["bpfo"] + o["bpfi"] == pytest.approx(g.n_elements)


def test_velocity_of_a_pure_sine_matches_the_analytic_value():
    fs, f0, a_g = 5000.0, 50.0, 1.0                         # 1 g peak at 50 Hz
    t = np.arange(int(fs * 4)) / fs
    acc = a_g * np.sin(2 * np.pi * f0 * t)
    v_peak_mm_s = a_g * P.G * 1000 / (2 * np.pi * f0)     # integrate: a/omega
    assert P.velocity_rms_mm_s(acc, fs) == pytest.approx(v_peak_mm_s / math.sqrt(2), rel=0.02)


def test_severity_zones():
    assert P.severity_zone(1.0)["zone"] == "A"
    assert P.severity_zone(2.0)["zone"] == "B"
    assert P.severity_zone(3.0)["zone"] == "C"
    assert P.severity_zone(9.9)["zone"] == "D"
    assert P.severity_zone(float("nan"))["zone"] is None


def _orders_signal(fs, shaft, amps, secs=8, noise=0.01, seed=0):
    t = np.arange(int(fs * secs)) / fs
    x = sum(a * np.sin(2 * np.pi * o * shaft * t + o) for o, a in amps.items())
    return x + noise * np.random.default_rng(seed).normal(size=len(t))


@pytest.mark.parametrize("amps,expected", [
    ({1: 1.0, 2: 0.1}, "imbalance"),
    ({1: 1.0, 2: 0.8}, "misalignment"),
    ({1: 1.0, 2: 0.3, 3: 0.5, 4: 0.4}, "looseness"),
    ({0.5: 0.5, 1: 1.0}, "looseness"),
])
def test_rotating_rules(amps, expected):
    fs, shaft = 60.0, 20.0 / 3                              # a phone at 60 Hz on a machine at 400 rpm
    f, a = P.spectrum(_orders_signal(fs, shaft, amps), fs)
    assert P.rotating_rules(f, a, shaft)["fault_class"] == expected


def test_shaft_speed_estimate_finds_the_dominant_1x():
    fs, shaft = 60.0, 17.0
    f, a = P.spectrum(_orders_signal(fs, shaft, {1: 1.0, 2: 0.2}), fs)
    assert P.estimate_shaft_hz(f, a) == pytest.approx(shaft, abs=0.2)


def test_bearing_rule_finds_the_modulated_defect_frequency():
    fs, shaft = 12000.0, 29.95
    geo = P.BEARINGS["SKF6205-CWRU"]
    t = np.arange(int(fs * 1.0)) / fs
    for defect, cls in (("bpfi", "inner_race"), ("bpfo", "outer_race")):
        fd = geo.orders()[defect] * shaft
        carrier = np.sin(2 * np.pi * 3500 * t)               # resonance inside the 2-5.5 kHz demodulation band
        x = (1 + 0.8 * (np.sin(2 * np.pi * fd * t) > 0.95)) * carrier + 0.05 * np.random.default_rng(1).normal(size=len(t))
        assert P.bearing_rule(P.bearing_defect_scores(x, fs, shaft, geo))["fault_class"] == cls
