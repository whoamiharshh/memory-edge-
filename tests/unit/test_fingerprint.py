"""Unit tests for edge/fingerprint.py. Synthetic signals with analytically known answers; no dataset needed."""
import numpy as np
import pytest
import scipy.io as sio

from edge import fingerprint as fp

T = np.arange(fp.WINDOW) / fp.FS
K_CYCLES = 341                                   # whole cycles per window -> exact sine statistics
F_SINE = fp.FS * K_CYCLES / fp.WINDOW            # ~999 Hz
BAND0 = 5                                        # first spectral band column
ENV0 = BAND0 + fp.N_BANDS
ORD0 = ENV0 + fp.N_ENV_BANDS


def sine(amp=1.0, f=F_SINE):
    return amp * np.sin(2 * np.pi * f * T)


def am_signal(order_name, rpm=1797.0, carrier=3500.0, seed=0):
    """Carrier inside the envelope band-pass, amplitude-modulated at one defect frequency."""
    fmod = fp.DEFECT_ORDERS[order_name] * rpm / 60.0
    noise = np.random.default_rng(seed).normal(0, 0.05, fp.WINDOW)
    return (1 + 0.8 * np.cos(2 * np.pi * fmod * T)) * np.sin(2 * np.pi * carrier * T) + noise


# ---- constants / layout ---------------------------------------------------------------------------
def test_feature_layout_is_consistent():
    assert fp.DIM == len(fp.FEATURE_NAMES) == 5 + fp.N_BANDS + fp.N_ENV_BANDS + len(fp.DEFECT_ORDERS)
    assert fp.FEATURE_NAMES[fp.PHYSICS_SLICE] == [f"order_{k}" for k in fp.DEFECT_ORDERS]
    assert fp.PHYSICS_SLICE.start == ORD0


# ---- windows --------------------------------------------------------------------------------------
def test_windows_count_and_hop():
    x = np.arange(10_000, dtype=float)
    w = fp.windows(x)
    assert w.shape == (1 + (10_000 - fp.WINDOW) // fp.HOP, fp.WINDOW)
    for i in range(len(w)):
        assert w[i, 0] == i * fp.HOP
        np.testing.assert_array_equal(w[i], x[i * fp.HOP:i * fp.HOP + fp.WINDOW])


def test_windows_short_signal_is_empty():
    w = fp.windows(np.zeros(fp.WINDOW - 1))
    assert w.shape == (0, fp.WINDOW)


def test_windows_exact_length_gives_one():
    assert fp.windows(np.zeros(fp.WINDOW)).shape == (1, fp.WINDOW)


# ---- time-domain features -------------------------------------------------------------------------
def test_sine_time_domain_statistics():
    f = fp.features(sine(2.0))
    assert f.shape == (fp.DIM,)
    assert np.all(np.isfinite(f))
    assert f[0] == pytest.approx(np.log10(2.0 / np.sqrt(2)), abs=1e-3)   # log10 RMS
    assert f[2] == pytest.approx(np.sqrt(2), rel=1e-2)                   # crest factor
    assert f[3] == pytest.approx(np.log10(1.5), abs=1e-3)                # log10 kurtosis (sine = 1.5)
    assert f[4] == pytest.approx(0.0, abs=1e-3)                          # skewness


def test_gaussian_noise_kurtosis_near_three():
    x = np.random.default_rng(1).normal(0, 1, fp.WINDOW)
    assert fp.features(x)[3] == pytest.approx(np.log10(3.0), abs=0.05)


def test_dc_offset_does_not_change_features():
    x = am_signal("bpfi")
    np.testing.assert_allclose(fp.features(x + 5.0, rpm=1797.0), fp.features(x, rpm=1797.0), atol=1e-6)


def test_amplitude_scaling_shifts_only_level_features():
    x = np.random.default_rng(2).normal(0, 1, fp.WINDOW)
    a, b = fp.features(x), fp.features(2.0 * x)
    assert b[0] - a[0] == pytest.approx(np.log10(2.0), abs=1e-9)                  # rms
    assert b[1] - a[1] == pytest.approx(np.log10(2.0), abs=1e-9)                  # peak
    np.testing.assert_allclose(b[2:5], a[2:5], atol=1e-9)                         # crest, kurt, skew: scale-free
    np.testing.assert_allclose(b[BAND0:ENV0] - a[BAND0:ENV0], 2 * np.log10(2.0), atol=1e-6)  # power bands


def test_features_are_deterministic():
    x = am_signal("bsf")
    np.testing.assert_array_equal(fp.features(x, rpm=1750.0), fp.features(x, rpm=1750.0))


# ---- spectral bands -------------------------------------------------------------------------------
def test_sine_energy_lands_in_its_band():
    bands = fp.features(sine())[BAND0:ENV0]
    expected = int(np.searchsorted(fp.BAND_EDGES, F_SINE, side="right")) - 1
    assert int(np.argmax(bands)) == expected


# ---- physics (order) features ---------------------------------------------------------------------
def test_order_features_without_rpm_are_zero():
    np.testing.assert_array_equal(fp.order_features(sine(), float("nan")), np.zeros(len(fp.DEFECT_ORDERS)))
    np.testing.assert_array_equal(fp.order_features(sine(), 0.0), np.zeros(len(fp.DEFECT_ORDERS)))
    np.testing.assert_array_equal(fp.features(sine())[fp.PHYSICS_SLICE], np.zeros(len(fp.DEFECT_ORDERS)))


@pytest.mark.parametrize("defect", ["bpfo", "bpfi", "bsf"])
def test_modulation_at_defect_frequency_dominates_that_order(defect):
    orders = fp.order_features(am_signal(defect) - am_signal(defect).mean(), 1797.0)
    defects = list(fp.DEFECT_ORDERS)[1:]          # skip shaft_1x, as bench.retrieval_vib's rule does
    assert defects[int(np.argmax(orders[1:]))] == defect


def test_order_features_follow_shaft_speed():
    """The same physical defect at a different speed must still map to the same order."""
    x = am_signal("bpfo", rpm=1730.0)
    right = fp.order_features(x, 1730.0)
    wrong = fp.order_features(x, 1400.0)          # wrong speed -> the defect peak falls outside the tolerance
    i = list(fp.DEFECT_ORDERS).index("bpfo")
    assert right[i] > wrong[i] + 1.0


# ---- batch ----------------------------------------------------------------------------------------
def test_features_batch_shapes():
    ws = np.stack([sine(), am_signal("bpfi")])
    out = fp.features_batch(ws, rpm=1797.0)
    assert out.shape == (2, fp.DIM)
    np.testing.assert_array_equal(out[1], fp.features(ws[1], rpm=1797.0))
    assert fp.features_batch(np.empty((0, fp.WINDOW))).shape == (0, fp.DIM)


# ---- baseline -------------------------------------------------------------------------------------
def test_baseline_needs_five_windows():
    with pytest.raises(ValueError):
        fp.Baseline.fit(np.zeros((4, fp.DIM)))


def test_baseline_z_scores_and_std_floor():
    rng = np.random.default_rng(3)
    healthy = rng.normal(5.0, 2.0, (200, fp.DIM))
    healthy[:, 0] = 7.0                                   # constant feature -> std floored, no div by zero
    b = fp.Baseline.fit(healthy)
    assert b.std[0] == 1e-6
    z = b.z(healthy)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(z[:, 1:].std(axis=0), 1.0, atol=1e-9)
    assert np.all(np.isfinite(z))


def test_baseline_roundtrip_and_version_check():
    b = fp.Baseline.fit(np.random.default_rng(4).normal(size=(10, fp.DIM)))
    b2 = fp.Baseline.from_dict(b.to_dict())
    np.testing.assert_array_equal(b.mean, b2.mean)
    np.testing.assert_array_equal(b.std, b2.std)
    bad = b.to_dict() | {"fp_version": "fp-v0"}
    with pytest.raises(ValueError, match="fp-v0"):
        fp.Baseline.from_dict(bad)


# ---- CWRU loader (synthetic .mat files in CWRU's key layout) --------------------------------------
def _write_mat(path, x, rpm=None):
    num = path.stem.zfill(3)
    d = {f"X{num}_DE_time": x.reshape(-1, 1)}
    if rpm is not None:
        d[f"X{num}RPM"] = np.array([[rpm]])
    sio.savemat(path, d)


def test_load_cwru_12k_file_keeps_signal_and_reads_rpm(tmp_path):
    x = np.random.default_rng(5).normal(size=8000)
    _write_mat(tmp_path / "105.mat", x, rpm=1797)
    y, rpm = fp.load_cwru(tmp_path / "105.mat")
    np.testing.assert_allclose(y, x)
    assert rpm == 1797.0


def test_load_cwru_missing_rpm_is_nan(tmp_path):
    _write_mat(tmp_path / "130.mat", np.zeros(100))
    assert np.isnan(fp.load_cwru(tmp_path / "130.mat")[1])


def test_load_cwru_normal_file_is_antialiased_decimation(tmp_path):
    """Normal files are treated as 48 kHz (documented assumption) and decimated x4 to 12 kHz.
    A 1 kHz tone must survive; a 10 kHz tone (above the new 6 kHz Nyquist) must be filtered, not aliased."""
    fs_in = 48_000
    t = np.arange(fs_in) / fs_in
    _write_mat(tmp_path / "97.mat", np.sin(2 * np.pi * 1000 * t), rpm=1796)
    _write_mat(tmp_path / "98.mat", np.sin(2 * np.pi * 10_000 * t), rpm=1772)
    keep, _ = fp.load_cwru(tmp_path / "97.mat")
    alias, _ = fp.load_cwru(tmp_path / "98.mat")
    assert len(keep) == fs_in // 4
    mid = slice(1000, -1000)                              # ignore filter edge effects
    assert np.sqrt(np.mean(keep[mid] ** 2)) == pytest.approx(1 / np.sqrt(2), rel=0.02)
    assert np.sqrt(np.mean(alias[mid] ** 2)) < 0.01
