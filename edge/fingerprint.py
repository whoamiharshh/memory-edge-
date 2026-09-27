"""Vibration fingerprint: deterministic signal processing, no learned model.

A window of raw acceleration becomes a small vector of physically meaningful features:
  time domain   : RMS, peak, crest factor, kurtosis, skewness
  frequency     : log energy in log-spaced FFT bands (spectral shape)
  envelope      : log energy of the envelope spectrum in bands (classic bearing-fault analysis:
                  impacts from a defect amplitude-modulate a structural resonance)
Every feature is later z-scored against a healthy baseline, so distances mean "how many standard
deviations away from healthy" rather than raw units.

FP_VERSION must change whenever the feature definition changes: vectors from different versions
are not comparable, and the version is stored on every point.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import scipy.io as sio
import scipy.signal as ss
import scipy.stats as st

FP_VERSION = "fp-v2"
FS = 12_000                 # analysis sampling rate (Hz)
WINDOW = 4096               # samples per window (~0.34 s at 12 kHz)
HOP = 2048                  # 50 % overlap
N_BANDS = 12                # spectral bands
N_ENV_BANDS = 6             # envelope-spectrum bands
BAND_EDGES = np.geomspace(20.0, FS / 2, N_BANDS + 1)
ENV_EDGES = np.geomspace(10.0, 500.0, N_ENV_BANDS + 1)   # bearing defect rates live here at ~1800 rpm
# Bearing defect frequencies as multiples of shaft speed. Verified Fact: CWRU Bearing Information page,
# drive-end bearing SKF 6205-2RS JEM. Physics features need shaft speed (tachometer / drive speed) as input.
DEFECT_ORDERS = {"shaft_1x": 1.0, "bpfo": 3.5848, "bpfi": 5.4152, "bsf": 4.7135}
ENV_BANDPASS = (2000.0, 5500.0)   # demodulation band (structural resonance region); prototype parameter
N_HARMONICS = 3
ORDER_TOL = 0.03                  # +-3 % around each harmonic, absorbs slip and speed error
FEATURE_NAMES = (["rms", "peak", "crest", "kurtosis", "skewness"]
                 + [f"band{i}" for i in range(N_BANDS)] + [f"env{i}" for i in range(N_ENV_BANDS)]
                 + [f"order_{k}" for k in DEFECT_ORDERS])
PHYSICS_SLICE = slice(len(FEATURE_NAMES) - len(DEFECT_ORDERS), len(FEATURE_NAMES))
DIM = len(FEATURE_NAMES)
_EPS = 1e-12

# CWRU normal-baseline files: the CWRU pages do not state their sampling rate; secondary sources say 48 kHz.
# Assumption (documented in README): treat 97-100 as 48 kHz and decimate x4 to 12 kHz.
CWRU_48K_FILES = {97, 98, 99, 100}


def load_cwru(path: str | pathlib.Path) -> tuple[np.ndarray, float]:
    """Return (drive-end signal at FS Hz, rpm) for one CWRU .mat file."""
    path = pathlib.Path(path)
    m = sio.loadmat(path)
    key = next(k for k in m if k.endswith("DE_time"))
    x = m[key].ravel().astype(np.float64)
    rpm_keys = [k for k in m if k.endswith("RPM")]
    rpm = float(m[rpm_keys[0]].ravel()[0]) if rpm_keys else float("nan")
    if int(path.stem) in CWRU_48K_FILES:
        x = ss.decimate(x, 4, ftype="fir", zero_phase=True)   # anti-aliased 48 kHz -> 12 kHz
    return x, rpm


def windows(x: np.ndarray, size: int = WINDOW, hop: int = HOP) -> np.ndarray:
    """Split a signal into overlapping windows, shape (n, size). Trailing partial window dropped."""
    if len(x) < size:
        return np.empty((0, size))
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _band_log_energy(freqs: np.ndarray, power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = np.log10(power[sel].sum() + _EPS)
    return out


_SOS = ss.butter(4, ENV_BANDPASS, btype="bandpass", fs=FS, output="sos")


def order_features(w: np.ndarray, rpm: float, fs: float = FS) -> np.ndarray:
    """Envelope-spectrum energy at bearing defect frequencies (order domain), relative to the median
    envelope level, log10. Independent of which physical bearing (same geometry) produced the signal."""
    if not np.isfinite(rpm) or rpm <= 0:
        return np.zeros(len(DEFECT_ORDERS))
    band = ss.sosfiltfilt(_SOS, w) if fs == FS else w
    env = np.abs(ss.hilbert(band))
    env = env - env.mean()
    espec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freqs = np.fft.rfftfreq(len(env), 1.0 / fs)
    ref = np.median(espec[(freqs > 10) & (freqs < 1000)]) + _EPS
    shaft = rpm / 60.0
    out = []
    for order in DEFECT_ORDERS.values():
        tot = 0.0
        for h in range(1, N_HARMONICS + 1):
            f0 = h * order * shaft
            sel = (freqs >= f0 * (1 - ORDER_TOL)) & (freqs <= f0 * (1 + ORDER_TOL))
            tot += espec[sel].max() if sel.any() else 0.0
        out.append(np.log10(tot / ref + _EPS))
    return np.asarray(out)


def features(w: np.ndarray, fs: float = FS, rpm: float = float("nan")) -> np.ndarray:
    """Raw (un-normalised) fingerprint of one window. Shape (DIM,). rpm enables the physics features."""
    w = np.asarray(w, dtype=np.float64)
    w = w - w.mean()
    rms = float(np.sqrt(np.mean(w ** 2)))
    peak = float(np.max(np.abs(w)))
    crest = peak / (rms + _EPS)
    kurt = float(st.kurtosis(w, fisher=False))
    skew = float(st.skew(w))
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
    freqs = np.fft.rfftfreq(len(w), 1.0 / fs)
    bands = _band_log_energy(freqs, spec, BAND_EDGES)
    env = np.abs(ss.hilbert(w))
    env = env - env.mean()
    espec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    ebands = _band_log_energy(freqs, espec, ENV_EDGES)
    return np.concatenate([[np.log10(rms + _EPS), np.log10(peak + _EPS), crest, np.log10(kurt), skew],
                           bands, ebands, order_features(w, rpm, fs)])


def features_batch(ws: np.ndarray, fs: float = FS, rpm: float = float("nan")) -> np.ndarray:
    return np.stack([features(w, fs, rpm) for w in ws]) if len(ws) else np.empty((0, DIM))


@dataclass(frozen=True)
class Baseline:
    """Per-machine healthy statistics used to z-score fingerprints."""
    mean: np.ndarray
    std: np.ndarray
    fp_version: str = FP_VERSION

    @classmethod
    def fit(cls, healthy_raw: np.ndarray, fp_version: str = FP_VERSION) -> "Baseline":
        healthy_raw = np.asarray(healthy_raw, dtype=np.float64)
        if len(healthy_raw) < 5:
            raise ValueError("need at least 5 healthy windows to fit a baseline")
        std = healthy_raw.std(axis=0)
        return cls(mean=healthy_raw.mean(axis=0), std=np.where(std < 1e-6, 1e-6, std), fp_version=fp_version)

    def z(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fp_version": self.fp_version}

    @classmethod
    def from_dict(cls, d: dict, fp_version: str = FP_VERSION) -> "Baseline":
        if d.get("fp_version") != fp_version:
            raise ValueError(f"baseline built with {d.get('fp_version')}, code is {fp_version}")
        return cls(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]), fp_version=fp_version)
