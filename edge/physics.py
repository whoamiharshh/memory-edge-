"""Physics and signal maths for diagnosis support. Deterministic, no learned model; every rule states its source.

1. Bearing defect frequencies from geometry (standard kinematic equations, n rolling elements of diameter d on
   pitch diameter D, contact angle phi, shaft frequency fr):
       FTF  = fr/2 * (1 - d/D cos phi)                      cage
       BPFO = n*fr/2 * (1 - d/D cos phi)                    outer race
       BPFI = n*fr/2 * (1 + d/D cos phi)                    inner race
       BSF  = D*fr/(2d) * (1 - (d/D cos phi)^2)             ball spin; a ball defect hits both races -> 2*BSF
   Checked against the CWRU Bearing Data Center table for the SKF 6205 (tests/unit/test_physics.py): 3.5848 /
   5.4152 / 0.3983 / 4.7135 (=2*BSF) x shaft speed.
2. Vibration velocity from acceleration (integration in the frequency domain, 10-1000 Hz band) and a severity zone
   using the ISO 10816-3 zone boundaries for group 2 machines (15-300 kW) on rigid foundations: A/B 1.4, B/C 2.8,
   C/D 4.5 mm/s RMS (values from secondary sources; the standard itself was not reproduced). For other machine
   groups, foundations or small motors the zone is INDICATIVE only, and the UI says so.
3. Classic rotating-machinery rules on the order spectrum (shaft speed must be known or estimated):
       imbalance      dominant 1x, little 2x                (radial, once-per-revolution force)
       misalignment   strong 2x relative to 1x               (commonly 2x >= ~0.5 x 1x)
       looseness      many harmonics and/or 0.5x sub-harmonics
   These are textbook heuristics (e.g. ISO 18436-2 training material; vendor application notes). They are HINTS with
   no measured accuracy on our data; the technician confirms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

G = 9.80665
_EPS = 1e-12
ISO10816_G2_RIGID_MM_S = (1.4, 2.8, 4.5)           # A/B, B/C, C/D boundaries (velocity RMS)
ZONE_TEXT = {"A": "newly commissioned level", "B": "acceptable for unrestricted long-term operation",
             "C": "unsatisfactory for long-term operation; plan corrective action",
             "D": "severe enough to cause damage"}


@dataclass(frozen=True)
class BearingGeometry:
    n_elements: int
    ball_d: float          # any length unit, same as pitch_d
    pitch_d: float
    contact_deg: float = 0.0
    name: str = ""

    def orders(self) -> dict[str, float]:
        """Defect frequencies as multiples of shaft speed."""
        r = self.ball_d / self.pitch_d * math.cos(math.radians(self.contact_deg))
        bsf = self.pitch_d / (2 * self.ball_d) * (1 - r * r)
        return {"ftf": 0.5 * (1 - r), "bpfo": self.n_elements / 2 * (1 - r),
                "bpfi": self.n_elements / 2 * (1 + r), "bsf": 2 * bsf}


# Known geometries. SKF 6205 from the CWRU Bearing Data Center (inches). HUST 6204-6208 from Hong & Thuan, BMC Res
# Notes 2023 (mm): ball diameter and count as published; pitch diameter = (bore + outside diameter) / 2, the usual
# approximation for deep-groove ball bearings (the paper gives no pitch diameter). Contact angle 0 (deep groove).
BEARINGS = {
    "SKF6205-CWRU": BearingGeometry(9, 0.3126, 1.537, 0.0, "SKF 6205-2RS JEM (CWRU drive end)"),
    "6204": BearingGeometry(8, 7.6, (20 + 47) / 2, 0.0, "6204 (HUST)"),
    "6205": BearingGeometry(9, 7.8, (25 + 52) / 2, 0.0, "6205 (HUST)"),
    "6206": BearingGeometry(9, 9.0, (30 + 62) / 2, 0.0, "6206 (HUST)"),
    "6207": BearingGeometry(9, 11.0, (35 + 72) / 2, 0.0, "6207 (HUST)"),
    "6208": BearingGeometry(9, 12.0, (40 + 80) / 2, 0.0, "6208 (HUST)"),
}


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum (Hann window, amplitude-corrected) of a zero-mean signal."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    w = np.hanning(len(x))
    amp = 2.0 * np.abs(np.fft.rfft(x * w)) / (w.sum() + _EPS)
    return np.fft.rfftfreq(len(x), 1.0 / fs), amp


def peak_near(freqs: np.ndarray, amp: np.ndarray, f0: float, tol: float = 0.03) -> float:
    """Largest amplitude within +-tol*f0 of f0, but never a window narrower than half a frequency bin: otherwise a
    low frequency (e.g. a ~10 Hz cage rate at 3 Hz resolution) can fall between bins and read as 'no energy' in
    one window and real energy in the next (seen on HUST; it made one feature a 13-million-sigma outlier)."""
    half_bin = 0.5 * (freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    w = max(tol * f0, half_bin)
    sel = (freqs >= f0 - w) & (freqs <= f0 + w)
    return float(amp[sel].max()) if sel.any() else 0.0


def velocity_rms_mm_s(acc_g: np.ndarray, fs: float, band: tuple[float, float] = (10.0, 1000.0)) -> float:
    """RMS vibration velocity (mm/s) from acceleration in g, by integration in the frequency domain over `band`
    (the ISO 10816 measurement band). Returns nan if the band lies above Nyquist."""
    lo, hi = band[0], min(band[1], fs / 2 * 0.98)
    if hi <= lo:
        return float("nan")
    a = (np.asarray(acc_g, dtype=np.float64) - np.mean(acc_g)) * G * 1000.0          # mm/s^2
    spec = np.fft.rfft(a)
    f = np.fft.rfftfreq(len(a), 1.0 / fs)
    sel = (f >= lo) & (f <= hi)
    v = np.zeros_like(spec)
    v[sel] = spec[sel] / (2j * np.pi * f[sel])
    vel = np.fft.irfft(v, n=len(a))
    return float(np.sqrt(np.mean(vel ** 2)))


def severity_zone(v_rms_mm_s: float, bounds: tuple[float, float, float] = ISO10816_G2_RIGID_MM_S) -> dict:
    if not math.isfinite(v_rms_mm_s):
        return {"zone": None, "text": "velocity not measurable at this sampling rate"}
    zone = "A" if v_rms_mm_s < bounds[0] else "B" if v_rms_mm_s < bounds[1] else "C" if v_rms_mm_s < bounds[2] else "D"
    return {"zone": zone, "velocity_mm_s": round(v_rms_mm_s, 3), "text": ZONE_TEXT[zone],
            "reference": "ISO 10816-3 group 2 rigid boundaries 1.4/2.8/4.5 mm/s (indicative for other machines)"}


def estimate_shaft_hz(freqs: np.ndarray, amp: np.ndarray, lo: float = 2.0, hi: float | None = None) -> float | None:
    """Rough shaft speed when no tachometer exists: the strongest spectral peak in [lo, hi]. For rotating machines
    with imbalance this is usually 1x; it is an ESTIMATE and the UI says so."""
    hi = hi or freqs[-1]
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any() or amp[sel].max() <= 0:
        return None
    return float(freqs[sel][int(np.argmax(amp[sel]))])


def order_amplitudes(freqs: np.ndarray, amp: np.ndarray, shaft_hz: float, orders=(0.5, 1, 2, 3, 4)) -> dict[str, float]:
    return {f"{o}x": peak_near(freqs, amp, o * shaft_hz) for o in orders if o * shaft_hz < freqs[-1]}


def rotating_rules(freqs: np.ndarray, amp: np.ndarray, shaft_hz: float | None) -> dict:
    """Imbalance / misalignment / looseness hint from the order spectrum. Returns fault_class + why (or unknown)."""
    if not shaft_hz:
        return {"fault_class": "unknown", "why": "shaft speed unknown", "rule": None}
    o = order_amplitudes(freqs, amp, shaft_hz)
    blind = [k for k, m in (("2x", 2), ("3x", 3), ("4x", 4)) if m * shaft_hz >= freqs[-1]]
    if "2x" in blind:                      # above Nyquist: silence there is NOT evidence of "no misalignment"
        return {"fault_class": "imbalance" if o.get("1x", 0) > _EPS else "unknown",
                "why": f"1x at {shaft_hz:.1f} Hz; 2x ({2 * shaft_hz:.0f} Hz) is above the {freqs[-1]:.0f} Hz Nyquist "
                       "limit, so misalignment and looseness are NOT assessable at this sampling rate",
                "rule": "textbook order-spectrum heuristic; no measured accuracy", "not_assessable": blind,
                "orders": {k: round(v, 6) for k, v in o.items()}}
    a1, a2 = o.get("1x", 0.0), o.get("2x", 0.0)
    harmonics = sum(1 for k in ("3x", "4x") if o.get(k, 0.0) > 0.25 * max(a1, _EPS))
    sub = o.get("0.5x", 0.0) > 0.25 * max(a1, _EPS)
    if a1 <= _EPS:
        return {"fault_class": "unknown", "why": "no energy at 1x", "rule": None, "orders": o}
    if sub or harmonics >= 2:
        cls, why = "looseness", f"{harmonics} strong higher harmonics{' and a 0.5x sub-harmonic' if sub else ''}"
    elif a2 >= 0.5 * a1:
        cls, why = "misalignment", f"2x is {a2 / a1:.2f} of 1x (>= 0.5)"
    else:
        cls, why = "imbalance", f"1x dominates (2x is {a2 / a1:.2f} of 1x)"
    return {"fault_class": cls, "why": why, "rule": "textbook order-spectrum heuristic; no measured accuracy",
            "orders": {k: round(v, 6) for k, v in o.items()}}


_SOS_CACHE: dict[tuple, np.ndarray] = {}


def envelope_spectrum(x: np.ndarray, fs: float, band: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Envelope (demodulation) spectrum: band-pass around a structural resonance, Hilbert envelope, spectrum.
    Default band 2-5.5 kHz (the CWRU prototype band), clipped below Nyquist."""
    band = band or (2000.0, 5500.0)
    hi = min(band[1], fs / 2 * 0.9)
    y = np.asarray(x, dtype=np.float64) - np.mean(x)
    if hi > band[0]:
        key = (round(fs), band[0], round(hi))
        if key not in _SOS_CACHE:
            _SOS_CACHE[key] = ss.butter(4, (band[0], hi), btype="bandpass", fs=fs, output="sos")
        y = ss.sosfiltfilt(_SOS_CACHE[key], y)
    env = np.abs(ss.hilbert(y))
    env = env - env.mean()
    p = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    return np.fft.rfftfreq(len(env), 1.0 / fs), p


def kurtogram_band(x: np.ndarray, fs: float, lo: float = 1000.0, step: float = 1.4) -> tuple[float, float]:
    """A simple kurtogram: of a fixed grid of bands between `lo` and 0.45 fs (widths of one and two grid steps), the
    band whose band-passed signal is most impulsive (highest kurtosis) - where defect impacts ring the structure.
    Measured (spike/kurtogram.py, per recording): CWRU outer race 67 % -> 92 %, overall 67 % -> 75 %; HUST unchanged
    at 97.6 %."""
    import scipy.stats as st
    edges = [lo]
    while edges[-1] * step < 0.45 * fs:
        edges.append(edges[-1] * step)
    cands = list(zip(edges[:-1], edges[1:])) + [(a, c) for a, c in zip(edges[:-2], edges[2:])]
    if not cands:
        return (2000.0, 5500.0)
    y = np.asarray(x, dtype=np.float64) - np.mean(x)
    best, bk = cands[0], -np.inf
    for a, b in cands:
        k = float(st.kurtosis(ss.sosfiltfilt(ss.butter(4, (a, b), btype="bandpass", fs=fs, output="sos"), y)))
        if k > bk:
            best, bk = (a, b), k
    return best


def bearing_defect_scores(x: np.ndarray, fs: float, shaft_hz: float, geometry: BearingGeometry,
                          n_harmonics: int = 3, tol: float = 0.03, band: tuple[float, float] | None = None) -> dict[str, float]:
    """log10 of envelope energy at each defect frequency (+harmonics) relative to the median envelope level."""
    f, p = envelope_spectrum(x, fs, band)
    ref = float(np.median(p[(f > 10) & (f < 1000)])) + _EPS
    out = {}
    for name, order in geometry.orders().items():
        tot = sum(peak_near(f, p, h * order * shaft_hz, tol) for h in range(1, n_harmonics + 1))
        out[name] = math.log10(tot / ref + _EPS)
    return out


DEFECT_TO_CLASS = {"bpfo": "outer_race", "bpfi": "inner_race", "bsf": "ball", "ftf": "cage"}


BEARING_DOMINANT = 1.0      # log10 ratio (10x the median envelope level): a defect frequency clearly stands out


def bearing_rule(scores: dict[str, float]) -> dict:
    """Which bearing defect frequency dominates the envelope spectrum (FTF excluded: a cage fault is rare and not in
    our fault classes)."""
    cand = {k: v for k, v in scores.items() if k in ("bpfo", "bpfi", "bsf")}
    best = max(cand, key=cand.get)
    return {"fault_class": DEFECT_TO_CLASS[best], "why": f"{best.upper()} envelope energy dominates ({cand[best]:.2f})",
            "scores": {k: round(v, 3) for k, v in scores.items()}, "dominant": cand[best] >= BEARING_DOMINANT}


def combine(diag: dict) -> dict:
    """Make the two rules and the severity zone read correctly together (display text only)."""
    b, r, s = diag.get("bearing"), diag.get("rotating"), diag.get("severity") or {}
    if b and b.get("dominant") and r:
        r["why"] += " - SECONDARY: a bearing defect frequency dominates; defect impacts also create harmonics"
    if b and b.get("dominant") and s.get("zone") in ("A", "B"):
        s["text"] += ("; early bearing defects show in envelope/acceleration long before overall velocity rises, "
                      "so a low zone does not mean the bearing is healthy")
    return diag
