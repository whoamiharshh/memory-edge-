"""Signal profiles: one memory engine for many kinds of edge device (PS3 names robots, industrial systems, kiosks,
vehicles, mobile devices).

Every profile turns ITS input into the same 27-float fingerprint slot, z-scored later against that device's own
healthy baseline. Everything downstream (novelty gate, outcome verifier, policy, outbox, sync, cloud tallies) is
profile-agnostic. `fp_version` names the profile's feature definition: fingerprints of different versions are never
compared with each other (the cloud groups by component + confirmed fault class, K2).

  bearing-12k     fp-v2   CWRU-style bearing vibration at 12 kHz (the original pipeline, unchanged)
  rotating-hf     fp-rh1  any accelerometer >= 2 kHz on a rotating machine; bearing geometry from config; physics:
                          defect frequencies, order spectrum (imbalance/misalignment/looseness), velocity severity
  lowrate-accel   fp-lr1  phone / IMU / telematics accelerometer at ~20-1000 Hz, 1 or 3 axes (m/s^2): shaft-order
                          rules + indicative velocity; cannot see bearing defect frequencies (far above Nyquist)
  force-torque    fp-ft1  robot wrist force/torque, 6 channels (Fx Fy Fz Tx Ty Tz)
  events          fp-ev1  kiosks, vehicles, apps: error / event codes per time bucket; "fixed" = the codes stay away
  telemetry       fp-tm1  vehicle / machine counters and histograms between readouts
  acoustic        fp-ac1  a MICROPHONE (phone at 44.1/48 kHz, or any mic) next to a rotating machine: the same
                          defect-frequency and order physics as rotating-hf, on sound (bench/acoustic_uottawa.py)
A machine card (edge/machine_card.py) makes a profile machine-specific: the manufacturer's bearing geometry, speed,
severity table / limits and mains frequency (profile.apply_card).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import scipy.signal as ss
import scipy.stats as st

from edge import fault_hint
from edge import fingerprint as fp
from edge import physics as P

DIM = 27
_EPS = 1e-12


def _log(x: float) -> float:
    return math.log10(abs(float(x)) + _EPS)


def _stats(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    kurt = float(st.kurtosis(x, fisher=False)) if np.std(x) > 0 else 3.0
    skew = float(st.skew(x)) if np.std(x) > 0 else 0.0
    return [_log(rms), _log(peak), peak / (rms + _EPS), _log(kurt), skew]


def _bands(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float, n: int) -> list[float]:
    edges = np.geomspace(lo, hi, n + 1)
    return [_log(power[(freqs >= a) & (freqs < b)].sum()) for a, b in zip(edges[:-1], edges[1:])]


def _resample(x: np.ndarray, fs: float, target: float) -> np.ndarray:
    if abs(fs - target) < 1e-6:
        return x
    frac = Fraction(int(round(target)), int(round(fs))).limit_denominator(64)
    return ss.resample_poly(x, frac.numerator, frac.denominator, axis=0)


@dataclass
class Profile:
    name: str
    fp_version: str
    description: str
    default_component: str
    input_units: str
    analysis_fs: float | None          # None = any rate
    window: int                        # samples per window at analysis_fs (or per bucket for events)
    detects: list[str]
    cannot: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    min_std: float = 0.05              # baseline spread floor (log10 units); see fingerprint.Baseline.fit
    normal_factor: float | None = None # healthy radius factor; None = the gate default (2.0, swept on CWRU)
    target_false_alarm: float | None = None   # set the radius from healthy data at this alarm rate (events)
    learned_detector: bool = False     # edge/local_detector.py: learn from this machine's confirmed faults

    # --- to implement per profile ---
    def features(self, x, fs: float, rpm: float | None = None) -> np.ndarray:
        raise NotImplementedError

    def hint(self, raw_features: np.ndarray, z: np.ndarray | None = None) -> dict:
        """Fault-class hint for one window. `z` = the window's z-scores against THIS machine's healthy baseline, so
        rules can ask "what changed?" instead of applying absolute textbook thresholds."""
        return {"fault_class": "unknown", "why": "no physics rule for this profile; the technician classifies",
                "measured_accuracy": None}

    GROWN_Z = 3.0      # a shaft order "grew" when it is > 3 standard deviations above this machine's healthy level


    def diagnose(self, x, fs: float, rpm: float | None = None) -> dict:
        return {}

    def measured(self, fault_class: str) -> float | None:
        """Measured accuracy of this profile's hint for a class, if a benchmark measured it."""
        return None

    # Chosen on CWRU (the tuning data): healthy-window p99 of the strongest defect envelope score + 0.25 margin
    # (spike/signature_threshold.py). Checked on HUST (held out), per recording: 12/15 healthy below, 36/42 faulty above.
    SIGNATURE_THRESHOLD = 1.63

    def signature_score(self, raw_features: np.ndarray) -> float | None:
        """Strength of a physical fault signature in one window (None if this profile cannot tell)."""
        return None

    def fault_signature(self, raw_features: np.ndarray) -> bool | None:
        """Does this window carry a physical fault signature (True), clearly not (False), or unknown (None)?
        Used to word the suggestion when an episode opens at an untaught operating point."""
        s = self.signature_score(raw_features)
        return None if s is None else bool(s >= self.SIGNATURE_THRESHOLD)

    def describe(self) -> dict:
        return {"name": self.name, "fp_version": self.fp_version, "description": self.description,
                "input_units": self.input_units, "analysis_fs": self.analysis_fs, "window": self.window,
                "default_component": self.default_component, "detects": self.detects, "cannot": self.cannot,
                "params": self.params}

    card: Any = None                    # edge/machine_card.MachineCard, set by apply_card

    def apply_card(self, card) -> None:
        """Use the manufacturer's data from the machine card: bearing geometry (defect frequencies), nominal speed
        (when no tachometer), severity table / limits. The CWRU profile keeps its measured SKF 6205 geometry."""
        self.card = card
        if card is None:
            return
        g = card.geometry()
        if g is not None and hasattr(self, "geometry") and self.name != "bearing-12k":
            self.geometry = g
            self.params = self.params | {"bearing_from_card": g.name}
        if card.nominal_rpm and hasattr(self, "shaft_hz") and not self.shaft_hz:
            self.shaft_hz = card.nominal_rpm / 60.0

    def _severity(self, v_mm_s: float) -> dict:
        if self.card is not None:
            bounds, ref = self.card.severity_bounds()
            return P.severity_zone(v_mm_s, bounds, ref)
        return P.severity_zone(v_mm_s)

    def order_features(self, x, fs: float, rpm: float | None = None) -> list[float] | None:
        """Order-domain features for the fleet-learned fault hint (physics.order_features), or None when this
        profile has no bearing geometry, no shaft speed or too short a segment."""
        geo = getattr(self, "geometry", None)
        x = np.asarray(x, dtype=np.float64)
        shaft = rpm / 60.0 if rpm and rpm > 0 else getattr(self, "shaft_hz", None)
        if geo is None or not shaft or x.ndim != 1 or len(x) < P.ORDER_FEATURES_MIN_S * fs:
            return None
        return P.order_features(x, fs, float(shaft), geo)

    def windows(self, x: np.ndarray, fs: float) -> list[np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        if self.analysis_fs:
            x = _resample(x, fs, self.analysis_fs)
        hop = self.window // 2
        n = 1 + (len(x) - self.window) // hop if len(x) >= self.window else 0
        return [x[i * hop:i * hop + self.window] for i in range(n)]



def _grown_orders(z, idx: dict[str, int], blind: tuple = ()) -> dict | None:
    """The relative order rule (bench/motor_rules.py): which shaft order grew most versus the machine's own healthy
    state. 1x -> imbalance, 2x -> misalignment, 0.5x / 3x -> looseness; nothing grew -> not a shaft-rate fault.
    Measured on real drive-fed motors, where the ABSOLUTE textbook rules called every motor, healthy ones included,
    'looseness' (electrical harmonics of the drive sit on shaft orders)."""
    if z is None:
        return None
    zz = {k: float(z[i]) for k, i in idx.items() if k not in blind}
    best = max(zz, key=zz.get)
    if zz[best] <= Profile.GROWN_Z:
        return {"fault_class": "unknown", "why": "no shaft order grew above this machine's healthy level (largest: "
                f"{best} at {zz[best]:.1f} sigma): not a shaft-rate fault - inspect", "measured_accuracy": None}
    cls = {"1x": "imbalance", "2x": "misalignment"}.get(best, "looseness")
    return {"fault_class": cls, "why": f"{best} grew {zz[best]:.1f} sigma above this machine's healthy level "
                                       "(relative order rule)", "measured_accuracy": None,
            "grown_sigma": {k: round(v, 1) for k, v in zz.items()}}


# --------------------------------------------------------------------------------------------------------------
class BearingCWRU(Profile):
    def __init__(self, **params):
        super().__init__("bearing-12k", fp.FP_VERSION, "Bearing vibration at 12 kHz with SKF 6205 defect orders "
                         "(the CWRU-calibrated pipeline).", "bearing", "g", float(fp.FS), fp.WINDOW,
                         ["healthy vs abnormal", "inner race / outer race / ball hint (measured, K2)",
                          "severity zone (velocity)"], ["cage faults", "other bearing geometries"], params,
                         min_std=1e-6)   # unchanged: the CWRU numbers (K2/K3/sweep) were measured with it
        self.geometry = P.BEARINGS["SKF6205-CWRU"]
        self.shaft_hz = None

    def features(self, x, fs, rpm=None):
        return fp.features(_resample(np.asarray(x, dtype=np.float64), fs, fp.FS), fp.FS,
                           float("nan") if rpm is None else rpm)

    def hint(self, raw_features, z=None):
        return fault_hint.suggest(raw_features)

    def measured(self, fault_class):
        return fault_hint.MEASURED["per_class"].get(fault_class)

    def signature_score(self, raw_features):
        orders = np.asarray(raw_features)[fp.PHYSICS_SLICE][1:]          # bpfo, bpfi, bsf envelope log-ratios
        return None if not np.any(orders) else float(orders.max())

    def diagnose(self, x, fs, rpm=None):
        """Physics shown to the technician (display only; the hint above is the measured CWRU rule)."""
        x = _resample(np.asarray(x, dtype=np.float64), fs, fp.FS)
        out = {"severity": self._severity(P.velocity_rms_mm_s(x, fp.FS))}
        if rpm and rpm > 0:
            shaft, geo = rpm / 60.0, P.BEARINGS["SKF6205-CWRU"]
            freqs, amp = P.spectrum(x, fp.FS)
            out |= {"shaft_hz": round(shaft, 3), "shaft_source": "tachometer (recording)",
                    "defect_frequencies_hz": {k: round(v * shaft, 2) for k, v in geo.orders().items()},
                    "bearing": P.bearing_rule(P.bearing_defect_scores(x, fp.FS, shaft, geo, band=P.kurtogram_band(x, fp.FS)))
                               | {"band_hz": [round(v) for v in P.kurtogram_band(x, fp.FS)]},
                    "rotating": P.rotating_rules(freqs, amp, shaft)}
        return P.combine(out)


# --------------------------------------------------------------------------------------------------------------
class RotatingHF(Profile):
    """Any rotating machine with an accelerometer (in g) sampled >= 2 kHz. Analysis at 12.8 kHz, 4096 samples."""

    def __init__(self, bearing: str = "SKF6205-CWRU", shaft_hz: float | None = None, geometry: dict | None = None,
                 **params):
        self._init_rotating("rotating-hf", "fp-rh1", "Rotating machine, accelerometer >= 2 kHz", "g", 12800.0, 4096,
                            bearing, shaft_hz, geometry, params)

    def _init_rotating(self, name, fpv, what, units, fsa, window, bearing, shaft_hz, geometry, params):
        geo = P.custom_geometry(geometry) if geometry else P.BEARINGS[bearing]
        super().__init__(name, fpv, f"{what}; bearing geometry '{geo.name}'; physics from defect frequencies and the "
                         "order spectrum.", "bearing", units, fsa, window,
                         ["healthy vs abnormal", "inner/outer race/ball (envelope at geometry-derived defect "
                          "frequencies)", "imbalance / misalignment / looseness (order rules)", "severity zone"],
                         ["cage faults (reported, not classified)", "electrical faults"],
                         {"bearing": bearing, "shaft_hz": shaft_hz, **({"geometry": geometry} if geometry else {}),
                          **params})
        self.geometry = geo
        self.shaft_hz = shaft_hz

    def _shaft(self, rpm, freqs=None, amp=None):
        if rpm and rpm > 0:
            return rpm / 60.0
        if self.shaft_hz:
            return float(self.shaft_hz)
        return P.estimate_shaft_hz(freqs, amp, 5.0, 200.0) if freqs is not None else None

    def features(self, x, fs, rpm=None):
        fsa = self.analysis_fs
        x = _resample(np.asarray(x, dtype=np.float64), fs, fsa)
        x = x - x.mean()
        freqs, amp = P.spectrum(x, fsa)
        power = amp ** 2
        shaft = self._shaft(rpm, freqs, amp)
        out = _stats(x)                                                           # 0:5
        out += _bands(freqs, power, 10.0, min(fsa / 2, 6000.0), 10)               # 5:15
        if shaft:
            s = P.bearing_defect_scores(x, fsa, shaft, self.geometry)
            out += [s["bpfo"], s["bpfi"], s["bsf"], s["ftf"]]                     # 15:19
            o = P.order_amplitudes(freqs, amp, shaft, (0.5, 1, 2, 3))
            out += [_log(o.get(k, 0.0)) for k in ("0.5x", "1x", "2x", "3x")]      # 19:23
        else:
            out += [0.0] * 8
        out.append(_log(P.velocity_rms_mm_s(x, fsa)))                             # 23
        hf = power[freqs > 2000].sum() / (power.sum() + _EPS)
        out.append(_log(hf))                                                      # 24
        out.append(_log((freqs * power).sum() / (power.sum() + _EPS)))            # 25 spectral centroid
        env = np.abs(ss.hilbert(x))
        out.append(_log(st.kurtosis(env, fisher=False)))                          # 26 envelope impulsiveness
        return np.asarray(out, dtype=np.float64)

    def measured(self, fault_class):
        return HF_MEASURED.get(fault_class)

    def signature_score(self, raw_features):
        r = np.asarray(raw_features)
        return None if not np.any(r[15:23]) else float(max(r[15:18]))

    BEARING_MIN_SCORE = 1.0      # log10 ratio over the median envelope level (10x) to call a bearing defect

    def hint(self, raw_features, z=None):
        r = np.asarray(raw_features)
        defects = {"bpfo": r[15], "bpfi": r[16], "bsf": r[17]}
        if not np.any(r[15:23]):
            return {"fault_class": "unknown", "why": "no shaft speed (give rpm or shaft_hz)", "measured_accuracy": None}
        best = max(defects, key=defects.get)
        # (tried: also requiring the defect line to have GROWN vs the healthy baseline - it cut the HUST hint from
        # 97.6 % to 78.6 % and only helped on healthy windows, which never open an episode; reverted)
        if defects[best] >= self.BEARING_MIN_SCORE:
            cls = P.DEFECT_TO_CLASS[best]
            return {"fault_class": cls, "why": f"{best.upper()} ({self.geometry.name}) envelope energy "
                                               f"{10 ** defects[best]:.0f}x the median", "measured_accuracy":
                    HF_MEASURED.get(cls), "measured_source": "bench/results/hust_holdout.json"}
        rel = _grown_orders(z, {"0.5x": 19, "1x": 20, "2x": 21, "3x": 22})
        if rel is not None:
            return rel
        o = {k: 10 ** r[i] for k, i in (("0.5x", 19), ("1x", 20), ("2x", 21), ("3x", 22))}
        a1 = max(o["1x"], _EPS)
        if o["0.5x"] > 0.25 * a1 or o["3x"] > 0.25 * a1:
            cls, why = "looseness", "strong sub-harmonic or higher harmonics"
        elif o["2x"] >= 0.5 * a1:
            cls, why = "misalignment", f"2x is {o['2x'] / a1:.2f} of 1x"
        else:
            cls, why = "imbalance", f"1x dominates; no bearing defect frequency stands out"
        return {"fault_class": cls, "why": why + " (absolute order-spectrum heuristic: no baseline)",
                "measured_accuracy": None}

    def diagnose(self, x, fs, rpm=None):
        x = _resample(np.asarray(x, dtype=np.float64), fs, self.analysis_fs)
        freqs, amp = P.spectrum(x, self.analysis_fs)
        shaft = self._shaft(rpm, freqs, amp)
        out = {"severity": self._severity(P.velocity_rms_mm_s(x, self.analysis_fs)),
               "shaft_hz": round(shaft, 3) if shaft else None,
               "defect_frequencies_hz": {k: round(v * shaft, 2) for k, v in self.geometry.orders().items()} if shaft else None}
        if shaft:
            band = P.kurtogram_band(x, self.analysis_fs)
            out["bearing"] = P.bearing_rule(P.bearing_defect_scores(x, self.analysis_fs, shaft, self.geometry, band=band)) | {
                "band_hz": [round(v) for v in band]}
            out["rotating"] = P.rotating_rules(freqs, amp, shaft)
        return P.combine(out)


HF_MEASURED: dict[str, float] = {}      # filled from bench/results/hust_holdout.json when present (see _load_hf)


def _load_hf() -> None:
    import json
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "bench" / "results" / "hust_holdout.json"
    try:
        HF_MEASURED.update(json.loads(p.read_text())["physics_hint"]["per_class"])
    except (OSError, KeyError, ValueError):
        pass


_load_hf()


# --------------------------------------------------------------------------------------------------------------
class Acoustic(RotatingHF):
    """A microphone near a rotating machine (a phone's microphone at 44.1/48 kHz, or any mic). Sound carries the same
    bearing impacts and shaft orders as vibration, at a rate a phone CAN deliver (its motion sensor is capped at
    60 Hz in browsers). Analysis at 16 kHz, 8192-sample windows (0.51 s). A microphone is not a calibrated vibration
    sensor: no ISO velocity zone. Measured on real microphone recordings in bench/acoustic_uottawa.py."""

    def __init__(self, bearing: str = "SKF6205-CWRU", shaft_hz: float | None = None, geometry: dict | None = None,
                 **params):
        self._init_rotating("acoustic", "fp-ac1", "Microphone next to a rotating machine (phone mic or any mic)",
                            "audio (uncalibrated)", 16000.0, 8192, bearing, shaft_hz, geometry, params)
        self.default_component = params.get("component", "bearing")
        self.cannot = ["calibrated severity (a microphone is not an ISO vibration sensor)",
                       "faults on machines louder than the one you listen to (background noise)"]

    def features(self, x, fs, rpm=None):
        f = super().features(x, fs, rpm)
        y = _resample(np.asarray(x, dtype=np.float64), fs, self.analysis_fs) if fs != self.analysis_fs else np.asarray(x, dtype=np.float64)
        freqs, amp = P.spectrum(y, self.analysis_fs)
        f[23] = _log(np.sqrt((amp[(freqs >= 2000) & (freqs <= 7500)] ** 2).sum()))   # replaces velocity (no calibration)
        return f

    def measured(self, fault_class):
        return None

    def diagnose(self, x, fs, rpm=None):
        out = super().diagnose(x, fs, rpm)
        out["severity"] = {"zone": None, "text": "no ISO severity zone from a microphone (not a calibrated vibration "
                                                 "sensor); use the healthy-radius and fault-frequency evidence"}
        return out


# --------------------------------------------------------------------------------------------------------------
class LowRateAccel(Profile):
    """Phone / IMU / telematics accelerometer. Input (n,) or (n, 3) in m/s^2 (the browser DeviceMotion unit)."""

    def __init__(self, shaft_hz: float | None = None, window: int = 256, **params):
        super().__init__("lowrate-accel", "fp-lr1", "Low-rate accelerometer (phone, IMU, telematics), ~20-1000 Hz, "
                         "1 or 3 axes. Sees shaft-rate problems; not bearing defect frequencies.", "fan", "m/s^2",
                         None, window,
                         ["healthy vs abnormal", "imbalance / misalignment / looseness (order rules)",
                          "indicative velocity level"],
                         ["bearing defect frequencies (above the ~30 Hz Nyquist of a 60 Hz phone)",
                          "calibrated severity (phone sensors are not calibrated)"],
                         {"shaft_hz": shaft_hz, "window": window, **params})
        self.shaft_hz = shaft_hz

    def windows(self, x, fs):
        x = np.asarray(x, dtype=np.float64)
        hop = self.window // 2
        n = 1 + (len(x) - self.window) // hop if len(x) >= self.window else 0
        return [x[i * hop:i * hop + self.window] for i in range(n)]

    def _split(self, x):
        x = np.asarray(x, dtype=np.float64)
        axes = x if x.ndim == 2 else x[:, None]
        axes = axes - axes.mean(axis=0)                  # removes gravity (a constant per axis)
        mag = np.sqrt((axes ** 2).sum(axis=1)) if axes.shape[1] > 1 else axes[:, 0]
        sig = axes[:, int(np.argmax(axes.std(axis=0)))]  # the axis with most motion, for the spectrum
        return axes, mag, sig

    def _shaft(self, rpm, freqs, amp):
        if rpm and rpm > 0:
            return rpm / 60.0
        if self.shaft_hz:
            return float(self.shaft_hz)
        return P.estimate_shaft_hz(freqs, amp, 2.0, None)

    last_fs: float | None = None       # sampling rate of the latest window; the hint needs it for Nyquist limits

    def features(self, x, fs, rpm=None):
        self.last_fs = float(fs)
        axes, mag, sig = self._split(x)
        freqs, amp = P.spectrum(sig, fs)
        power = amp ** 2
        shaft = self._shaft(rpm, freqs, amp)
        per_axis = [_log(np.sqrt(np.mean(axes[:, i] ** 2))) for i in range(axes.shape[1])]
        per_axis += [per_axis[-1]] * (3 - len(per_axis))
        out = _stats(mag)                                                         # 0:5
        out += per_axis[:3]                                                       # 5:8
        out += _bands(freqs, power, 0.5, fs / 2, 10)                              # 8:18
        dom = P.estimate_shaft_hz(freqs, amp, 0.5, None) or 0.0
        out.append(_log(dom))                                                     # 18
        out.append(_log((freqs * power).sum() / (power.sum() + _EPS)))            # 19
        tot = float(np.sqrt(power.sum())) + _EPS
        if shaft:
            o = P.order_amplitudes(freqs, amp, shaft, (0.5, 1, 2, 3))
            out += [_log(o.get(k, 0.0) / tot) for k in ("0.5x", "1x", "2x", "3x")]  # 20:24
            out.append(_log(shaft))                                                # 24
        else:
            out += [0.0] * 5
        out.append(float(np.mean(np.abs(np.diff(np.sign(sig))) > 0)))             # 25 zero-crossing rate
        out.append(_log(P.velocity_rms_mm_s(sig / P.G, fs, (1.0, 1000.0))))       # 26 indicative velocity
        return np.asarray(out, dtype=np.float64)

    def hint(self, raw_features, z=None):
        r = np.asarray(raw_features)
        if not np.any(r[20:25]):
            return {"fault_class": "unknown", "why": "no shaft speed", "measured_accuracy": None}
        o = {k: 10 ** r[i] for k, i in (("0.5x", 20), ("1x", 21), ("2x", 22), ("3x", 23))}
        shaft = 10 ** r[24]
        nyq = (self.last_fs or 0.0) / 2
        blind = [k for k, m in (("2x", 2), ("3x", 3)) if nyq and m * shaft >= 0.98 * nyq]
        a1 = max(o["1x"], _EPS)
        note = (f"; {', '.join(blind)} ({', '.join(f'{m * shaft:.0f} Hz' for m in (2, 3) if f'{m}x' in blind)}) above "
                f"the {nyq:.0f} Hz Nyquist limit: misalignment/looseness NOT assessable at this sampling rate"
                if blind else "")
        rel = _grown_orders(z, {"0.5x": 20, "1x": 21, "2x": 22, "3x": 23}, tuple(blind))
        # the phone's order features are SHARES of the total: imbalance lifts 1x and the total together, so the share
        # barely moves. If the overall level grew (> GROWN_Z) the current order PATTERN names the fault (below);
        # if nothing grew at all, the relative verdict ("no shaft order grew") stands.
        level_grew = z is not None and float(max(z[0], *z[5:8])) > self.GROWN_Z
        if rel is not None and (rel["fault_class"] != "unknown" or not level_grew):
            return rel | {"why": rel["why"] + note + " (phone-grade sensor)", "not_assessable": blind}
        if o["0.5x"] > 0.25 * a1 or ("3x" not in blind and o["3x"] > 0.25 * a1):
            cls, why = "looseness", "strong sub-harmonic or higher harmonics"
        elif "2x" not in blind and o["2x"] >= 0.5 * a1:
            cls, why = "misalignment", f"2x is {o['2x'] / a1:.2f} of 1x"
        else:
            cls, why = "imbalance", f"1x dominates at {shaft:.1f} Hz"
        return {"fault_class": cls, "why": why + note + " (order-spectrum heuristic, phone-grade sensor)",
                "measured_accuracy": None, "not_assessable": blind}

    def diagnose(self, x, fs, rpm=None):
        axes, mag, sig = self._split(x)
        freqs, amp = P.spectrum(sig, fs)
        shaft = self._shaft(rpm, freqs, amp)
        return {"shaft_hz": round(shaft, 3) if shaft else None, "shaft_source": "rpm" if rpm else
                ("configured" if self.shaft_hz else "estimated from the strongest peak"),
                "rotating": P.rotating_rules(freqs, amp, shaft),
                "severity": P.severity_zone(P.velocity_rms_mm_s(sig / P.G, fs, (1.0, 1000.0))) |
                {"reference": "INDICATIVE only: phone accelerometers are not calibrated vibration sensors"}}


# --------------------------------------------------------------------------------------------------------------
class ForceTorque(Profile):
    """Robot wrist force/torque, (n, 6) = Fx Fy Fz (N) Tx Ty Tz (N·m), any short window (>= 5 samples)."""

    def __init__(self, window: int = 15, **params):
        super().__init__("force-torque", "fp-ft1", "Robot force/torque sensor, 6 channels.", "robot_gripper",
                         "N, N·m", None, window, ["healthy vs abnormal motion (collision, obstruction, slip)",
                                                   "subtle failures once the robot has confirmed examples (learned "
                                                   "detector, bench/robot_model.py)"],
                         ["failure type (the technician classifies)"], {"window": window, **params},
                         learned_detector=True)

    def windows(self, x, fs):
        x = np.asarray(x, dtype=np.float64)
        return [x[i:i + self.window] for i in range(0, len(x) - self.window + 1, max(1, self.window // 2))]

    def features(self, x, fs=1.0, rpm=None):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 6)
        sl = lambda v: math.copysign(math.log1p(abs(v)), v)            # signed log: forces span 0.1 N .. 1000 N
        out = []
        for c in range(6):
            out += [sl(x[:, c].mean()), sl(x[:, c].std()), sl(np.abs(x[:, c]).max())]         # 0:18
        f, t = np.linalg.norm(x[:, :3], axis=1), np.linalg.norm(x[:, 3:], axis=1)
        out += [sl(f.mean()), sl(f.max()), sl(f.std()), sl(t.mean()), sl(t.max()), sl(t.std())]  # 18:24
        out += [sl(f[-1] - f[0]), sl(t[-1] - t[0]), sl(np.abs(np.diff(f)).max() if len(f) > 1 else 0.0)]  # 24:27
        return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------------------------------------------
class Events(Profile):
    """Kiosks, vehicles, apps: each window is a time bucket of error/event codes, e.g.
    {"codes": {"PRN-JAM": 3, "NET-TIMEOUT": 1}, "severity": 2}. Codes are hashed into 24 signed buckets (feature
    hashing, like a bag of words), plus total count, distinct codes and max severity. The healthy baseline is normal
    buckets (few or no errors); 'fixed' = N buckets in a row back to normal, i.e. the error stayed away."""

    N_HASH = 24

    def __init__(self, window: int = 1, **params):
        super().__init__("events", "fp-ev1", "Error / event codes per time bucket (kiosk, vehicle DTCs, app "
                         "crashes).", "kiosk", "codes per bucket", None, window,
                         ["new vs known error patterns", "fix verified when the codes stay away for N buckets"],
                         ["physical diagnosis (the codes carry the meaning)"], {"window": window, **params},
                         target_false_alarm=0.01)   # 1 % of unseen healthy buckets (bench/events_hdfs.py)

    @staticmethod
    def _bucket(code: str) -> tuple[int, float]:
        h = hashlib.sha1(code.strip().upper().encode()).digest()
        return h[0] % Events.N_HASH, 1.0 if h[1] % 2 == 0 else -1.0

    def features(self, x, fs=1.0, rpm=None):
        codes = dict(x.get("codes") or {})
        v = np.zeros(self.N_HASH)
        for code, n in codes.items():
            i, s = self._bucket(code)
            v[i] += s * math.log1p(max(0.0, float(n)))
        total = sum(max(0.0, float(n)) for n in codes.values())
        return np.concatenate([v, [math.log1p(total), math.log1p(len(codes)), float(x.get("severity", 0))]])

    def hint(self, raw_features, z=None):
        return {"fault_class": "unknown", "why": "event codes: the technician names the fault", "measured_accuracy": None}


# --------------------------------------------------------------------------------------------------------------
class Telemetry(Profile):
    """Vehicle / machine telemetry with cumulative counters and histograms (e.g. SCANIA Component X readouts).
    One window = the INCREMENT between two consecutive readouts: {"dt": time between readouts, "counters": {name:
    increment}, "histograms": {name: [bin increments]}}. Up to 8 counters (log rate each) and 6 histograms (log total
    rate, normalised centre and spread of the bin distribution); missing ones are 0; plus log dt = 27 numbers."""

    N_COUNTERS, N_HISTS = 8, 6

    def __init__(self, counters: list[str] | None = None, histograms: list[str] | None = None, **params):
        super().__init__("telemetry", "fp-tm1", "Vehicle/machine telemetry: cumulative counters and histograms between "
                         "readouts.", "engine", "counter increments", None, 1,
                         ["unusual use or load pattern vs this vehicle's own history"],
                         ["which part will fail (a trained fleet model can add a risk hint)"],
                         {"counters": counters, "histograms": histograms, **params})
        self.counters, self.histograms = counters, histograms

    def features(self, x, fs=1.0, rpm=None):
        dt = max(float(x.get("dt", 1.0)), 1e-6)
        cs = x.get("counters") or {}
        names = self.counters or sorted(cs)
        out = [math.log10(max(float(cs.get(n, 0.0)), 0.0) / dt + 1.0) for n in names[: self.N_COUNTERS]]
        out += [0.0] * (self.N_COUNTERS - len(out))
        hs = x.get("histograms") or {}
        hnames = self.histograms or sorted(hs)
        for n in hnames[: self.N_HISTS]:
            b = np.clip(np.asarray(hs.get(n, []), dtype=np.float64), 0, None)
            tot = float(b.sum())
            if tot <= 0 or len(b) < 2:
                out += [0.0, 0.0, 0.0]
                continue
            idx = np.arange(len(b)) / (len(b) - 1)
            c = float((idx * b).sum() / tot)
            out += [math.log10(tot / dt + 1.0), c, float(np.sqrt(((idx - c) ** 2 * b).sum() / tot))]
        out += [0.0, 0.0, 0.0] * (self.N_HISTS - min(len(hnames), self.N_HISTS))
        out.append(math.log10(dt + 1.0))
        return np.asarray(out, dtype=np.float64)


REGISTRY = {"bearing-12k": BearingCWRU, "rotating-hf": RotatingHF, "lowrate-accel": LowRateAccel,
            "force-torque": ForceTorque, "events": Events, "telemetry": Telemetry, "acoustic": Acoustic}
FP_VERSIONS = {"fp-v2": "bearing-12k", "fp-rh1": "rotating-hf", "fp-lr1": "lowrate-accel", "fp-ft1": "force-torque",
               "fp-ev1": "events", "fp-tm1": "telemetry", "fp-ac1": "acoustic"}


def make(name: str = "bearing-12k", **params) -> Profile:
    if name not in REGISTRY:
        raise ValueError(f"unknown profile {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
