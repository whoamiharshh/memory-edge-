"""Vehicles on real data: a device with the telemetry profile, fed a real SCANIA truck's readouts, computes the same
risk probability as the trained model's own evaluation (bench/vehicle_scania.py); and the model file is plain
coefficients trained on real data (no pickle)."""
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCANIA = ROOT / "data" / "raw" / "scania" / "test_operational_readouts.csv"
needs_scania = pytest.mark.skipif(not SCANIA.exists(), reason="SCANIA data not downloaded (data/fetch_scania.py)")


def test_model_file_is_plain_coefficients_trained_on_real_data():
    m = json.loads((ROOT / "knowledge" / "vehicle_risk_model.json").read_text())
    assert "REAL data" in m["about"] and m["kind"].startswith("logistic regression")
    assert len(m["coef"]) == len(m["mean"]) == len(m["scale"]) == 27 + 27 + 4


@needs_scania
def test_device_risk_equals_the_trained_models_probability(tmp_path):
    import pandas as pd
    from bench.vehicle_scania import COUNTERS, HISTS, truck_features, windows
    from edge import vehicle_risk
    from edge.device import Device, DeviceConfig
    from shared.embed import HashEmbedder
    m = vehicle_risk.model()
    df = pd.read_csv(SCANIA, nrows=20000)
    checked = 0
    for vid, g in df.groupby("vehicle_id"):
        if len(g) < 13:
            continue
        W, age = windows(g)
        truck = {"W": W, "age": age, "n": len(g)}
        x = (truck_features(truck) - np.asarray(m["mean"])) / np.asarray(m["scale"])
        expect = 1 / (1 + np.exp(-(np.dot(m["coef"], x) + m["intercept"])))
        d = Device(DeviceConfig(device_id="t", site_id="s", machine_id=str(vid), root=tmp_path / str(vid),
                                profile="telemetry", profile_params={"counters": COUNTERS, "histograms": list(HISTS)},
                                component="engine"), HashEmbedder())
        try:
            d.fit_baseline(W[:10])
            d._tm_windows = 10
            for w in W[10:]:
                d.ingest_window(w)
                d._tm_windows += 1
            got = vehicle_risk.risk(W[-1], d.baseline, age, d._tm_windows + 1)
        finally:
            d.close()
        assert got["probability"] == pytest.approx(float(expect), abs=1e-4)
        checked += 1
        if checked == 3:
            break
    assert checked == 3
