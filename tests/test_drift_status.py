import mlflow
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from flowradar.serving.api import create_app
from flowradar.serving.drift import drift_status
from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.registry import load_registered_model, train_and_register_model
DATES = pd.date_range("2024-01-01", periods=640, freq="D")
TRAIN_DAYS = 300
WINDOWS = {"current_days": 56, "reference_days": 182}

def _series(shift_from=None, seed=7):
    # two segments, weekly seasonal, optionally level-shifted in the freelancer segment
    rng = np.random.default_rng(seed)
    t = np.arange(640)
    frames = []
    for segment, level, noise in (("salaried", 1000.0, 15.0), ("freelancer", 3000.0, 100.0)):
        inflow = level + 600 + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, 640)
        outflow = 500 + rng.normal(0, 15, 640)
        if segment == "freelancer" and shift_from is not None:
            inflow[shift_from:] += 1500.0
        frames.append(
            pd.DataFrame(
                {"date": DATES, "segment": segment, "inflow": inflow, "outflow": outflow, "net": inflow - outflow}
            )
        )
    return pd.concat(frames, ignore_index=True)

@pytest.fixture(scope="module")
def booster(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("drift")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    series = _series()
    train_and_register_model(series[pd.to_datetime(series["date"]) <= DATES[TRAIN_DAYS - 1]], num_boost_round=50)
    yield load_registered_model("latest")
    patch.undo()

def _service(booster, history):
    return ForecastService(booster, history, train_end=DATES[TRAIN_DAYS - 1], model_version=3)

def _by_segment(status):
    return {s["segment"]: s for s in status["segments"]}

def test_stable_data_raises_no_alert(booster):
    status = drift_status(_service(booster, _series()), **WINDOWS)
    assert status["retrain"] is False
    assert {s["status"] for s in status["segments"]} == {"ok"}
    assert status["model_version"] == 3

def test_a_level_shift_in_one_segment_is_attributed_to_it(booster):
    status = drift_status(_service(booster, _series(shift_from=560)), **WINDOWS)
    by = _by_segment(status)
    assert by["freelancer"]["status"] == "alert"
    assert by["freelancer"]["data_drift_detected"] is True
    assert by["salaried"]["status"] == "ok"
    assert status["retrain"] is True
    assert any("freelancer" in reason for reason in status["reasons"])

def test_too_little_out_of_sample_history_means_the_error_monitor_is_unassessed(booster):
    # the champion's training cutoff is day 300, so only 340 days follow it and the default
    # windows (60 + 180) fit, but a very long reference window does not
    status = drift_status(_service(booster, _series()), current_days=56, reference_days=400)
    assert all(s["error_assessed"] is False for s in status["segments"])
    assert all(s["error_ratio"] is None for s in status["segments"])

def test_a_segment_with_nothing_to_assess_is_unassessed_not_ok(booster):
    history = _series()
    history.loc[history["segment"] == "salaried", ["inflow", "outflow", "net"]] = np.nan
    status = drift_status(_service(booster, history), **WINDOWS)
    assert _by_segment(status)["salaried"]["status"] == "unassessed"

def test_the_endpoint_returns_the_status_as_json(booster):
    client = TestClient(create_app(_service(booster, _series(shift_from=560))))
    response = client.get("/drift", params=WINDOWS)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == 3
    assert body["as_of"] == DATES[639].strftime("%Y-%m-%d")
    assert body["train_end"] == DATES[TRAIN_DAYS - 1].strftime("%Y-%m-%d")
    assert body["retrain"] is True
    assert {s["segment"] for s in body["segments"]} == {"salaried", "freelancer"}

def test_the_endpoint_rejects_out_of_range_windows_and_modes(booster):
    client = TestClient(create_app(_service(booster, _series())))
    assert client.get("/drift", params={"current_days": 5}).status_code == 422
    assert client.get("/drift", params={"reference_days": 1000}).status_code == 422
    assert client.get("/drift", params={"mode": "sometimes"}).status_code == 422