import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from flowradar.serving.api import create_app
from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.registry import load_registered_model, train_and_register_model

TARGETS = ["net", "inflow", "outflow"]
DATES = pd.date_range("2024-01-01", periods=400, freq="D")
TRAIN_DAYS = 270
SPIKE_DAY = 385


def _series(seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(400)
    frames = []
    for segment, level, noise in (("salaried", 1000.0, 15.0), ("freelancer", 3000.0, 150.0)):
        net = level + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, 400)
        frames.append(
            pd.DataFrame(
                {"date": DATES, "segment": segment, "net": net, "inflow": net + 500, "outflow": 500.0}
            )
        )
    return pd.concat(frames, ignore_index=True)


def _with_spike(series):
    out = series.copy()
    mask = (out["segment"] == "freelancer") & (pd.to_datetime(out["date"]) == DATES[SPIKE_DAY])
    out.loc[mask, ["net", "inflow"]] += 3000.0
    return out


@pytest.fixture(scope="module")
def booster(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    series = _series()
    train = series[pd.to_datetime(series["date"]) <= DATES[TRAIN_DAYS - 1]]
    train_and_register_model(train, num_boost_round=50)
    yield load_registered_model("latest")
    patch.undo()


def _client(booster, history):
    service = ForecastService(booster, history, train_end=DATES[TRAIN_DAYS - 1], model_version=7)
    return TestClient(create_app(service))


def test_health_reports_the_served_model(booster):
    body = _client(booster, _series()).get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == 7
    assert body["horizon"] == 30
    assert body["segments"] == ["freelancer", "salaried"]
    assert body["as_of"] == DATES[399].strftime("%Y-%m-%d")
    assert body["train_end_date"] == DATES[TRAIN_DAYS - 1].strftime("%Y-%m-%d")


def test_forecast_returns_30_calibrated_days_per_segment(booster):
    response = _client(booster, _series()).get("/forecast")
    assert response.status_code == 200
    body = response.json()
    assert body["alpha"] == 0.1
    assert len(body["points"]) == 60
    for p in body["points"]:
        assert p["interval_calibrated"] is True
        assert p["lower"] < p["forecast"] < p["upper"]


def test_forecast_segment_filter(booster):
    body = _client(booster, _series()).get("/forecast", params={"segment": "salaried"}).json()
    assert len(body["points"]) == 30
    assert {p["segment"] for p in body["points"]} == {"salaried"}


def test_unknown_segment_is_a_404(booster):
    assert _client(booster, _series()).get("/forecast", params={"segment": "nope"}).status_code == 404
    assert _client(booster, _series()).get("/anomalies", params={"segment": "nope"}).status_code == 404


def test_alpha_out_of_range_is_a_422(booster):
    assert _client(booster, _series()).get("/forecast", params={"alpha": 1.5}).status_code == 422


def test_missing_forecast_becomes_null_not_a_crash(booster):
    history = _series()
    history.loc[pd.to_datetime(history["date"]) == DATES[399], TARGETS] = np.nan
    response = _client(booster, history).get("/forecast")
    assert response.status_code == 200
    points = response.json()["points"]
    assert all(p["forecast"] is None for p in points if p["horizon_days"] == 30)
    assert all(p["forecast"] is not None for p in points if p["horizon_days"] == 1)


def test_anomalies_flags_the_injected_spike(booster):
    body = _client(booster, _with_spike(_series())).get("/anomalies").json()
    assert body["alpha"] == 0.01 and body["days"] == 30
    assert body["flagged"] >= 1
    spike_date = DATES[SPIKE_DAY].strftime("%Y-%m-%d")
    row = next(p for p in body["points"] if p["segment"] == "freelancer" and p["date"] == spike_date)
    assert row["status"] == "anomaly"


def test_only_flagged_returns_just_the_anomalies(booster):
    body = _client(booster, _with_spike(_series())).get("/anomalies", params={"only_flagged": True}).json()
    assert len(body["points"]) == body["flagged"] >= 1
    assert all(p["status"] == "anomaly" for p in body["points"])


def test_strict_alpha_without_enough_calibration_is_reported(booster):
    body = _client(booster, _series()).get("/anomalies", params={"alpha": 0.001, "days": 90}).json()
    assert body["flagged"] == 0
    assert all(p["status"] == "not_calibrated" for p in body["points"])


def test_days_out_of_range_is_a_422(booster):
    assert _client(booster, _series()).get("/anomalies", params={"days": 0}).status_code == 422