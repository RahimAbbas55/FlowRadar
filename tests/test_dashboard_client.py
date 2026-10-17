import mlflow
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from flowradar.serving.api import create_app
from flowradar.serving.forecaster import ForecastService
from flowradar.dashboard.client import ApiError, FlowRadarClient
from flowradar.tracking.registry import load_registered_model, train_and_register_model
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
            pd.DataFrame({"date": DATES, "segment": segment, "net": net, "inflow": net + 500, "outflow": 500.0})
        )
    return pd.concat(frames, ignore_index=True)

def _with_spike(series):
    out = series.copy()
    mask = (out["segment"] == "freelancer") & (pd.to_datetime(out["date"]) == DATES[SPIKE_DAY])
    out.loc[mask, ["net", "inflow"]] += 3000.0
    return out

@pytest.fixture(scope="module")
def booster(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dashclient")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    series = _series()
    train_and_register_model(series[pd.to_datetime(series["date"]) <= DATES[TRAIN_DAYS - 1]], num_boost_round=50)
    yield load_registered_model("latest")
    patch.undo()

@pytest.fixture
def client(booster):
    service = ForecastService(booster, _with_spike(_series()), train_end=DATES[TRAIN_DAYS - 1], model_version=5)
    return FlowRadarClient(http=TestClient(create_app(service)))

class _Response:
    status_code = 200
    text = "{}"

    def json(self):
        return {}

class _Recorder:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        return _Response()

def test_health_describes_the_served_model(client):
    body = client.health()
    assert body["model_version"] == 5
    assert body["horizon"] == 30

def test_forecast_can_be_limited_to_one_segment(client):
    body = client.forecast(segment="salaried")
    assert len(body["points"]) == 30
    assert {p["segment"] for p in body["points"]} == {"salaried"}

def test_anomalies_filters_are_passed_to_the_api(client):
    body = client.anomalies(only_flagged=True)
    assert body["flagged"] >= 1
    assert all(p["status"] == "anomaly" for p in body["points"])

def test_drift_returns_a_decision_and_one_row_per_segment(client):
    body = client.drift()
    assert isinstance(body["retrain"], bool)
    assert len(body["segments"]) == 2

def test_an_unknown_segment_becomes_an_api_error_carrying_the_servers_message(client):
    with pytest.raises(ApiError, match="unknown segment"):
        client.forecast(segment="nope")

def test_a_validation_error_is_reported_with_its_status_code(client):
    with pytest.raises(ApiError, match="422"):
        client.forecast(alpha=1.5)

def test_an_unreachable_server_becomes_an_api_error():
    with pytest.raises(ApiError, match="cannot reach"):
        FlowRadarClient(base_url="http://127.0.0.1:9", timeout=2.0).health()

def test_parameters_left_unset_are_not_sent():
    recorder = _Recorder()
    client = FlowRadarClient(http=recorder)
    client.anomalies(segment="a", only_flagged=True)
    client.forecast()
    assert recorder.calls[0] == ("/anomalies", {"segment": "a", "only_flagged": True})
    assert recorder.calls[1] == ("/forecast", {})