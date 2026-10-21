import copy
import json
import mlflow
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from flowradar.serving.api import create_app
from flowradar.dashboard.snapshot import SnapshotClient
from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.registry import load_registered_model, train_and_register_model
from flowradar.dashboard.build_snapshot import build_snapshot, validate_snapshot, write_snapshot
DATES = pd.date_range("2024-01-01", periods=400, freq="D")
TRAIN_DAYS = 270
META = {"built_at": "2026-10-21", "generator_version": "0.3.0", "seed": 42}

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

@pytest.fixture(scope="module")
def service(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("snapshot")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    series = _series()
    train_and_register_model(series[pd.to_datetime(series["date"]) <= DATES[TRAIN_DAYS - 1]], num_boost_round=50)
    booster = load_registered_model("latest")
    yield ForecastService(booster, series, train_end=DATES[TRAIN_DAYS - 1], model_version=7)
    patch.undo()

@pytest.fixture(scope="module")
def snapshot(service):
    return build_snapshot(service, dict(META))

def test_a_built_snapshot_is_complete(snapshot):
    assert validate_snapshot(snapshot) == []
    assert snapshot["meta"] == META
    assert snapshot["health"]["model_version"] == 7

def test_validation_reports_each_kind_of_gap(snapshot):
    broken = copy.deepcopy(snapshot)
    del broken["drift"]
    assert any("missing section" in p for p in validate_snapshot(broken))

    broken = copy.deepcopy(snapshot)
    del broken["forecast"]["salaried"]["0.2"]
    assert any("forecast salaried alpha 0.2 is missing" in p for p in validate_snapshot(broken))

    broken = copy.deepcopy(snapshot)
    broken["anomalies"]["salaried"]["0.05"]["7"]["points"].pop()
    assert any("anomalies salaried alpha 0.05 days 7 has" in p for p in validate_snapshot(broken))

    broken = copy.deepcopy(snapshot)
    del broken["meta"]["seed"]
    assert any("meta is missing seed" in p for p in validate_snapshot(broken))

def test_the_snapshot_is_strict_json_without_nan(snapshot):
    json.dumps(snapshot, allow_nan=False)

def test_writing_creates_the_folder_and_round_trips(snapshot, tmp_path):
    path = write_snapshot(snapshot, tmp_path / "nested" / "snapshot.json")
    assert json.loads(path.read_text()) == snapshot

def test_a_snapshot_with_nan_is_refused_at_write_time(snapshot, tmp_path):
    broken = copy.deepcopy(snapshot)
    broken["health"]["horizon"] = float("nan")
    with pytest.raises(ValueError):
        write_snapshot(broken, tmp_path / "snapshot.json")

def test_the_snapshot_returns_exactly_what_the_live_api_returns(service, snapshot, tmp_path):
    stored = SnapshotClient(write_snapshot(snapshot, tmp_path / "snapshot.json"))
    live = TestClient(create_app(service))

    assert stored.health() == live.get("/health").json()
    assert stored.drift() == live.get("/drift").json()
    for segment, alpha in (("salaried", 0.1), ("freelancer", 0.2), ("salaried", 0.05)):
        params = {"segment": segment, "alpha": alpha}
        assert stored.forecast(**params) == live.get("/forecast", params=params).json()
    for segment, alpha, days in (("salaried", 0.01, 30), ("freelancer", 0.05, 7), ("salaried", 0.1, 90)):
        params = {"segment": segment, "alpha": alpha, "days": days}
        assert stored.anomalies(**params) == live.get("/anomalies", params=params).json()

    merged = {"alpha": 0.05, "days": 30, "only_flagged": True}
    assert stored.anomalies(**merged) == live.get("/anomalies", params=merged).json()
    assert stored.forecast(alpha=0.1) == live.get("/forecast", params={"alpha": 0.1}).json()