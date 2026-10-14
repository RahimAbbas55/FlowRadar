import mlflow
import numpy as np
import pandas as pd
import pytest

from flowradar.serving.champion import ensure_champion, load_forecast_service
from flowradar.tracking.promotion import get_champion_version
from flowradar.tracking.registry import train_and_register_model

DATES = pd.date_range("2024-01-01", periods=400, freq="D")
MODEL = "flowradar-lightgbm"


def _history(seed=3):
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


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")


def test_registered_version_records_its_training_cutoff_and_horizon(mlflow_store):
    history = _history()
    version = train_and_register_model(history[history["date"] < DATES[200]], num_boost_round=20)
    assert version.tags["train_end_date"] == DATES[199].date().isoformat()
    assert version.tags["horizon"] == "30"


def test_ensure_champion_trains_and_promotes_when_none_exists(mlflow_store):
    champion = ensure_champion(_history(), calibration_days=90, num_boost_round=20)
    assert int(champion.version) == 1
    # 90 days held back means the model trained through day index 309
    assert champion.tags["train_end_date"] == DATES[309].date().isoformat()


def test_ensure_champion_does_not_retrain_when_one_exists(mlflow_store):
    history = _history()
    ensure_champion(history, num_boost_round=20)
    ensure_champion(history, num_boost_round=20)
    assert len(mlflow.MlflowClient().search_model_versions(f"name='{MODEL}'")) == 1


def test_load_forecast_service_serves_the_champion(mlflow_store):
    history = _history()
    ensure_champion(history, calibration_days=90, num_boost_round=20)
    service = load_forecast_service(history)
    assert service.model_version == 1
    assert service.train_end == DATES[309]
    assert len(service.forecast()) == 60


def test_load_forecast_service_without_a_champion_raises(mlflow_store):
    with pytest.raises(RuntimeError, match="no champion"):
        load_forecast_service(_history())


def test_champion_without_a_training_cutoff_tag_is_rejected(mlflow_store):
    history = _history()
    ensure_champion(history, num_boost_round=20)
    mlflow.MlflowClient().delete_model_version_tag(MODEL, "1", "train_end_date")
    with pytest.raises(RuntimeError, match="train_end_date"):
        load_forecast_service(history)


def test_calibration_window_shorter_than_the_horizon_is_rejected(mlflow_store):
    with pytest.raises(ValueError):
        ensure_champion(_history(), calibration_days=10)