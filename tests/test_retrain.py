import mlflow
import numpy as np
import pandas as pd
import pytest

from flowradar.evaluation.holdout import build_prediction_log
from flowradar.flows.retrain import retrain_flow
from flowradar.tracking.promotion import get_champion_version, promote_if_better
from flowradar.tracking.registry import load_registered_model, train_and_register_model


def _series(n_days=450, shift_start=None, shift=300.0, seed=5):
    # weekly-seasonal inflow with noise, optionally level-shifted from shift_start onward
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    inflow = 1500 + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 15, n_days)
    if shift_start is not None:
        inflow[shift_start:] += shift
    outflow = 500 + rng.normal(0, 15, n_days)
    return pd.DataFrame(
        {"date": dates, "segment": "salaried", "inflow": inflow, "outflow": outflow, "net": inflow - outflow}
    )


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")


def _champion_and_log(series):
    # champion trained on the first 180 days only, so its log from day 180 on is out-of-sample
    v1 = train_and_register_model(series.iloc[:180], num_boost_round=50)
    promote_if_better(v1, challenger_score=0.5, champion_score=None)
    booster = load_registered_model("champion")
    return build_prediction_log(booster, series, start_date=series["date"].iloc[180])


# windows are multiples of 7 so weekly seasonality can't look like drift
_WINDOWS = {"current_days": 56, "reference_days": 182}


def test_no_retrain_on_stable_data(mlflow_store):
    series = _series()
    log = _champion_and_log(series)
    result = retrain_flow(series, log, num_boost_round=50, **_WINDOWS)
    assert result["retrained"] is False
    assert get_champion_version().version == "1"


def test_retrains_and_promotes_after_level_shift(mlflow_store):
    series = _series(shift_start=380)
    log = _champion_and_log(series)
    result = retrain_flow(series, log, holdout_days=30, num_boost_round=50, **_WINDOWS)
    assert result["decision"]["retrain"] is True
    assert result["retrained"] is True
    assert result["challenger_version"] == 2
    assert result["promoted"] is True
    assert get_champion_version().version == "2"


def test_force_retrains_on_stable_data(mlflow_store):
    series = _series()
    log = _champion_and_log(series)
    result = retrain_flow(series, log, num_boost_round=50, force=True, **_WINDOWS)
    assert result["retrained"] is True
    assert result["challenger_version"] == 2
    assert result["champion_version"] == 1