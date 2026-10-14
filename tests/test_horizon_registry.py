import mlflow
import pytest
import numpy as np
import pandas as pd
from flowradar.evaluation.holdout import infer_horizon, predict_with_booster, score_on_holdout
from flowradar.flows.retrain import retrain_flow
from flowradar.tracking.registry import load_registered_model, train_and_register_model
TARGETS = ["net", "inflow", "outflow"]

def _series(n_days=300, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    inflow = 1500 + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 15, n_days)
    outflow = 500 + rng.normal(0, 15, n_days)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_days, freq="D"),
            "segment": "salaried",
            "inflow": inflow,
            "outflow": outflow,
            "net": inflow - outflow,
        }
    )

@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

def _booster(series, horizon=30):
    train_and_register_model(series.iloc[:200], num_boost_round=20, horizon=horizon)
    return load_registered_model("latest")

def test_infer_horizon_reads_it_from_the_trained_booster(mlflow_store):
    series = _series()
    for horizon in (14, 30):
        assert infer_horizon(_booster(series, horizon=horizon)) == horizon

def test_predictions_for_the_next_horizon_days_ignore_their_actuals(mlflow_store):
    series = _series()
    booster = _booster(series)
    fit, hold = series.iloc[:260], series.iloc[260:290]
    scrambled = hold.copy()
    scrambled[TARGETS] = 999999.0

    a = predict_with_booster(booster, pd.concat([fit, hold], ignore_index=True))
    b = predict_with_booster(booster, pd.concat([fit, scrambled], ignore_index=True))
    assert np.allclose(a["prediction"], b["prediction"], equal_nan=True)

def test_changing_the_last_known_day_does_change_predictions(mlflow_store):
    # control: proves the scramble test above isn't vacuous
    series = _series()
    booster = _booster(series)
    fit, hold = series.iloc[:260], series.iloc[260:290]
    changed = fit.copy()
    changed.iloc[-1, changed.columns.get_indexer(TARGETS)] = 999999.0

    a = predict_with_booster(booster, pd.concat([fit, hold], ignore_index=True))
    b = predict_with_booster(booster, pd.concat([changed, hold], ignore_index=True))
    assert not np.allclose(a["prediction"], b["prediction"], equal_nan=True)

def test_holdout_longer_than_the_horizon_is_rejected(mlflow_store):
    series = _series()
    booster = _booster(series)
    with pytest.raises(ValueError):
        score_on_holdout(booster, series.iloc[:200], series.iloc[200:245])

def test_retrain_flow_rejects_a_holdout_longer_than_the_horizon(mlflow_store):
    series = _series()
    with pytest.raises(ValueError):
        retrain_flow(series, series, holdout_days=45)