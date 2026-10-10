import mlflow
import numpy as np
import pandas as pd
import pytest

from flowradar.evaluation.holdout import build_prediction_log, predict_with_booster, score_on_holdout
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


def _booster(train_df):
    train_and_register_model(train_df, num_boost_round=50)
    return load_registered_model("latest")


def test_predictions_are_nan_during_warmup(mlflow_store):
    series = _series()
    booster = _booster(series.iloc[:180])
    preds = predict_with_booster(booster, series)
    assert preds["prediction"].iloc[:28].isna().all()
    assert preds["prediction"].iloc[28:].notna().all()


def test_prediction_log_columns_and_start_date(mlflow_store):
    series = _series()
    booster = _booster(series.iloc[:180])
    start = series["date"].iloc[180]
    log = build_prediction_log(booster, series, start_date=start)
    assert list(log.columns) == ["segment", "date", "actual", "prediction"]
    assert log["date"].min() == start
    assert len(log) == 270
    assert log["prediction"].notna().all()


def test_score_on_holdout_is_finite_and_positive(mlflow_store):
    series = _series()
    booster = _booster(series.iloc[:180])
    score = score_on_holdout(booster, series.iloc[:420], series.iloc[420:])
    assert np.isfinite(score)
    assert score > 0


def test_model_trained_on_new_regime_scores_better(mlflow_store):
    series = _series(shift_start=380)
    old_booster = _booster(series.iloc[:180])
    new_booster = _booster(series.iloc[:420])
    fit_df, holdout_df = series.iloc[:420], series.iloc[420:]
    assert score_on_holdout(new_booster, fit_df, holdout_df) < score_on_holdout(
        old_booster, fit_df, holdout_df
    )