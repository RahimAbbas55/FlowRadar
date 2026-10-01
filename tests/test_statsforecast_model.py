import numpy as np
import pandas as pd
from flowradar.evaluation.backtest import make_walk_forward_folds
from flowradar.models.statsforecast_model import statsforecast_predict

def _seasonal_series(n_days=200, segment="salaried"):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = 40 * np.sin(2 * np.pi * t / 7)
    net = 1000 + weekly
    return pd.DataFrame(
        {"date": dates, "segment": segment, "net": net, "inflow": net + 500, "outflow": 500.0}
    )

def test_ets_prediction_length_matches_test():
    df = _seasonal_series()
    folds = make_walk_forward_folds(df, horizon=14, n_folds=1, step_days=14)
    fold = folds[0]
    preds = statsforecast_predict(fold.train, fold.test, model_name="ets")
    assert len(preds) == len(fold.test)

def test_arima_prediction_length_matches_test():
    df = _seasonal_series()
    folds = make_walk_forward_folds(df, horizon=14, n_folds=1, step_days=14)
    fold = folds[0]
    preds = statsforecast_predict(fold.train, fold.test, model_name="arima")
    assert len(preds) == len(fold.test)

def test_predictions_are_finite():
    df = _seasonal_series()
    folds = make_walk_forward_folds(df, horizon=14, n_folds=1, step_days=14)
    fold = folds[0]
    preds = statsforecast_predict(fold.train, fold.test, model_name="ets")
    assert np.isfinite(preds).all()

def test_predictions_aligned_to_correct_segment():
    # two segments with very different levels; a misaligned merge would mix them up
    salaried = _seasonal_series(segment="salaried")
    freelancer = _seasonal_series(segment="freelancer")
    freelancer["net"] = freelancer["net"] + 10000  # clearly distinct level

    df = pd.concat([salaried, freelancer], ignore_index=True)
    folds = make_walk_forward_folds(df, horizon=14, n_folds=1, step_days=14)
    fold = folds[0]
    preds = statsforecast_predict(fold.train, fold.test, model_name="ets")

    test_with_preds = fold.test.copy()
    test_with_preds["prediction"] = preds.values
    salaried_preds = test_with_preds[test_with_preds["segment"] == "salaried"]["prediction"]
    freelancer_preds = test_with_preds[test_with_preds["segment"] == "freelancer"]["prediction"]
    assert freelancer_preds.mean() > salaried_preds.mean() + 5000

def test_unknown_model_name_raises():
    import pytest

    df = _seasonal_series()
    folds = make_walk_forward_folds(df, horizon=14, n_folds=1, step_days=14)
    fold = folds[0]
    with pytest.raises(ValueError):
        statsforecast_predict(fold.train, fold.test, model_name="not_a_model")