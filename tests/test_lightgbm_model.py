import numpy as np
import pandas as pd
from flowradar.evaluation.backtest import make_walk_forward_folds
from flowradar.models.lightgbm_model import lightgbm_predict

def _trending_series(n_days=400, segment="salaried"):
    # a clear weekly-seasonal, upward-trending series so a real model should
    # noticeably beat seasonal-naive, giving a meaningful sanity check
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = 50 * np.sin(2 * np.pi * t / 7)
    trend = t * 0.5
    net = 1000 + trend + weekly
    return pd.DataFrame(
        {"date": dates, "segment": segment, "net": net, "inflow": net + 500, "outflow": 500.0}
    )

def test_prediction_length_matches_test():
    df = _trending_series()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)
    fold = folds[0]
    preds = lightgbm_predict(fold.train, fold.test, num_boost_round=20)
    assert len(preds) == len(fold.test)

def test_determinism_same_seed():
    df = _trending_series()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)
    fold = folds[0]
    a = lightgbm_predict(fold.train, fold.test, num_boost_round=20, seed=7)
    b = lightgbm_predict(fold.train, fold.test, num_boost_round=20, seed=7)
    pd.testing.assert_series_equal(a, b)

def test_predictions_are_finite():
    df = _trending_series()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)
    fold = folds[0]
    preds = lightgbm_predict(fold.train, fold.test, num_boost_round=20)
    assert np.isfinite(preds).all()

def test_beats_seasonal_naive_on_clear_trend():
    from flowradar.evaluation.metrics import mase
    from flowradar.models.seasonal_naive import seasonal_naive_predict

    df = _trending_series()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)
    fold = folds[0]

    lgb_preds = lightgbm_predict(fold.train, fold.test, num_boost_round=100)
    naive_preds = seasonal_naive_predict(fold.train, fold.test, season_length=7)

    actual = fold.test["net"].reset_index(drop=True)
    lgb_score = mase(actual, lgb_preds.reset_index(drop=True), fold.train["net"])
    naive_score = mase(actual, naive_preds.reset_index(drop=True), fold.train["net"])

    # on a clear upward trend, a trained model should beat a trend-blind naive baseline
    assert lgb_score < naive_score

def test_training_excludes_warmup_nan_rows():
    # with only 60 days of train data and a 28-day lag feature, most train rows
    # are warm-up — this should still run without error, not crash on all-NaN features
    df = _trending_series(n_days=90)
    folds = make_walk_forward_folds(df, horizon=20, n_folds=1, step_days=20)
    fold = folds[0]
    preds = lightgbm_predict(fold.train, fold.test, num_boost_round=20)
    assert len(preds) == len(fold.test)