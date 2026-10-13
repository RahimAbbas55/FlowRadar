import math
from functools import partial
import numpy as np
import pandas as pd
import pytest
from flowradar.evaluation.backtest import make_walk_forward_folds
from flowradar.evaluation.comparison import run_model_comparison
from flowradar.models.lightgbm_model import lightgbm_predict
from flowradar.models.seasonal_naive import seasonal_naive_predict
from flowradar.models.statsforecast_model import statsforecast_predict
TARGETS = ["net", "inflow", "outflow"]
MODELS = {
    "seasonal_naive": seasonal_naive_predict,
    "lightgbm": partial(lightgbm_predict, num_boost_round=30),
    "ets": partial(statsforecast_predict, model_name="ets"),
    "arima": partial(statsforecast_predict, model_name="arima"),
}

def _series(n_days=300, seed=3, segment="salaried", level=1000.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    net = level + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 15, n_days)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_days, freq="D"),
            "segment": segment,
            "net": net,
            "inflow": net + 500,
            "outflow": 500.0,
        }
    )

def _fold():
    return make_walk_forward_folds(_series(), horizon=30, n_folds=1, step_days=30)[0]

@pytest.mark.parametrize("name", list(MODELS))
def test_scrambling_test_actuals_does_not_change_predictions(name):
    fold = _fold()
    scrambled = fold.test.copy()
    scrambled[TARGETS] = 999999.0
    a = MODELS[name](fold.train, fold.test)
    b = MODELS[name](fold.train, scrambled)
    assert np.allclose(a, b, equal_nan=True)

@pytest.mark.parametrize("name", list(MODELS))
def test_changing_recent_training_actuals_does_change_predictions(name):
    # control: proves the scramble test above isn't vacuous
    fold = _fold()
    changed = fold.train.copy()
    changed.loc[changed.index[-30:], TARGETS] += 1000.0
    a = MODELS[name](fold.train, fold.test)
    b = MODELS[name](changed, fold.test)
    assert not np.allclose(a, b, equal_nan=True)

def test_seasonal_naive_uses_most_recent_known_week():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": np.arange(100.0), "inflow": 0.0, "outflow": 0.0}
    )
    preds = seasonal_naive_predict(df.iloc[:80], df.iloc[80:94])
    for i in range(14):
        expected = 80 + i - 7 * math.ceil((i + 1) / 7)
        assert preds.iloc[i] == expected

def test_seasonal_naive_aligns_predictions_to_the_right_segment():
    df = pd.concat(
        [_series(segment="salaried", level=100.0), _series(segment="freelancer", level=10000.0, seed=4)],
        ignore_index=True,
    )
    fold = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)[0]
    preds = seasonal_naive_predict(fold.train, fold.test)
    scored = fold.test.assign(prediction=preds.values)
    assert scored[scored["segment"] == "freelancer"]["prediction"].mean() > 9000
    assert scored[scored["segment"] == "salaried"]["prediction"].mean() < 200

def test_lightgbm_rejects_a_test_window_longer_than_the_horizon():
    fold = _fold()  # test window reaches 30 days past the cutoff
    with pytest.raises(ValueError):
        lightgbm_predict(fold.train, fold.test, horizon=10, num_boost_round=10)

def test_every_forecast_in_the_comparison_starts_the_day_after_its_training_cutoff():
    calls = []

    def spy(train, test):
        train_max = pd.to_datetime(train["date"]).max()
        calls.append(
            (
                (pd.to_datetime(test["date"]).min() - train_max).days,
                (pd.to_datetime(test["date"]).max() - train_max).days,
            )
        )
        return pd.Series(0.0, index=test.index)

    run_model_comparison(
        _series(), horizon=14, n_folds=1, step_days=14, calib_days=14, models={"spy": spy}
    )
    assert len(calls) == 2  # one calibration forecast, one test forecast
    for first_gap, last_gap in calls:
        assert first_gap == 1
        assert last_gap <= 14

def test_calibration_longer_than_horizon_is_rejected():
    with pytest.raises(ValueError):
        run_model_comparison(
            _series(), horizon=14, n_folds=1, step_days=14, calib_days=30,
            models={"naive": seasonal_naive_predict},
        )