import pytest
import numpy as np
import pandas as pd
from flowradar.evaluation.backtest import make_walk_forward_folds
from flowradar.evaluation.comparison import run_model_comparison
from flowradar.models.lightgbm_model import lightgbm_predict
from flowradar.models.statsforecast_model import statsforecast_predict
TARGETS = ["net", "inflow", "outflow"]

def _series(n_days=200, seed=2, segment="salaried"):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    net = 1000 + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 15, n_days)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_days, freq="D"),
            "segment": segment,
            "net": net,
            "inflow": net + 500,
            "outflow": 500.0,
        }
    )

def test_missing_actuals_are_excluded_from_scoring():
    df = _series()
    df.loc[175:179, TARGETS] = np.nan  # five data-gap days inside the 30-day test window

    def const(train, test):
        return pd.Series(1000.0, index=test.index)

    result = run_model_comparison(
        df, horizon=30, n_folds=1, step_days=30, calib_days=30, models={"const": const}
    )
    assert result["n_scored"].iloc[0] == 25
    assert np.isfinite(result["mase"].iloc[0])

def test_mase_is_averaged_per_segment():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    df = pd.concat(
        [
            pd.DataFrame(
                {"date": dates, "segment": seg, "net": rng.normal(0, scale, 200), "inflow": 0.0, "outflow": 0.0}
            )
            for seg, scale in (("small", 1.0), ("large", 1000.0))
        ],
        ignore_index=True,
    )

    def cheat(train, test):
        return test["net"] + 1.0  # every forecast is off by exactly 1

    result = run_model_comparison(
        df, horizon=30, n_folds=1, step_days=30, calib_days=30, models={"cheat": cheat}
    )
    fold = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)[0]
    scales = [
        fold.train[fold.train["segment"] == s].sort_values("date")["net"].diff().abs().mean()
        for s in ("small", "large")
    ]
    expected = np.mean([1.0 / s for s in scales])
    assert result["mase"].iloc[0] == pytest.approx(expected)

def test_statsforecast_tolerates_gaps_in_training_data():
    df = _series()
    df.loc[100:102, TARGETS] = np.nan
    df.loc[168:169, TARGETS] = np.nan  # a gap right at the training cutoff
    fold = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)[0]
    preds = statsforecast_predict(fold.train, fold.test, model_name="ets")
    assert len(preds) == len(fold.test)
    assert np.isfinite(preds).all()

def test_lightgbm_trains_when_labels_have_gaps():
    df = _series()
    df.loc[100:104, TARGETS] = np.nan
    fold = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)[0]
    preds = lightgbm_predict(fold.train, fold.test, num_boost_round=10)
    assert len(preds) == len(fold.test)
    assert np.isfinite(preds).all()