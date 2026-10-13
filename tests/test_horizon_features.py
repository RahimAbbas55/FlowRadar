import numpy as np
import pandas as pd
import pytest
from flowradar.features.horizon_features import (
    add_horizon_lag_features,
    build_horizon_features,
    horizon_safe_lags,
)
from flowradar.features.pipeline import build_features
TARGETS = ["net", "inflow", "outflow"]

def _frame(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "segment": "salaried",
            "net": rng.normal(0, 100, n),
            "inflow": rng.normal(500, 50, n),
            "outflow": rng.normal(500, 50, n),
        }
    )

def _feature_cols(df):
    return [c for c in df.columns if "_lag_" in c or "_roll_" in c]

def _unchanged_after_scramble(build, first_row, last_row, t=100):
    # overwrites the targets in rows first_row..last_row, then checks target row t's features
    df = _frame()
    scrambled = df.copy()
    scrambled.loc[first_row:last_row, TARGETS] = 999999.0
    a, b = build(df), build(scrambled)
    cols = _feature_cols(a)
    return np.allclose(
        a.loc[t, cols].astype(float), b.loc[t, cols].astype(float), equal_nan=True
    )

def test_horizon_safe_lags_values():
    assert horizon_safe_lags(30) == [30, 35, 42, 49]
    assert horizon_safe_lags(1) == [1, 7, 14, 21]
    assert horizon_safe_lags(7) == [7, 14, 21]

def test_lag_values_match_shifted_series():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "segment": "salaried",
            "net": np.arange(100.0),
            "inflow": 0.0,
            "outflow": 0.0,
        }
    )
    row = add_horizon_lag_features(df, horizon=30).iloc[60]
    assert row["net_lag_30"] == 30
    assert row["net_lag_35"] == 25
    assert row["net_lag_42"] == 18
    assert row["net_lag_49"] == 11

def test_rolling_mean_covers_only_rows_at_least_horizon_back():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "segment": "salaried",
            "net": np.arange(100.0),
            "inflow": 0.0,
            "outflow": 0.0,
        }
    )
    row = add_horizon_lag_features(df, horizon=30).iloc[60]
    assert row["net_roll_mean_7"] == 27.0  # mean of rows 24..30
    assert row["net_roll_mean_28"] == 16.5  # mean of rows 3..30

def test_scrambling_the_forecast_window_does_not_change_features():
    # forecasting 30 days ahead for row 100 means rows 71..100 are unknown at forecast time
    assert _unchanged_after_scramble(lambda d: build_horizon_features(d, horizon=30), 71, 100)

def test_scrambling_the_last_known_day_does_change_features():
    # row 70 is the most recent known day for row 100, so the boundary is exactly the horizon
    assert not _unchanged_after_scramble(
        lambda d: build_horizon_features(d, horizon=30), 70, 70
    )

def test_old_pipeline_fails_the_same_check():
    # documents the defect: the one-step pipeline uses actuals from inside the horizon
    assert not _unchanged_after_scramble(build_features, 71, 100)

def test_lags_do_not_cross_segments():
    a = _frame().assign(segment="salaried")
    b = _frame(seed=2).assign(segment="freelancer")
    out = add_horizon_lag_features(pd.concat([a, b], ignore_index=True), horizon=30)
    freelancer = out[out["segment"] == "freelancer"]
    assert freelancer["net_lag_30"].iloc[:30].isna().all()
    assert freelancer["net_lag_30"].iloc[30] == freelancer["net"].iloc[0]

def test_nan_propagates_through_a_data_gap():
    df = _frame()
    df.loc[50, "net"] = np.nan
    out = add_horizon_lag_features(df, horizon=30)
    assert pd.isna(out.iloc[80]["net_lag_30"])

def test_invalid_horizon_raises():
    with pytest.raises(ValueError):
        horizon_safe_lags(0)
    with pytest.raises(ValueError):
        add_horizon_lag_features(_frame(), horizon=0)

def test_calendar_features_are_included():
    out = build_horizon_features(_frame(), horizon=30)
    for col in ["is_weekend", "day_of_month", "is_working_day", "net_lag_30"]:
        assert col in out.columns