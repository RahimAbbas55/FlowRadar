import pandas as pd
import numpy as np
from flowradar.features.lag_features import add_lag_features

def test_lag_1_matches_previous_day():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": range(10), "inflow": 0.0, "outflow": 0.0}
    )
    out = add_lag_features(df, lags=[1], rolling_windows=[])
    assert out["net_lag_1"].iloc[5] == out["net"].iloc[4]

def test_lag_does_not_cross_segments():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.concat(
        [
            pd.DataFrame({"date": dates, "segment": "salaried", "net": [10, 20, 30, 40, 50], "inflow": 0.0, "outflow": 0.0}),
            pd.DataFrame({"date": dates, "segment": "freelancer", "net": [1, 2, 3, 4, 5], "inflow": 0.0, "outflow": 0.0}),
        ],
        ignore_index=True,
    )
    out = add_lag_features(df, lags=[1], rolling_windows=[])
    first_freelancer_row = out[out["segment"] == "freelancer"].iloc[0]
    # first day of freelancer's lag must be NaN, not leaked from salaried's last value
    assert pd.isna(first_freelancer_row["net_lag_1"])

def test_rolling_mean_excludes_current_day():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": [100] * 9 + [10000], "inflow": 0.0, "outflow": 0.0}
    )
    out = add_lag_features(df, lags=[], rolling_windows=[3])
    # the huge spike on the last day must not pull its own rolling mean upward
    assert out["net_roll_mean_3"].iloc[-1] < 200

def test_nan_propagates_through_data_gap():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    net = [10.0] * 10
    net[4] = np.nan
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": net, "inflow": 0.0, "outflow": 0.0})
    out = add_lag_features(df, lags=[1], rolling_windows=[])
    assert pd.isna(out["net_lag_1"].iloc[5])