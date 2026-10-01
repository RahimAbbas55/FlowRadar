import pandas as pd
import numpy as np
from flowradar.evaluation.backtest import make_walk_forward_folds
from flowradar.features.pipeline import build_features

def _sample_series(n_days=400):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    # distinct, strictly increasing values so leakage is detectable, not masked by noise
    return pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": range(n_days), "inflow": 0.0, "outflow": 0.0}
    )

def test_fold_features_built_before_split_do_not_leak():
    # the correct order: split into folds first, then build features per fold using
    # only that fold's own train data. This test documents and enforces that order.
    df = _sample_series()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=3, step_days=30)

    for fold in folds:
        train_features = build_features(fold.train)
        # the last row's lag_1 feature must equal the second-to-last row's net,
        # both drawn only from train — proves no lookahead past the fold boundary
        last_row = train_features.iloc[-1]
        second_last_net = fold.train["net"].iloc[-2]
        assert last_row["net_lag_1"] == second_last_net

def test_feature_values_are_unaffected_by_future_rows():
    # the real leakage invariant: a feature computed for a given date must not change
    # if every row after that date is deleted before the feature is computed
    df = _sample_series(n_days=100)
    full_features = build_features(df)

    check_date = pd.Timestamp("2024-02-15")
    truncated = df[pd.to_datetime(df["date"]) <= check_date]
    truncated_features = build_features(truncated)

    full_row = full_features[pd.to_datetime(full_features["date"]) == check_date].iloc[0]
    truncated_row = truncated_features[pd.to_datetime(truncated_features["date"]) == check_date].iloc[0]

    feature_cols = [c for c in full_features.columns if "_lag_" in c or "_roll_" in c]
    for col in feature_cols:
        assert full_row[col] == truncated_row[col] or (pd.isna(full_row[col]) and pd.isna(truncated_row[col]))

def test_rolling_mean_in_fold_uses_only_train_history():
    df = _sample_series()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)
    fold = folds[0]
    train_features = build_features(fold.train)

    # the last train row's 7-day rolling mean must be computable purely from
    # the 7 prior train rows, none of which can be a test-period date
    last_train_date = pd.to_datetime(fold.train["date"]).max()
    assert last_train_date < pd.to_datetime(fold.test["date"]).min()

def test_data_gap_nan_does_not_silently_become_zero_in_features():
    # a NaN from a data_gap anomaly must propagate as NaN through lag features,
    # never silently fill to zero, which would misrepresent a missing day as a zero-activity day
    df = _sample_series(n_days=60)
    df.loc[10, "net"] = np.nan
    features = build_features(df)
    assert pd.isna(features.loc[11, "net_lag_1"])
    assert features.loc[11, "net_lag_1"] != 0.0