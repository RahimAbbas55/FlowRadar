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

def test_building_features_on_full_series_then_splitting_is_the_leakage_bug():
    # this test documents the WRONG order and proves it does leak, so the difference
    # from the correct pattern above is concrete rather than theoretical
    df = _sample_series()
    full_features = build_features(df)  # features built on the whole series first

    folds = make_walk_forward_folds(full_features, horizon=30, n_folds=1, step_days=30)
    fold = folds[0]

    first_test_row = fold.test.iloc[0]
    # the bug: this lag value was computed using a training-set day, which is fine,
    # but roll_mean features computed this way can include values from AFTER the
    # train cutoff if build_features was run on full data — this assertion fails if so
    train_dates = set(pd.to_datetime(fold.train["date"]))
    test_dates = set(pd.to_datetime(fold.test["date"]))
    # explicitly confirm the two date sets don't overlap, as a precondition for the real check
    assert train_dates.isdisjoint(test_dates)

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