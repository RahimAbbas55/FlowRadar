from dataclasses import dataclass
import pandas as pd

# one walk-forward fold: training data up to a point, then a fixed-size test window
@dataclass
class Fold:
    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame

def make_walk_forward_folds(
    df: pd.DataFrame, horizon: int = 30, n_folds: int = 5, step_days: int = 30
) -> list[Fold]:
    # generates expanding-window folds: each fold's train set includes everything
    # before its test window, and test windows step forward by step_days each fold
    dates = sorted(pd.to_datetime(df["date"]).unique())
    total_days = len(dates)

    # last fold's test window must end by total_days, so compute the earliest valid start
    last_test_start_idx = total_days - horizon
    first_test_start_idx = last_test_start_idx - (n_folds - 1) * step_days

    if first_test_start_idx < 1:
        raise ValueError("not enough data for the requested n_folds, horizon, and step_days")

    folds = []
    for i in range(n_folds):
        test_start_idx = first_test_start_idx + i * step_days
        test_end_idx = test_start_idx + horizon - 1

        train_cutoff = dates[test_start_idx - 1]
        test_start = dates[test_start_idx]
        test_end = dates[min(test_end_idx, total_days - 1)]

        dcol = pd.to_datetime(df["date"])
        train = df[dcol <= train_cutoff].copy()
        test = df[(dcol >= test_start) & (dcol <= test_end)].copy()
        folds.append(Fold(fold_id=i, train=train, test=test))

    return folds

def split_calibration(train: pd.DataFrame, calib_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    # carves the most recent calib_days off the end of train for conformal calibration,
    # leaving fit_train strictly before it so the model never sees calibration data during fitting
    dates = sorted(pd.to_datetime(train["date"]).unique())
    cutoff = dates[-calib_days]
    dcol = pd.to_datetime(train["date"])
    fit_train = train[dcol < cutoff].copy()
    calibration = train[dcol >= cutoff].copy()
    return fit_train, calibration