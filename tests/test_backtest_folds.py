import pandas as pd
from flowradar.evaluation.backtest import make_walk_forward_folds

def _sample_df(n_days=300):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "segment": "salaried", "net": range(n_days)})

def test_correct_number_of_folds():
    df = _sample_df()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=5, step_days=30)
    assert len(folds) == 5

def test_folds_are_expanding():
    df = _sample_df()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=5, step_days=30)
    for i in range(len(folds) - 1):
        assert len(folds[i].train) < len(folds[i + 1].train)

def test_train_never_overlaps_test():
    df = _sample_df()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=5, step_days=30)
    for fold in folds:
        train_max_date = pd.to_datetime(fold.train["date"]).max()
        test_min_date = pd.to_datetime(fold.test["date"]).min()
        assert train_max_date < test_min_date

def test_test_window_length_matches_horizon():
    df = _sample_df()
    folds = make_walk_forward_folds(df, horizon=30, n_folds=3, step_days=30)
    for fold in folds[:-1]:  # last fold may be shorter if it hits the data boundary
        assert len(fold.test) == 30

def test_raises_when_not_enough_data():
    df = _sample_df(n_days=50)
    import pytest

    with pytest.raises(ValueError):
        make_walk_forward_folds(df, horizon=30, n_folds=5, step_days=30)