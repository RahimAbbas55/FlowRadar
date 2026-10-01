import pandas as pd
from flowradar.evaluation.backtest import make_walk_forward_folds
from flowradar.models.seasonal_naive import seasonal_naive_predict

def test_prediction_matches_value_one_week_prior():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": range(60)})
    folds = make_walk_forward_folds(df, horizon=7, n_folds=1, step_days=7)
    fold = folds[0]

    preds = seasonal_naive_predict(fold.train, fold.test, season_length=7)
    expected_first = fold.test["net"].iloc[0] - 7  # net is just the row index here
    assert preds.iloc[0] == expected_first

def test_prediction_length_matches_test():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": range(90)})
    folds = make_walk_forward_folds(df, horizon=30, n_folds=1, step_days=30)
    fold = folds[0]

    preds = seasonal_naive_predict(fold.train, fold.test, season_length=7)
    assert len(preds) == len(fold.test)