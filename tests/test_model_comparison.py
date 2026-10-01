import numpy as np
import pandas as pd
from flowradar.evaluation.comparison import run_model_comparison
from flowradar.models.seasonal_naive import seasonal_naive_predict

def _seasonal_series(n_days=300):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = 40 * np.sin(2 * np.pi * t / 7)
    net = 1000 + weekly
    return pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": net, "inflow": net + 500, "outflow": 500.0}
    )

def test_comparison_runs_all_default_models():
    df = _seasonal_series()
    result = run_model_comparison(df, horizon=14, n_folds=1, step_days=14, calib_days=14)
    assert set(result["model"].unique()) == {"seasonal_naive", "lightgbm", "ets", "arima"}

def test_comparison_has_one_row_per_model_per_fold():
    df = _seasonal_series()
    result = run_model_comparison(df, horizon=14, n_folds=2, step_days=14, calib_days=14)
    assert len(result) == 4 * 2

def test_comparison_columns_present():
    df = _seasonal_series()
    result = run_model_comparison(
        df, horizon=14, n_folds=1, step_days=14, calib_days=14,
        models={"seasonal_naive": seasonal_naive_predict},
    )
    for col in ["mase", "smape", "interval_coverage", "mean_interval_width"]:
        assert col in result.columns

def test_restricting_to_one_model():
    df = _seasonal_series()
    result = run_model_comparison(
        df, horizon=14, n_folds=1, step_days=14, calib_days=14,
        models={"seasonal_naive": seasonal_naive_predict},
    )
    assert set(result["model"].unique()) == {"seasonal_naive"}