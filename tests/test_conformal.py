import numpy as np
import pandas as pd
from flowradar.evaluation.conformal import conformal_interval, conformal_quantile

def test_quantile_zero_for_perfect_residuals():
    residuals = pd.Series([0.0] * 20)
    assert conformal_quantile(residuals, alpha=0.1) == 0.0

def test_quantile_increases_with_wider_residuals():
    narrow = pd.Series(np.random.default_rng(1).normal(0, 1, 100))
    wide = pd.Series(np.random.default_rng(1).normal(0, 10, 100))
    assert conformal_quantile(wide, alpha=0.1) > conformal_quantile(narrow, alpha=0.1)

def test_interval_is_symmetric_around_point_prediction():
    preds = pd.Series([100.0, 200.0, 300.0])
    lower, upper = conformal_interval(preds, q=10.0)
    assert (preds - lower == 10.0).all()
    assert (upper - preds == 10.0).all()

def test_empty_residuals_returns_nan():
    residuals = pd.Series([], dtype=float)
    assert np.isnan(conformal_quantile(residuals))