import pandas as pd
import numpy as np
from flowradar.evaluation.metrics import interval_coverage, mase, mean_interval_width, smape

def test_mase_perfect_forecast_is_zero():
    actual = pd.Series([10, 20, 30, 40])
    predicted = pd.Series([10, 20, 30, 40])
    train = pd.Series(range(100))
    assert mase(actual, predicted, train) == 0.0

def test_mase_worse_than_naive_exceeds_one():
    actual = pd.Series([10.0, 20.0, 30.0])
    predicted = pd.Series([100.0, 200.0, 300.0])  # wildly off
    train = pd.Series([1.0] * 50)  # naive errors near zero, train is flat
    train_with_variation = pd.Series(np.linspace(1, 50, 50))
    score = mase(actual, predicted, train_with_variation)
    assert score > 1.0

def test_smape_zero_for_identical_series():
    actual = pd.Series([5.0, 10.0, 15.0])
    predicted = pd.Series([5.0, 10.0, 15.0])
    assert smape(actual, predicted) == 0.0

def test_smape_max_is_200_for_opposite_sign_equal_magnitude():
    actual = pd.Series([10.0])
    predicted = pd.Series([-10.0])
    assert smape(actual, predicted) == 200.0

def test_interval_coverage_all_inside():
    actual = pd.Series([5, 10, 15])
    lower = pd.Series([0, 5, 10])
    upper = pd.Series([10, 15, 20])
    assert interval_coverage(actual, lower, upper) == 1.0

def test_interval_coverage_partial():
    actual = pd.Series([5, 100, 15])
    lower = pd.Series([0, 5, 10])
    upper = pd.Series([10, 15, 20])
    assert interval_coverage(actual, lower, upper) == pytest_approx_two_thirds()

def pytest_approx_two_thirds():
    return 2 / 3

def test_mean_interval_width():
    lower = pd.Series([0, 0, 0])
    upper = pd.Series([10, 20, 30])
    assert mean_interval_width(lower, upper) == 20.0