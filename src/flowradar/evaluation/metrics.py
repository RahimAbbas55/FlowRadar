import numpy as np
import pandas as pd

'''
    Mean absolute scaled error: MAE of the forecast, scaled by the naive
    seasonal forecast's MAE on the training set, making scores comparable across segments
'''
def mase(actual: pd.Series, predicted: pd.Series, train_actual: pd.Series, season_length: int = 1) -> float:
    naive_errors = train_actual.diff(season_length).abs().dropna()
    scale = naive_errors.mean()
    if scale == 0 or np.isnan(scale):
        return float("nan")
    forecast_errors = (actual.values - predicted.values)
    return float(np.mean(np.abs(forecast_errors)) / scale)

'''
    Symmetric mean absolute percentage error, expressed as a percentage
    note: unstable near zero actual+predicted values, a known limitation of sMAPE generally   
'''
def smape(actual: pd.Series, predicted: pd.Series) -> float:
    denom = (actual.abs() + predicted.abs()).replace(0, np.nan)
    ratio = (actual - predicted).abs() / denom
    return float(ratio.mean() * 200)

# Fraction of actual values falling inside the predicted [lower, upper] interval
def interval_coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    inside = (actual >= lower) & (actual <= upper)
    return float(inside.mean())
'''
    Average width of the predicted interval, useful alongside coverage since
    a very wide interval can trivially achieve high coverage without being useful
'''
def mean_interval_width(lower: pd.Series, upper: pd.Series) -> float:
    return float((upper - lower).mean())