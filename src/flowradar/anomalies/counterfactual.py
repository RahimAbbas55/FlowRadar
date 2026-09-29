import pandas as pd

'''
    The series as it looked after drift but before any anomaly injection,
    preserved so backtests can separate forecast error from anomaly-detector error
'''
def generate_counterfactual(drifted_series: pd.DataFrame) -> pd.DataFrame:
    return drifted_series.copy()