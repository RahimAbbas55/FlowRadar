import pandas as pd

# flags a day as anomalous when the actual value falls outside the calibrated interval
def flag_anomalies(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    return (actual < lower) | (actual > upper)
'''
    Signed distance outside the interval: 0 when inside, positive above upper,
    negative below lower — magnitude indicates how severe the flagged anomaly is
'''
def flag_anomalies_with_distance(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
    distance = pd.Series(0.0, index=actual.index)
    above = actual > upper
    below = actual < lower
    distance[above] = actual[above] - upper[above]
    distance[below] = actual[below] - lower[below]

    return pd.DataFrame(
        {
            "actual": actual,
            "lower": lower,
            "upper": upper,
            "is_anomaly": above | below,
            "distance": distance,
        }
    )