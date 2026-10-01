import pandas as pd
from flowradar.features.calendar_features import add_calendar_features
from flowradar.features.lag_features import add_lag_features

# Combines calendar and lag features into the full feature table for modeling
def build_features(series: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(series)
    out = add_lag_features(out)
    return out.reset_index(drop=True)