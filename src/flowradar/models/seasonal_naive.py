import numpy as np
import pandas as pd

def seasonal_naive_predict(train: pd.DataFrame, test: pd.DataFrame, season_length: int = 7) -> pd.Series:
    # horizon-safe: forecasts are made at the training cutoff T. For a target date d,
    # uses the value from the most recent same-weekday date at or before T, which is
    # d - season_length * ceil((d - T) / season_length). Never reads test actuals
    train = train[["segment", "date", "net"]].copy()
    train["date"] = pd.to_datetime(train["date"])
    cutoff = train["date"].max()

    lookup = test[["segment", "date"]].copy()
    lookup["date"] = pd.to_datetime(lookup["date"])
    days_ahead = (lookup["date"] - cutoff).dt.days
    steps = np.ceil(days_ahead / season_length).clip(lower=1)
    lookup["source_date"] = lookup["date"] - pd.to_timedelta(steps * season_length, unit="D")

    known = train.rename(columns={"date": "source_date", "net": "prediction"})
    merged = lookup.merge(known, on=["segment", "source_date"], how="left")
    return pd.Series(merged["prediction"].values, index=test.index)