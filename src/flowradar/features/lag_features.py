import pandas as pd

# Adds lag and rolling-mean features per segment, never crossing segment boundaries
def add_lag_features(
    df: pd.DataFrame, lags: list[int] = [1, 7, 14, 28], rolling_windows: list[int] = [7, 28]
) -> pd.DataFrame:
    out = df.sort_values(["segment", "date"]).copy()
    for col in ["net", "inflow", "outflow"]:
        grouped = out.groupby("segment")[col]
        for lag in lags:
            out[f"{col}_lag_{lag}"] = grouped.shift(lag)
        for window in rolling_windows:
            # shift(1) first so the current day's own value never leaks into its own rolling mean
            out[f"{col}_roll_mean_{window}"] = (
                grouped.shift(1).rolling(window).mean()
            )

    return out