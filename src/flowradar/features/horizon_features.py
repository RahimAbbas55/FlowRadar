import math
import pandas as pd
from flowradar.features.calendar_features import add_calendar_features
_TARGET_COLS = ["net", "inflow", "outflow"]

def horizon_safe_lags(horizon: int, n_weekly: int = 3) -> list[int]:
    # every lag is at least the forecast horizon, so no feature can use a value that
    # would still be unknown when forecasting horizon days ahead. Beyond the horizon
    # lag itself, lags are weekly multiples so day-of-week alignment is kept
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    first_weekly = 7 * math.ceil(horizon / 7)
    return sorted({horizon} | {first_weekly + 7 * i for i in range(n_weekly)})


def add_horizon_lag_features(
    df: pd.DataFrame, horizon: int = 30, rolling_windows: tuple[int, ...] = (7, 28)
) -> pd.DataFrame:
    # lag and rolling-mean features per segment, all shifted by at least the horizon
    out = df.sort_values(["segment", "date"]).copy()
    lags = horizon_safe_lags(horizon)

    for col in _TARGET_COLS:
        grouped = out.groupby("segment")[col]
        for lag in lags:
            out[f"{col}_lag_{lag}"] = grouped.shift(lag)
        for window in rolling_windows:
            # shifted by the full horizon first, so a window never reaches inside the horizon
            out[f"{col}_roll_mean_{window}"] = grouped.transform(
                lambda s, w=window: s.shift(horizon).rolling(w).mean()
            )
    return out


def build_horizon_features(series: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    # calendar features are known in advance at any horizon, lag features are horizon-safe
    out = add_calendar_features(series)
    out = add_horizon_lag_features(out, horizon=horizon)
    return out.reset_index(drop=True)