import math
import pandas as pd
from flowradar.features.calendar_features import add_calendar_features, add_event_features
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
    df: pd.DataFrame,
    horizon: int = 30,
    rolling_windows: tuple[int, ...] = (7, 28),
    max_gap: int = 3,
) -> pd.DataFrame:
    # lag and rolling-mean features per segment, all shifted by at least the horizon.
    # Data gaps of up to max_gap days are bridged with the last known value before the
    # features are computed, so one missing day does not blank a month of forecasts. The
    # fill only looks backwards, and the target columns themselves are never filled
    out = df.sort_values(["segment", "date"]).copy()
    lags = horizon_safe_lags(horizon)

    for col in _TARGET_COLS:
        if max_gap > 0:
            filled = out.groupby("segment")[col].ffill(limit=max_gap)
        else:
            filled = out[col]
        grouped = filled.groupby(out["segment"])
        for lag in lags:
            out[f"{col}_lag_{lag}"] = grouped.shift(lag)
        for window in rolling_windows:
            # shifted by the full horizon first, so a window never reaches inside the horizon
            out[f"{col}_roll_mean_{window}"] = grouped.transform(
                lambda s, w=window: s.shift(horizon).rolling(w).mean()
            )
    return out

def build_horizon_features(series: pd.DataFrame, horizon: int = 30, max_gap: int = 3) -> pd.DataFrame:
    # calendar and event features are known in advance at any horizon, lag features are horizon-safe
    out = add_calendar_features(series)
    out = add_event_features(out)
    out = add_horizon_lag_features(out, horizon=horizon, max_gap=max_gap)
    return out.reset_index(drop=True)