import math

import numpy as np
import pandas as pd

from flowradar.evaluation.conformal import conformal_quantile
from flowradar.evaluation.holdout import infer_horizon, predict_with_booster

_COLUMNS = ["segment", "date", "net", "inflow", "outflow"]


def calibration_residuals(booster, history: pd.DataFrame, train_end) -> pd.DataFrame:
    # residuals of the booster's own horizon-ahead forecasts on days after its training
    # cutoff. Those days were never seen in training, so the residuals are out-of-sample
    preds = predict_with_booster(booster, history)
    actual = history[["segment", "date", "net"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])

    merged = actual.merge(preds, on=["segment", "date"])
    merged = merged[merged["date"] > pd.Timestamp(train_end)].copy()
    merged["residual"] = merged["net"] - merged["prediction"]
    merged = merged.dropna(subset=["residual"])
    return merged[["segment", "date", "residual"]].reset_index(drop=True)


def future_frame(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # empty rows (NaN targets) for the `horizon` days after history ends, for every segment
    last = pd.to_datetime(history["date"]).max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    rows = [
        {"segment": s, "date": d, "net": np.nan, "inflow": np.nan, "outflow": np.nan}
        for s in sorted(history["segment"].unique())
        for d in dates
    ]
    return pd.DataFrame(rows)


class ForecastService:
    def __init__(self, booster, history: pd.DataFrame, train_end, model_version=None, min_calibration: int = 20):
        history = history[_COLUMNS].copy()
        history["date"] = pd.to_datetime(history["date"])
        self.history = history.sort_values(["segment", "date"]).reset_index(drop=True)
        self.last_date = self.history["date"].max()
        self.train_end = pd.Timestamp(train_end)
        if self.train_end >= self.last_date:
            raise ValueError("no out-of-sample history after the training cutoff to calibrate intervals on")

        self.booster = booster
        self.horizon = infer_horizon(booster)
        self.model_version = model_version
        self.min_calibration = min_calibration
        self.segments = sorted(self.history["segment"].unique())
        self.residuals = calibration_residuals(booster, self.history, self.train_end)

    def interval_half_widths(self, alpha: float = 0.1) -> dict:
        # per-segment conformal half-widths. A segment needs min_calibration residuals and
        # enough for the finite-sample level to be valid at this alpha (n >= 1/alpha - 1),
        # otherwise its width is NaN rather than a silently weaker interval
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        needed = max(self.min_calibration, math.ceil(1 / alpha) - 1)
        widths = {}
        for segment in self.segments:
            res = self.residuals.loc[self.residuals["segment"] == segment, "residual"]
            widths[segment] = conformal_quantile(res, alpha=alpha) if len(res) >= needed else float("nan")
        return widths

    def forecast(self, alpha: float = 0.1) -> pd.DataFrame:
        # every feature of a future day is at least `horizon` days old, so the empty future
        # rows never feed a feature. A NaN input in a day's lookback gives a NaN forecast
        combined = pd.concat([self.history, future_frame(self.history, self.horizon)], ignore_index=True)
        preds = predict_with_booster(self.booster, combined)
        preds = preds[preds["date"] > self.last_date].copy()

        half = preds["segment"].map(self.interval_half_widths(alpha))
        out = preds.rename(columns={"prediction": "forecast"})
        out["lower"] = out["forecast"] - half
        out["upper"] = out["forecast"] + half
        out["interval_calibrated"] = half.notna()
        out["horizon_days"] = (out["date"] - self.last_date).dt.days
        out["model_version"] = self.model_version
        columns = ["segment", "date", "horizon_days", "forecast", "lower", "upper",
                   "interval_calibrated", "model_version"]
        return out[columns].sort_values(["segment", "date"]).reset_index(drop=True)