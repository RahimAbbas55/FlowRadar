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

    def _half_widths(self, residuals: pd.DataFrame, alpha: float) -> dict:
        # per-segment conformal half-widths. A segment needs min_calibration residuals and
        # enough for the finite-sample level to be valid at this alpha (n >= 1/alpha - 1),
        # otherwise its width is NaN rather than a silently weaker interval
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        needed = max(self.min_calibration, math.ceil(1 / alpha) - 1)
        widths = {}
        for segment in self.segments:
            res = residuals.loc[residuals["segment"] == segment, "residual"]
            widths[segment] = conformal_quantile(res, alpha=alpha) if len(res) >= needed else float("nan")
        return widths

    def interval_half_widths(self, alpha: float = 0.1) -> dict:
        return self._half_widths(self.residuals, alpha)

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

    def anomaly_flags(self, alpha: float = 0.01, days: int = 30) -> pd.DataFrame:
        # judges the most recent `days` against a strict conformal interval calibrated only on
        # out-of-sample days BEFORE that window. Calibrating on the days being judged would let
        # the largest residual set the width, so no judged day could ever exceed it
        if days < 1:
            raise ValueError("days must be at least 1")
        window_start = self.last_date - pd.Timedelta(days=days - 1)
        widths = self._half_widths(self.residuals[self.residuals["date"] < window_start], alpha)

        preds = predict_with_booster(self.booster, self.history)
        actual = self.history[["segment", "date", "net"]]
        rows = actual.merge(preds, on=["segment", "date"])
        # days up to the training cutoff are in-sample for the booster, so they are never judged
        rows = rows[(rows["date"] >= window_start) & (rows["date"] > self.train_end)].copy()
        rows = rows.rename(columns={"net": "actual", "prediction": "forecast"})

        half = rows["segment"].map(widths).astype(float)
        rows["lower"] = rows["forecast"] - half
        rows["upper"] = rows["forecast"] + half

        above = (rows["actual"] > rows["upper"]).to_numpy()
        below = (rows["actual"] < rows["lower"]).to_numpy()
        status = np.select(
            [
                rows["actual"].isna().to_numpy(),
                rows["forecast"].isna().to_numpy(),
                half.isna().to_numpy(),
                above | below,
            ],
            ["no_actual", "no_forecast", "not_calibrated", "anomaly"],
            default="normal",
        )
        rows["status"] = status
        outside = np.where(above, rows["actual"] - rows["upper"], rows["actual"] - rows["lower"])
        rows["distance"] = np.where(status == "anomaly", outside, np.where(status == "normal", 0.0, np.nan))

        columns = ["segment", "date", "actual", "forecast", "lower", "upper", "status", "distance"]
        return rows[columns].sort_values(["segment", "date"]).reset_index(drop=True)