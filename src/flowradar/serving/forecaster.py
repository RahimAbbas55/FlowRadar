import math
import numpy as np
import pandas as pd
from flowradar.evaluation.conformal import conformal_quantile
from flowradar.evaluation.holdout import infer_horizon, predict_with_booster
_COLUMNS = ["segment", "date", "net", "inflow", "outflow"]
_SCALES = ("constant", "relative")

def calibration_residuals(booster, history: pd.DataFrame, train_end) -> pd.DataFrame:
    # residuals of the booster's own horizon-ahead forecasts on days after its training
    # cutoff. Those days were never seen in training, so the residuals are out-of-sample.
    # The forecast is kept too, because relative-scale intervals need it
    preds = predict_with_booster(booster, history)
    actual = history[["segment", "date", "net"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])

    merged = actual.merge(preds, on=["segment", "date"])
    merged = merged[merged["date"] > pd.Timestamp(train_end)].copy()
    merged["residual"] = merged["net"] - merged["prediction"]
    merged = merged.dropna(subset=["residual"])
    return merged[["segment", "date", "prediction", "residual"]].reset_index(drop=True)

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
    def __init__(
        self,
        booster,
        history: pd.DataFrame,
        train_end,
        model_version=None,
        min_calibration: int = 20,
        interval_scale: str = "constant",
    ):
        if interval_scale not in _SCALES:
            raise ValueError(f"interval_scale must be one of {_SCALES}, got {interval_scale!r}")
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
        self.interval_scale = interval_scale
        self.segments = sorted(self.history["segment"].unique())
        self.residuals = calibration_residuals(booster, self.history, self.train_end)

    def with_interval_scale(self, interval_scale: str) -> "ForecastService":
        # the same model and history with a different interval scale, so two modes can be
        # compared on one champion
        return ForecastService(
            self.booster, self.history, self.train_end, self.model_version,
            self.min_calibration, interval_scale,
        )

    def _calibrate(self, residuals: pd.DataFrame, alpha: float) -> dict:
        # per-segment (q, floor). constant: q is the conformal quantile of |residual|.
        # relative: q is the quantile of |residual| / (|forecast| + floor), where floor is the
        # segment's median |forecast|. A segment needs min_calibration residuals and enough for
        # the finite-sample level to be valid at this alpha (n >= 1/alpha - 1), otherwise q is
        # NaN rather than a silently weaker interval
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        needed = max(self.min_calibration, math.ceil(1 / alpha) - 1)
        params = {}
        for segment in self.segments:
            res = residuals[residuals["segment"] == segment]
            if len(res) < needed:
                params[segment] = (float("nan"), 0.0)
            elif self.interval_scale == "constant":
                params[segment] = (conformal_quantile(res["residual"], alpha=alpha), 0.0)
            else:
                floor = float(res["prediction"].abs().median())
                floor = floor if floor > 0 else 1.0
                scores = res["residual"].abs() / (res["prediction"].abs() + floor)
                params[segment] = (conformal_quantile(scores, alpha=alpha), floor)
        return params

    def _bounds(self, segment: pd.Series, forecast: pd.Series, params: dict):
        # q per row (NaN when the segment is not calibrated) and the half-width per row
        q = segment.map(lambda s: params[s][0]).astype(float)
        if self.interval_scale == "constant":
            return q, q
        floor = segment.map(lambda s: params[s][1]).astype(float)
        return q, q * (forecast.abs() + floor)

    def interval_half_widths(self, alpha: float = 0.1) -> dict:
        # per-segment calibration factor: a half-width in constant mode, a multiplier on
        # (|forecast| + floor) in relative mode
        return {segment: q for segment, (q, _) in self._calibrate(self.residuals, alpha).items()}

    def forecast(self, alpha: float = 0.1) -> pd.DataFrame:
        # every feature of a future day is at least `horizon` days old, so the empty future
        # rows never feed a feature. A NaN input in a day's lookback gives a NaN forecast
        combined = pd.concat([self.history, future_frame(self.history, self.horizon)], ignore_index=True)
        preds = predict_with_booster(self.booster, combined)
        preds = preds[preds["date"] > self.last_date].copy()

        params = self._calibrate(self.residuals, alpha)
        q, half = self._bounds(preds["segment"], preds["prediction"], params)
        out = preds.rename(columns={"prediction": "forecast"})
        out["lower"] = out["forecast"] - half
        out["upper"] = out["forecast"] + half
        out["interval_calibrated"] = q.notna()
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
        params = self._calibrate(self.residuals[self.residuals["date"] < window_start], alpha)

        preds = predict_with_booster(self.booster, self.history)
        actual = self.history[["segment", "date", "net"]]
        rows = actual.merge(preds, on=["segment", "date"])
        # days up to the training cutoff are in-sample for the booster, so they are never judged
        rows = rows[(rows["date"] >= window_start) & (rows["date"] > self.train_end)].copy()
        rows = rows.rename(columns={"net": "actual", "prediction": "forecast"})

        q, half = self._bounds(rows["segment"], rows["forecast"], params)
        rows["lower"] = rows["forecast"] - half
        rows["upper"] = rows["forecast"] + half

        above = (rows["actual"] > rows["upper"]).to_numpy()
        below = (rows["actual"] < rows["lower"]).to_numpy()
        status = np.select(
            [
                rows["actual"].isna().to_numpy(),
                rows["forecast"].isna().to_numpy(),
                q.isna().to_numpy(),
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