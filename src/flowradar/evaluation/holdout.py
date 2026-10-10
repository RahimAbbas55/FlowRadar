import numpy as np
import pandas as pd

from flowradar.evaluation.metrics import mase
from flowradar.features.pipeline import build_features


def predict_with_booster(booster, series: pd.DataFrame) -> pd.DataFrame:
    # one-step-ahead predictions for every row, using only backward-looking features.
    # Rows without full lag history get NaN, matching how training dropped warm-up rows
    features = build_features(series)
    names = booster.feature_name()
    lag_cols = [n for n in names if n != "segment"]

    X = features[names].copy()
    if "segment" in X.columns:
        X["segment"] = X["segment"].astype("category")

    valid = X[lag_cols].notna().all(axis=1)
    preds = pd.Series(np.nan, index=X.index)
    if valid.any():
        preds.loc[valid] = booster.predict(X.loc[valid])

    out = features[["segment", "date"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["prediction"] = preds
    return out.reset_index(drop=True)


def score_on_holdout(booster, fit_df: pd.DataFrame, holdout_df: pd.DataFrame) -> float:
    # mean per-segment MASE on the holdout, with fit_df as the lag history. Per-segment
    # scoring stops a large-scale segment from dominating a pooled score
    combined = pd.concat([fit_df, holdout_df], ignore_index=True)
    preds = predict_with_booster(booster, combined)

    holdout = holdout_df[["segment", "date", "net"]].copy()
    holdout["date"] = pd.to_datetime(holdout["date"])
    merged = holdout.merge(preds, on=["segment", "date"], how="left")

    scores = []
    for segment, group in merged.groupby("segment"):
        group = group.dropna(subset=["net", "prediction"])
        if group.empty:
            continue
        history = fit_df[fit_df["segment"] == segment].sort_values("date")["net"]
        score = mase(group["net"], group["prediction"], history)
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else float("nan")


def build_prediction_log(booster, series: pd.DataFrame, start_date) -> pd.DataFrame:
    # (segment, date, actual, prediction) rows from start_date onward, for the error monitor.
    # Only out-of-sample if the booster was trained on data before start_date
    preds = predict_with_booster(booster, series)
    actual = series[["segment", "date", "net"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])

    log = actual.merge(preds, on=["segment", "date"]).rename(columns={"net": "actual"})
    log = log[log["date"] >= pd.Timestamp(start_date)]
    return log[["segment", "date", "actual", "prediction"]].reset_index(drop=True)