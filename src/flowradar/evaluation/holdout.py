import re
import numpy as np
import pandas as pd
from flowradar.evaluation.metrics import mase
from flowradar.features.horizon_features import build_horizon_features

def infer_horizon(booster) -> int:
    # the pipeline always includes a lag equal to the horizon, so the smallest lag
    # among the booster's feature names is the horizon it was trained for
    lags = [
        int(m.group(1))
        for name in booster.feature_name()
        if (m := re.search(r"_lag_(\d+)$", name))
    ]
    if not lags:
        raise ValueError("booster has no lag features, cannot infer its horizon")
    return min(lags)

def predict_with_booster(booster, series: pd.DataFrame) -> pd.DataFrame:
    # for every row, the direct forecast made `horizon` days earlier: all features use data
    # at least `horizon` days old, so a row's prediction never reads its own actual.
    # Rows without full feature history get NaN, matching how training dropped warm-up rows
    features = build_horizon_features(series, horizon=infer_horizon(booster))
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
    # mean per-segment MASE on the holdout, with fit_df as the known history. The holdout
    # may not reach further past fit_df than the horizon, or its features would use holdout actuals
    horizon = infer_horizon(booster)
    span = (pd.to_datetime(holdout_df["date"]).max() - pd.to_datetime(fit_df["date"]).max()).days
    if span > horizon:
        raise ValueError(
            f"holdout reaches {span} days past the fit data but the booster's horizon is {horizon}"
        )

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
    # Each prediction is the booster's `horizon`-day-ahead forecast. Only out-of-sample
    # if the booster was trained on data before start_date
    preds = predict_with_booster(booster, series)
    actual = series[["segment", "date", "net"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])

    log = actual.merge(preds, on=["segment", "date"]).rename(columns={"net": "actual"})
    log = log[log["date"] >= pd.Timestamp(start_date)]
    return log[["segment", "date", "actual", "prediction"]].reset_index(drop=True)