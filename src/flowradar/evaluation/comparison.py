import math
import numpy as np
import pandas as pd
from functools import partial
from flowradar.evaluation.backtest import make_walk_forward_folds, split_calibration
from flowradar.evaluation.conformal import segment_quantiles
from flowradar.evaluation.metrics import interval_coverage, mase, mean_interval_width, smape
from flowradar.models.lightgbm_model import lightgbm_predict
from flowradar.models.seasonal_naive import seasonal_naive_predict
from flowradar.models.statsforecast_model import statsforecast_predict

def _default_models(horizon: int) -> dict:
    # every model matches the (train, test) -> predictions contract
    return {
        "seasonal_naive": seasonal_naive_predict,
        "lightgbm": partial(lightgbm_predict, horizon=horizon),
        "ets": partial(statsforecast_predict, model_name="ets"),
        "arima": partial(statsforecast_predict, model_name="arima"),
    }

def _finite_mean(values) -> float:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")

def _score_fold(
    predict_fn,
    fit_train: pd.DataFrame,
    calibration: pd.DataFrame,
    full_train: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
):
    # calibration residuals come from forecasting the calibration slice from fit_train
    # (1..calib_days ahead). Test predictions are made from full_train, so they are also
    # 1..horizon days ahead and the interval is calibrated for the horizon it is used at.
    # Each segment gets its own conformal quantile, because one pooled quantile is set by the
    # largest segment and under-covers it
    calib_preds = predict_fn(fit_train, calibration)
    residuals = pd.DataFrame(
        {
            "segment": calibration["segment"].values,
            "residual": calibration["net"].values - calib_preds.values,
        }
    )
    q_by_segment = segment_quantiles(residuals, alpha=alpha)

    test_preds = predict_fn(full_train, test)
    scored = pd.DataFrame(
        {
            "segment": test["segment"].values,
            "actual": test["net"].values,
            "prediction": test_preds.values,
        }
    )
    scored["q"] = scored["segment"].map(q_by_segment)
    scored["lower"] = scored["prediction"] - scored["q"]
    scored["upper"] = scored["prediction"] + scored["q"]
    # rows with a missing actual (data gap), a missing prediction, or no valid interval for their
    # segment are excluded, not scored as misses. n_scored and segments_scored say what remains
    scored = scored.dropna(subset=["actual", "prediction", "q"])

    mases, coverages, width_mases = [], [], []
    for segment, group in scored.groupby("segment"):
        history = full_train.loc[full_train["segment"] == segment].sort_values("date")["net"]
        mases.append(mase(group["actual"], group["prediction"], history))
        coverages.append(interval_coverage(group["actual"], group["lower"], group["upper"]))
        # width in MASE units: the segment's naive one-step error is the unit, so segments
        # with very different scales can be compared
        scale = history.diff().abs().mean()
        width = mean_interval_width(group["lower"], group["upper"])
        width_mases.append(width / scale if pd.notna(scale) and scale > 0 else float("nan"))

    return {
        "mase": _finite_mean(mases),
        "smape": smape(scored["actual"], scored["prediction"]),
        "interval_coverage": interval_coverage(scored["actual"], scored["lower"], scored["upper"]),
        "min_segment_coverage": min(coverages) if coverages else float("nan"),
        "mean_interval_width": mean_interval_width(scored["lower"], scored["upper"]),
        "mean_width_mase": _finite_mean(width_mases),
        "n_scored": len(scored),
        "segments_scored": int(scored["segment"].nunique()),
    }

def run_model_comparison(
    df: pd.DataFrame,
    horizon: int = 30,
    n_folds: int = 3,
    step_days: int = 30,
    calib_days: int = 30,
    alpha: float = 0.1,
    models: dict | None = None,
) -> pd.DataFrame:
    # runs every model through every fold, one row per (model, fold) so variance
    # across folds stays visible
    if calib_days > horizon:
        raise ValueError("calib_days must not exceed horizon, or calibration would use longer horizons than the test")
    needed = math.ceil(1 / alpha) - 1
    if calib_days < needed:
        raise ValueError(
            f"calib_days ({calib_days}) gives too few calibration residuals for alpha={alpha}, "
            f"which needs at least {needed}"
        )

    model_fns = models or _default_models(horizon)
    folds = make_walk_forward_folds(df, horizon=horizon, n_folds=n_folds, step_days=step_days)

    rows = []
    for fold in folds:
        fit_train, calibration = split_calibration(fold.train, calib_days=calib_days)
        for model_name, predict_fn in model_fns.items():
            scores = _score_fold(predict_fn, fit_train, calibration, fold.train, fold.test, alpha=alpha)
            rows.append({"model": model_name, "fold_id": fold.fold_id, **scores})

    return pd.DataFrame(rows)