import pandas as pd
from functools import partial
from flowradar.evaluation.backtest import make_walk_forward_folds, split_calibration
from flowradar.evaluation.conformal import conformal_interval, conformal_quantile
from flowradar.evaluation.metrics import interval_coverage, mase, mean_interval_width, smape
from flowradar.models.lightgbm_model import lightgbm_predict
from flowradar.models.seasonal_naive import seasonal_naive_predict
from flowradar.models.statsforecast_model import statsforecast_predict

# default model set run by the comparison, each matching the shared (train, test) -> predictions contract
_MODELS = {
    "seasonal_naive": seasonal_naive_predict,
    "lightgbm": lightgbm_predict,
    "ets": partial(statsforecast_predict, model_name="ets"),
    "arima": partial(statsforecast_predict, model_name="arima"),
}

def _score_fold(predict_fn, fit_train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame, alpha: float):
    # calibrates on the held-out calibration slice, then scores point + interval predictions on test
    calib_preds = predict_fn(fit_train, calibration)
    residuals = calibration["net"].reset_index(drop=True) - calib_preds.reset_index(drop=True)
    q = conformal_quantile(residuals, alpha=alpha)

    test_preds = predict_fn(fit_train, test)
    lower, upper = conformal_interval(test_preds, q)

    actual = test["net"].reset_index(drop=True)
    return {
        "mase": mase(actual, test_preds.reset_index(drop=True), fit_train["net"]),
        "smape": smape(actual, test_preds.reset_index(drop=True)),
        "interval_coverage": interval_coverage(actual, lower.reset_index(drop=True), upper.reset_index(drop=True)),
        "mean_interval_width": mean_interval_width(lower.reset_index(drop=True), upper.reset_index(drop=True)),
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
    # runs every model through every fold, returning one row per (model, fold) so
    # variance across folds stays visible rather than hidden behind an average
    model_fns = models or _MODELS
    folds = make_walk_forward_folds(df, horizon=horizon, n_folds=n_folds, step_days=step_days)

    rows = []
    for fold in folds:
        fit_train, calibration = split_calibration(fold.train, calib_days=calib_days)
        for model_name, predict_fn in model_fns.items():
            scores = _score_fold(predict_fn, fit_train, calibration, fold.test, alpha=alpha)
            rows.append({"model": model_name, "fold_id": fold.fold_id, **scores})

    return pd.DataFrame(rows)