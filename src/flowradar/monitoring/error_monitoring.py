import numpy as np
import pandas as pd

from flowradar.monitoring.data_drift import make_windows


def _mae(df: pd.DataFrame) -> tuple[float, int]:
    # mean absolute error over rows where both actual and prediction exist
    errors = (df["actual"] - df["prediction"]).abs().dropna()
    if len(errors) == 0:
        return float("nan"), 0
    return float(errors.mean()), len(errors)


def monitor_forecast_error(
    log: pd.DataFrame,
    current_days: int = 60,
    reference_days: int = 180,
    ratio_threshold: float = 1.5,
    min_obs: int = 14,
) -> pd.DataFrame:
    # log columns: segment, date, actual, prediction. One row per segment comparing
    # recent MAE against the reference window's MAE, triggering when the ratio is too high
    reference, current = make_windows(log, current_days, reference_days)

    rows = []
    for segment in sorted(log["segment"].unique()):
        ref_mae, n_ref = _mae(reference[reference["segment"] == segment])
        cur_mae, n_cur = _mae(current[current["segment"] == segment])

        assessed = n_ref >= min_obs and n_cur >= min_obs
        if not assessed:
            ratio = float("nan")
        elif ref_mae == 0:
            ratio = float("inf") if cur_mae > 0 else 1.0
        else:
            ratio = cur_mae / ref_mae

        rows.append(
            {
                "segment": segment,
                "reference_mae": ref_mae,
                "current_mae": cur_mae,
                "error_ratio": ratio,
                "n_reference": n_ref,
                "n_current": n_cur,
                "assessed": assessed,
                "triggered": bool(assessed and ratio > ratio_threshold),
            }
        )
    return pd.DataFrame(rows)


def error_trigger_fired(result: pd.DataFrame) -> bool:
    # a retrain is recommended if any assessed segment's error ratio breached the threshold
    return bool(result["triggered"].any())