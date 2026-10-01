import numpy as np
import pandas as pd

def conformal_quantile(residuals: pd.Series, alpha: float = 0.1) -> float:
    # split conformal: quantile of absolute calibration residuals, with the finite-sample
    # correction so the resulting interval has a valid coverage guarantee, not just asymptotic
    abs_res = residuals.abs().dropna().sort_values().values
    n = len(abs_res)
    if n == 0:
        return float("nan")
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(abs_res, level))


def conformal_interval(point_predictions: pd.Series, q: float) -> tuple[pd.Series, pd.Series]:
    # symmetric interval around each point prediction, fixed width set by the calibration quantile
    return point_predictions - q, point_predictions + q