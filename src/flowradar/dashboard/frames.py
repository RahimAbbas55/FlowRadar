import pandas as pd
_FORECAST_COLUMNS = ["segment", "date", "horizon_days", "forecast", "lower", "upper", "interval_calibrated"]
_ANOMALY_COLUMNS = ["segment", "date", "actual", "forecast", "lower", "upper", "status", "distance"]
_STATUSES = ["anomaly", "normal", "no_actual", "no_forecast", "not_calibrated"]

def _frame(points: list, columns: list[str], numeric: list[str]) -> pd.DataFrame:
    # JSON null becomes NaN and dates become datetimes, so charts and tables can use the data directly
    df = pd.DataFrame(points)
    if df.empty:
        return pd.DataFrame(columns=columns)
    df["date"] = pd.to_datetime(df["date"])
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[columns].sort_values(["segment", "date"]).reset_index(drop=True)

def forecast_frame(payload: dict) -> pd.DataFrame:
    return _frame(payload["points"], _FORECAST_COLUMNS, ["forecast", "lower", "upper"])

def anomaly_frame(payload: dict) -> pd.DataFrame:
    return _frame(payload["points"], _ANOMALY_COLUMNS, ["actual", "forecast", "lower", "upper", "distance"])

def drift_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload["segments"])

def anomaly_counts(df: pd.DataFrame) -> dict:
    # a count for every status, including zeros, so the dashboard layout never changes shape
    counts = df["status"].value_counts() if len(df) else pd.Series(dtype=int)
    return {status: int(counts.get(status, 0)) for status in _STATUSES}

def flagged_table(df: pd.DataFrame) -> pd.DataFrame:
    # the flagged days, newest first, with the figures rounded for display
    flagged = df[df["status"] == "anomaly"].sort_values("date", ascending=False)
    columns = ["date", "segment", "actual", "forecast", "lower", "upper", "distance"]
    return flagged[columns].round(2).reset_index(drop=True)

def status_banner(drift: dict) -> tuple[str, str]:
    # one (level, message) pair for the top of the page. A segment that could not be assessed
    # is never reported as healthy
    segments = drift["segments"]
    if drift["retrain"]:
        return "alert", "Retrain recommended: " + "; ".join(drift["reasons"])

    alerts = [s["segment"] for s in segments if s["status"] == "alert"]
    if alerts:
        return "alert", "Drift signals in: " + ", ".join(alerts)

    unassessed = [s["segment"] for s in segments if s["status"] == "unassessed"]
    if segments and len(unassessed) == len(segments):
        return "unknown", "Drift could not be assessed for any segment"
    if unassessed:
        return "warning", "No drift detected, but not assessed for: " + ", ".join(unassessed)

    message = "No drift detected"
    if any(not s["error_assessed"] for s in segments):
        message += " (forecast error monitor not assessed, it needs more out-of-sample history)"
    return "ok", message

def model_caption(health: dict) -> str:
    version = health["model_version"]
    name = f"Model v{version}" if version is not None else "Model (unversioned)"
    return (
        f"{name} \u00b7 trained through {health['train_end_date']} \u00b7 "
        f"data through {health['as_of']} \u00b7 {health['horizon']}-day horizon"
    )

def snapshot_note(meta: dict) -> str:
    # tells the viewer that the numbers are stored results, not a live model
    built = meta.get("built_at", "an unknown date")
    return (
        "This deployment serves a precomputed snapshot, not a live model. Forecasts, anomaly flags and drift "
        f"status were computed once on {built} from synthetic data (generator v{meta.get('generator_version', '?')}, "
        f"seed {meta.get('seed', '?')}), so the controls choose between stored results."
    )