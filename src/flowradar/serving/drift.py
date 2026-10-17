import pandas as pd
from flowradar.evaluation.holdout import build_prediction_log
from flowradar.monitoring.data_drift import drift_by_segment, make_windows
from flowradar.monitoring.decision import decide_retrain
from flowradar.monitoring.error_monitor import monitor_forecast_error
from flowradar.serving.forecaster import ForecastService

def drift_status(
    service: ForecastService,
    current_days: int = 60,
    reference_days: int = 180,
    drift_share_threshold: float = 0.5,
    error_ratio_threshold: float = 1.5,
    mode: str = "any",
) -> dict:
    # runs the same monitors as the retrain flow on the served history. The error log keeps
    # only days after the champion's training cutoff, so every logged forecast is out-of-sample
    reference, current = make_windows(service.history, current_days, reference_days)
    drift = drift_by_segment(reference, current, drift_share_threshold=drift_share_threshold)

    log = build_prediction_log(service.booster, service.history, start_date=service.train_end + pd.Timedelta(days=1))
    error = monitor_forecast_error(
        log,
        current_days=current_days,
        reference_days=reference_days,
        ratio_threshold=error_ratio_threshold,
    )

    decision = decide_retrain(drift, error, mode=mode)
    drift_by = drift.set_index("segment")
    error_by = error.set_index("segment")

    segments = []
    for segment in service.segments:
        d = drift_by.loc[segment] if segment in drift_by.index else None
        e = error_by.loc[segment] if segment in error_by.index else None
        data_ok = d is not None and pd.notna(d["share_drifted"])
        error_ok = e is not None and bool(e["assessed"])
        drifted = bool(d["drift_detected"]) if data_ok else False
        degraded = bool(e["triggered"]) if error_ok else False

        if not data_ok and not error_ok:
            status = "unassessed"
        elif drifted or degraded:
            status = "alert"
        else:
            status = "ok"

        segments.append(
            {
                "segment": segment,
                "status": status,
                "data_drift_assessed": bool(data_ok),
                "share_drifted": float(d["share_drifted"]) if data_ok else None,
                "data_drift_detected": drifted,
                "error_assessed": bool(error_ok),
                "error_ratio": float(e["error_ratio"]) if error_ok else None,
                "error_degraded": degraded,
            }
        )

    return {
        "as_of": service.last_date,
        "model_version": service.model_version,
        "train_end": service.train_end,
        "current_days": current_days,
        "reference_days": reference_days,
        "retrain": bool(decision["retrain"]),
        "mode": decision["mode"],
        "reasons": decision["reasons"],
        "segments": segments,
    }