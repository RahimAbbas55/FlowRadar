from prefect import flow, task

from flowradar.monitoring.data_drift import drift_by_segment, make_windows
from flowradar.monitoring.decision import decide_retrain
from flowradar.monitoring.error_monitor import monitor_forecast_error


@task(name="check-data-drift")
def check_data_drift(series, current_days: int, reference_days: int, drift_share_threshold: float):
    # per-segment Evidently drift on the raw signals between the two windows
    reference, current = make_windows(series, current_days, reference_days)
    return drift_by_segment(reference, current, drift_share_threshold=drift_share_threshold)


@task(name="check-forecast-error")
def check_forecast_error(prediction_log, current_days: int, reference_days: int, ratio_threshold: float):
    # per-segment recent-vs-reference MAE ratio from the prediction log
    return monitor_forecast_error(
        prediction_log,
        current_days=current_days,
        reference_days=reference_days,
        ratio_threshold=ratio_threshold,
    )


@task(name="decide-retrain")
def make_decision(drift_result, error_result, mode: str):
    return decide_retrain(drift_result, error_result, mode=mode)


# series and prediction_log are DataFrames and deliberately have no type hints,
# so Prefect's parameter validation never has to reason about DataFrame types
@flow(name="retrain-decision", log_prints=True)
def retrain_decision_flow(
    series,
    prediction_log,
    current_days: int = 60,
    reference_days: int = 180,
    drift_share_threshold: float = 0.5,
    error_ratio_threshold: float = 1.5,
    mode: str = "any",
) -> dict:
    drift_result = check_data_drift(series, current_days, reference_days, drift_share_threshold)
    error_result = check_forecast_error(
        prediction_log, current_days, reference_days, error_ratio_threshold
    )
    decision = make_decision(drift_result, error_result, mode)

    print(f"retrain={decision['retrain']} reasons={decision['reasons']}")

    # plain records rather than DataFrames, so the result is easy to log and serialize
    return {
        **decision,
        "drift": drift_result.to_dict("records"),
        "error": error_result.to_dict("records"),
    }