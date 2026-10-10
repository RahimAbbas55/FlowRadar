import pandas as pd

_VALID_MODES = {"any", "both"}


def decide_retrain(
    drift_result: pd.DataFrame, error_result: pd.DataFrame, mode: str = "any"
) -> dict:
    # combines the two monitors into one retrain decision. "any" retrains when either
    # signal fires, "both" only when data drift and forecast error degradation both fire
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")

    drifted = sorted(
        drift_result.loc[drift_result["drift_detected"].astype(bool), "segment"]
    )
    degraded = sorted(error_result.loc[error_result["triggered"].astype(bool), "segment"])

    drift_fired = len(drifted) > 0
    error_fired = len(degraded) > 0
    retrain = (drift_fired or error_fired) if mode == "any" else (drift_fired and error_fired)

    reasons = []
    if drift_fired:
        reasons.append(f"data drift in: {', '.join(drifted)}")
    if error_fired:
        reasons.append(f"forecast error degraded in: {', '.join(degraded)}")
    if not retrain and (drift_fired or error_fired):
        reasons.append("only one signal fired and mode 'both' requires both")

    return {
        "retrain": bool(retrain),
        "mode": mode,
        "drift_fired": drift_fired,
        "error_fired": error_fired,
        "drifted_segments": drifted,
        "degraded_segments": degraded,
        "reasons": reasons,
    }