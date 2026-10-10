import math

import mlflow
from mlflow.exceptions import MlflowException

from flowradar.tracking.mlflow_tracking import _ensure_tracking_uri_set
from flowradar.tracking.registry import CHAMPION_ALIAS, _MODEL_NAME


def should_promote(
    challenger_score: float | None,
    champion_score: float | None,
    min_improvement: float = 0.02,
) -> tuple[bool, str]:
    # lower MASE is better: the challenger must beat the champion by at least
    # min_improvement (as a fraction) so noise alone can't flip the champion
    if challenger_score is None or math.isnan(challenger_score):
        return False, "challenger score is missing or NaN"
    if champion_score is None:
        return True, "no existing champion"
    if math.isnan(champion_score):
        return True, "champion score is NaN, cannot be compared"

    threshold = champion_score * (1 - min_improvement)
    if challenger_score < threshold:
        return True, f"challenger {challenger_score:.4f} beats threshold {threshold:.4f}"
    return False, f"challenger {challenger_score:.4f} does not beat threshold {threshold:.4f}"


def get_champion_version():
    # returns the version currently holding the champion alias, or None if there isn't one
    _ensure_tracking_uri_set()
    client = mlflow.MlflowClient()
    try:
        return client.get_model_version_by_alias(_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None


def promote_if_better(
    challenger_version,
    challenger_score: float,
    champion_score: float | None,
    min_improvement: float = 0.02,
) -> dict:
    # records the challenger's holdout score on its version, then moves the champion
    # alias to it only if the gate says so. Both scores must come from the same holdout.
    _ensure_tracking_uri_set()
    client = mlflow.MlflowClient()
    version = str(challenger_version.version)

    client.set_model_version_tag(_MODEL_NAME, version, "holdout_mase", str(challenger_score))

    promote, reason = should_promote(challenger_score, champion_score, min_improvement)
    if promote:
        client.set_registered_model_alias(_MODEL_NAME, CHAMPION_ALIAS, version)

    return {"promoted": promote, "reason": reason, "version": int(version)}