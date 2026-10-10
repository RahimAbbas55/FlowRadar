import math

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# raw signals where Phase 1 injects drift; calendar features cannot drift
DRIFT_COLUMNS = ["inflow", "outflow", "net"]


def _extract_drifted_share(result: dict) -> float:
    # pulls the share of drifted columns out of Evidently's result dict. This is the
    # only field read from Evidently's output, so format changes only break one place
    for metric in result.get("metrics", []):
        name = f"{metric.get('metric_name', '')}{metric.get('metric_id', '')}"
        if "DriftedColumnsCount" in name:
            value = metric.get("value")
            if isinstance(value, dict) and "share" in value:
                return float(value["share"])
    raise ValueError(
        "could not find the DriftedColumnsCount share in Evidently output, inspect snapshot.dict()"
    )


def make_windows(
    df: pd.DataFrame, current_days: int = 60, reference_days: int = 180
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # current = the most recent current_days, reference = the reference_days
    # immediately before it, so the two windows never overlap
    dates = pd.to_datetime(df["date"])
    current_start = dates.max() - pd.Timedelta(days=current_days - 1)
    reference_start = current_start - pd.Timedelta(days=reference_days)
    reference = df[(dates >= reference_start) & (dates < current_start)].copy()
    current = df[dates >= current_start].copy()
    return reference, current


def compute_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    drift_share_threshold: float = 0.5,
) -> dict:
    # runs Evidently's drift preset on one reference/current pair and applies our own
    # threshold to the share of drifted columns
    cols = columns or DRIFT_COLUMNS
    ref = reference[cols].dropna().reset_index(drop=True)
    cur = current[cols].dropna().reset_index(drop=True)

    if ref.empty or cur.empty:
        # cannot assess drift without data in both windows
        return {
            "share_drifted": float("nan"),
            "drift_detected": False,
            "n_reference": len(ref),
            "n_current": len(cur),
        }

    definition = DataDefinition(numerical_columns=cols)
    ref_ds = Dataset.from_pandas(ref, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur, data_definition=definition)

    snapshot = Report([DataDriftPreset()]).run(cur_ds, ref_ds)
    share = _extract_drifted_share(snapshot.dict())

    return {
        "share_drifted": share,
        "drift_detected": bool(share >= drift_share_threshold),
        "n_reference": len(ref),
        "n_current": len(cur),
    }


def drift_by_segment(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    drift_share_threshold: float = 0.5,
) -> pd.DataFrame:
    # one row per segment, since segments have different distributions and pooling
    # them would flag drift just from mixing segments
    rows = []
    for segment in sorted(current["segment"].unique()):
        result = compute_data_drift(
            reference[reference["segment"] == segment],
            current[current["segment"] == segment],
            columns=columns,
            drift_share_threshold=drift_share_threshold,
        )
        rows.append({"segment": segment, **result})
    return pd.DataFrame(rows)