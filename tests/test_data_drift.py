import numpy as np
import pandas as pd
import pytest
from flowradar.monitoring.data_drift import (
    _extract_drifted_share,
    compute_data_drift,
    drift_by_segment,
    make_windows,
)

def _frame(n, mean, seed, segment="salaried"):
    # three independent columns so the drift verdict doesn't depend on net = inflow - outflow
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "segment": segment,
            "inflow": rng.normal(mean, 10, n),
            "outflow": rng.normal(mean, 10, n),
            "net": rng.normal(mean, 10, n),
        }
    )

def test_extract_share_from_expected_structure():
    result = {
        "metrics": [
            {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 2.0, "share": 0.6667}}
        ]
    }
    assert _extract_drifted_share(result) == pytest.approx(0.6667)

def test_extract_raises_when_metric_missing():
    with pytest.raises(ValueError):
        _extract_drifted_share({"metrics": [{"metric_name": "SomethingElse", "value": 1}]})

def test_no_drift_for_same_distribution():
    result = compute_data_drift(_frame(300, 100, seed=1), _frame(100, 100, seed=2))
    assert result["drift_detected"] is False

def test_drift_detected_for_shifted_distribution():
    result = compute_data_drift(_frame(300, 100, seed=1), _frame(100, 200, seed=2))
    assert result["drift_detected"] is True
    assert result["share_drifted"] == pytest.approx(1.0)

def test_nan_rows_are_dropped_not_crashing():
    current = _frame(100, 100, seed=2)
    current.loc[:9, ["inflow", "outflow", "net"]] = np.nan  # simulated data_gap days
    result = compute_data_drift(_frame(300, 100, seed=1), current)
    assert result["n_current"] == 90
    assert 0.0 <= result["share_drifted"] <= 1.0

def test_empty_window_returns_nan_without_drift():
    empty = _frame(100, 100, seed=2).iloc[0:0]
    result = compute_data_drift(_frame(300, 100, seed=1), empty)
    assert np.isnan(result["share_drifted"])
    assert result["drift_detected"] is False

def test_windows_do_not_overlap():
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": 0.0})
    reference, current = make_windows(df, current_days=60, reference_days=180)
    assert pd.to_datetime(reference["date"]).max() < pd.to_datetime(current["date"]).min()

def test_current_window_is_most_recent_days():
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": 0.0})
    reference, current = make_windows(df, current_days=60, reference_days=180)
    assert len(current) == 60
    assert len(reference) == 180
    assert pd.to_datetime(current["date"]).max() == dates.max()

def test_by_segment_flags_only_the_drifted_segment():
    reference = pd.concat(
        [_frame(300, 100, seed=1, segment="salaried"), _frame(300, 100, seed=3, segment="freelancer")],
        ignore_index=True,
    )
    current = pd.concat(
        [_frame(100, 100, seed=2, segment="salaried"), _frame(100, 200, seed=4, segment="freelancer")],
        ignore_index=True,
    )
    result = drift_by_segment(reference, current).set_index("segment")
    assert bool(result.loc["freelancer", "drift_detected"]) is True
    assert bool(result.loc["salaried", "drift_detected"]) is False

def test_by_segment_returns_one_row_per_segment():
    reference = pd.concat(
        [_frame(300, 100, seed=1, segment="salaried"), _frame(300, 100, seed=3, segment="freelancer")],
        ignore_index=True,
    )
    current = pd.concat(
        [_frame(100, 100, seed=2, segment="salaried"), _frame(100, 100, seed=4, segment="freelancer")],
        ignore_index=True,
    )
    assert len(drift_by_segment(reference, current)) == 2