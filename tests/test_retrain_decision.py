import pandas as pd
import pytest

from flowradar.monitoring.decision import decide_retrain


def _drift(rows):
    return pd.DataFrame(rows, columns=["segment", "drift_detected"])


def _err(rows):
    return pd.DataFrame(rows, columns=["segment", "triggered"])


def test_no_signals_means_no_retrain():
    result = decide_retrain(
        _drift([("salaried", False), ("freelancer", False)]),
        _err([("salaried", False), ("freelancer", False)]),
    )
    assert result["retrain"] is False
    assert result["reasons"] == []


def test_drift_only_retrains_in_any_mode():
    result = decide_retrain(
        _drift([("salaried", False), ("freelancer", True)]),
        _err([("salaried", False), ("freelancer", False)]),
        mode="any",
    )
    assert result["retrain"] is True
    assert result["drifted_segments"] == ["freelancer"]
    assert result["degraded_segments"] == []


def test_error_only_retrains_in_any_mode():
    result = decide_retrain(
        _drift([("salaried", False)]),
        _err([("salaried", True)]),
        mode="any",
    )
    assert result["retrain"] is True
    assert result["degraded_segments"] == ["salaried"]


def test_drift_only_does_not_retrain_in_both_mode():
    result = decide_retrain(
        _drift([("freelancer", True)]),
        _err([("freelancer", False)]),
        mode="both",
    )
    assert result["retrain"] is False
    assert any("requires both" in r for r in result["reasons"])


def test_both_signals_retrain_in_both_mode():
    result = decide_retrain(
        _drift([("freelancer", True)]),
        _err([("freelancer", True)]),
        mode="both",
    )
    assert result["retrain"] is True


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        decide_retrain(_drift([("salaried", False)]), _err([("salaried", False)]), mode="sometimes")