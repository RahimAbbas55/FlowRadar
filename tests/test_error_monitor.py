import numpy as np
import pandas as pd

from flowradar.monitoring.error_monitor import error_trigger_fired, monitor_forecast_error


def _log(segment, ref_err, cur_err, n_days=400, current_days=60):
    # the last current_days carry cur_err, everything before carries ref_err
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    err = np.where(np.arange(n_days) >= n_days - current_days, cur_err, ref_err)
    return pd.DataFrame(
        {"date": dates, "segment": segment, "actual": 100.0, "prediction": 100.0 + err}
    )


def test_stable_error_does_not_trigger():
    result = monitor_forecast_error(_log("salaried", ref_err=5.0, cur_err=5.0))
    assert result["error_ratio"].iloc[0] == 1.0
    assert bool(result["triggered"].iloc[0]) is False


def test_degraded_error_triggers():
    result = monitor_forecast_error(_log("salaried", ref_err=5.0, cur_err=15.0))
    assert abs(result["error_ratio"].iloc[0] - 3.0) < 1e-9
    assert bool(result["triggered"].iloc[0]) is True


def test_improved_error_does_not_trigger():
    result = monitor_forecast_error(_log("salaried", ref_err=10.0, cur_err=1.0))
    assert result["error_ratio"].iloc[0] < 1.0
    assert bool(result["triggered"].iloc[0]) is False


def test_nan_rows_are_dropped_from_counts():
    log = _log("salaried", ref_err=5.0, cur_err=5.0)
    log.loc[log.index[-10:], "actual"] = np.nan  # simulated data_gap days
    result = monitor_forecast_error(log)
    assert result["n_current"].iloc[0] == 50


def test_too_few_observations_is_not_assessed():
    result = monitor_forecast_error(_log("salaried", ref_err=5.0, cur_err=50.0), current_days=10)
    assert bool(result["assessed"].iloc[0]) is False
    assert np.isnan(result["error_ratio"].iloc[0])
    assert bool(result["triggered"].iloc[0]) is False


def test_only_the_degraded_segment_triggers():
    log = pd.concat(
        [
            _log("salaried", ref_err=5.0, cur_err=5.0),
            _log("freelancer", ref_err=5.0, cur_err=20.0),
        ],
        ignore_index=True,
    )
    result = monitor_forecast_error(log).set_index("segment")
    assert bool(result.loc["freelancer", "triggered"]) is True
    assert bool(result.loc["salaried", "triggered"]) is False


def test_zero_reference_error_with_nonzero_current_triggers():
    result = monitor_forecast_error(_log("salaried", ref_err=0.0, cur_err=5.0))
    assert result["error_ratio"].iloc[0] == float("inf")
    assert bool(result["triggered"].iloc[0]) is True


def test_error_trigger_fired_reflects_any_segment():
    stable = monitor_forecast_error(_log("salaried", ref_err=5.0, cur_err=5.0))
    degraded = monitor_forecast_error(_log("salaried", ref_err=5.0, cur_err=15.0))
    assert error_trigger_fired(stable) is False
    assert error_trigger_fired(degraded) is True