import pytest
import pandas as pd
from flowradar.dashboard.frames import (
    anomaly_counts,
    anomaly_frame,
    drift_frame,
    flagged_table,
    forecast_frame,
    model_caption,
    snapshot_note,
    status_banner,
)

def _point(segment, date, **values):
    return {"segment": segment, "date": date, **values}

def _anomaly_payload():
    return {
        "points": [
            _point("b", "2024-03-02", actual=5.0, forecast=4.0, lower=1.0, upper=9.0, status="normal", distance=0.0),
            _point("a", "2024-03-02", actual=50.0, forecast=4.0, lower=1.0, upper=9.0, status="anomaly", distance=41.0),
            _point("a", "2024-03-01", actual=None, forecast=4.0, lower=None, upper=None, status="not_calibrated", distance=None),
            _point("a", "2024-03-03", actual=60.0, forecast=4.0, lower=1.0, upper=9.0, status="anomaly", distance=51.0),
        ]
    }

def _segment(name, status, data=True, error=True):
    return {
        "segment": name, "status": status, "data_drift_assessed": data, "share_drifted": None,
        "data_drift_detected": status == "alert", "error_assessed": error, "error_ratio": None,
        "error_degraded": False,
    }

def _drift(segments, retrain=False, reasons=()):
    return {"retrain": retrain, "reasons": list(reasons), "segments": segments}

def test_forecast_frame_parses_dates_sorts_and_turns_null_into_nan():
    payload = {
        "points": [
            _point("b", "2024-02-02", horizon_days=2, forecast=2.0, lower=None, upper=None, interval_calibrated=False),
            _point("a", "2024-02-02", horizon_days=2, forecast=1.0, lower=0.0, upper=2.0, interval_calibrated=True),
            _point("a", "2024-02-01", horizon_days=1, forecast=1.5, lower=0.5, upper=2.5, interval_calibrated=True),
        ]
    }
    df = forecast_frame(payload)
    assert list(df["segment"]) == ["a", "a", "b"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].iloc[0] < df["date"].iloc[1]
    assert pd.isna(df["lower"].iloc[2]) and pd.isna(df["upper"].iloc[2])

def test_an_empty_forecast_payload_gives_an_empty_frame_with_the_columns():
    df = forecast_frame({"points": []})
    assert df.empty
    assert {"segment", "date", "forecast", "lower", "upper"} <= set(df.columns)

def test_anomaly_frame_keeps_nulls_as_nan_and_sorts():
    df = anomaly_frame(_anomaly_payload())
    assert list(df["segment"]) == ["a", "a", "a", "b"]
    assert pd.isna(df["actual"].iloc[0]) and pd.isna(df["lower"].iloc[0])
    assert df["date"].iloc[0] == pd.Timestamp("2024-03-01") 

def test_anomaly_counts_include_every_status_even_when_zero():
    counts = anomaly_counts(anomaly_frame(_anomaly_payload()))
    assert counts == {"anomaly": 2, "normal": 1, "no_actual": 0, "no_forecast": 0, "not_calibrated": 1}
    assert anomaly_counts(anomaly_frame({"points": []}))["anomaly"] == 0

def test_flagged_table_lists_only_anomalies_newest_first():
    table = flagged_table(anomaly_frame(_anomaly_payload()))
    assert list(table["date"]) == [pd.Timestamp("2024-03-03"), pd.Timestamp("2024-03-02")]
    assert list(table["actual"]) == [60.0, 50.0]
    assert "status" not in table.columns

def test_drift_frame_has_one_row_per_segment():
    df = drift_frame(_drift([_segment("a", "ok"), _segment("b", "alert")]))
    assert list(df["segment"]) == ["a", "b"]

def test_banner_alerts_with_the_reasons_when_a_retrain_is_recommended():
    level, message = status_banner(
        _drift([_segment("a", "alert")], retrain=True, reasons=["data drift in: a"])
    )
    assert level == "alert"
    assert "data drift in: a" in message

def test_banner_alerts_on_a_segment_signal_without_a_retrain_decision():
    level, message = status_banner(_drift([_segment("a", "ok"), _segment("b", "alert")]))
    assert level == "alert" and "b" in message

def test_banner_is_unknown_when_nothing_could_be_assessed():
    level, _ = status_banner(_drift([_segment("a", "unassessed", False, False)]))
    assert level == "unknown"

def test_banner_warns_when_only_some_segments_could_be_assessed():
    level, message = status_banner(_drift([_segment("a", "ok"), _segment("b", "unassessed", False, False)]))
    assert level == "warning" and "b" in message

def test_banner_says_when_the_error_monitor_was_not_assessed():
    level, message = status_banner(_drift([_segment("a", "ok", True, False)]))
    assert level == "ok"
    assert "not assessed" in message

def test_banner_is_plainly_ok_when_both_monitors_were_assessed():
    level, message = status_banner(_drift([_segment("a", "ok")]))
    assert (level, message) == ("ok", "No drift detected")

def test_model_caption_names_the_version_and_dates():
    health = {"model_version": 3, "train_end_date": "2024-07-03", "as_of": "2024-12-30", "horizon": 30}
    assert model_caption(health) == "Model v3 \u00b7 trained through 2024-07-03 \u00b7 data through 2024-12-30 \u00b7 30-day horizon"
    assert model_caption({**health, "model_version": None}).startswith("Model (unversioned)")

def test_the_snapshot_note_says_the_results_are_precomputed():
    note = snapshot_note({"built_at": "2026-10-20", "generator_version": "0.3.0", "seed": 42})
    assert "precomputed snapshot" in note
    assert "2026-10-20" in note and "seed 42" in note