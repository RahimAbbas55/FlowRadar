import json
import math
import pytest
import pandas as pd
from flowradar.dashboard.charts import anomaly_chart, forecast_chart

def _forecast_df():
    dates = pd.date_range("2024-02-01", periods=4, freq="D")
    rows = []
    for segment, base in (("a", 100.0), ("b", 9000.0)):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "segment": segment, "date": d, "horizon_days": i + 1, "forecast": base + i,
                    "lower": base + i - 10 if i < 3 else float("nan"),
                    "upper": base + i + 10 if i < 3 else float("nan"),
                    "interval_calibrated": i < 3,
                }
            )
    return pd.DataFrame(rows)

def _anomaly_df():
    dates = pd.date_range("2024-03-01", periods=4, freq="D")
    nan = float("nan")
    rows = [
        {"segment": "a", "date": dates[0], "actual": 50.0, "forecast": 5.0, "lower": 0.0, "upper": 10.0, "status": "anomaly"},
        {"segment": "a", "date": dates[1], "actual": 6.0, "forecast": 5.0, "lower": 0.0, "upper": 10.0, "status": "normal"},
        {"segment": "a", "date": dates[2], "actual": 7.0, "forecast": 5.0, "lower": nan, "upper": nan, "status": "not_calibrated"},
        {"segment": "b", "date": dates[0], "actual": 1.0, "forecast": 1.0, "lower": 0.0, "upper": 2.0, "status": "normal"},
    ]
    return pd.DataFrame(rows)

def _rows(spec):
    datasets = spec.get("datasets")
    if datasets:
        return next(iter(datasets.values()))
    return spec["layer"][0]["data"]["values"]

def _mark(layer):
    mark = layer["mark"]
    return mark if isinstance(mark, str) else mark["type"]

def test_forecast_chart_is_a_band_under_a_line_for_one_segment():
    spec = forecast_chart(_forecast_df(), "a").to_dict()
    assert [_mark(layer) for layer in spec["layer"]] == ["area", "line"]
    rows = _rows(spec)
    assert len(rows) == 4
    assert max(row["forecast"] for row in rows) < 1000  # segment b's rows are not in the chart

def test_forecast_chart_serialises_missing_bounds_as_non_numbers():
    rows = _rows(forecast_chart(_forecast_df(), "a").to_dict())
    uncalibrated = [row for row in rows if row["horizon_days"] == 4][0]
    for key in ("lower", "upper"):
        value = uncalibrated[key]
        assert value is None or (isinstance(value, float) and math.isnan(value))

def test_an_unknown_segment_is_rejected_by_both_charts():
    with pytest.raises(ValueError, match="no rows for segment"):
        forecast_chart(_forecast_df(), "nope")
    with pytest.raises(ValueError, match="no rows for segment"):
        anomaly_chart(_anomaly_df(), "nope")

def test_anomaly_chart_has_a_flagged_layer_that_filters_on_status():
    spec = anomaly_chart(_anomaly_df(), "a").to_dict()
    assert [_mark(layer) for layer in spec["layer"]] == ["area", "line", "point"]
    assert "anomaly" in json.dumps(spec["layer"][2]["transform"])
    assert len(_rows(spec)) == 3