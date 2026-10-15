import numpy as np
import pandas as pd
from flowradar.anomalies.injectors import (
    inject_business_closure,
    inject_duplicate_debit,
    inject_large_unplanned_payment,
)

def _df(outflows, segment="small_business"):
    dates = pd.date_range("2024-01-01", periods=len(outflows), freq="D")
    df = pd.DataFrame({"date": dates, "segment": segment, "inflow": 500.0, "outflow": outflows})
    df["net"] = df["inflow"] - df["outflow"]
    return df

def test_large_payment_on_a_zero_outflow_day_uses_the_typical_payment():
    df = _df([0, 0, 100, 100, 100, 0])
    out, event = inject_large_unplanned_payment(df, "small_business", df["date"].iloc[0], magnitude=3.0)
    assert out["outflow"].iloc[0] == 300.0  # 3 x the median non-zero outflow of 100
    assert event.expected_value == 0.0
    assert event.actual_value == 300.0

def test_large_payment_on_a_normal_day_behaves_as_before():
    df = _df([0, 0, 100, 100, 100, 0])
    out, event = inject_large_unplanned_payment(df, "small_business", df["date"].iloc[2], magnitude=3.0)
    assert out["outflow"].iloc[2] == 300.0
    assert event.expected_value == 100.0

def test_duplicate_debit_on_a_zero_outflow_day_uses_the_typical_payment():
    df = _df([0, 0, 100, 100, 100, 0])
    out, event = inject_duplicate_debit(df, "small_business", df["date"].iloc[0], magnitude=2.0)
    assert out["outflow"].iloc[0] == 200.0
    assert event.actual_value != event.expected_value

def test_duplicate_debit_on_a_normal_day_behaves_as_before():
    df = _df([0, 0, 100, 100, 100, 0])
    out, _ = inject_duplicate_debit(df, "small_business", df["date"].iloc[2], magnitude=1.0)
    assert out["outflow"].iloc[2] == 200.0  # the original 100 plus a duplicate of 100

def test_a_data_gap_day_is_not_overwritten_by_a_payment_anomaly():
    df = _df([0, 0, 100, 100, 100, 0])
    df.loc[0, ["inflow", "outflow", "net"]] = np.nan
    out, event = inject_large_unplanned_payment(df, "small_business", df["date"].iloc[0], magnitude=3.0)
    assert np.isnan(out["outflow"].iloc[0])
    assert np.isnan(event.expected_value)

def test_closure_leaves_data_gap_days_missing():
    df = _df([100.0] * 10)
    df.loc[3, ["inflow", "outflow", "net"]] = np.nan
    out, _ = inject_business_closure(df, "small_business", df["date"].iloc[2], df["date"].iloc[4])
    assert out["inflow"].iloc[2] == 0.0
    assert np.isnan(out["inflow"].iloc[3])
    assert out["inflow"].iloc[4] == 0.0