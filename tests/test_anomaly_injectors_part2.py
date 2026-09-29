import pandas as pd
import numpy as np
from datetime import date

from flowradar.anomalies.injectors import (
    inject_business_closure,
    inject_client_payment_delay,
    inject_data_gap,
)

def _sample_df():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "segment": "freelancer",
            "inflow": [500.0] * 20,
            "outflow": [40.0] * 20,
        }
    )
    df["net"] = df["inflow"] - df["outflow"]
    return df

def test_client_payment_delay_zeroes_and_shifts():
    df = _sample_df()
    out, event = inject_client_payment_delay(df, "freelancer", date(2024, 1, 5), delay_days=4)
    original_day = out[out["date"] == "2024-01-05"].iloc[0]
    delayed_day = out[out["date"] == "2024-01-09"].iloc[0]
    assert original_day["inflow"] == 0.0
    assert delayed_day["inflow"] == 1000.0  # 500 base + 500 delayed
    assert event.cause_label == "client_payment_delay"

def test_business_closure_zeroes_inflow_across_range():
    df = _sample_df()
    out, event = inject_business_closure(df, "freelancer", date(2024, 1, 5), date(2024, 1, 8))
    closed = out[
        (pd.to_datetime(out["date"]) >= pd.Timestamp("2024-01-05"))
        & (pd.to_datetime(out["date"]) <= pd.Timestamp("2024-01-08"))
    ]
    assert (closed["inflow"] == 0.0).all()
    assert event.end_date == date(2024, 1, 8)

def test_data_gap_sets_nan_both_components():
    df = _sample_df()
    out, event = inject_data_gap(df, "freelancer", date(2024, 1, 5), date(2024, 1, 6))
    gap = out[
        (pd.to_datetime(out["date"]) >= pd.Timestamp("2024-01-05"))
        & (pd.to_datetime(out["date"]) <= pd.Timestamp("2024-01-06"))
    ]
    assert gap["inflow"].isna().all()
    assert gap["outflow"].isna().all()
    assert gap["net"].isna().all()
    assert event.cause_label == "data_gap"

def test_data_gap_does_not_affect_other_segments():
    df = pd.concat([_sample_df(), _sample_df().assign(segment="salaried")], ignore_index=True)
    out, _ = inject_data_gap(df, "freelancer", date(2024, 1, 5), date(2024, 1, 6))
    salaried_row = out[(out["segment"] == "salaried") & (out["date"] == "2024-01-05")].iloc[0]
    assert not np.isnan(salaried_row["inflow"])