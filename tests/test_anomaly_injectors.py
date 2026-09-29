import pandas as pd
from datetime import date
from flowradar.anomalies.injectors import (
    inject_duplicate_debit,
    inject_large_unplanned_payment,
    inject_payroll_delay,
)

def _sample_df():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "segment": "salaried",
            "inflow": [0.0] * 20,
            "outflow": [50.0] * 20,
        }
    )
    df.loc[df["date"] == "2024-01-10", "inflow"] = 2800.0
    df["net"] = df["inflow"] - df["outflow"]
    return df


def test_large_unplanned_payment_spikes_outflow():
    df = _sample_df()
    out, event = inject_large_unplanned_payment(df, "salaried", date(2024, 1, 5), magnitude=4.0)
    row = out[out["date"] == "2024-01-05"].iloc[0]
    assert row["outflow"] == 200.0
    assert event.cause_label == "large_unplanned_payment"
    assert event.expected_value == 50.0
    assert event.actual_value == 200.0


def test_payroll_delay_zeroes_and_shifts_inflow():
    df = _sample_df()
    out, event = inject_payroll_delay(df, "salaried", date(2024, 1, 10), delay_days=3)
    original_day = out[out["date"] == "2024-01-10"].iloc[0]
    delayed_day = out[out["date"] == "2024-01-13"].iloc[0]
    assert original_day["inflow"] == 0.0
    assert delayed_day["inflow"] == 2800.0
    assert event.expected_value == 2800.0
    assert event.actual_value == 0.0


def test_duplicate_debit_adds_to_outflow():
    df = _sample_df()
    out, event = inject_duplicate_debit(df, "salaried", date(2024, 1, 5), magnitude=1.0)
    row = out[out["date"] == "2024-01-05"].iloc[0]
    assert row["outflow"] == 100.0  # doubled
    assert event.cause_label == "duplicate_or_suspicious_debit"


def test_net_recomputed_after_each_injector():
    df = _sample_df()
    out, _ = inject_large_unplanned_payment(df, "salaried", date(2024, 1, 5), magnitude=2.0)
    assert (out["net"] == (out["inflow"] - out["outflow"]).round(2)).all()


def test_only_targets_specified_segment():
    df = pd.concat([_sample_df(), _sample_df().assign(segment="freelancer")], ignore_index=True)
    out, _ = inject_large_unplanned_payment(df, "salaried", date(2024, 1, 5), magnitude=3.0)
    freelancer_row = out[(out["segment"] == "freelancer") & (out["date"] == "2024-01-05")].iloc[0]
    assert freelancer_row["outflow"] == 50.0