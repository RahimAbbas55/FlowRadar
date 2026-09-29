import pandas as pd
from flowradar.anomalies.events import AnomalyEvent

# Locates the single row for a segment and date
def _row_mask(df: pd.DataFrame, segment: str, event_date) -> pd.Series:
    return (df["segment"] == segment) & (
        pd.to_datetime(df["date"]) == pd.Timestamp(event_date)
    )

# One-off large outflow spike on a single day
def inject_large_unplanned_payment(
    df: pd.DataFrame, segment: str, event_date, magnitude: float
) -> tuple[pd.DataFrame, AnomalyEvent]:
    out = df.copy()
    mask = _row_mask(out, segment, event_date)
    expected = float(out.loc[mask, "outflow"].iloc[0])
    actual = expected * magnitude
    out.loc[mask, "outflow"] = actual
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    event = AnomalyEvent(
        segment, "large_unplanned_payment", "large_unplanned_payment",
        "outflow", event_date, expected, actual,
    )
    return out, event

# Zeroes the expected payday inflow and moves it to a later date
def inject_payroll_delay(
    df: pd.DataFrame, segment: str, event_date, delay_days: int
) -> tuple[pd.DataFrame, AnomalyEvent]:
    out = df.copy()
    mask = _row_mask(out, segment, event_date)
    expected = float(out.loc[mask, "inflow"].iloc[0])
    out.loc[mask, "inflow"] = 0.0

    delayed_date = pd.Timestamp(event_date) + pd.Timedelta(days=delay_days)
    delayed_mask = (out["segment"] == segment) & (
        pd.to_datetime(out["date"]) == delayed_date
    )
    if delayed_mask.any():
        out.loc[delayed_mask, "inflow"] = out.loc[delayed_mask, "inflow"] + expected

    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    event = AnomalyEvent(
        segment, "payroll_delay", "payroll_delay", "inflow", event_date, expected, 0.0,
    )
    return out, event
'''
    Simulates a duplicate or fraud-like debit, similar shape to large_unplanned_payment
    but a distinct cause label for evaluation purposes
'''
def inject_duplicate_debit(
    df: pd.DataFrame, segment: str, event_date, magnitude: float
) -> tuple[pd.DataFrame, AnomalyEvent]:
    out = df.copy()
    mask = _row_mask(out, segment, event_date)
    expected = float(out.loc[mask, "outflow"].iloc[0])
    actual = expected + (expected * magnitude)
    out.loc[mask, "outflow"] = actual
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    event = AnomalyEvent(
        segment, "duplicate_or_suspicious_debit", "duplicate_or_suspicious_debit",
        "outflow", event_date, expected, actual,
    )
    return out, event

# Invoice payment arrives late, same mechanism as payroll_delay but different cause label
def inject_client_payment_delay(
    df: pd.DataFrame, segment: str, event_date, delay_days: int
) -> tuple[pd.DataFrame, AnomalyEvent]:
    out = df.copy()
    mask = _row_mask(out, segment, event_date)
    expected = float(out.loc[mask, "inflow"].iloc[0])
    out.loc[mask, "inflow"] = 0.0

    delayed_date = pd.Timestamp(event_date) + pd.Timedelta(days=delay_days)
    delayed_mask = (out["segment"] == segment) & (
        pd.to_datetime(out["date"]) == delayed_date
    )
    if delayed_mask.any():
        out.loc[delayed_mask, "inflow"] = out.loc[delayed_mask, "inflow"] + expected

    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    event = AnomalyEvent(
        segment, "client_payment_delay", "client_payment_delay",
        "inflow", event_date, expected, 0.0,
    )
    return out, event

# Zeroes inflow across a date range, simulating a trading closure
def inject_business_closure(
    df: pd.DataFrame, segment: str, start_date, end_date
) -> tuple[pd.DataFrame, AnomalyEvent]:
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    mask = (
        (out["segment"] == segment)
        & (dates >= pd.Timestamp(start_date))
        & (dates <= pd.Timestamp(end_date))
    )
    expected = float(out.loc[mask, "inflow"].mean()) if mask.any() else 0.0
    out.loc[mask, "inflow"] = 0.0
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    event = AnomalyEvent(
        segment, "business_closure", "business_closure", "inflow",
        start_date, expected, 0.0, end_date=end_date,
    )
    return out, event

# Sets both components to NaN across a range, simulating a genuine data outage
def inject_data_gap(
    df: pd.DataFrame, segment: str, start_date, end_date
) -> tuple[pd.DataFrame, AnomalyEvent]:
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    mask = (
        (out["segment"] == segment)
        & (dates >= pd.Timestamp(start_date))
        & (dates <= pd.Timestamp(end_date))
    )
    out.loc[mask, ["inflow", "outflow", "net"]] = float("nan")
    event = AnomalyEvent(
        segment, "data_gap", "data_gap", "both", start_date, 0.0, 0.0, end_date=end_date,
    )
    return out, event