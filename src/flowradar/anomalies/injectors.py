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