import pandas as pd
from flowradar.anomalies.events import AnomalyEvent

def _row_mask(df: pd.DataFrame, segment: str, event_date) -> pd.Series:
    # locates the single row for a segment and date
    return (df["segment"] == segment) & (
        pd.to_datetime(df["date"]) == pd.Timestamp(event_date)
    )

def _typical_outflow(df: pd.DataFrame, segment: str) -> float:
    # median of the segment's non-zero daily outflows: the size of an ordinary payment
    values = df.loc[df["segment"] == segment, "outflow"]
    values = values[values > 0]
    return float(values.median()) if len(values) else 0.0

def _delay_inflow(df: pd.DataFrame, segment: str, event_date, delay_days: int):
    # removes the inflow due on event_date and adds it to the day delay_days later. Returns the
    # new frame, the amount moved, and the arrival date (None if it falls after the data ends)
    out = df.copy()
    mask = _row_mask(out, segment, event_date)
    expected = float(out.loc[mask, "inflow"].iloc[0])
    out.loc[mask, "inflow"] = 0.0

    delayed_date = pd.Timestamp(event_date) + pd.Timedelta(days=delay_days)
    delayed_mask = (out["segment"] == segment) & (
        pd.to_datetime(out["date"]) == delayed_date
    )
    arrival = None
    if delayed_mask.any():
        out.loc[delayed_mask, "inflow"] = out.loc[delayed_mask, "inflow"] + expected
        arrival = delayed_date.date()

    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    return out, expected, arrival

def inject_large_unplanned_payment(
    df: pd.DataFrame, segment: str, event_date, magnitude: float
) -> tuple[pd.DataFrame, AnomalyEvent]:
    # one-off large outflow spike on a single day
    out = df.copy()
    mask = _row_mask(out, segment, event_date)
    expected = float(out.loc[mask, "outflow"].iloc[0])
    if pd.isna(expected):
        # a data-gap day stays a gap, and the scheduler drops the event
        event = AnomalyEvent(
            segment, "large_unplanned_payment", "large_unplanned_payment",
            "outflow", event_date, expected, expected,
        )
        return out, event

    # on a day with nothing due, "large" means a multiple of the segment's usual payment,
    # because a multiple of zero would change nothing
    base = expected if expected > 0 else _typical_outflow(out, segment)
    actual = base * magnitude
    out.loc[mask, "outflow"] = actual
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    event = AnomalyEvent(
        segment, "large_unplanned_payment", "large_unplanned_payment",
        "outflow", event_date, expected, actual,
    )
    return out, event

def inject_payroll_delay(
    df: pd.DataFrame, segment: str, event_date, delay_days: int
) -> tuple[pd.DataFrame, AnomalyEvent]:
    # zeroes the expected payday inflow and moves it to a later date
    out, expected, arrival = _delay_inflow(df, segment, event_date, delay_days)
    event = AnomalyEvent(
        segment, "payroll_delay", "payroll_delay", "inflow", event_date, expected, 0.0,
        arrival_date=arrival,
    )
    return out, event

def inject_duplicate_debit(
    df: pd.DataFrame, segment: str, event_date, magnitude: float
) -> tuple[pd.DataFrame, AnomalyEvent]:
    # simulates a duplicate or fraud-like debit, similar shape to large_unplanned_payment
    # but a distinct cause label for evaluation purposes
    out = df.copy()
    mask = _row_mask(out, segment, event_date)
    expected = float(out.loc[mask, "outflow"].iloc[0])
    if pd.isna(expected):
        event = AnomalyEvent(
            segment, "duplicate_or_suspicious_debit", "duplicate_or_suspicious_debit",
            "outflow", event_date, expected, expected,
        )
        return out, event

    # a duplicate of a payment that was due is a multiple of that payment. With nothing due,
    # it is a multiple of the segment's usual payment
    base = expected if expected > 0 else _typical_outflow(out, segment)
    actual = expected + base * magnitude
    out.loc[mask, "outflow"] = actual
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    event = AnomalyEvent(
        segment, "duplicate_or_suspicious_debit", "duplicate_or_suspicious_debit",
        "outflow", event_date, expected, actual,
    )
    return out, event

def inject_client_payment_delay(
    df: pd.DataFrame, segment: str, event_date, delay_days: int
) -> tuple[pd.DataFrame, AnomalyEvent]:
    # invoice payment arrives late, same mechanism as payroll_delay but different cause label
    out, expected, arrival = _delay_inflow(df, segment, event_date, delay_days)
    event = AnomalyEvent(
        segment, "client_payment_delay", "client_payment_delay",
        "inflow", event_date, expected, 0.0, arrival_date=arrival,
    )
    return out, event

def inject_business_closure(
    df: pd.DataFrame, segment: str, start_date, end_date
) -> tuple[pd.DataFrame, AnomalyEvent]:
    # zeroes inflow across a date range, simulating a trading closure. Days that are
    # already missing (data gaps) stay missing instead of becoming zero
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    mask = (
        (out["segment"] == segment)
        & (dates >= pd.Timestamp(start_date))
        & (dates <= pd.Timestamp(end_date))
    )
    expected = float(out.loc[mask, "inflow"].mean()) if mask.any() else 0.0
    out.loc[mask & out["inflow"].notna(), "inflow"] = 0.0
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    event = AnomalyEvent(
        segment, "business_closure", "business_closure", "inflow",
        start_date, expected, 0.0, end_date=end_date,
    )
    return out, event

def inject_data_gap(
    df: pd.DataFrame, segment: str, start_date, end_date
) -> tuple[pd.DataFrame, AnomalyEvent]:
    # sets both components to NaN across a range, simulating a genuine data outage
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