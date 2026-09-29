from datetime import timedelta

import numpy as np
import pandas as pd

from flowradar.anomalies.injectors import (
    inject_business_closure,
    inject_client_payment_delay,
    inject_data_gap,
    inject_duplicate_debit,
    inject_large_unplanned_payment,
    inject_payroll_delay,
)
from flowradar.config import GeneratorConfig

# restricts each anomaly type to segments where its cause makes sense; None means any segment
_SEGMENT_ELIGIBILITY = {
    "large_unplanned_payment": None,
    "duplicate_or_suspicious_debit": None,
    "payroll_delay": {"salaried"},
    "client_payment_delay": {"freelancer"},
    "business_closure": {"freelancer", "small_business"},
    "data_gap": None,
}


def _eligible_segments(anomaly_type: str, configured_segments: list[str]) -> list[str]:
    allowed = _SEGMENT_ELIGIBILITY[anomaly_type]
    return configured_segments if allowed is None else [s for s in configured_segments if s in allowed]


def _nearest_inflow_day(df: pd.DataFrame, segment: str, target_date):
    # finds the closest date on or after target_date where this segment has nonzero inflow,
    # since payroll_delay and client_payment_delay both need a real inflow day to delay
    seg_df = df[
        (df["segment"] == segment) & (pd.to_datetime(df["date"]) >= pd.Timestamp(target_date))
    ]
    seg_df = seg_df[seg_df["inflow"] > 0]
    if seg_df.empty:
        return None
    return pd.to_datetime(seg_df["date"].iloc[0]).date()


def schedule_and_apply_anomalies(df: pd.DataFrame, config: GeneratorConfig, seed: int):
    # applies anomalies as they're scheduled, since delay-type anomalies need to read
    # the current series state on the chosen day before overwriting it
    rng = np.random.default_rng(seed)
    out = df.copy()
    events = []

    total_days = config.n_days * len(config.segments)
    target_count = max(1, round(config.anomalies.target_rate * total_days))
    types = config.anomalies.types
    buffer_days = 14

    for i in range(target_count):
        anomaly_type = types[i % len(types)]
        candidates = _eligible_segments(anomaly_type, config.segments)
        if not candidates:
            continue
        segment = str(rng.choice(candidates))

        offset = int(rng.integers(buffer_days, max(config.n_days - buffer_days, buffer_days + 1)))
        event_date = config.start_date + timedelta(days=offset)
        magnitude = float(rng.uniform(config.anomalies.magnitude_min, config.anomalies.magnitude_max))

        if anomaly_type == "large_unplanned_payment":
            out, event = inject_large_unplanned_payment(out, segment, event_date, magnitude)
        elif anomaly_type == "duplicate_or_suspicious_debit":
            out, event = inject_duplicate_debit(out, segment, event_date, magnitude)
        elif anomaly_type == "payroll_delay":
            real_date = _nearest_inflow_day(out, segment, event_date)
            if real_date is None:
                continue
            out, event = inject_payroll_delay(out, segment, real_date, delay_days=int(rng.integers(2, 8)))
        elif anomaly_type == "client_payment_delay":
            real_date = _nearest_inflow_day(out, segment, event_date)
            if real_date is None:
                continue
            out, event = inject_client_payment_delay(out, segment, real_date, delay_days=int(rng.integers(2, 8)))
        elif anomaly_type == "business_closure":
            end_date = event_date + timedelta(days=int(rng.integers(2, 7)))
            out, event = inject_business_closure(out, segment, event_date, end_date)
        elif anomaly_type == "data_gap":
            end_date = event_date + timedelta(days=int(rng.integers(1, 3)))
            out, event = inject_data_gap(out, segment, event_date, end_date)
        else:
            continue

        events.append(event)

    return out, events