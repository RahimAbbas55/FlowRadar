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


def _nearest_inflow_day(df: pd.DataFrame, segment: str, target_date, blocked=frozenset()):
    # finds the closest date on or after target_date where this segment has nonzero inflow
    # and no earlier anomaly has touched it, since a delay or closure must remove money
    # that was originally due, not money an earlier delay moved there
    seg_df = df[
        (df["segment"] == segment) & (pd.to_datetime(df["date"]) >= pd.Timestamp(target_date))
    ]
    seg_df = seg_df[seg_df["inflow"] > 0]
    if blocked:
        seg_df = seg_df[~pd.to_datetime(seg_df["date"]).dt.date.isin(blocked)]
    if seg_df.empty:
        return None
    return pd.to_datetime(seg_df["date"].iloc[0]).date()


def _span(start, end) -> set:
    return {start + timedelta(days=i) for i in range((end - start).days + 1)}


def schedule_and_apply_anomalies(df: pd.DataFrame, config: GeneratorConfig, seed: int):
    # applies anomalies as they're scheduled, since delay-type anomalies need to read
    # the current series state on the chosen day before overwriting it. Every day an event
    # touches is recorded, and no later event may start on, land on or overlap one
    rng = np.random.default_rng(seed)
    out = df.copy()
    events = []
    touched = {segment: set() for segment in config.segments}

    total_days = config.n_days * len(config.segments)
    target_count = max(1, round(config.anomalies.target_rate * total_days))
    types = config.anomalies.types
    buffer_days = 14
    last_date = pd.to_datetime(out["date"]).max().date()

    for i in range(target_count):
        anomaly_type = types[i % len(types)]
        candidates = _eligible_segments(anomaly_type, config.segments)
        if not candidates:
            continue
        segment = str(rng.choice(candidates))

        offset = int(rng.integers(buffer_days, max(config.n_days - buffer_days, buffer_days + 1)))
        event_date = config.start_date + timedelta(days=offset)
        magnitude = float(rng.uniform(config.anomalies.magnitude_min, config.anomalies.magnitude_max))

        if anomaly_type in ("large_unplanned_payment", "duplicate_or_suspicious_debit"):
            footprint = {event_date}
            if footprint & touched[segment]:
                continue
            inject = (
                inject_large_unplanned_payment
                if anomaly_type == "large_unplanned_payment"
                else inject_duplicate_debit
            )
            out, event = inject(out, segment, event_date, magnitude)

        elif anomaly_type in ("payroll_delay", "client_payment_delay"):
            real_date = _nearest_inflow_day(out, segment, event_date, blocked=touched[segment])
            if real_date is None:
                continue
            delay_days = int(rng.integers(2, 8))
            footprint = {real_date, real_date + timedelta(days=delay_days)}
            if footprint & touched[segment]:
                continue
            inject = inject_payroll_delay if anomaly_type == "payroll_delay" else inject_client_payment_delay
            out, event = inject(out, segment, real_date, delay_days=delay_days)

        elif anomaly_type == "business_closure":
            # starts on a day that has inflow, so the closure always removes something
            real_date = _nearest_inflow_day(out, segment, event_date, blocked=touched[segment])
            if real_date is None:
                continue
            end_date = min(real_date + timedelta(days=int(rng.integers(2, 7))), last_date)
            footprint = _span(real_date, end_date)
            if footprint & touched[segment]:
                continue
            out, event = inject_business_closure(out, segment, real_date, end_date)

        elif anomaly_type == "data_gap":
            end_date = event_date + timedelta(days=int(rng.integers(1, 3)))
            footprint = _span(event_date, end_date)
            if footprint & touched[segment]:
                continue
            out, event = inject_data_gap(out, segment, event_date, end_date)

        else:
            continue

        # an anomaly that changes nothing must not be labelled. Data gaps are exempt because
        # their effect is the missing value itself
        if anomaly_type != "data_gap" and (
            pd.isna(event.expected_value) or event.expected_value == event.actual_value
        ):
            continue

        touched[segment] |= footprint
        events.append(event)

    return out, events