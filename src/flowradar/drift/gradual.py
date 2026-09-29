import pandas as pd
import numpy as np
from flowradar.drift.events import DriftEvent

# ramps a component's multiplier linearly from 1.0 to magnitude over duration_days,
# then holds at magnitude for all dates after the ramp completes
def apply_gradual_drift(df: pd.DataFrame , event: DriftEvent) -> pd.DataFrame:
    out = df.copy()
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    start = pd.Timestamp(event.start_date)
    end = start + pd.Timedelta(days=event.duration_days)

    days_since_start = (dates - start).dt.days.clip(lower=0)
    progress = np.clip(days_since_start / max(event.duration_days, 1), 0.0, 1.0)
    multiplier = 1.0 + (event.magnitude - 1.0) * progress

    mask = (out["segment"] == event.segment) & (dates >= start)
    out.loc[mask, event.component] = out.loc[mask, event.component] * multiplier[mask]
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    return out