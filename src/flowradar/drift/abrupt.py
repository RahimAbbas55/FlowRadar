import pandas as pd
from flowradar.drift.events import DriftEvent

def apply_abrupt_shift(df: pd.DataFrame , event: DriftEvent) -> pd.DataFrame:
    out = df.copy()
    mask = (out["segment"] == event.segment) & (
        pd.to_datetime(out["date"]) >= pd.Timestamp(event.start_date)
    )
    out.loc[mask, event.component] = out.loc[mask, event.component] * event.magnitude
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    return out