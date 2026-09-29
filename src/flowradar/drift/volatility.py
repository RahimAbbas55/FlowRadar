import numpy as np
import pandas as pd
from flowradar.drift.events import DriftEvent

# adds extra multiplicative noise to a component from start_date onward,
# simulating increased volatility rather than a change in mean level
def apply_volatility_shift(df: pd.DataFrame, event: DriftEvent, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    mask = (out["segment"] == event.segment) & (dates >= pd.Timestamp(event.start_date))

    n_affected = mask.sum()
    extra_noise = rng.normal(0, event.magnitude, size=n_affected)
    out.loc[mask, event.component] = np.maximum(
        out.loc[mask, event.component] * (1 + extra_noise), 0
    )
    out["net"] = (out["inflow"] - out["outflow"]).round(2)
    return out