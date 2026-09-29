from datetime import timedelta
import numpy as np
import pandas as pd
from flowradar.config import GeneratorConfig
from flowradar.drift.abrupt import apply_abrupt_shift
from flowradar.drift.events import DriftEvent
from flowradar.drift.gradual import apply_gradual_drift
from flowradar.drift.volatility import apply_volatility_shift

# maps drift type to its application function
_APPLICATORS = {
    "abrupt": apply_abrupt_shift,
    "gradual": apply_gradual_drift,
    "volatility": apply_volatility_shift,
}


def schedule_drift_events(config: GeneratorConfig, seed: int) -> list[DriftEvent]:
    rng = np.random.default_rng(seed)
    events = []
    post_baseline_days = config.n_days - config.drift.clean_baseline_days

    for segment in config.segments:
        counts = {
            "abrupt": config.drift.abrupt_events_per_segment,
            "gradual": config.drift.gradual_events_per_segment,
            "volatility": config.drift.volatility_events_per_segment,
        }
        for drift_type, count in counts.items():
            for _ in range(count):
                offset_days = int(rng.integers(0, max(post_baseline_days, 1)))
                start = config.start_date + timedelta(
                    days=config.drift.clean_baseline_days + offset_days
                )
                component = rng.choice(["inflow", "outflow"])
                magnitude = float(rng.choice([0.6, 0.75, 1.3, 1.5, 1.8]))
                duration = int(rng.integers(14, 60)) if drift_type == "gradual" else 0
                events.append(
                    DriftEvent(segment, drift_type, component, start, magnitude, duration)
                )
    return events

# applies every scheduled drift event in sequence, so later events compound on earlier ones
def apply_drift_events(
    df: pd.DataFrame, events: list[DriftEvent], seed: int
) -> pd.DataFrame:
    out = df
    for i, event in enumerate(events):
        applicator = _APPLICATORS[event.drift_type]
        if event.drift_type == "volatility":
            out = applicator(out, event, seed=seed + i)
        else:
            out = applicator(out, event)
    return out