from dataclasses import dataclass
from datetime import date

@dataclass
class AnomalyEvent:
    # ground truth record for one injected anomaly, including the facts Phase 7 needs later
    segment: str
    anomaly_type: str
    cause_label: str
    component: str  # "inflow", "outflow", or "both"
    event_date: date
    expected_value: float
    actual_value: float
    end_date: date | None = None  # set for span-based anomalies like business_closure
    arrival_date: date | None = None  # set for delays: the day the delayed money arrives