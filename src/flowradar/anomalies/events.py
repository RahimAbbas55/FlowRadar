from dataclasses import dataclass
from datetime import date

@dataclass
class AnomalyEvent:
    segment: str
    anomaly_type: str
    cause_label: str
    component: str  # "inflow", "outflow", or "both"
    event_date: date
    expected_value: float
    actual_value: float