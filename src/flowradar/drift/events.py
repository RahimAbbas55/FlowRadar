from dataclasses import dataclass
from datetime import date

# custom data class for the drift events
@dataclass
class DriftEvent:
    segment : str
    drift_type : str # abrupt | gradual | volatility
    component : str # inflow | outflow
    start_date : date
    magnitude : str # interpretation depends on drift_type