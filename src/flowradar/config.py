import hashlib
import json
from datetime import date
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, model_validator

# drift injection settings
class DriftConfig(BaseModel):
    clean_baseline_days: int = 183
    abrupt_events_per_segment: int = 1
    gradual_events_per_segment: int = 1
    volatility_events_per_segment: int = 1

# anomaly injection settings
class AnomalyConfig(BaseModel):
    target_rate: float = Field(0.015, ge=0.0, le=0.1)
    magnitude_min: float = 1.5
    magnitude_max: float = 6.0
    types: list[str] = [
        "large_unplanned_payment",
        "payroll_delay",
        "duplicate_or_suspicious_debit",
        "client_payment_delay",
        "business_closure",
        "data_gap",
    ]

# top level generator config
class GeneratorConfig(BaseModel):
    seed: int = 42
    start_date: date = date(2022, 1, 1)
    n_days: int = Field(1095, ge=365)
    segments: list[str] = ["salaried", "freelancer", "small_business"]
    noise_scale: float = Field(0.08, ge=0.0)
    drift: DriftConfig = DriftConfig()
    anomalies: AnomalyConfig = AnomalyConfig()

    @model_validator(mode="after")
    def _check(self):
        if self.drift.clean_baseline_days >= self.n_days:
            raise ValueError("clean_baseline_days must be less than n_days")
        if self.anomalies.magnitude_min >= self.anomalies.magnitude_max:
            raise ValueError("magnitude_min must be below magnitude_max")
        return self

    def config_hash(self) -> str:
        # stable hash for the manifest
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_config(path: str | Path) -> GeneratorConfig:
    with open(path) as f:
        return GeneratorConfig(**yaml.safe_load(f))