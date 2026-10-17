import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from flowradar.serving.drift import drift_status
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from flowradar.dataset_io import load_dataset
from flowradar.serving.champion import ensure_champion, load_forecast_service
from flowradar.serving.forecaster import ForecastService

class _Model(BaseModel):
    # model_version would otherwise clash with pydantic's protected "model_" namespace
    model_config = ConfigDict(protected_namespaces=())

class HealthResponse(_Model):
    status: str
    model_version: int | None
    as_of: str
    train_end_date: str
    horizon: int
    segments: list[str]

class ForecastPoint(_Model):
    segment: str
    date: str
    horizon_days: int
    forecast: float | None
    lower: float | None
    upper: float | None
    interval_calibrated: bool

class ForecastResponse(_Model):
    model_version: int | None
    as_of: str
    horizon: int
    alpha: float
    points: list[ForecastPoint]

class AnomalyPoint(_Model):
    segment: str
    date: str
    actual: float | None
    forecast: float | None
    lower: float | None
    upper: float | None
    status: str
    distance: float | None

class AnomalyResponse(_Model):
    model_version: int | None
    as_of: str
    alpha: float
    days: int
    flagged: int
    points: list[AnomalyPoint]

class SegmentDrift(_Model):
    segment: str
    status: str
    data_drift_assessed: bool
    share_drifted: float | None
    data_drift_detected: bool
    error_assessed: bool
    error_ratio: float | None
    error_degraded: bool

class DriftResponse(_Model):
    as_of: str
    model_version: int | None
    train_end: str
    current_days: int
    reference_days: int
    retrain: bool
    mode: str
    reasons: list[str]
    segments: list[SegmentDrift]

def _clean(value):
    # JSON cannot carry NaN or numpy scalars, so missing values become null
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value

def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]

def _day(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")

def create_app(service: ForecastService) -> FastAPI:
    app = FastAPI(
        title="FlowRadar",
        description="30-day cash flow forecasts with conformal intervals, and recent-day anomaly flags",
        version="0.1.0",
    )

    def check_segment(segment: str | None) -> None:
        if segment is not None and segment not in service.segments:
            raise HTTPException(
                status_code=404, detail=f"unknown segment {segment!r}, expected one of {service.segments}"
            )

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_version=service.model_version,
            as_of=_day(service.last_date),
            train_end_date=_day(service.train_end),
            horizon=service.horizon,
            segments=service.segments,
        )

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(segment: str | None = None, alpha: float = Query(0.1, gt=0, lt=1)):
        check_segment(segment)
        df = service.forecast(alpha=alpha)
        if segment is not None:
            df = df[df["segment"] == segment]
        columns = ["segment", "date", "horizon_days", "forecast", "lower", "upper", "interval_calibrated"]
        return ForecastResponse(
            model_version=service.model_version,
            as_of=_day(service.last_date),
            horizon=service.horizon,
            alpha=alpha,
            points=_records(df[columns]),
        )

    @app.get("/anomalies", response_model=AnomalyResponse)
    def anomalies(
        segment: str | None = None,
        alpha: float = Query(0.01, gt=0, lt=0.5),
        days: int = Query(30, ge=1, le=90),
        only_flagged: bool = False,
    ):
        check_segment(segment)
        flags = service.anomaly_flags(alpha=alpha, days=days)
        if segment is not None:
            flags = flags[flags["segment"] == segment]
        flagged = int((flags["status"] == "anomaly").sum())
        if only_flagged:
            flags = flags[flags["status"] == "anomaly"]
        return AnomalyResponse(
            model_version=service.model_version,
            as_of=_day(service.last_date),
            alpha=alpha,
            days=days,
            flagged=flagged,
            points=_records(flags),
        )
        
    @app.get("/drift", response_model=DriftResponse)
    def drift(
        current_days: int = Query(60, ge=14, le=120),
        reference_days: int = Query(180, ge=60, le=400),
        mode: str = Query("any", pattern="^(any|both)$"),
    ):
        status = drift_status(
            service, current_days=current_days, reference_days=reference_days, mode=mode
        )
        status["as_of"] = _day(status["as_of"])
        status["train_end"] = _day(status["train_end"])
        status["segments"] = _records(pd.DataFrame(status["segments"]))
        return DriftResponse(**status)

    return app

def build_app() -> FastAPI:
    # uvicorn entry point: loads the saved dataset, makes sure a champion exists, and serves it.
    # 180 held-back days leaves enough calibration residuals for a strict anomaly alpha
    data_dir = os.environ.get("FLOWRADAR_DATA_DIR", "data/default")
    calibration_days = int(os.environ.get("FLOWRADAR_CALIBRATION_DAYS", "180"))
    history = load_dataset(data_dir)["series"]
    ensure_champion(history, calibration_days=calibration_days)
    return create_app(load_forecast_service(history))