import datetime
from flowradar.dashboard.snapshot import ANOMALY_ALPHAS, FORECAST_ALPHAS, REVIEW_WINDOWS, alpha_key
LAST_DAY = datetime.date(2024, 12, 30)

def _forecast(segment, alpha):
    points = [
        {
            "segment": segment, "date": (LAST_DAY + datetime.timedelta(days=i)).isoformat(),
            "horizon_days": i, "forecast": 100.0 + i, "lower": 90.0 + i, "upper": 110.0 + i,
            "interval_calibrated": True,
        }
        for i in range(1, 31)
    ]
    return {"model_version": 1, "as_of": LAST_DAY.isoformat(), "horizon": 30, "alpha": alpha, "points": points}

def _anomalies(segment, alpha, days):
    points = []
    for i in range(days):
        flagged = i == days - 1
        points.append(
            {
                "segment": segment, "date": (LAST_DAY - datetime.timedelta(days=days - 1 - i)).isoformat(),
                "actual": 50.0 if flagged else 5.0, "forecast": 5.0, "lower": 0.0, "upper": 10.0,
                "status": "anomaly" if flagged else "normal", "distance": 40.0 if flagged else 0.0,
            }
        )
    return {
        "model_version": 1, "as_of": LAST_DAY.isoformat(), "alpha": alpha, "days": days,
        "flagged": 1, "points": points,
    }

def full_snapshot(segments=("freelancer", "salaried")) -> dict:
    # a complete snapshot with a stored result for every choice the dashboard offers
    return {
        "meta": {"built_at": "2026-10-21", "generator_version": "0.3.0", "seed": 42},
        "health": {
            "status": "ok", "model_version": 1, "as_of": LAST_DAY.isoformat(),
            "train_end_date": "2024-03-01", "horizon": 30, "segments": list(segments),
        },
        "drift": {
            "as_of": LAST_DAY.isoformat(), "model_version": 1, "train_end": "2024-03-01",
            "current_days": 60, "reference_days": 180, "retrain": False, "mode": "any", "reasons": [],
            "segments": [
                {
                    "segment": s, "status": "ok", "data_drift_assessed": True, "share_drifted": 0.1,
                    "data_drift_detected": False, "error_assessed": True, "error_ratio": 1.0,
                    "error_degraded": False,
                }
                for s in segments
            ],
        },
        "forecast": {s: {alpha_key(a): _forecast(s, a) for a in FORECAST_ALPHAS} for s in segments},
        "anomalies": {
            s: {alpha_key(a): {str(d): _anomalies(s, a, d) for d in REVIEW_WINDOWS} for a in ANOMALY_ALPHAS}
            for s in segments
        },
    }