from dataclasses import asdict
from datetime import timedelta
import pandas as pd
from flowradar.anomalies.counterfactual import generate_counterfactual
from flowradar.anomalies.scheduler import schedule_and_apply_anomalies
from flowradar.calendar_uk import holiday_set
from flowradar.config import GeneratorConfig
from flowradar.drift.scheduler import apply_drift_events, schedule_drift_events
from flowradar.segments.base_series import generate_base_series
'''
=> Ties every stage together: base series, drift,
=> a counterfactual snapshot, anomalies, plus ground truth labels and a manifest
'''
def generate_dataset(config: GeneratorConfig) -> dict:
    base = generate_base_series(config)

    drift_events = schedule_drift_events(config, seed=config.seed + 1000)
    drifted = apply_drift_events(base, drift_events, seed=config.seed + 2000)

    # captured after drift, before anomalies, so it isolates anomaly effects only
    counterfactual = generate_counterfactual(drifted)

    anomalous, anomaly_events = schedule_and_apply_anomalies(drifted, config, seed=config.seed + 3000)

    end_date = config.start_date + timedelta(days=config.n_days - 1)
    hols = holiday_set(config.start_date.year, end_date.year)
    calendar_df = (
        pd.DataFrame({"date": list(hols.keys()), "holiday_name": list(hols.values())})
        .sort_values("date")
        .reset_index(drop=True)
    )

    drift_df = pd.DataFrame([asdict(e) for e in drift_events])
    anomaly_df = pd.DataFrame([asdict(e) for e in anomaly_events])

    manifest = {
        "seed": config.seed,
        "config_hash": config.config_hash(),
        "generator_version": "0.1.0",
        "n_days": config.n_days,
        "segments": config.segments,
        "n_drift_events": len(drift_events),
        "n_anomaly_events": len(anomaly_events),
    }

    return {
        "series": anomalous,
        "counterfactual": counterfactual,
        "anomaly_labels": anomaly_df,
        "drift_events": drift_df,
        "calendar": calendar_df,
        "manifest": manifest,
    }