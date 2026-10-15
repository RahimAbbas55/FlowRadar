import os
import mlflow
import tempfile
import numpy as np
import pandas as pd
from flowradar.config import DriftConfig, GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.evaluation.anomaly_eval import build_detection_report, expand_ground_truth_to_days
from flowradar.serving.champion import ensure_champion, load_forecast_service
from flowradar.serving.forecaster import ForecastService

def clip_labels_to_window(labels: pd.DataFrame, start, end) -> pd.DataFrame:
    # keeps only the part of each label that falls inside [start, end]. Labels entirely
    # outside the window are dropped, spans that cross an edge are shortened
    out = labels.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    out["end_date"] = pd.to_datetime(out["end_date"]).fillna(out["event_date"])
    out["event_date"] = out["event_date"].clip(lower=pd.Timestamp(start))
    out["end_date"] = out["end_date"].clip(upper=pd.Timestamp(end))
    return out[out["event_date"] <= out["end_date"]].reset_index(drop=True)

def _keys(df: pd.DataFrame) -> set:
    return set(zip(df["segment"], df["date"]))

def evaluate_detection(
    service: ForecastService, anomaly_labels: pd.DataFrame, alpha: float = 0.01, days: int = 90
) -> dict:
    # judges the last `days` days with the service's out-of-sample calibrated interval and
    # scores the flags against the injected labels that fall inside that window
    flags = service.anomaly_flags(alpha=alpha, days=days)
    if flags.empty:
        raise ValueError("no days to judge, the window lies inside the training data")
    start, end = flags["date"].min(), flags["date"].max()

    labels = clip_labels_to_window(anomaly_labels, start, end)
    labels = labels[labels["segment"].isin(service.segments)]
    value_labels = labels[labels["anomaly_type"] != "data_gap"]
    gap_labels = labels[labels["anomaly_type"] == "data_gap"]
    if value_labels.empty:
        raise ValueError("no value-based anomaly labels fall inside the evaluation window")

    detected = flags[["segment", "date"]].copy()
    detected["is_anomaly"] = flags["status"] == "anomaly"
    report = build_detection_report(detected, value_labels)

    truth = expand_ground_truth_to_days(value_labels)
    judged = flags[flags["status"].isin(["normal", "anomaly"])]
    clean_judged = len(_keys(judged) - _keys(truth))
    false_positives = report["overall"]["false_positives"]

    # data gaps are not judged against an interval, they are detected by the missing value itself
    gap_days = _keys(expand_ground_truth_to_days(gap_labels)) if len(gap_labels) else set()
    missing_days = _keys(flags[flags["status"] == "no_actual"])

    return {
        "alpha": alpha,
        "window_start": start,
        "window_end": end,
        "overall": report["overall"],
        "by_type": report["by_type"],
        "clean_days_judged": clean_judged,
        "false_positive_rate": false_positives / clean_judged if clean_judged else float("nan"),
        "status_counts": flags["status"].value_counts().to_dict(),
        "missing_data": {
            "gap_days": len(gap_days),
            "detected": len(gap_days & missing_days),
            "unexplained": len(missing_days - gap_days),
        },
    }

def alpha_sweep(
    service: ForecastService, anomaly_labels: pd.DataFrame, alphas=(0.1, 0.05, 0.01), days: int = 90
) -> pd.DataFrame:
    # the precision/recall trade-off: a smaller alpha is a wider interval and fewer flags
    rows = []
    for alpha in alphas:
        r = evaluate_detection(service, anomaly_labels, alpha=alpha, days=days)
        o = r["overall"]
        rows.append(
            {
                "alpha": alpha,
                "precision": o["precision"],
                "recall": o["recall"],
                "true_positives": o["true_positives"],
                "false_positives": o["false_positives"],
                "false_negatives": o["false_negatives"],
                "flagged": o["true_positives"] + o["false_positives"],
                "false_positive_rate": r["false_positive_rate"],
                "not_calibrated_days": r["status_counts"].get("not_calibrated", 0),
            }
        )
    return pd.DataFrame(rows)

def main() -> None:
    # drift-free 5-year dataset: train on the first 1000 days, calibrate on the next 300,
    # judge the last 525. Drift is switched off so false alarms reflect the detector, not drift
    config = GeneratorConfig(
        n_days=1825,
        drift=DriftConfig(
            abrupt_events_per_segment=0, gradual_events_per_segment=0, volatility_events_per_segment=0
        ),
    )
    dataset = generate_dataset(config)

    # a throwaway registry, so this never touches or reuses another champion
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    mlflow.set_tracking_uri(f"sqlite:///{workdir}/mlflow.db")

    history = dataset["series"]
    ensure_champion(history, calibration_days=825)
    service = load_forecast_service(history)

    labels = dataset["anomaly_labels"]
    print(f"trained through {service.train_end.date()}, judging the last 525 days")
    print(alpha_sweep(service, labels, alphas=(0.1, 0.05, 0.01), days=525).to_string(index=False))

    detail = evaluate_detection(service, labels, alpha=0.05, days=525)
    print("\nby anomaly type at alpha=0.05")
    print(detail["by_type"].to_string(index=False))
    print("\nmissing-data rule:", detail["missing_data"])
    print("status counts:", detail["status_counts"])

if __name__ == "__main__":
    main()