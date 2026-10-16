import pandas as pd

def expand_ground_truth_to_days(anomaly_labels: pd.DataFrame) -> pd.DataFrame:
    # expands span-based anomalies (business_closure, data_gap) into one row per affected
    # day, and adds the arrival day of delayed payments, so every anomalous day is a row
    rows = []
    for _, row in anomaly_labels.iterrows():
        start = pd.Timestamp(row["event_date"])
        end = pd.Timestamp(row["end_date"]) if pd.notna(row["end_date"]) else start
        for d in pd.date_range(start, end, freq="D"):
            rows.append({"segment": row["segment"], "date": d, "anomaly_type": row["anomaly_type"]})

        arrival = row.get("arrival_date")
        if arrival is not None and pd.notna(arrival):
            rows.append(
                {"segment": row["segment"], "date": pd.Timestamp(arrival), "anomaly_type": row["anomaly_type"]}
            )
    return pd.DataFrame(rows)

def evaluate_detector(
    detected: pd.DataFrame, anomaly_labels: pd.DataFrame
) -> dict:
    # detected: columns [segment, date, is_anomaly] — detector's day-level flags
    # anomaly_labels: Phase 1's ground truth anomaly_labels table
    # returns overall precision, recall, and the supporting counts
    ground_truth_days = expand_ground_truth_to_days(anomaly_labels)
    ground_truth_keys = set(
        zip(ground_truth_days["segment"], pd.to_datetime(ground_truth_days["date"]))
    )

    detected = detected.copy()
    detected["date"] = pd.to_datetime(detected["date"])
    flagged = detected[detected["is_anomaly"]]
    flagged_keys = set(zip(flagged["segment"], flagged["date"]))

    true_positives = len(flagged_keys & ground_truth_keys)
    false_positives = len(flagged_keys - ground_truth_keys)
    false_negatives = len(ground_truth_keys - flagged_keys)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else float("nan")
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else float("nan")

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
    }

def evaluate_by_type(detected: pd.DataFrame, anomaly_labels: pd.DataFrame) -> pd.DataFrame:
    # scores precision/recall separately for each anomaly type, since detection
    # difficulty varies a lot by mechanism (a spike vs a missing-data gap)
    rows = []
    for anomaly_type in anomaly_labels["anomaly_type"].unique():
        type_labels = anomaly_labels[anomaly_labels["anomaly_type"] == anomaly_type]
        result = evaluate_detector(detected, type_labels)
        rows.append({"anomaly_type": anomaly_type, **result})
    return pd.DataFrame(rows).sort_values("anomaly_type").reset_index(drop=True)

def build_detection_report(detected: pd.DataFrame, anomaly_labels: pd.DataFrame) -> dict:
    # combines overall and per-type evaluation into one report, the single entry
    # point for Phase 3's results rather than requiring both calls separately
    return {
        "overall": evaluate_detector(detected, anomaly_labels),
        "by_type": evaluate_by_type(detected, anomaly_labels),
    }