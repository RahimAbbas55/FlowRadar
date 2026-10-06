import pandas as pd

'''
    Expands span-based anomalies (business_closure, data_gap) into one row per
    affected day, so point and span anomalies can be matched against detector
    output on equal footing
'''
def expand_ground_truth_to_days(anomaly_labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in anomaly_labels.iterrows():
        start = pd.Timestamp(row["event_date"])
        end = pd.Timestamp(row["end_date"]) if pd.notna(row["end_date"]) else start
        for d in pd.date_range(start, end, freq="D"):
            rows.append({"segment": row["segment"], "date": d, "anomaly_type": row["anomaly_type"]})
    return pd.DataFrame(rows)

'''
    Detected: columns [segment, date, is_anomaly] — detector's day-level flags
    anomaly_labels: Phase 1's ground truth anomaly_labels table
    returns overall precision, recall, and the supporting counts
'''
def evaluate_detector(
    detected: pd.DataFrame, anomaly_labels: pd.DataFrame
) -> dict:
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