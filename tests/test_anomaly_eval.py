import pandas as pd
from flowradar.evaluation.anomaly_eval import evaluate_detector, expand_ground_truth_to_days

def _labels(rows):
    return pd.DataFrame(rows)

def test_expand_point_anomaly_is_single_day():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": None, "anomaly_type": "large_unplanned_payment"}
    ])
    expanded = expand_ground_truth_to_days(labels)
    assert len(expanded) == 1
    assert expanded["date"].iloc[0] == pd.Timestamp("2024-01-05")

def test_expand_span_anomaly_covers_full_range():
    labels = _labels([
        {"segment": "freelancer", "event_date": "2024-01-05", "end_date": "2024-01-08", "anomaly_type": "business_closure"}
    ])
    expanded = expand_ground_truth_to_days(labels)
    assert len(expanded) == 4  # 5th, 6th, 7th, 8th

def test_perfect_detector_scores_precision_and_recall_1():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": None, "anomaly_type": "large_unplanned_payment"},
    ])
    detected = pd.DataFrame(
        {"segment": ["salaried", "salaried"], "date": ["2024-01-05", "2024-01-06"], "is_anomaly": [True, False]}
    )
    result = evaluate_detector(detected, labels)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["true_positives"] == 1

def test_false_positive_reduces_precision_not_recall():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": None, "anomaly_type": "large_unplanned_payment"},
    ])
    detected = pd.DataFrame(
        {"segment": ["salaried", "salaried"], "date": ["2024-01-05", "2024-01-06"], "is_anomaly": [True, True]}
    )
    result = evaluate_detector(detected, labels)
    assert result["recall"] == 1.0
    assert result["precision"] == 0.5

def test_missed_anomaly_reduces_recall_not_precision():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": None, "anomaly_type": "large_unplanned_payment"},
        {"segment": "salaried", "event_date": "2024-01-10", "end_date": None, "anomaly_type": "payroll_delay"},
    ])
    detected = pd.DataFrame(
        {"segment": ["salaried"], "date": ["2024-01-05"], "is_anomaly": [True]}
    )
    result = evaluate_detector(detected, labels)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5

def test_segment_mismatch_does_not_count_as_true_positive():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": None, "anomaly_type": "large_unplanned_payment"},
    ])
    detected = pd.DataFrame(
        {"segment": ["freelancer"], "date": ["2024-01-05"], "is_anomaly": [True]}
    )
    result = evaluate_detector(detected, labels)
    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1