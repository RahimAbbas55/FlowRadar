import pandas as pd
from flowradar.evaluation.anomaly_eval import build_detection_report, evaluate_by_type

def _labels(rows):
    return pd.DataFrame(rows)

def test_by_type_returns_one_row_per_type():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": None, "anomaly_type": "large_unplanned_payment"},
        {"segment": "freelancer", "event_date": "2024-01-10", "end_date": None, "anomaly_type": "client_payment_delay"},
    ])
    detected = pd.DataFrame(
        {"segment": ["salaried", "freelancer"], "date": ["2024-01-05", "2024-01-10"], "is_anomaly": [True, False]}
    )
    result = evaluate_by_type(detected, labels)
    assert set(result["anomaly_type"]) == {"large_unplanned_payment", "client_payment_delay"}
    assert len(result) == 2

def test_by_type_scores_each_type_independently():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": None, "anomaly_type": "large_unplanned_payment"},
        {"segment": "salaried", "event_date": "2024-01-10", "end_date": None, "anomaly_type": "payroll_delay"},
    ])
    # detector catches the first type perfectly, misses the second entirely
    detected = pd.DataFrame(
        {"segment": ["salaried"], "date": ["2024-01-05"], "is_anomaly": [True]}
    )
    result = evaluate_by_type(detected, labels)
    large_row = result[result["anomaly_type"] == "large_unplanned_payment"].iloc[0]
    payroll_row = result[result["anomaly_type"] == "payroll_delay"].iloc[0]
    assert large_row["recall"] == 1.0
    assert payroll_row["recall"] == 0.0

def test_report_contains_overall_and_by_type():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": None, "anomaly_type": "large_unplanned_payment"},
    ])
    detected = pd.DataFrame(
        {"segment": ["salaried"], "date": ["2024-01-05"], "is_anomaly": [True]}
    )
    report = build_detection_report(detected, labels)
    assert "overall" in report
    assert "by_type" in report
    assert report["overall"]["recall"] == 1.0
    assert len(report["by_type"]) == 1

def test_data_gap_type_scored_like_any_other():
    labels = _labels([
        {"segment": "salaried", "event_date": "2024-01-05", "end_date": "2024-01-06", "anomaly_type": "data_gap"},
    ])
    detected = pd.DataFrame(
        {"segment": ["salaried", "salaried"], "date": ["2024-01-05", "2024-01-06"], "is_anomaly": [False, False]}
    )
    result = evaluate_by_type(detected, labels)
    row = result[result["anomaly_type"] == "data_gap"].iloc[0]
    assert row["recall"] == 0.0
    assert row["false_negatives"] == 2