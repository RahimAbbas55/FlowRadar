import pandas as pd
from flowradar.evaluation.anomaly_detection import flag_anomalies, flag_anomalies_with_distance

def test_flags_values_outside_interval():
    actual = pd.Series([5, 15, 25, 8])
    lower = pd.Series([0, 0, 0, 0])
    upper = pd.Series([10, 10, 10, 10])
    flags = flag_anomalies(actual, lower, upper)
    assert list(flags) == [False, True, True, False]

def test_flags_values_below_interval():
    actual = pd.Series([-5, 5])
    lower = pd.Series([0, 0])
    upper = pd.Series([10, 10])
    flags = flag_anomalies(actual, lower, upper)
    assert list(flags) == [True, False]

def test_distance_zero_when_inside():
    actual = pd.Series([5.0])
    lower = pd.Series([0.0])
    upper = pd.Series([10.0])
    out = flag_anomalies_with_distance(actual, lower, upper)
    assert out["distance"].iloc[0] == 0.0
    assert out["is_anomaly"].iloc[0] == False

def test_distance_positive_above_upper():
    actual = pd.Series([15.0])
    lower = pd.Series([0.0])
    upper = pd.Series([10.0])
    out = flag_anomalies_with_distance(actual, lower, upper)
    assert out["distance"].iloc[0] == 5.0
    assert out["is_anomaly"].iloc[0] == True

def test_distance_negative_below_lower():
    actual = pd.Series([-5.0])
    lower = pd.Series([0.0])
    upper = pd.Series([10.0])
    out = flag_anomalies_with_distance(actual, lower, upper)
    assert out["distance"].iloc[0] == -5.0
    assert out["is_anomaly"].iloc[0] == True

# exactly on the boundary counts as inside, consistent with interval_coverage's >= / <=
def test_boundary_values_are_not_flagged():
    actual = pd.Series([0.0, 10.0])
    lower = pd.Series([0.0, 0.0])
    upper = pd.Series([10.0, 10.0])
    flags = flag_anomalies(actual, lower, upper)
    assert list(flags) == [False, False]