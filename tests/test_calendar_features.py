import pandas as pd
from flowradar.features.calendar_features import add_calendar_features

def test_weekend_flagged_correctly():
    dates = pd.date_range("2024-06-01", periods=7, freq="D")  # Sat through Fri
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": 0.0})
    out = add_calendar_features(df)
    assert out.loc[out["date"] == "2024-06-01", "is_weekend"].iloc[0] == 1  # Saturday
    assert out.loc[out["date"] == "2024-06-03", "is_weekend"].iloc[0] == 0  # Monday

def test_christmas_flagged_as_holiday():
    dates = pd.date_range("2024-12-20", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": 0.0})
    out = add_calendar_features(df)
    assert out.loc[out["date"] == "2024-12-25", "is_holiday"].iloc[0] == 1

def test_holiday_is_not_working_day():
    dates = pd.date_range("2024-12-20", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": 0.0})
    out = add_calendar_features(df)
    assert out.loc[out["date"] == "2024-12-25", "is_working_day"].iloc[0] == 0