import pytest
import numpy as np
import pandas as pd
from datetime import date
from flowradar.features.calendar_features import add_event_features
from flowradar.features.horizon_features import add_horizon_lag_features, build_horizon_features
TARGETS = ["net", "inflow", "outflow"]

def _days(start="2024-01-01", end="2024-12-31"):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates, "segment": "salaried", "net": 0.0, "inflow": 0.0, "outflow": 0.0})

def _ramp(n=100):
    # every target equals its row number, so a lagged value identifies the row it came from
    values = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "segment": "salaried",
            "net": values,
            "inflow": values,
            "outflow": values,
        }
    )

def _flagged(out, col):
    return set(pd.to_datetime(out.loc[out[col] == 1, "date"]).dt.date)

def test_payday_flag_follows_the_working_day_rule():
    flagged = _flagged(add_event_features(_days()), "is_payday")
    assert len(flagged) == 12
    assert date(2024, 8, 30) in flagged and date(2024, 8, 31) not in flagged  # 31 Aug is a Saturday
    assert date(2024, 11, 29) in flagged  # 30 Nov is a Saturday
    assert date(2024, 3, 28) in flagged  # 31 Mar is a Sunday and 29 Mar is Good Friday
    assert date(2024, 12, 31) in flagged

def test_tax_dates_are_flagged():
    flagged = _flagged(add_event_features(_days()), "is_tax_date")
    assert flagged == {date(2024, 1, 31), date(2024, 7, 31)}

def test_vat_dates_are_flagged():
    flagged = _flagged(add_event_features(_days()), "is_vat_date")
    assert flagged == {date(2024, 2, 7), date(2024, 5, 7), date(2024, 8, 7), date(2024, 11, 7)}

def test_event_flags_are_added_by_the_feature_pipeline():
    out = build_horizon_features(_days())
    for col in ("is_payday", "is_tax_date", "is_vat_date"):
        assert col in out.columns

def test_event_flags_exist_for_future_days_with_no_data_yet():
    df = _days()
    df.loc[df.index[-30:], TARGETS] = np.nan
    out = build_horizon_features(df, horizon=30)
    assert out.iloc[-1]["is_payday"] == 1  # 31 Dec is a payday, and it has no actuals yet

def test_a_short_gap_is_bridged_with_the_last_known_value():
    df = _ramp()
    df.loc[50:51, TARGETS] = np.nan
    out = add_horizon_lag_features(df, horizon=30)
    assert out.iloc[80]["net_lag_30"] == 49.0  # row 50 is bridged with row 49
    assert out.iloc[81]["net_lag_30"] == 49.0
    assert pd.isna(out.iloc[50]["net"])  # the label itself stays missing

def test_a_gap_longer_than_the_limit_stays_missing():
    df = _ramp()
    df.loc[50:54, TARGETS] = np.nan  # five days, and the limit is three
    out = add_horizon_lag_features(df, horizon=30)
    assert out.iloc[80]["net_lag_30"] == 49.0
    assert out.iloc[82]["net_lag_30"] == 49.0
    assert pd.isna(out.iloc[83]["net_lag_30"])
    assert pd.isna(out.iloc[84]["net_lag_30"])

def test_gap_filling_never_looks_forward():
    # row 100's features may only use rows up to 70, even when the gap sits right at row 70
    df = _ramp(n=120)
    df.loc[69:70, TARGETS] = np.nan
    changed = df.copy()
    changed.loc[71:100, TARGETS] = np.nan

    a = add_horizon_lag_features(df, horizon=30)
    b = add_horizon_lag_features(changed, horizon=30)
    cols = [c for c in a.columns if "_lag_" in c or "_roll_" in c]
    assert np.allclose(a.loc[100, cols].astype(float), b.loc[100, cols].astype(float), equal_nan=True)

def test_rolling_mean_is_defined_across_a_short_gap():
    df = _ramp()
    df.loc[40:41, TARGETS] = np.nan
    out = add_horizon_lag_features(df, horizon=30)
    # the window for row 75 covers rows 39 to 45, and the gap rows count as 39
    assert out.iloc[75]["net_roll_mean_7"] == pytest.approx((39 + 39 + 39 + 42 + 43 + 44 + 45) / 7)

def test_a_max_gap_of_zero_turns_the_fill_off():
    df = _ramp()
    df.loc[50:51, TARGETS] = np.nan
    out = add_horizon_lag_features(df, horizon=30, max_gap=0)
    assert pd.isna(out.iloc[80]["net_lag_30"])