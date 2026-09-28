import pandas as pd
from flowradar.segments.small_business import generate_small_business

def test_output_shape_and_columns():
    df = generate_small_business(start_date="2024-01-01", n_days=365, seed=1)
    assert len(df) == 365
    assert list(df.columns) == ["date", "segment", "inflow", "outflow"]
    assert (df["segment"] == "small_business").all()

def test_determinism_same_seed():
    a = generate_small_business(start_date="2024-01-01", n_days=100, seed=42)
    b = generate_small_business(start_date="2024-01-01", n_days=100, seed=42)
    pd.testing.assert_frame_equal(a, b)

def test_bank_holiday_sales_near_zero():
    df = generate_small_business(start_date="2024-01-01", n_days=365, seed=1)
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    from datetime import date

    xmas_sales = df.loc[df["date_only"] == date(2024, 12, 25), "inflow"].iloc[0]
    typical_sales = df["inflow"].median()
    assert xmas_sales < typical_sales * 0.2

def test_monday_supplier_payment_present():
    df = generate_small_business(start_date="2024-01-01", n_days=365, seed=1)
    df["weekday"] = pd.to_datetime(df["date"]).dt.weekday
    mon_outflow = df.loc[df["weekday"] == 0, "outflow"].mean()
    other_outflow = df.loc[df["weekday"] != 0, "outflow"].mean()
    assert mon_outflow > other_outflow

def test_no_negative_values():
    df = generate_small_business(start_date="2024-01-01", n_days=365, seed=1)
    assert (df["inflow"] >= 0).all()
    assert (df["outflow"] >= 0).all()