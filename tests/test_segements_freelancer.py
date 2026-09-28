import pandas as pd
from flowradar.segments.freelancer import generate_freelancer

def test_output_shape_and_columns():
    df = generate_freelancer(start_date="2024-01-01", n_days=365, seed=1)
    assert len(df) == 365
    assert list(df.columns) == ["date", "segment", "inflow", "outflow"]
    assert (df["segment"] == "freelancer").all()

def test_determinism_same_seed():
    a = generate_freelancer(start_date="2024-01-01", n_days=100, seed=42)
    b = generate_freelancer(start_date="2024-01-01", n_days=100, seed=42)
    pd.testing.assert_frame_equal(a, b)

def test_different_seed_differs():
    a = generate_freelancer(start_date="2024-01-01", n_days=100, seed=1)
    b = generate_freelancer(start_date="2024-01-01", n_days=100, seed=2)
    assert not a["inflow"].equals(b["inflow"])

# most days should have no invoice landing, unlike salaried's steady spend
def test_inflow_is_lumpy_not_daily():
    df = generate_freelancer(start_date="2024-01-01", n_days=365, seed=1)
    zero_inflow_days = (df["inflow"] == 0).sum()
    assert zero_inflow_days > 250

def test_tax_dates_produce_outflow_spike():
    df = generate_freelancer(start_date="2024-01-01", n_days=365, seed=1)
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    from datetime import date

    jan31 = df.loc[df["date_only"] == date(2024, 1, 31), "outflow"].iloc[0]
    typical = df["outflow"].median()
    assert jan31 > typical * 3

def test_no_negative_values():
    df = generate_freelancer(start_date="2024-01-01", n_days=365, seed=1)
    assert (df["inflow"] >= 0).all()
    assert (df["outflow"] >= 0).all()