import pandas as pd

from flowradar.segments.salaried import generate_salaried


def test_output_shape_and_columns():
    df = generate_salaried(start_date="2024-01-01", n_days=365, seed=1)
    assert len(df) == 365
    assert list(df.columns) == ["date", "segment", "inflow", "outflow"]
    assert (df["segment"] == "salaried").all()


def test_determinism_same_seed():
    a = generate_salaried(start_date="2024-01-01", n_days=100, seed=42)
    b = generate_salaried(start_date="2024-01-01", n_days=100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = generate_salaried(start_date="2024-01-01", n_days=100, seed=1)
    b = generate_salaried(start_date="2024-01-01", n_days=100, seed=2)
    assert not a["outflow"].equals(b["outflow"])


def test_payday_produces_inflow_spike():
    df = generate_salaried(start_date="2024-01-01", n_days=365, seed=1)
    nonzero_inflow_days = df.loc[df["inflow"] > 0]
    # roughly one payday per month across a year
    assert 11 <= len(nonzero_inflow_days) <= 13
    # payday inflow should dwarf typical daily outflow
    assert nonzero_inflow_days["inflow"].min() > df["outflow"].max()


def test_no_negative_values():
    df = generate_salaried(start_date="2024-01-01", n_days=365, seed=1)
    assert (df["inflow"] >= 0).all()
    assert (df["outflow"] >= 0).all()


def test_december_outflow_higher_than_june():
    df = generate_salaried(start_date="2024-01-01", n_days=365, seed=1)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    dec_avg = df.loc[df["month"] == 12, "outflow"].mean()
    jun_avg = df.loc[df["month"] == 6, "outflow"].mean()
    assert dec_avg > jun_avg