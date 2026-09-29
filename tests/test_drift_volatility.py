import pandas as pd
from datetime import date

from flowradar.drift.volatility import apply_volatility_shift
from flowradar.drift.events import DriftEvent


def _sample_df():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "segment": "salaried",
            "inflow": [100.0] * 100,
            "outflow": [50.0] * 100,
        }
    ).assign(net=lambda d: d["inflow"] - d["outflow"])


def test_variance_increases_after_start_date():
    df = _sample_df()
    event = DriftEvent("salaried", "volatility", "outflow", date(2024, 2, 1), 0.4)
    out = apply_volatility_shift(df, event, seed=1)

    before = out[pd.to_datetime(out["date"]) < pd.Timestamp("2024-02-01")]["outflow"]
    after = out[pd.to_datetime(out["date"]) >= pd.Timestamp("2024-02-01")]["outflow"]

    assert before.std() == 0.0  # untouched, still constant
    assert after.std() > 0.0


def test_no_negative_values():
    df = _sample_df()
    event = DriftEvent("salaried", "volatility", "outflow", date(2024, 1, 1), 2.0)
    out = apply_volatility_shift(df, event, seed=1)
    assert (out["outflow"] >= 0).all()


def test_determinism_same_seed():
    df = _sample_df()
    event = DriftEvent("salaried", "volatility", "outflow", date(2024, 1, 10), 0.3)
    a = apply_volatility_shift(df, event, seed=5)
    b = apply_volatility_shift(df, event, seed=5)
    pd.testing.assert_frame_equal(a, b)