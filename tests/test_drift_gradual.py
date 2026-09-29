import pandas as pd
from datetime import date

from flowradar.drift.gradual import apply_gradual_drift
from flowradar.drift.events import DriftEvent


def _sample_df():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "segment": "salaried",
            "inflow": [100.0] * 20,
            "outflow": [50.0] * 20,
        }
    ).assign(net=lambda d: d["inflow"] - d["outflow"])


def test_no_change_before_start_date():
    df = _sample_df()
    event = DriftEvent("salaried", "gradual", "outflow", date(2024, 1, 10), 2.0, duration_days=5)
    out = apply_gradual_drift(df, event)
    before = out[pd.to_datetime(out["date"]) < pd.Timestamp("2024-01-10")]
    assert (before["outflow"] == 50.0).all()


def test_ramp_increases_monotonically():
    df = _sample_df()
    event = DriftEvent("salaried", "gradual", "outflow", date(2024, 1, 5), 3.0, duration_days=5)
    out = apply_gradual_drift(df, event)
    ramp = out[
        (pd.to_datetime(out["date"]) >= pd.Timestamp("2024-01-05"))
        & (pd.to_datetime(out["date"]) <= pd.Timestamp("2024-01-10"))
    ]["outflow"].values
    assert all(ramp[i] <= ramp[i + 1] for i in range(len(ramp) - 1))


def test_holds_at_magnitude_after_ramp_completes():
    df = _sample_df()
    event = DriftEvent("salaried", "gradual", "outflow", date(2024, 1, 1), 2.0, duration_days=5)
    out = apply_gradual_drift(df, event)
    after = out[pd.to_datetime(out["date"]) >= pd.Timestamp("2024-01-06")]
    assert (after["outflow"].round(2) == 100.0).all()


def test_only_affects_target_segment():
    df = pd.concat(
        [_sample_df(), _sample_df().assign(segment="freelancer")], ignore_index=True
    )
    event = DriftEvent("salaried", "gradual", "outflow", date(2024, 1, 1), 2.0, duration_days=5)
    out = apply_gradual_drift(df, event)
    freelancer_out = out[out["segment"] == "freelancer"]["outflow"]
    assert (freelancer_out == 50.0).all()


def test_net_recomputed_after_drift():
    df = _sample_df()
    event = DriftEvent("salaried", "gradual", "inflow", date(2024, 1, 3), 1.5, duration_days=4)
    out = apply_gradual_drift(df, event)
    assert (out["net"] == (out["inflow"] - out["outflow"]).round(2)).all()