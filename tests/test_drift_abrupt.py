import pandas as pd
from datetime import date
from flowradar.drift.abrupt import apply_abrupt_shift
from flowradar.drift.events import DriftEvent

def _sample_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "segment": "salaried",
            "inflow": [100.0] * 10,
            "outflow": [50.0] * 10,
        }
    ).assign(net=lambda d: d["inflow"] - d["outflow"])


def test_shift_applies_only_after_start_date():
    df = _sample_df()
    event = DriftEvent("salaried", "abrupt", "outflow", date(2024, 1, 6), 1.5)
    out = apply_abrupt_shift(df, event)

    before = out[pd.to_datetime(out["date"]) < pd.Timestamp("2024-01-06")]
    after = out[pd.to_datetime(out["date"]) >= pd.Timestamp("2024-01-06")]

    assert (before["outflow"] == 50.0).all()
    assert (after["outflow"] == 75.0).all()


def test_shift_only_affects_target_segment():
    df = pd.concat(
        [_sample_df(), _sample_df().assign(segment="freelancer")], ignore_index=True
    )
    event = DriftEvent("salaried", "abrupt", "outflow", date(2024, 1, 1), 2.0)
    out = apply_abrupt_shift(df, event)

    salaried_out = out[out["segment"] == "salaried"]["outflow"]
    freelancer_out = out[out["segment"] == "freelancer"]["outflow"]

    assert (salaried_out == 100.0).all()
    assert (freelancer_out == 50.0).all()


def test_net_recomputed_after_shift():
    df = _sample_df()
    event = DriftEvent("salaried", "abrupt", "inflow", date(2024, 1, 3), 2.0)
    out = apply_abrupt_shift(df, event)
    assert (out["net"] == (out["inflow"] - out["outflow"]).round(2)).all()


def test_shift_does_not_mutate_original():
    df = _sample_df()
    event = DriftEvent("salaried", "abrupt", "outflow", date(2024, 1, 5), 1.5)
    apply_abrupt_shift(df, event)
    assert (df["outflow"] == 50.0).all()