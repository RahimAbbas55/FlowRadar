import numpy as np
import pandas as pd

from flowradar.flows.retrain_decision import retrain_decision_flow

SEGMENTS = ["salaried", "freelancer"]


def _series(shift_segment=None, n_days=400, current_days=60):
    # iid signals per segment, optionally level-shifted over the current window
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    frames = []
    for i, segment in enumerate(SEGMENTS):
        rng = np.random.default_rng(10 + i)
        df = pd.DataFrame(
            {
                "date": dates,
                "segment": segment,
                "inflow": rng.normal(100, 10, n_days),
                "outflow": rng.normal(100, 10, n_days),
                "net": rng.normal(100, 10, n_days),
            }
        )
        if segment == shift_segment:
            df.loc[df.index[-current_days:], ["inflow", "outflow", "net"]] += 100
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _log(degraded_segment=None, n_days=400, current_days=60):
    # constant forecast error, raised over the current window for one segment
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    frames = []
    for segment in SEGMENTS:
        err = np.full(n_days, 5.0)
        if segment == degraded_segment:
            err[-current_days:] = 20.0
        frames.append(
            pd.DataFrame(
                {"date": dates, "segment": segment, "actual": 100.0, "prediction": 100.0 + err}
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_flow_does_not_retrain_on_stable_data():
    result = retrain_decision_flow(_series(), _log())
    assert result["retrain"] is False
    assert result["drifted_segments"] == []
    assert result["degraded_segments"] == []


def test_flow_retrains_when_one_segment_shifts_and_degrades():
    result = retrain_decision_flow(
        _series(shift_segment="freelancer"), _log(degraded_segment="freelancer")
    )
    assert result["retrain"] is True
    assert result["drifted_segments"] == ["freelancer"]
    assert result["degraded_segments"] == ["freelancer"]