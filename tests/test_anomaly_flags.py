import mlflow
import numpy as np
import pandas as pd
import pytest

from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.registry import load_registered_model, train_and_register_model

TARGETS = ["net", "inflow", "outflow"]
DATES = pd.date_range("2024-01-01", periods=400, freq="D")
TRAIN_DAYS = 270
SPIKE_DAY = 385


def _series(seed=7):
    # two segments with very different scale and noise
    rng = np.random.default_rng(seed)
    t = np.arange(400)
    frames = []
    for segment, level, noise in (("salaried", 1000.0, 15.0), ("freelancer", 3000.0, 150.0)):
        net = level + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, 400)
        frames.append(
            pd.DataFrame(
                {"date": DATES, "segment": segment, "net": net, "inflow": net + 500, "outflow": 500.0}
            )
        )
    return pd.concat(frames, ignore_index=True)


def _with_spike(series):
    out = series.copy()
    mask = (out["segment"] == "freelancer") & (pd.to_datetime(out["date"]) == DATES[SPIKE_DAY])
    out.loc[mask, ["net", "inflow"]] += 3000.0
    return out


@pytest.fixture(scope="module")
def booster(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("anomaly")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    series = _series()
    train = series[pd.to_datetime(series["date"]) <= DATES[TRAIN_DAYS - 1]]
    train_and_register_model(train, num_boost_round=50)
    yield load_registered_model("latest")
    patch.undo()


def _service(booster, history):
    return ForecastService(booster, history, train_end=DATES[TRAIN_DAYS - 1])


def test_injected_spike_is_flagged(booster):
    flags = _service(booster, _with_spike(_series())).anomaly_flags(alpha=0.01, days=30)
    row = flags[(flags["segment"] == "freelancer") & (flags["date"] == DATES[SPIKE_DAY])].iloc[0]
    assert row["status"] == "anomaly"
    assert row["distance"] > 0


def test_few_false_alarms_on_clean_days(booster):
    flags = _service(booster, _with_spike(_series())).anomaly_flags(alpha=0.01, days=30)
    # one true anomaly, and a 1% interval should add well under four false alarms in 59 days
    assert (flags["status"] == "anomaly").sum() <= 4


def test_window_covers_only_the_requested_recent_days(booster):
    flags = _service(booster, _series()).anomaly_flags(alpha=0.01, days=30)
    assert len(flags) == 60
    assert flags["date"].min() == DATES[370]
    assert flags["date"].max() == DATES[399]


def test_too_strict_an_alpha_is_reported_as_not_calibrated(booster):
    flags = _service(booster, _with_spike(_series())).anomaly_flags(alpha=0.001, days=30)
    assert (flags["status"] == "not_calibrated").all()
    assert flags["lower"].isna().all()


def test_missing_actual_is_reported_not_guessed(booster):
    history = _series()
    history.loc[
        (history["segment"] == "salaried") & (pd.to_datetime(history["date"]) == DATES[380]), TARGETS
    ] = np.nan
    flags = _service(booster, history).anomaly_flags(alpha=0.01, days=30)
    row = flags[(flags["segment"] == "salaried") & (flags["date"] == DATES[380])].iloc[0]
    assert row["status"] == "no_actual"


def test_days_below_one_is_rejected(booster):
    with pytest.raises(ValueError):
        _service(booster, _series()).anomaly_flags(days=0)


def test_window_reaching_into_training_data_never_judges_in_sample_days(booster):
    flags = _service(booster, _series()).anomaly_flags(alpha=0.01, days=200)
    assert flags["date"].min() > DATES[TRAIN_DAYS - 1]
    assert len(flags) == 2 * (400 - TRAIN_DAYS)
    assert (flags["status"] == "not_calibrated").all()