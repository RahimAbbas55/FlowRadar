import mlflow
import pytest
import numpy as np
import pandas as pd
from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.registry import load_registered_model, train_and_register_model
DATES = pd.date_range("2024-01-01", periods=400, freq="D")  # day 0 is a Monday
TRAIN_DAYS = 270
QUIET_DAY = 386  # a Tuesday inside the last 30 days

def _series(seed=11):
    # noise proportional to the level: Mondays carry 20000, every other day 500, both with 8% noise
    rng = np.random.default_rng(seed)
    level = np.where(DATES.weekday == 0, 20000.0, 500.0)
    net = level * (1 + 0.08 * rng.normal(size=400))
    return pd.DataFrame(
        {"date": DATES, "segment": "salaried", "net": net, "inflow": net + 500, "outflow": 500.0}
    )

def _with_spike(series):
    out = series.copy()
    mask = pd.to_datetime(out["date"]) == DATES[QUIET_DAY]
    out.loc[mask, ["net", "inflow"]] += 300.0
    return out

@pytest.fixture(scope="module")
def booster(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scaled")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    series = _series()
    train_and_register_model(
        series[pd.to_datetime(series["date"]) <= DATES[TRAIN_DAYS - 1]], num_boost_round=50
    )
    yield load_registered_model("latest")
    patch.undo()

def _service(booster, history, scale, **kwargs):
    return ForecastService(
        booster, history, train_end=DATES[TRAIN_DAYS - 1], interval_scale=scale, **kwargs
    )

def _status(flags, day):
    return flags.loc[flags["date"] == DATES[day], "status"].iloc[0]

def test_unknown_scale_is_rejected(booster):
    with pytest.raises(ValueError):
        _service(booster, _series(), "sometimes")

def test_constant_mode_gives_every_day_the_same_width(booster):
    fc = _service(booster, _series(), "constant").forecast(alpha=0.1)
    assert (fc["upper"] - fc["lower"]).round(6).nunique() == 1

def test_relative_mode_gives_busy_days_wider_intervals(booster):
    fc = _service(booster, _series(), "relative").forecast(alpha=0.1)
    half = (fc["upper"] - fc["lower"]) / 2
    monday = fc["date"].dt.weekday == 0
    assert half[monday].mean() > 3 * half[~monday].mean()

def test_a_modest_spike_on_a_quiet_day_is_flagged_only_in_relative_mode(booster):
    history = _with_spike(_series())
    constant = _service(booster, history, "constant").anomaly_flags(alpha=0.05, days=30)
    relative = _service(booster, history, "relative").anomaly_flags(alpha=0.05, days=30)
    assert _status(constant, QUIET_DAY) == "normal"
    assert _status(relative, QUIET_DAY) == "anomaly"

def test_relative_mode_keeps_false_alarms_near_the_nominal_rate(booster):
    flags = _service(booster, _series(), "relative").anomaly_flags(alpha=0.1, days=30)
    assert (flags["status"] == "anomaly").mean() <= 0.3  # nominal is 10%

def test_relative_mode_needs_the_same_calibration_as_constant_mode(booster):
    service = _service(booster, _series(), "relative", min_calibration=500)
    fc = service.forecast()
    assert not fc["interval_calibrated"].any()
    assert fc["lower"].isna().all()
    assert (service.anomaly_flags(alpha=0.05, days=30)["status"] == "not_calibrated").all()

def test_with_interval_scale_leaves_the_original_unchanged(booster):
    service = _service(booster, _series(), "constant")
    other = service.with_interval_scale("relative")
    assert service.interval_scale == "constant"
    assert other.interval_scale == "relative"
    assert other.booster is service.booster