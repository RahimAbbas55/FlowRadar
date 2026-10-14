import mlflow
import numpy as np
import pandas as pd
import pytest

from flowradar.evaluation.holdout import predict_with_booster
from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.registry import load_registered_model, train_and_register_model

TARGETS = ["net", "inflow", "outflow"]
DATES = pd.date_range("2024-01-01", periods=400, freq="D")
TRAIN_DAYS = 270
HISTORY_DAYS = 370


def _series(seed=7):
    # two segments with very different scale and noise, so per-segment intervals matter
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


def _rows(series, start, stop):
    # rows for day indices start..stop-1 (stop is exclusive and may equal len(DATES))
    d = pd.to_datetime(series["date"])
    return series[(d >= DATES[start]) & (d <= DATES[stop - 1])].reset_index(drop=True)


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")


@pytest.fixture
def setup(mlflow_store):
    # booster trained on the first TRAIN_DAYS only, history known through HISTORY_DAYS
    series = _series()
    train_and_register_model(_rows(series, 0, TRAIN_DAYS), num_boost_round=50)
    return series, load_registered_model("latest"), _rows(series, 0, HISTORY_DAYS)


def _service(booster, history, **kwargs):
    return ForecastService(booster, history, train_end=DATES[TRAIN_DAYS - 1], **kwargs)


def test_residuals_cover_only_days_after_the_training_cutoff(setup):
    _, booster, history = setup
    res = _service(booster, history).residuals
    assert res["date"].min() > DATES[TRAIN_DAYS - 1]
    assert res["residual"].notna().all()
    assert (res.groupby("segment").size() == HISTORY_DAYS - TRAIN_DAYS).all()


def test_forecast_covers_the_horizon_for_every_segment(setup):
    _, booster, history = setup
    fc = _service(booster, history).forecast()
    assert (fc.groupby("segment").size() == 30).all()
    assert fc["date"].min() == DATES[HISTORY_DAYS]
    assert sorted(fc["horizon_days"].unique()) == list(range(1, 31))
    assert fc["forecast"].notna().all()
    assert ((fc["lower"] < fc["forecast"]) & (fc["forecast"] < fc["upper"])).all()


def test_future_forecast_matches_the_backtest_prediction_for_the_same_dates(setup):
    series, booster, history = setup
    fc = _service(booster, history).forecast()
    backtest = predict_with_booster(booster, series)  # includes the true last 30 days
    merged = fc.merge(backtest, on=["segment", "date"])
    assert len(merged) == 60
    assert np.allclose(merged["forecast"], merged["prediction"])


def test_changing_the_last_known_day_changes_only_the_far_end_of_the_forecast(setup):
    _, booster, history = setup
    changed = history.copy()
    changed.loc[pd.to_datetime(changed["date"]) == DATES[HISTORY_DAYS - 1], TARGETS] += 5000.0

    a = _service(booster, history).forecast()
    b = _service(booster, changed).forecast()
    near, far = a["horizon_days"] == 1, a["horizon_days"] == 30
    assert np.allclose(a.loc[near, "forecast"], b.loc[near, "forecast"])
    assert not np.allclose(a.loc[far, "forecast"], b.loc[far, "forecast"])


def test_intervals_are_calibrated_per_segment(setup):
    _, booster, history = setup
    widths = _service(booster, history).interval_half_widths()
    assert widths["freelancer"] > 3 * widths["salaried"]


def test_intervals_cover_most_of_the_true_future_values(setup):
    series, booster, history = setup
    fc = _service(booster, history).forecast()
    truth = _rows(series, HISTORY_DAYS, 400)
    truth["date"] = pd.to_datetime(truth["date"])
    merged = fc.merge(truth[["segment", "date", "net"]], on=["segment", "date"])
    assert len(merged) == 60

    covered = (merged["net"] >= merged["lower"]) & (merged["net"] <= merged["upper"])
    assert covered.mean() >= 0.75
    assert covered.groupby(merged["segment"]).mean().min() >= 0.7


def test_too_few_calibration_points_gives_no_interval(setup):
    _, booster, history = setup
    fc = _service(booster, history, min_calibration=500).forecast()
    assert not fc["interval_calibrated"].any()
    assert fc["lower"].isna().all() and fc["upper"].isna().all()
    assert fc["forecast"].notna().all()


def test_stricter_alpha_needs_more_calibration_points(setup):
    _, booster, history = setup
    service = _service(booster, history)
    assert all(np.isfinite(v) for v in service.interval_half_widths(alpha=0.1).values())
    assert all(np.isnan(v) for v in service.interval_half_widths(alpha=0.001).values())


def test_training_cutoff_at_or_after_the_history_end_is_rejected(setup):
    _, booster, history = setup
    with pytest.raises(ValueError):
        ForecastService(booster, history, train_end=DATES[HISTORY_DAYS - 1])


def test_a_data_gap_in_the_input_window_gives_a_nan_forecast_not_a_guess(setup):
    _, booster, history = setup
    gapped = history.copy()
    gapped.loc[pd.to_datetime(gapped["date"]) == DATES[HISTORY_DAYS - 1], TARGETS] = np.nan
    fc = _service(booster, gapped).forecast()
    assert fc.loc[fc["horizon_days"] == 30, "forecast"].isna().all()
    assert fc.loc[fc["horizon_days"] == 1, "forecast"].notna().all()