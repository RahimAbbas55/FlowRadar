import mlflow
import pytest
import numpy as np
import pandas as pd
from flowradar.evaluation.detector_report import alpha_sweep, clip_labels_to_window, evaluate_detection
from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.registry import load_registered_model, train_and_register_model
TARGETS = ["net", "inflow", "outflow"]
DATES = pd.date_range("2024-01-01", periods=400, freq="D")
TRAIN_DAYS = 270

def _series(seed=7):
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

def _with_anomalies(series):
    out = series.copy()
    d = pd.to_datetime(out["date"])
    out.loc[(out["segment"] == "freelancer") & (d == DATES[350]), ["net", "inflow"]] += 3000.0
    out.loc[(out["segment"] == "salaried") & (d == DATES[360]), ["net", "inflow"]] += 1000.0
    out.loc[(out["segment"] == "salaried") & d.isin([DATES[380], DATES[381]]), TARGETS] = np.nan
    return out

def _day(n):
    # day n of the series, allowed to lie beyond the last generated date
    return DATES[0] + pd.Timedelta(days=n)

def _label(segment, anomaly_type, start, end=None):
    return {
        "segment": segment,
        "anomaly_type": anomaly_type,
        "cause_label": anomaly_type,
        "component": "outflow",
        "event_date": _day(start),
        "expected_value": 0.0,
        "actual_value": 0.0,
        "end_date": _day(end) if end is not None else pd.NaT,
    }

LABELS = pd.DataFrame(
    [
        _label("freelancer", "large_unplanned_payment", 350),
        _label("salaried", "duplicate_or_suspicious_debit", 360),
        _label("salaried", "data_gap", 380, 381),
    ]
)

@pytest.fixture(scope="module")
def service(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("detector")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    series = _series()
    train_and_register_model(
        series[pd.to_datetime(series["date"]) <= DATES[TRAIN_DAYS - 1]], num_boost_round=50
    )
    booster = load_registered_model("latest")
    yield ForecastService(booster, _with_anomalies(series), train_end=DATES[TRAIN_DAYS - 1])
    patch.undo()

def test_clip_drops_outside_labels_and_clips_spans():
    labels = pd.DataFrame(
        [
            _label("salaried", "a", 50),
            _label("salaried", "business_closure", 295, 305),
            _label("salaried", "a", 350),
            _label("salaried", "business_closure", 395, 420),
        ]
    )
    clipped = clip_labels_to_window(labels, DATES[300], DATES[399])
    assert len(clipped) == 3
    assert clipped["event_date"].min() == DATES[300]
    assert clipped["end_date"].max() == DATES[399]
    assert (clipped["event_date"] <= clipped["end_date"]).all()

def test_injected_spikes_are_found(service):
    result = evaluate_detection(service, LABELS, alpha=0.1, days=100)
    assert result["overall"]["true_positives"] == 2
    assert result["overall"]["recall"] == 1.0
    assert set(result["by_type"]["anomaly_type"]) == {
        "large_unplanned_payment",
        "duplicate_or_suspicious_debit",
    }

def test_data_gaps_are_reported_separately_not_scored_against_the_interval(service):
    result = evaluate_detection(service, LABELS, alpha=0.1, days=100)
    assert result["missing_data"] == {"gap_days": 2, "detected": 2, "unexplained": 0}
    assert "data_gap" not in set(result["by_type"]["anomaly_type"])

def test_false_positive_rate_is_flags_over_clean_judged_days(service):
    result = evaluate_detection(service, LABELS, alpha=0.1, days=100)
    assert result["clean_days_judged"] > 0
    assert result["false_positive_rate"] == pytest.approx(
        result["overall"]["false_positives"] / result["clean_days_judged"]
    )

def test_labels_outside_the_window_raise(service):
    labels = pd.DataFrame([_label("salaried", "large_unplanned_payment", 50)])
    with pytest.raises(ValueError, match="no value-based"):
        evaluate_detection(service, labels, alpha=0.1, days=100)

def test_alpha_sweep_flags_fewer_days_as_alpha_shrinks(service):
    sweep = alpha_sweep(service, LABELS, alphas=(0.05, 0.1, 0.2), days=100)
    assert len(sweep) == 3
    ordered = sweep.sort_values("alpha", ascending=False)
    assert (ordered["flagged"].diff().dropna() <= 0).all()

def test_uncalibrated_days_are_counted_and_scored_as_misses(service):
    # 30 calibration residuals per segment cannot support alpha=0.01, which needs 99
    result = evaluate_detection(service, LABELS, alpha=0.01, days=100)
    assert result["status_counts"].get("not_calibrated", 0) > 0
    assert result["overall"]["true_positives"] == 0
    assert result["overall"]["recall"] == 0.0