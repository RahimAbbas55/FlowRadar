import os
import mlflow
import pytest
import numpy as np
import pandas as pd
from flowradar.evaluation.detector_report import evaluate_detection
from flowradar.evaluation.multi_seed import (
    alpha_summary,
    assign_size_bins,
    by_size_summary,
    by_type_summary,
    evaluate_seed,
    isolated_registry,
    judged_day_table,
    overall_summary,
    per_seed_summary,
    wilson_interval,
)
from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.registry import load_registered_model, train_and_register_model
TARGETS = ["net", "inflow", "outflow"]
DATES = pd.date_range("2024-01-01", periods=400, freq="D")
TRAIN_DAYS = 270
NAN = np.nan

def _table(rows):
    # rows: (seed, event_id, status, anomaly_type, size, half_width)
    df = pd.DataFrame(rows, columns=["seed", "event_id", "status", "anomaly_type", "size", "half_width"])
    df["segment"] = "salaried"
    df["flagged"] = df["status"] == "anomaly"
    df["anomalous"] = df["anomaly_type"].notna()
    df["echo_exposed"] = False
    df["size_bin"] = assign_size_bins(df["status"], df["size"], df["half_width"])
    df.loc[~df["anomalous"], "size_bin"] = None
    return df

def test_wilson_interval_known_values():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)

def test_wilson_interval_with_no_events_is_nan():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)

def test_size_bins_at_the_edges():
    size = pd.Series([24.9, 25.0, 50.0, 99.9, 100.0, 199.0, 200.0])
    bins = assign_size_bins(pd.Series(["normal"] * 7), size, pd.Series([100.0] * 7))
    assert list(bins) == ["<0.25", "0.25-0.5", "0.5-1", "0.5-1", "1-2", "1-2", ">=2"]

def test_effectless_and_unjudged_days_get_their_own_bins():
    bins = assign_size_bins(
        pd.Series(["normal", "not_calibrated", "no_actual"]),
        pd.Series([0.0, 500.0, NAN]),
        pd.Series([100.0, NAN, 100.0]),
    )
    assert list(bins) == ["no_effect", "unjudged", "unjudged"]

def test_overall_counts_are_pooled_and_effect_free_days_are_separated():
    rows = [
        (1, 0, "anomaly", "a", 100.0, 50.0),  # caught
        (1, 1, "anomaly", "a", 100.0, 50.0),  # caught
        (1, 2, "normal", "a", 100.0, 500.0),  # missed
        (1, 3, "normal", "b", 0.0, 50.0),  # labelled but changed nothing
        (1, NAN, "anomaly", None, 0.0, 50.0),  # false alarm
    ] + [(1, NAN, "normal", None, 0.0, 50.0)] * 6
    result = overall_summary(_table(rows))
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (2, 1, 2)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(0.5)
    assert result["effective_days"] == 3
    assert result["recall_effective"] == pytest.approx(2 / 3)
    assert result["clean_days_judged"] == 7
    assert result["false_positive_rate"] == pytest.approx(1 / 7)

def test_false_positive_rate_is_split_by_echo_exposure():
    rows = [(1, NAN, "anomaly", None, 0.0, 50.0)] * 2 + [(1, NAN, "normal", None, 0.0, 50.0)] * 6
    table = _table(rows)
    table["echo_exposed"] = [True] * 4 + [False] * 4  # two flagged and two clean exposed, four clean not
    result = overall_summary(table)
    assert result["false_positive_rate_echo_exposed"] == pytest.approx(0.5)
    assert result["false_positive_rate_unexposed"] == 0.0
    assert result["echo_exposed_share"] == pytest.approx(0.5)

def test_by_type_counts_events_and_reports_an_interval():
    rows = [
        (1, 0, "anomaly", "a", 100.0, 50.0),
        (1, 1, "anomaly", "a", 100.0, 50.0),
        (1, 2, "anomaly", "a", 100.0, 50.0),
        (1, 3, "normal", "a", 100.0, 50.0),
        (1, 4, "normal", "a", 0.0, 50.0),  # no effect, listed but not counted against recall
        (1, 5, "anomaly", "b", 100.0, 50.0),
    ]
    result = by_type_summary(_table(rows)).set_index("anomaly_type")
    a = result.loc["a"]
    assert (a["events"], a["events_with_effect"], a["events_detected"]) == (5, 4, 3)
    assert a["event_recall"] == pytest.approx(0.75)
    assert 0 <= a["ci_low"] < 0.75 < a["ci_high"] <= 1
    assert result.loc["b", "event_recall"] == 1.0

def test_detection_rate_by_size_bin():
    rows = [
        (1, 0, "anomaly", "a", 10.0, 100.0),
        (1, 1, "normal", "a", 10.0, 100.0),
        (1, 2, "anomaly", "a", 30.0, 100.0),
        (1, 3, "anomaly", "a", 70.0, 100.0),
        (1, 4, "anomaly", "a", 150.0, 100.0),
        (1, 5, "anomaly", "a", 500.0, 100.0),
        (1, 6, "normal", "a", 0.0, 100.0),
        (1, 7, "not_calibrated", "a", 100.0, NAN),
    ]
    result = by_size_summary(_table(rows)).set_index("size_bin")
    assert list(result.index) == ["unjudged", "no_effect", "<0.25", "0.25-0.5", "0.5-1", "1-2", ">=2"]
    assert result.loc["<0.25", "detection_rate"] == 0.5
    for bin_name in ("0.25-0.5", "0.5-1", "1-2", ">=2"):
        assert result.loc[bin_name, "detection_rate"] == 1.0
    assert result.loc["no_effect", "detection_rate"] == 0.0
    assert result.loc["unjudged", "anomalous_days"] == 1

def test_per_seed_and_alpha_summaries_have_one_row_per_group():
    rows = [(1, 0, "anomaly", "a", 100.0, 50.0), (1, NAN, "normal", None, 0.0, 50.0)]
    first, second = _table(rows).assign(seed=1, alpha=0.1), _table(rows).assign(seed=2, alpha=0.05)
    table = pd.concat([first, second], ignore_index=True)
    assert list(per_seed_summary(table)["seed"]) == [1, 2]
    assert list(alpha_summary(table)["alpha"]) == [0.1, 0.05]

def test_isolated_registry_restores_the_working_directory_and_tracking_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/outer.db")
    cwd, uri = os.getcwd(), mlflow.get_tracking_uri()
    with isolated_registry():
        assert os.getcwd() != cwd
        assert mlflow.get_tracking_uri() != uri
        assert "mlflow.db" in mlflow.get_tracking_uri()
    assert os.getcwd() == cwd
    assert mlflow.get_tracking_uri() == uri

def _series(seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(400)
    frames = []
    for segment, level, noise in (("salaried", 1000.0, 15.0), ("freelancer", 3000.0, 150.0)):
        net = level + 50 * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, 400)
        frames.append(
            pd.DataFrame({"date": DATES, "segment": segment, "net": net, "inflow": net + 500, "outflow": 500.0})
        )
    return pd.concat(frames, ignore_index=True)

def _with_anomalies(series):
    out = series.copy()
    d = pd.to_datetime(out["date"])
    out.loc[(out["segment"] == "freelancer") & (d == DATES[350]), ["net", "inflow"]] += 3000.0
    out.loc[(out["segment"] == "salaried") & (d == DATES[360]), ["net", "inflow"]] += 1000.0
    out.loc[(out["segment"] == "salaried") & d.isin([DATES[380], DATES[381]]), TARGETS] = np.nan
    return out

def _label(segment, anomaly_type, start, end=None):
    return {
        "segment": segment,
        "anomaly_type": anomaly_type,
        "event_date": DATES[start],
        "end_date": DATES[end] if end is not None else pd.NaT,
        "arrival_date": pd.NaT,
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
    tmp = tmp_path_factory.mktemp("multiseed")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    series = _series()
    train_and_register_model(series[pd.to_datetime(series["date"]) <= DATES[TRAIN_DAYS - 1]], num_boost_round=50)
    booster = load_registered_model("latest")
    yield ForecastService(booster, _with_anomalies(series), train_end=DATES[TRAIN_DAYS - 1])
    patch.undo()

def _dataset(labels=LABELS):
    return {"anomaly_labels": labels, "counterfactual": _series()}

def test_day_table_agrees_with_the_existing_evaluator(service):
    table = judged_day_table(service, _dataset(), alpha=0.1, days=100)
    reference = evaluate_detection(service, LABELS, alpha=0.1, days=100)["overall"]
    result = overall_summary(table.assign(seed=0))
    for key in ("true_positives", "false_positives", "false_negatives"):
        assert result[key] == reference[key]

def test_anomaly_size_is_measured_against_the_counterfactual(service):
    table = judged_day_table(service, _dataset(), alpha=0.1, days=100)
    spikes = table[table["anomalous"]].sort_values("segment")
    assert list(spikes["segment"]) == ["freelancer", "salaried"]
    assert spikes["size"].tolist() == pytest.approx([3000.0, 1000.0])
    assert (spikes["size_bin"] == ">=2").all()

def test_echo_exposure_covers_days_30_to_57_after_an_anomaly(service):
    table = judged_day_table(service, _dataset(), alpha=0.1, days=100)

    def exposed(segment, day):
        row = table[(table["segment"] == segment) & (table["date"] == DATES[day])]
        return bool(row["echo_exposed"].iloc[0])

    assert exposed("freelancer", 380) and exposed("freelancer", 399)  # 30 days after day 350
    assert not exposed("freelancer", 379)
    assert exposed("salaried", 390) and not exposed("salaried", 389)  # 30 days after day 360

def test_a_dataset_without_labels_still_produces_a_table(service):
    table = judged_day_table(service, _dataset(labels=pd.DataFrame()), alpha=0.1, days=100)
    assert not table["anomalous"].any()
    assert len(table) == 2 * 100

def test_evaluate_seed_returns_a_labelled_table_for_every_alpha():
    table = evaluate_seed(3, alphas=(0.1, 0.05), days=200, n_days=700, calibration_days=300, num_boost_round=20)
    assert set(table["alpha"]) == {0.1, 0.05}
    assert (table["seed"] == 3).all()
    assert len(table) == 2 * 3 * 200
    assert {"flagged", "anomalous", "size", "size_bin", "echo_exposed"} <= set(table.columns)