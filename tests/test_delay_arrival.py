import pytest
import pandas as pd
from flowradar.anomalies.injectors import inject_client_payment_delay, inject_payroll_delay
from flowradar.config import GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.dataset_io import load_dataset, save_dataset
from flowradar.evaluation.anomaly_eval import evaluate_detector, expand_ground_truth_to_days
from flowradar.evaluation.detector_report import clip_labels_to_window
DELAY_TYPES = {"payroll_delay", "client_payment_delay"}

def _df(n=30, segment="salaried"):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "segment": segment, "inflow": 0.0, "outflow": 50.0})
    df.loc[10, "inflow"] = 2800.0
    df["net"] = df["inflow"] - df["outflow"]
    return df

def _label(event, arrival=None, end=None, segment="salaried", anomaly_type="payroll_delay"):
    return {
        "segment": segment,
        "anomaly_type": anomaly_type,
        "event_date": pd.Timestamp(event),
        "end_date": pd.Timestamp(end) if end else pd.NaT,
        "arrival_date": pd.Timestamp(arrival) if arrival else pd.NaT,
    }

def test_payroll_delay_records_its_arrival_day():
    df = _df()
    out, event = inject_payroll_delay(df, "salaried", df["date"].iloc[10], delay_days=3)
    assert event.arrival_date == df["date"].iloc[13].date()
    assert out["inflow"].iloc[13] == 2800.0

def test_client_payment_delay_records_its_arrival_day():
    df = _df(segment="freelancer")
    out, event = inject_client_payment_delay(df, "freelancer", df["date"].iloc[10], delay_days=4)
    assert event.arrival_date == df["date"].iloc[14].date()
    assert out["inflow"].iloc[14] == 2800.0

def test_an_arrival_after_the_data_ends_is_none():
    df = _df(n=15)
    out, event = inject_payroll_delay(df, "salaried", df["date"].iloc[10], delay_days=7)
    assert event.arrival_date is None

def test_expansion_includes_the_arrival_day():
    labels = pd.DataFrame([_label("2024-01-11", arrival="2024-01-14")])
    days = expand_ground_truth_to_days(labels)
    assert sorted(days["date"]) == [pd.Timestamp("2024-01-11"), pd.Timestamp("2024-01-14")]

def test_expansion_still_works_without_an_arrival_column():
    labels = pd.DataFrame([_label("2024-01-11", arrival="2024-01-14")]).drop(columns="arrival_date")
    assert len(expand_ground_truth_to_days(labels)) == 1

def test_clip_keeps_the_arrival_of_an_event_that_starts_before_the_window():
    labels = pd.DataFrame([_label("2024-01-08", arrival="2024-01-11")])
    clipped = clip_labels_to_window(labels, "2024-01-10", "2024-01-20")
    assert len(clipped) == 1
    assert clipped["event_date"].iloc[0] == pd.Timestamp("2024-01-11")
    assert clipped["end_date"].iloc[0] == pd.Timestamp("2024-01-11")

def test_clip_clears_an_arrival_that_falls_after_the_window():
    labels = pd.DataFrame([_label("2024-01-18", arrival="2024-01-22")])
    clipped = clip_labels_to_window(labels, "2024-01-10", "2024-01-20")
    assert len(clipped) == 1
    assert pd.isna(clipped["arrival_date"].iloc[0])
    assert len(expand_ground_truth_to_days(clipped)) == 1

def test_flagging_the_arrival_day_is_a_true_positive():
    labels = pd.DataFrame([_label("2024-01-11", arrival="2024-01-14")])
    detected = pd.DataFrame(
        {
            "segment": ["salaried", "salaried"],
            "date": ["2024-01-11", "2024-01-14"],
            "is_anomaly": [True, True],
        }
    )
    result = evaluate_detector(detected, labels)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["recall"] == 1.0

@pytest.fixture(scope="module", params=[1, 2, 3, 4, 5])
def dataset(request):
    return generate_dataset(GeneratorConfig(seed=request.param))

def _net_change(dataset):
    series = dataset["series"].copy()
    series["date"] = pd.to_datetime(series["date"])
    counterfactual = dataset["counterfactual"].copy()
    counterfactual["date"] = pd.to_datetime(counterfactual["date"])
    merged = series.merge(counterfactual, on=["segment", "date"], suffixes=("_s", "_c"))
    merged["change"] = merged["net_s"] - merged["net_c"]
    return merged.set_index(["segment", "date"])["change"]

def test_arrival_days_receive_exactly_the_delayed_amount(dataset):
    change = _net_change(dataset)
    labels = dataset["anomaly_labels"]
    checked = 0
    for row in labels[labels["anomaly_type"].isin(DELAY_TYPES)].to_dict("records"):
        if pd.isna(row["arrival_date"]):
            continue
        arrival = (row["segment"], pd.Timestamp(row["arrival_date"]))
        event = (row["segment"], pd.Timestamp(row["event_date"]))
        assert change.loc[arrival] == pytest.approx(row["expected_value"], abs=0.02)
        assert change.loc[event] == pytest.approx(-row["expected_value"], abs=0.02)
        checked += 1
    assert checked > 0

def test_arrival_dates_survive_saving_and_loading(dataset, tmp_path):
    save_dataset(dataset, tmp_path / "d")
    loaded = load_dataset(tmp_path / "d")["anomaly_labels"]
    assert pd.api.types.is_datetime64_any_dtype(loaded["arrival_date"])
    original = pd.to_datetime(dataset["anomaly_labels"]["arrival_date"]).dropna().reset_index(drop=True)
    reloaded = loaded["arrival_date"].dropna().reset_index(drop=True)
    assert (original.values == reloaded.values).all()