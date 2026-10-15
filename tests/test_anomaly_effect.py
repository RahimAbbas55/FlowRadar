import pandas as pd
import pytest

from flowradar.config import GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.evaluation.anomaly_eval import expand_ground_truth_to_days


@pytest.fixture(scope="module", params=[1, 2, 3, 4, 5])
def dataset(request):
    return generate_dataset(GeneratorConfig(seed=request.param))


def _day_keys(label: dict) -> set:
    days = expand_ground_truth_to_days(pd.DataFrame([label]))
    return set(zip(days["segment"], days["date"]))


def test_every_value_anomaly_changes_the_series(dataset):
    series = dataset["series"].copy()
    series["date"] = pd.to_datetime(series["date"])
    counterfactual = dataset["counterfactual"].copy()
    counterfactual["date"] = pd.to_datetime(counterfactual["date"])

    merged = series.merge(counterfactual, on=["segment", "date"], suffixes=("_s", "_c"))
    diff = (merged["net_s"] - merged["net_c"]).abs()
    changed_rows = merged[(diff > 1e-6) | diff.isna()]
    changed = set(zip(changed_rows["segment"], changed_rows["date"]))

    labels = dataset["anomaly_labels"]
    value_labels = labels[labels["anomaly_type"] != "data_gap"]
    assert len(value_labels) > 0
    for label in value_labels.to_dict("records"):
        assert _day_keys(label) & changed, (
            f"{label['anomaly_type']} on {label['segment']} at {label['event_date']} changed nothing"
        )


def test_every_data_gap_day_is_missing_in_the_series(dataset):
    series = dataset["series"].copy()
    series["date"] = pd.to_datetime(series["date"])
    all_days = set(zip(series["segment"], series["date"]))
    missing = series[series["net"].isna()]
    missing_days = set(zip(missing["segment"], missing["date"]))

    labels = dataset["anomaly_labels"]
    for label in labels[labels["anomaly_type"] == "data_gap"].to_dict("records"):
        in_range = _day_keys(label) & all_days
        assert in_range <= missing_days