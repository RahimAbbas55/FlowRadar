import pandas as pd
import pytest

from flowradar.config import GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.evaluation.anomaly_eval import expand_ground_truth_to_days

DELAY_TYPES = {"payroll_delay", "client_payment_delay"}


@pytest.fixture(scope="module", params=[1, 2, 3, 4, 5])
def dataset(request):
    return generate_dataset(GeneratorConfig(seed=request.param))


def test_no_day_carries_two_labels(dataset):
    days = expand_ground_truth_to_days(dataset["anomaly_labels"])
    assert not days.duplicated(["segment", "date"]).any()


def test_delay_events_start_on_days_that_originally_had_inflow(dataset):
    # the original bug: a delay chained onto money an earlier delay had already moved
    counterfactual = dataset["counterfactual"].copy()
    counterfactual["date"] = pd.to_datetime(counterfactual["date"])
    inflow = counterfactual.set_index(["segment", "date"])["inflow"]

    labels = dataset["anomaly_labels"]
    delays = labels[labels["anomaly_type"].isin(DELAY_TYPES)]
    assert len(delays) > 0
    for row in delays.to_dict("records"):
        key = (row["segment"], pd.Timestamp(row["event_date"]))
        assert inflow.loc[key] > 0, f"{row['anomaly_type']} on {key} had no inflow to delay"