import pytest
import pandas as pd
from flowradar.config import GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.dataset_io import load_dataset, save_dataset

@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(GeneratorConfig(n_days=400))

def _assert_same_table(a, b):
    assert (a["date"].values == b["date"].values).all()
    pd.testing.assert_frame_equal(a.drop(columns="date"), b.drop(columns="date"), check_dtype=False)

def test_series_and_counterfactual_round_trip(dataset, tmp_path):
    save_dataset(dataset, tmp_path / "d")
    loaded = load_dataset(tmp_path / "d")
    _assert_same_table(dataset["series"], loaded["series"])
    _assert_same_table(dataset["counterfactual"], loaded["counterfactual"])

def test_manifest_and_keys_round_trip(dataset, tmp_path):
    save_dataset(dataset, tmp_path / "d")
    loaded = load_dataset(tmp_path / "d")
    assert set(loaded) == set(dataset)
    assert loaded["manifest"]["config_hash"] == dataset["manifest"]["config_hash"]
    assert loaded["manifest"]["seed"] == dataset["manifest"]["seed"]

def test_label_dates_are_parsed_back_to_datetimes(dataset, tmp_path):
    save_dataset(dataset, tmp_path / "d")
    loaded = load_dataset(tmp_path / "d")
    assert len(loaded["anomaly_labels"]) == len(dataset["anomaly_labels"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["anomaly_labels"]["event_date"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["drift_events"]["start_date"])

def test_empty_anomaly_table_round_trips(dataset, tmp_path):
    empty = {**dataset, "anomaly_labels": pd.DataFrame()}
    save_dataset(empty, tmp_path / "d")
    assert load_dataset(tmp_path / "d")["anomaly_labels"].empty

def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope")