import mlflow
import pytest
from flowradar.config import GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.models.seasonal_naive import seasonal_naive_predict
from flowradar.tracking.dataset_comparison import run_dataset_comparison, select_models

def test_no_names_means_every_default_model():
    assert select_models(None) is None

def test_named_models_are_selected():
    assert set(select_models(["lightgbm", "ets"])) == {"lightgbm", "ets"}

def test_an_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown models"):
        select_models(["lightgbm", "prophet"])

def test_a_single_dataset_can_be_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    dataset = generate_dataset(GeneratorConfig(n_days=400, segments=["salaried"]))
    results = run_dataset_comparison(
        dataset, horizon=14, n_folds=1, step_days=14, calib_days=14,
        models={"seasonal_naive": seasonal_naive_predict}, datasets=("observed",),
    )
    assert set(results["data"]) == {"observed"}

def test_an_unknown_dataset_label_is_rejected():
    with pytest.raises(ValueError, match="unknown datasets"):
        run_dataset_comparison({"manifest": {}}, datasets=("observed", "nonsense"))