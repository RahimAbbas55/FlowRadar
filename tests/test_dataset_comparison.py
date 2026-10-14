import pytest
import mlflow
import pandas as pd
from flowradar.config import GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.models.seasonal_naive import seasonal_naive_predict
from flowradar.tracking.dataset_comparison import run_dataset_comparison, summarize_comparison

def test_summary_averages_over_folds_and_sorts_by_mase():
    results = pd.DataFrame(
        {
            "data": ["observed"] * 4,
            "model": ["a", "a", "b", "b"],
            "fold_id": [0, 1, 0, 1],
            "mase": [1.0, 3.0, 0.5, 0.5],
            "smape": [10.0, 20.0, 5.0, 5.0],
            "interval_coverage": [0.9, 0.7, 0.9, 0.9],
            "mean_interval_width": [10.0, 10.0, 4.0, 4.0],
        }
    )
    summary = summarize_comparison(results)
    assert list(summary["model"]) == ["b", "a"]
    row_a = summary[summary["model"] == "a"].iloc[0]
    assert row_a["mase"] == pytest.approx(2.0)
    assert row_a["coverage"] == pytest.approx(0.8)
    assert row_a["n_folds"] == 2

def test_comparison_runs_on_both_series_and_tags_the_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    dataset = generate_dataset(GeneratorConfig(n_days=400, segments=["salaried"]))
    results = run_dataset_comparison(
        dataset, horizon=14, n_folds=1, step_days=14, calib_days=14,
        models={"seasonal_naive": seasonal_naive_predict},
    )
    assert set(results["data"]) == {"observed", "counterfactual"}
    assert len(results) == 2

    experiment = mlflow.get_experiment_by_name("flowradar-model-comparison")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert set(runs["tags.data"].dropna()) == {"observed", "counterfactual"}