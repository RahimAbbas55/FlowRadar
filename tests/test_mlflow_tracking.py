import numpy as np
import pandas as pd
import mlflow
from flowradar.tracking.mlflow_tracking import tracked_model_comparison
from flowradar.models.seasonal_naive import seasonal_naive_predict

def _seasonal_series(n_days=200):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = 40 * np.sin(2 * np.pi * t / 7)
    net = 1000 + weekly
    return pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": net, "inflow": net + 500, "outflow": 500.0}
    )

def test_tracked_comparison_returns_same_shape_as_untracked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    df = _seasonal_series()
    result = tracked_model_comparison(
        df, horizon=14, n_folds=1, step_days=14, calib_days=14,
        models={"seasonal_naive": seasonal_naive_predict},
    )
    assert len(result) == 1
    assert "mase" in result.columns

def test_tracked_comparison_creates_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    df = _seasonal_series()
    tracked_model_comparison(
        df, horizon=14, n_folds=1, step_days=14, calib_days=14,
        models={"seasonal_naive": seasonal_naive_predict},
    )

    experiment = mlflow.get_experiment_by_name("flowradar-model-comparison")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    # one parent run + one nested run for the single model/fold combination
    assert len(runs) == 2

def test_results_csv_logged_as_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    df = _seasonal_series()
    tracked_model_comparison(
        df, horizon=14, n_folds=1, step_days=14, calib_days=14,
        models={"seasonal_naive": seasonal_naive_predict},
    )

    experiment = mlflow.get_experiment_by_name("flowradar-model-comparison")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    parent_runs = runs[runs["tags.mlflow.parentRunId"].isna()]
    assert len(parent_runs) == 1
    artifacts = mlflow.artifacts.list_artifacts(run_id=parent_runs.iloc[0]["run_id"])
    artifact_names = [a.path for a in artifacts]
    assert "comparison_results.csv" in artifact_names