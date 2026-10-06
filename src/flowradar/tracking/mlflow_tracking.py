import os
import mlflow
import pandas as pd
from flowradar.evaluation.comparison import run_model_comparison
_EXPERIMENT_NAME = "flowradar-model-comparison"

'''
    Defaults to a local sqlite store, since recent MLflow versions deprecated
    the plain filesystem backend. If a test or the environment already called
    mlflow.set_tracking_uri(...), that choice is left alone.
'''
def _ensure_tracking_uri_set() -> None:
    current = mlflow.get_tracking_uri()
    if current is None or current.startswith("file:"):
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
        
'''
    Runs the model comparison and logs every (model, fold) result to MLflow as
    a nested run, plus the full results table as a CSV artifact on the parent run
'''
def tracked_model_comparison(
    df: pd.DataFrame,
    horizon: int = 30,
    n_folds: int = 3,
    step_days: int = 30,
    calib_days: int = 30,
    alpha: float = 0.1,
    models: dict | None = None,
) -> pd.DataFrame:
    _ensure_tracking_uri_set()
    mlflow.set_experiment(_EXPERIMENT_NAME)

    with mlflow.start_run() as parent_run:
        mlflow.log_params(
            {
                "horizon": horizon,
                "n_folds": n_folds,
                "step_days": step_days,
                "calib_days": calib_days,
                "alpha": alpha,
            }
        )

        results = run_model_comparison(
            df, horizon=horizon, n_folds=n_folds, step_days=step_days,
            calib_days=calib_days, alpha=alpha, models=models,
        )

        for _, row in results.iterrows():
            with mlflow.start_run(run_name=f"{row['model']}_fold{row['fold_id']}", nested=True):
                mlflow.log_param("model", row["model"])
                mlflow.log_param("fold_id", int(row["fold_id"]))
                mlflow.log_metric("mase", row["mase"])
                mlflow.log_metric("smape", row["smape"])
                mlflow.log_metric("interval_coverage", row["interval_coverage"])
                mlflow.log_metric("mean_interval_width", row["mean_interval_width"])

        results_path = "comparison_results.csv"
        results.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

    return results