import os
import mlflow
import pandas as pd
_EXPERIMENT_NAME = "flowradar-model-comparison"
_METRICS = ["mase", "smape", "interval_coverage", "min_segment_coverage", "mean_interval_width", "mean_width_mase"]

def _ensure_tracking_uri_set() -> None:
    # defaults to a local sqlite store, since recent MLflow versions deprecated
    # the plain filesystem backend. If a test or the environment already called
    # mlflow.set_tracking_uri(...), that choice is left alone.
    current = mlflow.get_tracking_uri()
    if current is None or current.startswith("file:"):
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")

def tracked_model_comparison(
    df: pd.DataFrame,
    horizon: int = 30,
    n_folds: int = 3,
    step_days: int = 30,
    calib_days: int = 30,
    alpha: float = 0.1,
    models: dict | None = None,
    tags: dict | None = None,
) -> pd.DataFrame:
    # runs the model comparison and logs every (model, fold) result to MLflow as a nested
    # run, per-model mean metrics on the parent run, and the full table as a CSV artifact
    from flowradar.evaluation.comparison import run_model_comparison
    _ensure_tracking_uri_set()
    mlflow.set_experiment(_EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "horizon": horizon,
                "n_folds": n_folds,
                "step_days": step_days,
                "calib_days": calib_days,
                "alpha": alpha,
            }
        )
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})

        results = run_model_comparison(
            df, horizon=horizon, n_folds=n_folds, step_days=step_days,
            calib_days=calib_days, alpha=alpha, models=models,
        )

        for _, row in results.iterrows():
            with mlflow.start_run(run_name=f"{row['model']}_fold{row['fold_id']}", nested=True):
                mlflow.log_param("model", row["model"])
                mlflow.log_param("fold_id", int(row["fold_id"]))
                for metric in _METRICS + ["n_scored", "segments_scored"]:
                    if pd.notna(row[metric]):
                        mlflow.log_metric(metric, float(row[metric]))

        for model, group in results.groupby("model"):
            for metric in _METRICS:
                value = group[metric].mean()
                if pd.notna(value):
                    mlflow.log_metric(f"{model}_{metric}_mean", float(value))

        results_path = "comparison_results.csv"
        results.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

    return results