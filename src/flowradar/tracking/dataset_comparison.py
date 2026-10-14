import pandas as pd
from pathlib import Path
from flowradar.config import GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.dataset_io import load_dataset, save_dataset
from flowradar.tracking.mlflow_tracking import tracked_model_comparison

def run_dataset_comparison(
    dataset: dict,
    horizon: int = 30,
    n_folds: int = 3,
    step_days: int = 30,
    calib_days: int = 30,
    alpha: float = 0.1,
    models: dict | None = None,
) -> pd.DataFrame:
    # runs the comparison on the observed series (with anomalies) and on the counterfactual
    # (drift but no anomalies), so model weakness and anomaly difficulty can be told apart
    manifest = dataset["manifest"]
    frames = []
    for label, key in (("observed", "series"), ("counterfactual", "counterfactual")):
        results = tracked_model_comparison(
            dataset[key],
            horizon=horizon,
            n_folds=n_folds,
            step_days=step_days,
            calib_days=calib_days,
            alpha=alpha,
            models=models,
            tags={"data": label, "config_hash": manifest["config_hash"], "seed": manifest["seed"]},
        )
        frames.append(results.assign(data=label))
    return pd.concat(frames, ignore_index=True)

def summarize_comparison(results: pd.DataFrame) -> pd.DataFrame:
    # one row per (data, model): means over folds, with MASE spread kept visible
    summary = results.groupby(["data", "model"]).agg(
        mase=("mase", "mean"),
        mase_std=("mase", "std"),
        smape=("smape", "mean"),
        coverage=("interval_coverage", "mean"),
        width=("mean_interval_width", "mean"),
        n_folds=("fold_id", "nunique"),
    )
    return summary.reset_index().sort_values(["data", "mase"]).reset_index(drop=True)

def main() -> None:
    # generates the default dataset, saves it under data/default (also used by the forecast
    # service), reloads it from disk, and runs the full comparison
    data_dir = Path("data/default")
    save_dataset(generate_dataset(GeneratorConfig()), data_dir)
    results = run_dataset_comparison(load_dataset(data_dir))
    print("nominal interval coverage is 90%")
    print(summarize_comparison(results).to_string(index=False))

if __name__ == "__main__":
    main()