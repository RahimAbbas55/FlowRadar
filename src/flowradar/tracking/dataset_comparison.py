import argparse
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
    # one row per (data, model): means over folds, with MASE spread kept visible.
    # min_coverage is the mean over folds of the worst segment's coverage in each fold
    named = {
        "mase": ("mase", "mean"),
        "mase_std": ("mase", "std"),
        "smape": ("smape", "mean"),
        "coverage": ("interval_coverage", "mean"),
        "width": ("mean_interval_width", "mean"),
        "n_folds": ("fold_id", "nunique"),
    }
    if "min_segment_coverage" in results.columns:
        named["min_coverage"] = ("min_segment_coverage", "mean")
    if "mean_width_mase" in results.columns:
        named["width_mase"] = ("mean_width_mase", "mean")
    summary = results.groupby(["data", "model"]).agg(**named)
    return summary.reset_index().sort_values(["data", "mase"]).reset_index(drop=True)

def main() -> None:
    # uses the dataset saved under data/default (creating it if missing, also used by the
    # forecast service) and runs the full comparison
    parser = argparse.ArgumentParser(description="model comparison on the saved default dataset")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = Path("data/default")
    if not (data_dir / "manifest.json").exists():
        save_dataset(generate_dataset(GeneratorConfig()), data_dir)
    results = run_dataset_comparison(
        load_dataset(data_dir),
        horizon=args.horizon,
        n_folds=args.folds,
        step_days=args.horizon,
        calib_days=args.horizon,
        alpha=args.alpha,
    )

    out = Path("results")
    out.mkdir(exist_ok=True)
    results.to_csv(out / "model_comparison.csv", index=False)
    summary = summarize_comparison(results)
    summary.to_csv(out / "model_comparison_summary.csv", index=False)
    print(f"nominal interval coverage is {1 - args.alpha:.0%}")
    print(summary.round(3).to_string(index=False))

if __name__ == "__main__":
    main()