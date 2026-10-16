import os
import mlflow
import argparse
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from flowradar.config import DriftConfig, GeneratorConfig
from flowradar.dataset import generate_dataset
from flowradar.evaluation.anomaly_eval import expand_ground_truth_to_days
from flowradar.evaluation.detector_report import clip_labels_to_window
from flowradar.serving.champion import ensure_champion, load_forecast_service
from flowradar.serving.forecaster import ForecastService
# a change in net below half a cent is rounding, not an anomaly
EFFECT_THRESHOLD = 0.005
JUDGED = ("normal", "anomaly")
SIZE_BIN_ORDER = ["unjudged", "no_effect", "<0.25", "0.25-0.5", "0.5-1", "1-2", ">=2"]
_NO_DRIFT = DriftConfig(
    abrupt_events_per_segment=0, gradual_events_per_segment=0, volatility_events_per_segment=0
)
_TABLE_COLUMNS = [
    "segment", "date", "status", "flagged", "anomalous", "anomaly_type", "event_id",
    "size", "half_width", "size_bin", "echo_exposed", "actual", "forecast", "lower", "upper",
]

@contextmanager
def isolated_registry():
    # runs in a throwaway directory with its own sqlite MLflow store, then restores the
    # working directory and tracking URI, so evaluation never touches a real registry
    previous_cwd = os.getcwd()
    previous_uri = mlflow.get_tracking_uri()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.chdir(tmp)
        mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
        try:
            yield Path(tmp)
        finally:
            os.chdir(previous_cwd)
            mlflow.set_tracking_uri(previous_uri)

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    # 95% Wilson score interval for a proportion, which stays sensible for small n and p near 0 or 1
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))

def assign_size_bins(status: pd.Series, size: pd.Series, half_width: pd.Series) -> pd.Series:
    # the anomaly's effect relative to the interval half-width. Below ratio 1 the anomaly is
    # smaller than the interval, so it cannot be flagged unless noise helps
    ratio = size / half_width
    conditions = [
        ~status.isin(JUDGED).to_numpy(),
        (size.isna() | (size < EFFECT_THRESHOLD)).to_numpy(),
        (ratio < 0.25).to_numpy(),
        (ratio < 0.5).to_numpy(),
        (ratio < 1).to_numpy(),
        (ratio < 2).to_numpy(),
    ]
    labels = ["unjudged", "no_effect", "<0.25", "0.25-0.5", "0.5-1", "1-2"]
    return pd.Series(np.select(conditions, labels, default=">=2"), index=size.index)

def _empty_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": pd.Series(dtype=object),
            "date": pd.Series(dtype="datetime64[ns]"),
            "anomaly_type": pd.Series(dtype=object),
            "event_id": pd.Series(dtype=float),
        }
    )

def _truth_days(labels, segments, start, end) -> pd.DataFrame:
    # one row per anomalous day inside the window, tagged with the event it belongs to.
    # Data gaps are excluded because they are detected by the missing value, not an interval
    if labels is None or labels.empty:
        return _empty_truth()
    clipped = clip_labels_to_window(labels, start, end)
    clipped = clipped[clipped["segment"].isin(segments) & (clipped["anomaly_type"] != "data_gap")]
    frames = []
    for event_id, (_, row) in enumerate(clipped.iterrows()):
        days = expand_ground_truth_to_days(pd.DataFrame([row]))
        days["event_id"] = float(event_id)
        frames.append(days)
    if not frames:
        return _empty_truth()
    truth = pd.concat(frames, ignore_index=True)
    truth["date"] = pd.to_datetime(truth["date"])
    return truth.drop_duplicates(["segment", "date"])[["segment", "date", "anomaly_type", "event_id"]]

def _echo_exposed_keys(labels, segments, horizon: int) -> set:
    # (segment, date) pairs whose lag features contain an anomalous day: the lags start at
    # the horizon and the 28-day rolling mean reaches horizon + 27 days back
    if labels is None or labels.empty:
        return set()
    value = labels[labels["segment"].isin(segments) & (labels["anomaly_type"] != "data_gap")]
    if value.empty:
        return set()
    days = expand_ground_truth_to_days(value)
    exposed = set()
    for segment, date in zip(days["segment"], pd.to_datetime(days["date"])):
        for k in range(horizon, horizon + 28):
            exposed.add((segment, date + pd.Timedelta(days=k)))
    return exposed

def judged_day_table(service: ForecastService, dataset: dict, alpha: float = 0.05, days: int = 525) -> pd.DataFrame:
    # one row per judged segment-day with the detector's verdict, the ground truth, and the
    # anomaly's true effect (series minus counterfactual) relative to the interval width
    flags = service.anomaly_flags(alpha=alpha, days=days)
    if flags.empty:
        raise ValueError("no days to judge, the window lies inside the training data")
    start, end = flags["date"].min(), flags["date"].max()

    labels = dataset["anomaly_labels"]
    truth = _truth_days(labels, service.segments, start, end)
    table = flags.merge(truth, on=["segment", "date"], how="left")

    counterfactual = dataset["counterfactual"][["segment", "date", "net"]].copy()
    counterfactual["date"] = pd.to_datetime(counterfactual["date"])
    counterfactual = counterfactual.rename(columns={"net": "net_counterfactual"})
    table = table.merge(counterfactual, on=["segment", "date"], how="left")

    table["flagged"] = table["status"] == "anomaly"
    table["anomalous"] = table["anomaly_type"].notna()
    table["half_width"] = (table["upper"] - table["lower"]) / 2
    table["size"] = (table["actual"] - table["net_counterfactual"]).abs()
    table["size_bin"] = assign_size_bins(table["status"], table["size"], table["half_width"])
    table.loc[~table["anomalous"], "size_bin"] = None

    exposed = _echo_exposed_keys(labels, service.segments, service.horizon)
    table["echo_exposed"] = [(s, d) in exposed for s, d in zip(table["segment"], table["date"])]
    return table[_TABLE_COLUMNS].reset_index(drop=True)

def _counts(table: pd.DataFrame) -> dict:
    flagged, anomalous = table["flagged"], table["anomalous"]
    judged = table["status"].isin(JUDGED)
    effective = anomalous & (table["size"] >= EFFECT_THRESHOLD)
    clean = ~anomalous & judged
    exposed, unexposed = clean & table["echo_exposed"], clean & ~table["echo_exposed"]

    def rate(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else float("nan")

    tp = int((flagged & anomalous).sum())
    fp = int((flagged & ~anomalous).sum())
    fn = int((~flagged & anomalous).sum())
    return {
        "anomalous_days": int(anomalous.sum()),
        "effective_days": int(effective.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": rate(tp, tp + fp),
        "recall": rate(tp, tp + fn),
        "recall_effective": rate((flagged & effective).sum(), effective.sum()),
        "clean_days_judged": int(clean.sum()),
        "false_positive_rate": rate(fp, clean.sum()),
        "false_positive_rate_echo_exposed": rate((flagged & exposed).sum(), exposed.sum()),
        "false_positive_rate_unexposed": rate((flagged & unexposed).sum(), unexposed.sum()),
        "echo_exposed_share": rate(exposed.sum(), clean.sum()),
    }

def overall_summary(table: pd.DataFrame) -> dict:
    # counts are pooled across seeds before any ratio is taken
    seeds = int(table["seed"].nunique()) if "seed" in table.columns else 1
    return {"seeds": seeds, **_counts(table)}

def per_seed_summary(table: pd.DataFrame) -> pd.DataFrame:
    keys = ("anomalous_days", "precision", "recall", "recall_effective", "false_positive_rate")
    rows = []
    for seed, group in table.groupby("seed"):
        counts = _counts(group)
        rows.append({"seed": seed, **{k: counts[k] for k in keys}})
    return pd.DataFrame(rows)

def alpha_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = [{"alpha": alpha, **_counts(group)} for alpha, group in table.groupby("alpha")]
    columns = ["alpha", "precision", "recall", "recall_effective", "false_positive_rate",
               "true_positives", "false_positives", "false_negatives"]
    return pd.DataFrame(rows)[columns].sort_values("alpha", ascending=False).reset_index(drop=True)

def by_type_summary(table: pd.DataFrame) -> pd.DataFrame:
    # event-level recall: an event is detected if any of its days was flagged. Events with no
    # effect on the series are listed but not counted against recall
    columns = ["anomaly_type", "events", "events_with_effect", "events_detected",
               "event_recall", "ci_low", "ci_high"]
    anomalous = table[table["anomalous"]]
    if anomalous.empty:
        return pd.DataFrame(columns=columns)

    events = (
        anomalous.groupby(["seed", "event_id"])
        .agg(anomaly_type=("anomaly_type", "first"), max_size=("size", "max"), detected=("flagged", "any"))
        .reset_index()
    )
    events["has_effect"] = events["max_size"] >= EFFECT_THRESHOLD

    rows = []
    for anomaly_type, group in events.groupby("anomaly_type"):
        effective = group[group["has_effect"]]
        n, detected = len(effective), int(effective["detected"].sum())
        low, high = wilson_interval(detected, n)
        rows.append(
            {
                "anomaly_type": anomaly_type,
                "events": len(group),
                "events_with_effect": n,
                "events_detected": detected,
                "event_recall": detected / n if n else float("nan"),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows, columns=columns)

def by_size_summary(table: pd.DataFrame) -> pd.DataFrame:
    # detection rate of anomalous days by effect size relative to the interval half-width
    anomalous = table[table["anomalous"]]
    grouped = anomalous.groupby("size_bin")["flagged"].agg(["size", "sum"]).reindex(SIZE_BIN_ORDER, fill_value=0)
    out = pd.DataFrame(
        {
            "size_bin": SIZE_BIN_ORDER,
            "anomalous_days": grouped["size"].astype(int).to_numpy(),
            "flagged": grouped["sum"].astype(int).to_numpy(),
        }
    )
    out["detection_rate"] = out["flagged"] / out["anomalous_days"].replace(0, np.nan)
    return out

def evaluate_seed(
    seed: int,
    alphas=(0.05,),
    days: int = 525,
    n_days: int = 1825,
    calibration_days: int = 825,
    num_boost_round: int = 200,
) -> pd.DataFrame:
    # one drift-free dataset, one champion trained in a throwaway registry, and the judged-day
    # table at each alpha. Drift is off so false alarms reflect the detector, not stale models
    dataset = generate_dataset(GeneratorConfig(n_days=n_days, seed=seed, drift=_NO_DRIFT))
    history = dataset["series"]
    with isolated_registry():
        ensure_champion(history, calibration_days=calibration_days, num_boost_round=num_boost_round)
        service = load_forecast_service(history)
        tables = [
            judged_day_table(service, dataset, alpha=alpha, days=days).assign(alpha=alpha, seed=seed)
            for alpha in alphas
        ]
    return pd.concat(tables, ignore_index=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="multi-seed anomaly detector evaluation on drift-free data")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05, help="alpha for the per-type and per-size tables")
    parser.add_argument("--out", default="results/detector_eval")
    args = parser.parse_args()

    alphas = sorted({0.1, 0.05, 0.01, args.alpha}, reverse=True)
    frames = []
    for seed in range(1, args.seeds + 1):
        print(f"seed {seed} of {args.seeds}", flush=True)
        frames.append(evaluate_seed(seed, alphas=alphas))
    table = pd.concat(frames, ignore_index=True)
    headline = table[table["alpha"] == args.alpha]

    outputs = {
        "alpha_sweep": alpha_summary(table),
        "overall": pd.DataFrame([overall_summary(headline)]),
        "per_seed": per_seed_summary(headline),
        "by_type": by_type_summary(headline),
        "by_size": by_size_summary(headline),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        print(f"\n{name} (alpha={args.alpha} unless a sweep)")
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()