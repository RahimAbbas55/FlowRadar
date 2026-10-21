import os
import json
import mlflow
import tempfile
import argparse
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from flowradar.config import GeneratorConfig
from flowradar.dashboard.frames import status_banner
from flowradar.dashboard.snapshot import ANOMALY_ALPHAS, FORECAST_ALPHAS, REVIEW_WINDOWS, alpha_key
from flowradar.dataset import generate_dataset
from flowradar.dataset_io import load_dataset, save_dataset
from flowradar.serving.api import create_app
from flowradar.serving.champion import ensure_champion, load_forecast_service
from flowradar.serving.forecaster import ForecastService

def _get(client: TestClient, path: str, **params) -> dict:
    response = client.get(path, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"{path} {params} returned {response.status_code}: {response.text[:200]}")
    return response.json()

def build_snapshot(service: ForecastService, meta: dict) -> dict:
    # queries the API app for every choice the dashboard offers, so the stored responses are
    # exactly what the live API returns
    client = TestClient(create_app(service))
    health = _get(client, "/health")
    snapshot = {
        "meta": meta,
        "health": health,
        "drift": _get(client, "/drift"),
        "forecast": {},
        "anomalies": {},
    }
    for segment in health["segments"]:
        snapshot["forecast"][segment] = {
            alpha_key(alpha): _get(client, "/forecast", segment=segment, alpha=alpha)
            for alpha in FORECAST_ALPHAS
        }
        snapshot["anomalies"][segment] = {
            alpha_key(alpha): {
                str(days): _get(client, "/anomalies", segment=segment, alpha=alpha, days=days)
                for days in REVIEW_WINDOWS
            }
            for alpha in ANOMALY_ALPHAS
        }
    return snapshot

def validate_snapshot(snapshot: dict) -> list[str]:
    # every problem that would make the dashboard fail on a stored choice, empty when complete
    problems = [
        f"missing section {name}"
        for name in ("meta", "health", "drift", "forecast", "anomalies")
        if name not in snapshot
    ]
    if problems:
        return problems

    health = snapshot["health"]
    if not health["segments"]:
        problems.append("the health payload lists no segments")
    for key in ("built_at", "generator_version", "seed"):
        if key not in snapshot["meta"]:
            problems.append(f"meta is missing {key}")

    for segment in health["segments"]:
        for alpha in FORECAST_ALPHAS:
            payload = snapshot["forecast"].get(segment, {}).get(alpha_key(alpha))
            label = f"forecast {segment} alpha {alpha_key(alpha)}"
            if payload is None:
                problems.append(f"{label} is missing")
            elif len(payload["points"]) != health["horizon"]:
                problems.append(f"{label} has {len(payload['points'])} points, expected {health['horizon']}")
        for alpha in ANOMALY_ALPHAS:
            for days in REVIEW_WINDOWS:
                payload = snapshot["anomalies"].get(segment, {}).get(alpha_key(alpha), {}).get(str(days))
                label = f"anomalies {segment} alpha {alpha_key(alpha)} days {days}"
                if payload is None:
                    problems.append(f"{label} is missing")
                elif len(payload["points"]) != days:
                    problems.append(f"{label} has {len(payload['points'])} points, expected {days}")
    return problems

def write_snapshot(snapshot: dict, path) -> Path:
    # compact strict JSON: a NaN that slipped through would fail here instead of in the browser
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, separators=(",", ":"), allow_nan=False))
    return out

@contextmanager
def _temporary_registry():
    # a throwaway MLflow store, so building never touches or reuses a real registry
    previous_cwd, previous_uri = os.getcwd(), mlflow.get_tracking_uri()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.chdir(tmp)
        mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
        try:
            yield
        finally:
            os.chdir(previous_cwd)
            mlflow.set_tracking_uri(previous_uri)

def main() -> None:
    parser = argparse.ArgumentParser(description="build the precomputed dashboard snapshot")
    parser.add_argument("--data-dir", default="data/default")
    parser.add_argument("--out", default="snapshot/snapshot.json")
    parser.add_argument("--calibration-days", type=int, default=300)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--regenerate", action="store_true", help="rebuild the dataset before using it")
    args = parser.parse_args()

    # paths are resolved first because the registry context changes the working directory
    data_dir, out = Path(args.data_dir).resolve(), Path(args.out).resolve()
    if args.regenerate or not (data_dir / "manifest.json").exists():
        save_dataset(generate_dataset(GeneratorConfig()), data_dir)
    dataset = load_dataset(data_dir)
    manifest = dataset["manifest"]

    with _temporary_registry():
        ensure_champion(
            dataset["series"], calibration_days=args.calibration_days, num_boost_round=args.num_boost_round
        )
        service = load_forecast_service(dataset["series"])
        meta = {
            "built_at": datetime.now(timezone.utc).date().isoformat(),
            "generator_version": manifest.get("generator_version"),
            "seed": manifest.get("seed"),
            "config_hash": manifest.get("config_hash"),
            "model_version": service.model_version,
            "train_end": service.train_end.date().isoformat(),
            "calibration_days": args.calibration_days,
            "num_boost_round": args.num_boost_round,
        }
        snapshot = build_snapshot(service, meta)

    problems = validate_snapshot(snapshot)
    if problems:
        raise SystemExit("the snapshot is incomplete:\n" + "\n".join(problems))
    write_snapshot(snapshot, out)

    level, message = status_banner(snapshot["drift"])
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"generator v{meta['generator_version']}, seed {meta['seed']}, model v{meta['model_version']}")
    print(f"trained through {meta['train_end']}, data through {snapshot['health']['as_of']}")
    print(f"drift banner: {level}: {message}")
    for segment in snapshot["health"]["segments"]:
        flagged = snapshot["anomalies"][segment]["0.05"]["30"]["flagged"]
        print(f"{segment}: {flagged} flagged days in the last 30 at 5% sensitivity")

if __name__ == "__main__":
    main()