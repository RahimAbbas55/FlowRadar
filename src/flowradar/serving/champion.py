import pandas as pd

from flowradar.dataset_io import load_dataset
from flowradar.evaluation.holdout import score_on_holdout
from flowradar.serving.forecaster import ForecastService
from flowradar.tracking.promotion import get_champion_version, promote_if_better
from flowradar.tracking.registry import load_registered_model, train_and_register_model


def ensure_champion(history: pd.DataFrame, calibration_days: int = 90, num_boost_round: int = 200, horizon: int = 30):
    # returns the current champion, training and promoting a first one if none exists. The
    # last calibration_days are held back from training so they can calibrate the intervals
    existing = get_champion_version()
    if existing is not None:
        return existing
    if calibration_days < horizon:
        raise ValueError("calibration_days must be at least the horizon")

    dates = pd.to_datetime(history["date"])
    cutoff = dates.max() - pd.Timedelta(days=calibration_days - 1)
    train_df = history[dates < cutoff]
    next_df = history[(dates >= cutoff) & (dates < cutoff + pd.Timedelta(days=horizon))]

    version = train_and_register_model(train_df, num_boost_round=num_boost_round, horizon=horizon)
    score = score_on_holdout(load_registered_model(int(version.version)), train_df, next_df)
    outcome = promote_if_better(version, challenger_score=score, champion_score=None)
    if not outcome["promoted"]:
        raise RuntimeError(f"initial champion could not be promoted: {outcome['reason']}")
    return get_champion_version()


def load_forecast_service(history: pd.DataFrame, min_calibration: int = 20) -> ForecastService:
    # builds a service around whichever version holds the champion alias
    champion = get_champion_version()
    if champion is None:
        raise RuntimeError("no champion model is registered, run ensure_champion first")
    train_end = champion.tags.get("train_end_date")
    if not train_end:
        raise RuntimeError(
            f"champion v{champion.version} has no train_end_date tag, so its calibration window is "
            "unknown. Register it again with the current registry code"
        )
    booster = load_registered_model("champion")
    return ForecastService(
        booster, history, train_end, model_version=int(champion.version), min_calibration=min_calibration
    )


def main() -> None:
    # smoke test against the dataset saved by the comparison script
    history = load_dataset("data/default")["series"]
    ensure_champion(history)
    service = load_forecast_service(history)
    print(f"champion v{service.model_version}, trained through {service.train_end.date()}")
    print(service.forecast().groupby("segment").head(3).to_string(index=False))


if __name__ == "__main__":
    main()