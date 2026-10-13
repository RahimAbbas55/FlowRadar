from datetime import timedelta

import pandas as pd

from flowradar.config import GeneratorConfig
from flowradar.drift.events import DriftEvent
from flowradar.drift.scheduler import apply_drift_events
from flowradar.evaluation.holdout import build_prediction_log, score_on_holdout
from flowradar.flows.retrain import retrain_flow
from flowradar.segments.base_series import generate_base_series
from flowradar.tracking.promotion import promote_if_better
from flowradar.tracking.registry import load_registered_model, train_and_register_model


def default_demo_config() -> GeneratorConfig:
    # two segments: small_business gets the drift, salaried is the untouched control
    return GeneratorConfig(n_days=800, segments=["salaried", "small_business"])


def default_demo_events(config: GeneratorConfig, shift_day: int = 700) -> list[DriftEvent]:
    # small business loses its main customer: inflow drops to 50% from shift_day onward
    start = config.start_date + timedelta(days=shift_day)
    return [DriftEvent("small_business", "abrupt", "inflow", start, 0.5)]


def run_drift_demo(
    config: GeneratorConfig,
    events: list[DriftEvent],
    checkpoints: tuple[int, ...] = (690, 800),
    champion_days: int = 365,
    holdout_days: int = 30,
    current_days: int = 56,
    reference_days: int = 182,
    mode: str = "both",
    num_boost_round: int = 100,
) -> dict:
    # runs the full loop on a drifted synthetic series: train an initial champion, then
    # at each checkpoint (a simulated "today") run the retrain flow on the data seen so far
    base = generate_base_series(config)
    series = apply_drift_events(base, events, seed=config.seed + 2000)
    dates = pd.to_datetime(series["date"])
    start = pd.Timestamp(config.start_date)

    def as_of(day: int) -> pd.Timestamp:
        return start + pd.Timedelta(days=day)

    # initial champion: trained on the first champion_days, scored on the 30 days after it
    train_df = series[dates < as_of(champion_days)]
    next_df = series[(dates >= as_of(champion_days)) & (dates < as_of(champion_days + 30))]
    v1 = train_and_register_model(train_df, num_boost_round=num_boost_round)
    initial_score = score_on_holdout(load_registered_model(int(v1.version)), train_df, next_df)
    outcome = promote_if_better(v1, challenger_score=initial_score, champion_score=None)
    if not outcome["promoted"]:
        raise RuntimeError(f"initial champion could not be promoted: {outcome['reason']}")

    # the log must start after the current champion's training cutoff to be out-of-sample
    log_start_day = champion_days
    timeline = []
    for day in checkpoints:
        visible = series[dates < as_of(day)]
        champion = load_registered_model("champion")
        log = build_prediction_log(champion, visible, start_date=as_of(log_start_day))

        result = retrain_flow(
            visible,
            log,
            holdout_days=holdout_days,
            num_boost_round=num_boost_round,
            current_days=current_days,
            reference_days=reference_days,
            mode=mode,
        )
        if result.get("promoted"):
            # the new champion trained on everything before its holdout
            log_start_day = day - holdout_days
        timeline.append({"as_of_day": day, "as_of_date": as_of(day), **result})

    return {"timeline": timeline, "series": series, "events": events}


def main() -> None:
    config = default_demo_config()
    result = run_drift_demo(config, default_demo_events(config))
    for step in result["timeline"]:
        print(
            f"{step['as_of_date'].date()}  retrained={step['retrained']}  "
            f"promoted={step.get('promoted')}  reasons={step['decision']['reasons']}"
        )


if __name__ == "__main__":
    main()