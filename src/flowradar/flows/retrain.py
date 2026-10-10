import pandas as pd
from prefect import flow, task

from flowradar.evaluation.holdout import score_on_holdout
from flowradar.flows.retrain_decision import retrain_decision_flow
from flowradar.tracking.promotion import get_champion_version, promote_if_better
from flowradar.tracking.registry import load_registered_model, train_and_register_model


@task(name="split-holdout")
def split_holdout(series, holdout_days: int):
    # the most recent holdout_days are held out from training for both models
    dates = pd.to_datetime(series["date"])
    cutoff = dates.max() - pd.Timedelta(days=holdout_days - 1)
    return series[dates < cutoff].copy(), series[dates >= cutoff].copy()


@task(name="train-challenger")
def train_challenger(fit_df, num_boost_round: int):
    return train_and_register_model(fit_df, num_boost_round=num_boost_round)


@task(name="score-challenger")
def score_challenger(version_number: int, fit_df, holdout_df):
    booster = load_registered_model(version_number)
    return score_on_holdout(booster, fit_df, holdout_df)


@task(name="score-champion")
def score_champion(fit_df, holdout_df):
    # scores the current champion on the same holdout, or reports that none exists yet
    champion = get_champion_version()
    if champion is None:
        return None, None
    booster = load_registered_model("champion")
    return int(champion.version), score_on_holdout(booster, fit_df, holdout_df)


@task(name="promote-if-better")
def promote(challenger, challenger_score, champion_score, min_improvement: float):
    return promote_if_better(challenger, challenger_score, champion_score, min_improvement)


# series and prediction_log are DataFrames with no type hints, as in the decision flow
@flow(name="retrain", log_prints=True)
def retrain_flow(
    series,
    prediction_log,
    holdout_days: int = 30,
    num_boost_round: int = 200,
    min_improvement: float = 0.02,
    force: bool = False,
    current_days: int = 60,
    reference_days: int = 180,
    drift_share_threshold: float = 0.5,
    error_ratio_threshold: float = 1.5,
    mode: str = "any",
) -> dict:
    decision = retrain_decision_flow(
        series,
        prediction_log,
        current_days=current_days,
        reference_days=reference_days,
        drift_share_threshold=drift_share_threshold,
        error_ratio_threshold=error_ratio_threshold,
        mode=mode,
    )

    if not decision["retrain"] and not force:
        print("no retrain needed")
        return {"retrained": False, "decision": decision}

    fit_df, holdout_df = split_holdout(series, holdout_days)
    challenger = train_challenger(fit_df, num_boost_round)
    challenger_score = score_challenger(int(challenger.version), fit_df, holdout_df)
    champion_version, champion_score = score_champion(fit_df, holdout_df)
    outcome = promote(challenger, challenger_score, champion_score, min_improvement)

    print(f"challenger v{challenger.version} promoted={outcome['promoted']}: {outcome['reason']}")
    return {
        "retrained": True,
        "decision": decision,
        "challenger_version": int(challenger.version),
        "challenger_score": challenger_score,
        "champion_version": champion_version,
        "champion_score": champion_score,
        "promoted": outcome["promoted"],
        "promotion_reason": outcome["reason"],
    }