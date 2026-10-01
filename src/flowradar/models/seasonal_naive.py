import pandas as pd

def seasonal_naive_predict(
    train: pd.DataFrame, test: pd.DataFrame, season_length: int = 7
) -> pd.Series:
    # predicts each test day's net as the net value exactly season_length days earlier,
    # computed per segment since each segment has its own independent series
    combined = pd.concat([train, test]).sort_values(["segment", "date"])
    combined["prediction"] = combined.groupby("segment")["net"].shift(season_length)

    test_dates = set(pd.to_datetime(test["date"]))
    mask = pd.to_datetime(combined["date"]).isin(test_dates)
    preds = combined.loc[mask].set_index(test.index.intersection(combined.loc[mask].index))

    # realign to test's original index and column order expected by the harness
    result = combined.loc[mask, "prediction"]
    result.index = test.index[: len(result)]
    return result