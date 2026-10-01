import pandas as pd


def seasonal_naive_predict(train: pd.DataFrame, test: pd.DataFrame, season_length: int = 7) -> pd.Series:
    # predicts each test day's net as the value exactly season_length days earlier,
    # per segment. Aligned back to test via an explicit (segment, date) merge, not
    # positional alignment, so this is safe regardless of row order or segment count
    combined = pd.concat([train, test]).sort_values(["segment", "date"]).copy()
    combined["date"] = pd.to_datetime(combined["date"])
    combined["prediction"] = combined.groupby("segment")["net"].shift(season_length)

    test_lookup = test[["segment", "date"]].copy()
    test_lookup["date"] = pd.to_datetime(test_lookup["date"])
    merged = test_lookup.merge(
        combined[["segment", "date", "prediction"]], on=["segment", "date"], how="left"
    )
    return pd.Series(merged["prediction"].values, index=test.index)