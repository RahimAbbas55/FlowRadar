import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
# Maps the model_name argument to its statsforecast class, each with weekly seasonality
_MODEL_CLASSES = {
    "ets": lambda: AutoETS(season_length=7),
    "arima": lambda: AutoARIMA(season_length=7),
}

def _to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    # statsforecast's required input shape: unique_id, ds, y
    return pd.DataFrame(
        {
            "unique_id": df["segment"],
            "ds": pd.to_datetime(df["date"]),
            "y": df["net"],
        }
    )

def statsforecast_predict(train: pd.DataFrame, test: pd.DataFrame, model_name: str = "ets") -> pd.Series:
    # fits one model per segment (unique_id) on train, forecasts horizon = len(test) days ahead
    if model_name not in _MODEL_CLASSES:
        raise ValueError(f"unknown model_name: {model_name}")

    train_long = _to_long_format(train)
    horizon = test["date"].nunique()

    sf = StatsForecast(models=[_MODEL_CLASSES[model_name]()], freq="D", n_jobs=1)
    sf.fit(train_long)
    forecast = sf.predict(h=horizon)

    model_col = [c for c in forecast.columns if c not in ("unique_id", "ds")][0]

    # map (segment, date) back to test's row order rather than relying on positional alignment,
    # since statsforecast's output order is per-unique_id blocks, not interleaved like test is
    forecast = forecast.rename(columns={"unique_id": "segment", "ds": "date", model_col: "prediction"})
    merged = test[["segment", "date"]].copy()
    merged["date"] = pd.to_datetime(merged["date"])
    forecast["date"] = pd.to_datetime(forecast["date"])
    merged = merged.merge(forecast, on=["segment", "date"], how="left")

    result = pd.Series(merged["prediction"].values, index=test.index)
    return result