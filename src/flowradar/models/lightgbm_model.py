import lightgbm as lgb
import pandas as pd

from flowradar.features.horizon_features import build_horizon_features

# columns that are never used as model inputs: identifiers and all three raw targets
_NON_FEATURE_COLS = {"date", "net", "inflow", "outflow"}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS and c != "segment"]


def lightgbm_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = 200,
    seed: int = 42,
    horizon: int = 30,
) -> pd.Series:
    # direct forecaster: every lag feature is at least `horizon` days old, so any test
    # row up to `horizon` days past the training cutoff is predicted from known data only
    last_train = pd.to_datetime(train["date"]).max()
    span = (pd.to_datetime(test["date"]).max() - last_train).days
    if span > horizon:
        raise ValueError(
            f"test window reaches {span} days past the training cutoff but horizon is "
            f"{horizon}, so its features would leak"
        )

    combined = pd.concat([train, test]).sort_values(["segment", "date"])
    features = build_horizon_features(combined, horizon=horizon)

    train_dates = set(pd.to_datetime(train["date"]))
    test_dates = set(pd.to_datetime(test["date"]))
    dcol = pd.to_datetime(features["date"])

    train_feat = features[dcol.isin(train_dates)].copy()
    test_feat = features[dcol.isin(test_dates)].copy()

    train_feat["segment"] = train_feat["segment"].astype("category")
    test_feat["segment"] = test_feat["segment"].astype("category").cat.set_categories(
        train_feat["segment"].cat.categories
    )

    feature_cols = _feature_columns(features)

    # drop warm-up rows with no valid lag history rather than filling them with a false signal
    train_feat = train_feat.dropna(subset=feature_cols)

    default_params = {"objective": "regression", "metric": "mae", "verbosity": -1, "seed": seed}
    final_params = {**default_params, **(params or {})}

    train_set = lgb.Dataset(
        train_feat[feature_cols + ["segment"]],
        label=train_feat["net"],
        categorical_feature=["segment"],
    )
    booster = lgb.train(final_params, train_set, num_boost_round=num_boost_round)

    preds_df = test_feat[["segment", "date"]].copy()
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df["prediction"] = booster.predict(test_feat[feature_cols + ["segment"]])

    # explicit (segment, date) merge, safe regardless of row order or segment count
    test_lookup = test[["segment", "date"]].copy()
    test_lookup["date"] = pd.to_datetime(test_lookup["date"])
    merged = test_lookup.merge(preds_df, on=["segment", "date"], how="left")
    return pd.Series(merged["prediction"].values, index=test.index)