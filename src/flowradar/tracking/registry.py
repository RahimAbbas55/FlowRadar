import mlflow
import mlflow.lightgbm
import pandas as pd
from flowradar.features.pipeline import build_features
from flowradar.tracking.mlflow_tracking import _ensure_tracking_uri_set
_MODEL_NAME = "flowradar-lightgbm"
_EXPERIMENT_NAME = "flowradar-model-comparison"

'''
    Fits a LightGBM booster on the full provided training set and registers it
    as a new version under one fixed model name, so versions accumulate over time
'''
def train_and_register_model(
    train: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = 200,
    seed: int = 42,
) -> mlflow.entities.model_registry.ModelVersion:
    import lightgbm as lgb

    _ensure_tracking_uri_set()
    mlflow.set_experiment(_EXPERIMENT_NAME)

    features = build_features(train)
    feature_cols = [
        c for c in features.columns if c not in {"date", "net", "inflow", "outflow", "segment"}
    ]
    features["segment"] = features["segment"].astype("category")
    features = features.dropna(subset=feature_cols)

    default_params = {"objective": "regression", "metric": "mae", "verbosity": -1, "seed": seed}
    final_params = {**default_params, **(params or {})}

    train_set = lgb.Dataset(
        features[feature_cols + ["segment"]], label=features["net"], categorical_feature=["segment"]
    )

    with mlflow.start_run(run_name="register_lightgbm") as run:
        booster = lgb.train(final_params, train_set, num_boost_round=num_boost_round)
        mlflow.log_params(final_params)
        mlflow.log_param("num_boost_round", num_boost_round)
        mlflow.lightgbm.log_model(booster, artifact_path="model", registered_model_name=_MODEL_NAME)

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    return latest

# loads a specific version, or the highest existing version number if "latest"
def load_registered_model(version: int | str = "latest"):
    _ensure_tracking_uri_set()
    client = mlflow.MlflowClient()

    if version == "latest":
        versions = client.search_model_versions(f"name='{_MODEL_NAME}'")
        version = max(int(v.version) for v in versions)

    return mlflow.lightgbm.load_model(f"models:/{_MODEL_NAME}/{version}")