import numpy as np
import pandas as pd
import mlflow
from flowradar.tracking.promotion import get_champion_version, promote_if_better, should_promote
from flowradar.tracking.registry import load_registered_model, train_and_register_model

def _series(n_days=200):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    net = 1000 + 40 * np.sin(2 * np.pi * t / 7)
    return pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": net, "inflow": net + 500, "outflow": 500.0}
    )

def test_promotes_when_no_champion():
    promote, _ = should_promote(0.8, None)
    assert promote is True

def test_nan_challenger_never_promotes():
    promote, _ = should_promote(float("nan"), 1.0)
    assert promote is False

def test_none_challenger_never_promotes():
    promote, _ = should_promote(None, 1.0)
    assert promote is False

def test_promotes_when_clearly_better():
    promote, _ = should_promote(0.8, 1.0, min_improvement=0.02)
    assert promote is True

def test_does_not_promote_below_min_improvement():
    # 0.99 vs 1.0 is only a 1% improvement, under the 2% margin
    promote, _ = should_promote(0.99, 1.0, min_improvement=0.02)
    assert promote is False

def test_does_not_promote_when_worse():
    promote, _ = should_promote(1.2, 1.0)
    assert promote is False

def test_promotes_when_champion_score_is_nan():
    promote, _ = should_promote(0.9, float("nan"))
    assert promote is True

def test_first_version_becomes_champion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    v1 = train_and_register_model(_series(), num_boost_round=20)
    assert get_champion_version() is None

    result = promote_if_better(v1, challenger_score=0.8, champion_score=None)
    assert result["promoted"] is True
    assert int(get_champion_version().version) == 1

def test_worse_challenger_leaves_champion_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    v1 = train_and_register_model(_series(), num_boost_round=20)
    promote_if_better(v1, challenger_score=0.8, champion_score=None)

    v2 = train_and_register_model(_series(), num_boost_round=20)
    result = promote_if_better(v2, challenger_score=0.9, champion_score=0.8)
    assert result["promoted"] is False
    assert int(get_champion_version().version) == 1

def test_better_challenger_moves_champion_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    v1 = train_and_register_model(_series(), num_boost_round=20)
    promote_if_better(v1, challenger_score=0.8, champion_score=None)

    v2 = train_and_register_model(_series(), num_boost_round=20)
    result = promote_if_better(v2, challenger_score=0.5, champion_score=0.8)
    assert result["promoted"] is True
    assert int(get_champion_version().version) == 2

def test_holdout_score_tagged_even_when_not_promoted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    v1 = train_and_register_model(_series(), num_boost_round=20)
    promote_if_better(v1, challenger_score=0.8, champion_score=None)
    v2 = train_and_register_model(_series(), num_boost_round=20)
    promote_if_better(v2, challenger_score=0.9, champion_score=0.8)

    client = mlflow.MlflowClient()
    tagged = client.get_model_version("flowradar-lightgbm", "2")
    assert tagged.tags["holdout_mase"] == "0.9"

def test_load_champion_returns_usable_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    v1 = train_and_register_model(_series(), num_boost_round=20)
    promote_if_better(v1, challenger_score=0.8, champion_score=None)
    assert load_registered_model("champion") is not None