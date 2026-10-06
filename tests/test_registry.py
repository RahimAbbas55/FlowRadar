import numpy as np
import pandas as pd
import mlflow
from flowradar.tracking.registry import load_registered_model, train_and_register_model

def _trending_series(n_days=200):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = 40 * np.sin(2 * np.pi * t / 7)
    net = 1000 + weekly
    return pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": net, "inflow": net + 500, "outflow": 500.0}
    )

def test_registration_returns_version_one_on_first_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    df = _trending_series()
    version = train_and_register_model(df, num_boost_round=20)
    assert int(version.version) == 1

def test_second_registration_increments_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    df = _trending_series()
    v1 = train_and_register_model(df, num_boost_round=20)
    v2 = train_and_register_model(df, num_boost_round=20)
    assert int(v2.version) == int(v1.version) + 1

def test_load_latest_returns_usable_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    df = _trending_series()
    train_and_register_model(df, num_boost_round=20)
    model = load_registered_model("latest")
    assert model is not None

def test_load_specific_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    df = _trending_series()
    v1 = train_and_register_model(df, num_boost_round=20)
    train_and_register_model(df, num_boost_round=20)  # v2
    model = load_registered_model(int(v1.version))
    assert model is not None