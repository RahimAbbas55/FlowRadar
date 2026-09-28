import pytest
from flowradar.config import GeneratorConfig, load_config

def test_default_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.n_days == 1095
    assert len(cfg.segments) == 3

def test_hash_is_stable_and_sensitive():
    a = GeneratorConfig()
    b = GeneratorConfig()
    c = GeneratorConfig(seed=7)
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()

def test_baseline_must_fit():
    with pytest.raises(ValueError):
        GeneratorConfig(n_days=400, drift={"clean_baseline_days": 500})

def test_magnitude_range_enforced():
    with pytest.raises(ValueError):
        GeneratorConfig(anomalies={"magnitude_min": 5.0, "magnitude_max": 3.0})