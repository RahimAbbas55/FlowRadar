import pytest
from flowradar.config import GeneratorConfig
from flowradar.segments.base_series import generate_base_series

def test_combined_shape():
    cfg = GeneratorConfig(n_days=365)
    df = generate_base_series(cfg)
    assert len(df) == 365 * 3
    assert set(df["segment"].unique()) == {"salaried", "freelancer", "small_business"}
    assert "net" in df.columns

def test_net_equals_inflow_minus_outflow():
    cfg = GeneratorConfig(n_days=365)
    df = generate_base_series(cfg)
    assert (df["net"] == (df["inflow"] - df["outflow"]).round(2)).all()

def test_determinism_full_config():
    cfg = GeneratorConfig(n_days=365, seed=7)
    a = generate_base_series(cfg)
    b = generate_base_series(cfg)
    import pandas as pd

    pd.testing.assert_frame_equal(a, b)

def test_unknown_segment_raises():
    cfg = GeneratorConfig(n_days=365, segments=["salaried", "not_a_segment"])
    with pytest.raises(ValueError):
        generate_base_series(cfg)