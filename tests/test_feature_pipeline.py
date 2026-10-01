import pandas as pd

from flowradar.features.pipeline import build_features


def test_pipeline_combines_both_feature_sets():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {"date": dates, "segment": "salaried", "net": range(60), "inflow": 0.0, "outflow": 0.0}
    )
    out = build_features(df)
    assert "is_weekend" in out.columns
    assert "net_lag_1" in out.columns
    assert "net_roll_mean_7" in out.columns
    assert len(out) == 60