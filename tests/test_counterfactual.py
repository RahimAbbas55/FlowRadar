import pandas as pd

from flowradar.anomalies.counterfactual import generate_counterfactual
from flowradar.anomalies.injectors import inject_large_unplanned_payment


def test_counterfactual_matches_pre_anomaly_series():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"date": dates, "segment": "salaried", "inflow": [0.0] * 10, "outflow": [50.0] * 10}
    )
    df["net"] = df["inflow"] - df["outflow"]

    counterfactual = generate_counterfactual(df)
    anomalous, _ = inject_large_unplanned_payment(df, "salaried", dates[3].date(), magnitude=5.0)

    # counterfactual is untouched by the anomaly applied afterward to a separate copy
    assert counterfactual["outflow"].iloc[3] == 50.0
    assert anomalous["outflow"].iloc[3] == 250.0


def test_counterfactual_is_independent_copy():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"date": dates, "segment": "salaried", "inflow": [0.0] * 5, "outflow": [50.0] * 5}
    )
    counterfactual = generate_counterfactual(df)
    counterfactual.loc[0, "outflow"] = 999.0
    assert df.loc[0, "outflow"] == 50.0