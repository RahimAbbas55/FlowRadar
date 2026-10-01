import pandas as pd

from flowradar.evaluation.backtest import split_calibration


def test_calibration_is_tail_of_train():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "segment": "salaried", "net": range(100)})
    fit_train, calibration = split_calibration(df, calib_days=20)
    assert len(calibration) == 20
    assert len(fit_train) == 80
    assert pd.to_datetime(fit_train["date"]).max() < pd.to_datetime(calibration["date"]).min()