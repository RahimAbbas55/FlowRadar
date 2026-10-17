import pytest
import numpy as np
import pandas as pd
from flowradar.evaluation.comparison import run_model_comparison
from flowradar.evaluation.conformal import segment_quantiles

def zero_model(train, test):
    return pd.Series(0.0, index=test.index)

def _frame(n, scales, seed=3):
    # one column of pure noise per segment, so with a zero forecast every residual is the value
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    frames = [
        pd.DataFrame(
            {"date": dates, "segment": segment, "net": rng.normal(0, scale, n), "inflow": 0.0, "outflow": 0.0}
        )
        for segment, scale in scales.items()
    ]
    return pd.concat(frames, ignore_index=True)

def test_segment_quantiles_scale_with_each_segments_residuals():
    rng = np.random.default_rng(1)
    residuals = pd.DataFrame(
        {
            "segment": ["small"] * 200 + ["large"] * 200,
            "residual": np.concatenate([rng.normal(0, 1, 200), rng.normal(0, 1000, 200)]),
        }
    )
    q = segment_quantiles(residuals, alpha=0.1)
    assert 600 < q["large"] / q["small"] < 1600

def test_segment_quantiles_need_enough_residuals():
    few = pd.DataFrame({"segment": "a", "residual": np.arange(8.0)})
    enough = pd.DataFrame({"segment": "a", "residual": np.arange(9.0)})
    assert np.isnan(segment_quantiles(few, alpha=0.1)["a"])
    assert np.isfinite(segment_quantiles(enough, alpha=0.1)["a"])

def test_segment_quantiles_of_an_empty_frame_is_empty():
    empty = pd.DataFrame(
        {"segment": pd.Series(dtype=object), "residual": pd.Series(dtype=float)}
    )
    assert segment_quantiles(empty) == {}

def test_coverage_holds_in_every_segment_not_just_overall():
    # a pooled quantile would leave the large segment near 80% and the small one near 100%
    df = _frame(1000, {"small": 1.0, "large": 1000.0})
    result = run_model_comparison(
        df, horizon=200, n_folds=1, step_days=200, calib_days=200, alpha=0.1, models={"zero": zero_model}
    )
    row = result.iloc[0]
    assert row["segments_scored"] == 2
    assert row["min_segment_coverage"] >= 0.84
    assert row["interval_coverage"] >= 0.84

def test_width_in_mase_units_is_comparable_across_scales():
    small = _frame(400, {"a": 1.0})
    large = small.assign(segment="b", net=small["net"] * 1000)
    both = pd.concat([small, large], ignore_index=True)
    kwargs = dict(
        horizon=30, n_folds=1, step_days=30, calib_days=30, alpha=0.1, models={"zero": zero_model}
    )
    alone = run_model_comparison(small, **kwargs).iloc[0]
    together = run_model_comparison(both, **kwargs).iloc[0]
    assert together["mean_width_mase"] == pytest.approx(alone["mean_width_mase"])
    assert together["mase"] == pytest.approx(alone["mase"])
    # the pooled width, by contrast, is dominated by the large segment
    assert together["mean_interval_width"] > 100 * alone["mean_interval_width"]

def test_a_segment_without_enough_calibration_is_excluded_not_given_a_weak_interval():
    df = _frame(300, {"good": 10.0, "gappy": 10.0})
    # rows 300-599 are the second segment, and days 240-269 are the calibration window
    df.loc[540:564, "net"] = np.nan
    result = run_model_comparison(
        df, horizon=30, n_folds=1, step_days=30, calib_days=30, alpha=0.1, models={"zero": zero_model}
    )
    row = result.iloc[0]
    assert row["segments_scored"] == 1
    assert row["n_scored"] == 30

def test_a_calibration_window_too_short_for_alpha_is_rejected():
    with pytest.raises(ValueError, match="calibration"):
        run_model_comparison(
            _frame(300, {"a": 1.0}), horizon=30, n_folds=1, step_days=30, calib_days=5,
            alpha=0.1, models={"zero": zero_model},
        )