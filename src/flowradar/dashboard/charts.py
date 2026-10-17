import altair as alt
import pandas as pd

def _one_segment(df: pd.DataFrame, segment: str, columns: list[str]) -> pd.DataFrame:
    data = df[df["segment"] == segment]
    if data.empty:
        raise ValueError(f"no rows for segment {segment!r}")
    return data[columns].reset_index(drop=True)

def forecast_chart(df: pd.DataFrame, segment: str, height: int = 320) -> alt.LayerChart:
    # the forecast line with its conformal interval as a band. Days without a calibrated
    # interval have null bounds, so the band simply has a gap there
    data = _one_segment(df, segment, ["date", "forecast", "lower", "upper", "horizon_days"])
    base = alt.Chart(data).encode(x=alt.X("date:T", title=None))
    band = base.mark_area(opacity=0.25).encode(
        y=alt.Y("lower:Q", title="net cash flow"), y2="upper:Q"
    )
    line = base.mark_line().encode(
        y="forecast:Q",
        tooltip=[
            alt.Tooltip("date:T"),
            alt.Tooltip("horizon_days:Q", title="days ahead"),
            alt.Tooltip("forecast:Q", format=",.0f"),
            alt.Tooltip("lower:Q", format=",.0f"),
            alt.Tooltip("upper:Q", format=",.0f"),
        ],
    )
    return (band + line).properties(height=height)

def anomaly_chart(df: pd.DataFrame, segment: str, height: int = 320) -> alt.LayerChart:
    # recent actuals against the anomaly interval, with flagged days marked in red. Days that
    # were not judged have null bounds and show as gaps in the band
    data = _one_segment(df, segment, ["date", "actual", "forecast", "lower", "upper", "status"])
    base = alt.Chart(data).encode(x=alt.X("date:T", title=None))
    band = base.mark_area(opacity=0.25).encode(
        y=alt.Y("lower:Q", title="net cash flow"), y2="upper:Q"
    )
    line = base.mark_line().encode(y="actual:Q")
    flagged = (
        base.transform_filter(alt.datum.status == "anomaly")
        .mark_point(color="red", filled=True, size=80)
        .encode(
            y="actual:Q",
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("actual:Q", format=",.0f"),
                alt.Tooltip("forecast:Q", format=",.0f"),
                alt.Tooltip("lower:Q", format=",.0f"),
                alt.Tooltip("upper:Q", format=",.0f"),
            ],
        )
    )
    return (band + line + flagged).properties(height=height)