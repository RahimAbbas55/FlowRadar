import streamlit as st
from flowradar.dashboard import settings
from flowradar.dashboard.charts import anomaly_chart, forecast_chart
from flowradar.dashboard.client import ApiError
from flowradar.dashboard.frames import (
    anomaly_counts,
    anomaly_frame,
    drift_frame,
    flagged_table,
    forecast_frame,
    model_caption,
    status_banner,
)

INTERVALS = {"80%": 0.2, "90%": 0.1, "95%": 0.05}
SENSITIVITY = {"Strict (1%)": 0.01, "Balanced (5%)": 0.05, "Sensitive (10%)": 0.1}
BANNERS = {"alert": st.error, "warning": st.warning, "unknown": st.info, "ok": st.success}
DRIFT_COLUMNS = ["segment", "status", "share_drifted", "data_drift_detected", "error_ratio", "error_degraded"]
st.set_page_config(page_title="FlowRadar", layout="wide")

@st.cache_data(ttl=120, show_spinner=False)
def fetch(_client, url, endpoint, params):
    # url and params form the cache key, the client object itself is not hashed
    return getattr(_client, endpoint)(**dict(params))

def get(client, url, endpoint, **params):
    return fetch(client, url, endpoint, tuple(sorted(params.items())))

client = settings.make_client()
url = settings.api_url()
st.title("FlowRadar")
st.caption("Cash flow forecasts, anomaly flags and drift status. All data is synthetic.")

try:
    health = get(client, url, "health")
except ApiError as exc:
    st.error(f"Could not load data from {url}. {exc}")
    st.stop()

st.caption(model_caption(health))

with st.sidebar:
    st.header("Controls")
    segment = st.selectbox("Segment", health["segments"], key="segment")
    interval_label = st.selectbox("Forecast interval", list(INTERVALS), index=1, key="interval")
    sensitivity_label = st.selectbox("Anomaly sensitivity", list(SENSITIVITY), index=0, key="sensitivity")
    days = st.slider("Days to review", min_value=7, max_value=90, value=30, key="days")

try:
    drift = get(client, url, "drift")
    forecast = forecast_frame(get(client, url, "forecast", segment=segment, alpha=INTERVALS[interval_label]))
    anomalies = anomaly_frame(
        get(client, url, "anomalies", segment=segment, alpha=SENSITIVITY[sensitivity_label], days=days)
    )
except ApiError as exc:
    st.error(f"Could not load data from {url}. {exc}")
    st.stop()

level, message = status_banner(drift)
BANNERS[level](message)

forecast_tab, recent_tab, drift_tab = st.tabs(["30-day forecast", "Recent days", "Drift and model"])

with forecast_tab:
    st.subheader(f"{segment}: next {health['horizon']} days")
    if forecast.empty:
        st.info("The API returned no forecast for this segment.")
    else:
        if not forecast["interval_calibrated"].any():
            st.warning("Not enough out-of-sample history to calibrate an interval, so only point forecasts are available.")
        missing = int(forecast["forecast"].isna().sum())
        if missing:
            st.warning(f"{missing} days have no forecast because recent input data is missing.")
        st.altair_chart(forecast_chart(forecast, segment), width="stretch")
        st.caption(
            f"The band is a {interval_label} conformal prediction interval, calibrated per segment on "
            "days the model never saw in training."
        )
        with st.expander("Forecast table"):
            st.dataframe(forecast.drop(columns="interval_calibrated").round(1), hide_index=True)

with recent_tab:
    st.subheader(f"{segment}: last {days} days")
    counts = anomaly_counts(anomalies)
    not_judged = counts["no_actual"] + counts["no_forecast"] + counts["not_calibrated"]
    flagged_col, normal_col, unjudged_col = st.columns(3)
    flagged_col.metric("Flagged days", counts["anomaly"])
    normal_col.metric("Judged normal", counts["normal"])
    unjudged_col.metric("Not judged", not_judged)

    if anomalies.empty:
        st.info("No recent days to review for this segment.")
    else:
        if counts["not_calibrated"] == len(anomalies):
            st.warning(
                "There is not enough out-of-sample history to calibrate this sensitivity. "
                "Try Balanced or Sensitive, or review fewer days."
            )
        st.altair_chart(anomaly_chart(anomalies, segment), width="stretch")
        flagged = flagged_table(anomalies)
        if flagged.empty:
            st.info("No days flagged in this window.")
        else:
            st.dataframe(flagged, hide_index=True)
    st.caption(
        "A day is flagged when the actual value falls outside the interval at the chosen sensitivity. "
        "Days with missing data or no calibrated interval are reported as not judged, never as normal."
    )

with drift_tab:
    st.subheader("Drift monitors")
    st.metric("Retrain recommended", "Yes" if drift["retrain"] else "No")
    st.dataframe(drift_frame(drift)[DRIFT_COLUMNS].round(3), hide_index=True)
    for reason in drift["reasons"]:
        st.write(f"- {reason}")
    st.caption(
        f"Each segment's last {drift['current_days']} days are compared with the {drift['reference_days']} "
        "days before them. The forecast error check needs enough out-of-sample history to run."
    )