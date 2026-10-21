import pytest
import datetime
from pathlib import Path
import streamlit as st
from streamlit.testing.v1 import AppTest
import flowradar.dashboard as dashboard_package
from flowradar.dashboard import settings
from flowradar.dashboard.client import ApiError
APP_PATH = str(Path(dashboard_package.__file__).parent / "app.py")

class FakeClient:
    # canned payloads in the shape the real API returns, recording every call it receives
    def __init__(self, retrain=False, fail=False, uncalibrated=False, no_interval=False):
        self.retrain, self.fail = retrain, fail
        self.uncalibrated, self.no_interval = uncalibrated, no_interval
        self.calls = []

    def _record(self, name, params):
        self.calls.append((name, params))
        if self.fail:
            raise ApiError("cannot reach the FlowRadar API (ConnectError): connection refused")

    def health(self):
        self._record("health", {})
        return {
            "status": "ok", "model_version": 5, "as_of": "2024-12-30", "train_end_date": "2024-07-03",
            "horizon": 30, "segments": ["freelancer", "salaried"],
        }

    def forecast(self, **params):
        self._record("forecast", params)
        calibrated = not self.no_interval
        points = [
            {
                "segment": params["segment"], "date": f"2025-01-{i:02d}", "horizon_days": i,
                "forecast": 100.0 + i, "lower": 90.0 + i if calibrated else None,
                "upper": 110.0 + i if calibrated else None, "interval_calibrated": calibrated,
            }
            for i in range(1, 31)
        ]
        return {"model_version": 5, "as_of": "2024-12-30", "horizon": 30, "alpha": params["alpha"], "points": points}

    def anomalies(self, **params):
        self._record("anomalies", params)
        days = params["days"]
        last = datetime.date(2024, 12, 30)
        points = []
        for i in range(days):
            day = last - datetime.timedelta(days=days - 1 - i)
            if self.uncalibrated:
                status, lower, upper = "not_calibrated", None, None
            else:
                status, lower, upper = ("anomaly" if i == days - 1 else "normal"), 0.0, 10.0
            points.append(
                {
                    "segment": params["segment"], "date": day.isoformat(),
                    "actual": 50.0 if status == "anomaly" else 5.0, "forecast": 5.0,
                    "lower": lower, "upper": upper, "status": status,
                    "distance": 40.0 if status == "anomaly" else (0.0 if status == "normal" else None),
                }
            )
        flagged = sum(p["status"] == "anomaly" for p in points)
        return {
            "model_version": 5, "as_of": "2024-12-30", "alpha": params["alpha"], "days": days,
            "flagged": flagged, "points": points,
        }

    def drift(self, **params):
        self._record("drift", params)
        alert = self.retrain
        segments = [
            {
                "segment": name, "status": "alert" if (alert and name == "freelancer") else "ok",
                "data_drift_assessed": True, "share_drifted": 0.9 if (alert and name == "freelancer") else 0.1,
                "data_drift_detected": alert and name == "freelancer", "error_assessed": True,
                "error_ratio": 1.0, "error_degraded": False,
            }
            for name in ("freelancer", "salaried")
        ]
        return {
            "as_of": "2024-12-30", "model_version": 5, "train_end": "2024-07-03", "current_days": 60,
            "reference_days": 180, "retrain": self.retrain, "mode": "any",
            "reasons": ["data drift in: freelancer"] if self.retrain else [], "segments": segments,
        }

class FakeSnapshotClient(FakeClient):
    def meta(self):
        return {"built_at": "2026-10-20", "generator_version": "0.3.0", "seed": 42}
    
@pytest.fixture
def use_client(monkeypatch):
    # installs a fake client for the app and clears Streamlit's cache so tests cannot share results
    def install(cls=FakeClient, **kwargs):
        fake = cls(**kwargs)
        monkeypatch.setattr(settings, "make_client", lambda: fake)
        monkeypatch.setattr(settings, "api_url", lambda: "http://fake")
        monkeypatch.setattr(settings, "source_label", lambda: "http://fake")
        st.cache_data.clear()
        return fake

    return install

def _run():
    return AppTest.from_file(APP_PATH, default_timeout=30).run()

def _metric(at, label):
    return next(m for m in at.metric if m.label == label)

def test_the_app_renders_without_an_exception(use_client):
    use_client()
    at = _run()
    assert not at.exception
    assert any("Model v5" in c.value for c in at.caption)

def test_the_first_segment_is_loaded_with_the_apis_default_choices(use_client):
    fake = use_client()
    _run()
    assert ("forecast", {"alpha": 0.1, "segment": "freelancer"}) in fake.calls
    assert ("anomalies", {"alpha": 0.01, "days": 30, "segment": "freelancer"}) in fake.calls

def test_changing_the_segment_loads_that_segments_data(use_client):
    fake = use_client()
    at = _run()
    at.selectbox(key="segment").select("salaried").run()
    assert not at.exception
    assert ("anomalies", {"alpha": 0.01, "days": 30, "segment": "salaried"}) in fake.calls

def test_the_sensitivity_choice_is_sent_to_the_api(use_client):
    fake = use_client()
    at = _run()
    at.selectbox(key="sensitivity").select("Balanced (5%)").run()
    assert fake.calls[-1][0] == "anomalies"
    assert fake.calls[-1][1]["alpha"] == 0.05

def test_flagged_days_are_counted(use_client):
    use_client()
    at = _run()
    assert _metric(at, "Flagged days").value == "1"
    assert _metric(at, "Judged normal").value == "29"
    assert _metric(at, "Not judged").value == "0"

def test_the_banner_is_an_error_with_the_reasons_when_a_retrain_is_recommended(use_client):
    use_client(retrain=True)
    at = _run()
    assert any("Retrain recommended" in e.value and "freelancer" in e.value for e in at.error)
    assert _metric(at, "Retrain recommended").value == "Yes"

def test_the_banner_is_a_success_when_nothing_drifted(use_client):
    use_client()
    at = _run()
    assert any("No drift detected" in s.value for s in at.success)
    assert not at.error

def test_an_api_failure_shows_one_error_and_stops_the_page(use_client):
    use_client(fail=True)
    at = _run()
    assert not at.exception
    assert any("cannot reach" in e.value for e in at.error)
    assert len(at.metric) == 0

def test_an_uncalibrated_sensitivity_is_explained_not_drawn_as_normal(use_client):
    use_client(uncalibrated=True)
    at = _run()
    assert any("calibrate" in w.value for w in at.warning)
    assert _metric(at, "Not judged").value == "30"
    assert _metric(at, "Judged normal").value == "0"

def test_a_forecast_without_an_interval_is_flagged_as_such(use_client):
    use_client(no_interval=True)
    at = _run()
    assert not at.exception
    assert any("calibrate an interval" in w.value for w in at.warning)

def test_the_api_address_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("FLOWRADAR_API_URL", raising=False)
    assert settings.api_url() == "http://localhost:8000"
    monkeypatch.setenv("FLOWRADAR_API_URL", "http://api:9000")
    assert settings.api_url() == "http://api:9000"

def test_the_review_window_choice_is_sent_to_the_api(use_client):
    fake = use_client()
    at = _run()
    at.select_slider(key="days").set_value(60).run()
    assert fake.calls[-1][0] == "anomalies"
    assert fake.calls[-1][1]["days"] == 60

def test_a_snapshot_deployment_says_so(use_client):
    use_client(FakeSnapshotClient)
    at = _run()
    assert any("precomputed snapshot" in info.value for info in at.info)

def test_a_live_api_deployment_shows_no_snapshot_note(use_client):
    use_client()
    at = _run()
    assert not any("precomputed snapshot" in info.value for info in at.info)