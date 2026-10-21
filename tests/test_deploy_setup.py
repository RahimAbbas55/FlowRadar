import json
import pytest
import streamlit as st
from pathlib import Path
from streamlit.testing.v1 import AppTest
from tests.snapshot_helpers import full_snapshot
from tests.test_docker_setup import DASHBOARD_APP, DASHBOARD_ENTRIES, ROOT, _inspect, _missing, _requirements
ENTRYPOINT = ROOT / "deploy" / "streamlit_app.py"
HEAVY = ["lightgbm", "mlflow", "evidently", "statsforecast", "prefect", "fastapi"]

@pytest.fixture(scope="module")
def deploy_report():
    return _inspect(
        DASHBOARD_ENTRIES + ["flowradar.dashboard.snapshot"], extra=[DASHBOARD_APP, ENTRYPOINT], heavy=HEAVY
    )

def test_every_package_the_deployed_app_imports_is_in_the_deploy_requirements(deploy_report):
    assert _missing(deploy_report["imports"], _requirements("deploy/requirements.txt")) == []

def test_the_deployed_app_loads_no_modelling_package(deploy_report):
    assert deploy_report["loaded"] == []

def test_streamlit_is_pinned_to_an_exact_version():
    lines = (ROOT / "deploy" / "requirements.txt").read_text().split()
    assert any(line.startswith("streamlit==") for line in lines)

@pytest.fixture
def snapshot_app(monkeypatch, tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(full_snapshot()))
    monkeypatch.delenv("FLOWRADAR_API_URL", raising=False)
    monkeypatch.setenv("FLOWRADAR_SNAPSHOT", str(path))
    st.cache_data.clear()
    return AppTest.from_file(str(ENTRYPOINT), default_timeout=60)

def test_the_entrypoint_renders_the_dashboard_from_a_snapshot(snapshot_app):
    at = snapshot_app.run()
    assert not at.exception
    assert any("Model v1" in c.value for c in at.caption)
    assert any("precomputed snapshot" in i.value for i in at.info)
    assert next(m for m in at.metric if m.label == "Flagged days").value == "1"

def test_every_choice_in_the_sidebar_has_a_stored_result(snapshot_app):
    at = snapshot_app.run()
    for key in ("segment", "interval", "sensitivity"):
        for label in at.selectbox(key=key).options:
            at.selectbox(key=key).select(label).run()
            assert not at.exception and not at.error, f"{key} = {label}"
    for days in (7, 14, 30, 60, 90):
        at.select_slider(key="days").set_value(days).run()
        assert not at.exception and not at.error, f"days = {days}"

def test_a_missing_snapshot_tells_the_user_how_to_build_it(monkeypatch, tmp_path):
    monkeypatch.delenv("FLOWRADAR_API_URL", raising=False)
    monkeypatch.setenv("FLOWRADAR_SNAPSHOT", str(tmp_path / "missing.json"))
    st.cache_data.clear()
    at = AppTest.from_file(str(ENTRYPOINT), default_timeout=60).run()
    assert not at.exception
    assert any("build_snapshot" in e.value for e in at.error)