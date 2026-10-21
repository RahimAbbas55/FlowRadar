import json
import streamlit as st
from pathlib import Path
from streamlit.testing.v1 import AppTest
from flowradar.dashboard.build_snapshot import validate_snapshot
ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "snapshot" / "snapshot.json"
ENTRYPOINT = ROOT / "deploy" / "streamlit_app.py"

def _reject(constant):
    raise ValueError(f"non-finite number {constant} in the snapshot")

def test_the_snapshot_is_committed():
    assert SNAPSHOT.exists(), (
        "run python -m flowradar.dashboard.build_snapshot and commit snapshot/snapshot.json"
    )

def test_the_snapshot_is_complete_strict_json():
    data = json.loads(SNAPSHOT.read_text(), parse_constant=_reject)
    assert validate_snapshot(data) == []

def test_the_snapshot_is_a_sensible_size():
    assert 10_000 < SNAPSHOT.stat().st_size < 2_000_000

def test_the_deploy_entrypoint_renders_the_committed_snapshot(monkeypatch):
    monkeypatch.delenv("FLOWRADAR_API_URL", raising=False)
    monkeypatch.setenv("FLOWRADAR_SNAPSHOT", str(SNAPSHOT))
    st.cache_data.clear()

    at = AppTest.from_file(str(ENTRYPOINT), default_timeout=60).run()
    assert not at.exception
    # a drift alert is drawn with st.error and is legitimate output, a failed data load is not
    assert not any("Could not load data" in e.value for e in at.error)
    assert any("Model v" in c.value for c in at.caption)

    other = [label for label in at.selectbox(key="segment").options][-1]
    at.selectbox(key="segment").select(other).run()
    assert not at.exception
    assert not any("Could not load data" in e.value for e in at.error)