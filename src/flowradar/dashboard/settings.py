import os
from pathlib import Path
from flowradar.dashboard.client import FlowRadarClient
from flowradar.dashboard.snapshot import SnapshotClient
# src/flowradar/dashboard/settings.py -> the repository root is three levels up from the package
DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[3] / "snapshot" / "snapshot.json"

def api_url() -> str:
    # where the FastAPI service lives, overridable per environment
    return os.environ.get("FLOWRADAR_API_URL", "http://localhost:8000")

def uses_live_api() -> bool:
    # the dashboard talks to the API only when an address is set, otherwise it reads the snapshot
    return bool(os.environ.get("FLOWRADAR_API_URL"))

def snapshot_path() -> Path:
    return Path(os.environ.get("FLOWRADAR_SNAPSHOT", DEFAULT_SNAPSHOT))

def source_label() -> str:
    return api_url() if uses_live_api() else f"snapshot {snapshot_path().name}"

def make_client():
    if uses_live_api():
        return FlowRadarClient(base_url=api_url())
    return SnapshotClient(snapshot_path())