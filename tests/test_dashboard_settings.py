from pathlib import Path
from flowradar.dashboard import settings
from flowradar.dashboard.client import FlowRadarClient
from flowradar.dashboard.snapshot import SnapshotClient
ROOT = Path(__file__).resolve().parent.parent

def test_without_an_api_address_the_dashboard_reads_the_snapshot(monkeypatch):
    monkeypatch.delenv("FLOWRADAR_API_URL", raising=False)
    monkeypatch.delenv("FLOWRADAR_SNAPSHOT", raising=False)
    assert settings.uses_live_api() is False
    assert isinstance(settings.make_client(), SnapshotClient)
    assert settings.source_label().startswith("snapshot")

def test_an_api_address_switches_to_the_live_api(monkeypatch):
    monkeypatch.setenv("FLOWRADAR_API_URL", "http://api:9000")
    assert settings.uses_live_api() is True
    assert isinstance(settings.make_client(), FlowRadarClient)
    assert settings.source_label() == "http://api:9000"

def test_the_snapshot_location_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOWRADAR_SNAPSHOT", str(tmp_path / "other.json"))
    assert settings.snapshot_path() == tmp_path / "other.json"

def test_the_default_snapshot_lives_at_the_repository_root():
    assert settings.DEFAULT_SNAPSHOT == ROOT / "snapshot" / "snapshot.json"