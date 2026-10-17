import os
from flowradar.dashboard.client import FlowRadarClient

def api_url() -> str:
    # where the FastAPI service lives, overridable per environment
    return os.environ.get("FLOWRADAR_API_URL", "http://localhost:8000")

def make_client() -> FlowRadarClient:
    return FlowRadarClient(base_url=api_url())