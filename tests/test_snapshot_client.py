import json
import pytest
from flowradar.dashboard.client import ApiError
from flowradar.dashboard.snapshot import SnapshotClient, alpha_key
SEGMENTS = ["freelancer", "salaried"]

def _point(segment, status):
    return {
        "segment": segment, "date": "2025-01-01", "actual": 1.0, "forecast": 1.0,
        "lower": 0.0, "upper": 2.0, "status": status, "distance": 0.0,
    }

def _forecast(segment, alpha):
    return {
        "model_version": 1, "as_of": "2024-12-30", "horizon": 30, "alpha": alpha,
        "points": [
            {"segment": segment, "date": "2025-01-01", "horizon_days": 1, "forecast": 1.0,
             "lower": 0.0, "upper": 2.0, "interval_calibrated": True}
        ],
    }

def _anomalies(segment, alpha, days):
    return {
        "model_version": 1, "as_of": "2024-12-30", "alpha": alpha, "days": days, "flagged": 1,
        "points": [_point(segment, "anomaly"), _point(segment, "normal")],
    }

def _snapshot():
    return {
        "meta": {"built_at": "2026-10-20", "generator_version": "0.3.0", "seed": 42},
        "health": {
            "status": "ok", "model_version": 1, "as_of": "2024-12-30",
            "train_end_date": "2024-03-01", "horizon": 30, "segments": SEGMENTS,
        },
        "drift": {
            "as_of": "2024-12-30", "model_version": 1, "train_end": "2024-03-01", "current_days": 60,
            "reference_days": 180, "retrain": False, "mode": "any", "reasons": [], "segments": [],
        },
        "forecast": {s: {alpha_key(a): _forecast(s, a) for a in (0.2, 0.1)} for s in SEGMENTS},
        "anomalies": {
            s: {alpha_key(a): {str(d): _anomalies(s, a, d) for d in (7, 30)} for a in (0.01, 0.05)}
            for s in SEGMENTS
        },
    }

@pytest.fixture
def client(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(_snapshot()))
    return SnapshotClient(path)

def test_alpha_keys_are_stable_for_the_dashboards_choices():
    assert [alpha_key(a) for a in (0.2, 0.1, 0.05, 0.01)] == ["0.2", "0.1", "0.05", "0.01"]

def test_a_returned_payload_cannot_change_the_stored_one(client):
    client.health()["segments"].append("intruder")
    assert client.health()["segments"] == SEGMENTS

def test_forecast_uses_the_apis_default_interval(client):
    assert client.forecast(segment="salaried")["alpha"] == 0.1

def test_an_alpha_written_with_a_trailing_zero_finds_the_same_result(client):
    assert client.forecast(segment="salaried", alpha=0.10)["alpha"] == 0.1

def test_forecast_without_a_segment_merges_every_segment_in_order(client):
    points = client.forecast()["points"]
    assert [p["segment"] for p in points] == SEGMENTS

def test_an_interval_the_snapshot_does_not_hold_names_what_it_has(client):
    with pytest.raises(ApiError, match="forecast interval alpha"):
        client.forecast(segment="salaried", alpha=0.3)

def test_an_unknown_segment_is_rejected(client):
    with pytest.raises(ApiError, match="no segment"):
        client.forecast(segment="nope")
    with pytest.raises(ApiError, match="no segment"):
        client.anomalies(segment="nope")

def test_anomalies_use_the_apis_defaults(client):
    body = client.anomalies(segment="salaried")
    assert (body["alpha"], body["days"]) == (0.01, 30)

def test_anomalies_return_the_stored_result_for_the_chosen_sensitivity_and_window(client):
    body = client.anomalies(segment="salaried", alpha=0.05, days=7)
    assert (body["alpha"], body["days"]) == (0.05, 7)

def test_only_flagged_filters_the_points_but_keeps_the_flagged_count(client):
    body = client.anomalies(segment="salaried", only_flagged=True)
    assert len(body["points"]) == 1
    assert body["points"][0]["status"] == "anomaly"
    assert body["flagged"] == 1

def test_anomalies_without_a_segment_merge_segments_and_sum_the_flagged_count(client):
    body = client.anomalies()
    assert len(body["points"]) == 4
    assert body["flagged"] == 2

def test_a_review_window_the_snapshot_does_not_hold_is_rejected(client):
    with pytest.raises(ApiError, match="review window"):
        client.anomalies(segment="salaried", days=45)

def test_the_default_drift_windows_are_served_and_others_are_refused(client):
    assert client.drift()["retrain"] is False
    assert client.drift(current_days=60, reference_days=180, mode="any")["mode"] == "any"
    with pytest.raises(ApiError, match="default drift windows"):
        client.drift(current_days=30)

def test_a_missing_snapshot_says_how_to_build_it(tmp_path):
    with pytest.raises(ApiError, match="build_snapshot"):
        SnapshotClient(tmp_path / "nope.json").health()

def test_a_corrupt_snapshot_is_reported_not_crashed_on(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text("{not json")
    with pytest.raises(ApiError, match="not valid JSON"):
        SnapshotClient(path).health()

def test_a_snapshot_missing_a_section_is_rejected(tmp_path):
    path = tmp_path / "snapshot.json"
    broken = _snapshot()
    del broken["drift"]
    path.write_text(json.dumps(broken))
    with pytest.raises(ApiError, match="missing sections"):
        SnapshotClient(path).health()

def test_meta_describes_how_the_snapshot_was_built(client):
    assert client.meta()["seed"] == 42