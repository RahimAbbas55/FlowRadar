from flowradar.config import GeneratorConfig
from flowradar.dataset import generate_dataset

def test_dataset_has_expected_keys():
    cfg = GeneratorConfig(n_days=730)
    result = generate_dataset(cfg)
    assert set(result.keys()) == {
        "series", "counterfactual", "anomaly_labels", "drift_events", "calendar", "manifest",
    }

def test_series_length_matches_config():
    cfg = GeneratorConfig(n_days=730)
    result = generate_dataset(cfg)
    assert len(result["series"]) == 730 * len(cfg.segments)

def test_determinism_full_pipeline():
    cfg = GeneratorConfig(n_days=730, seed=9)
    a = generate_dataset(cfg)
    b = generate_dataset(cfg)
    assert a["manifest"]["config_hash"] == b["manifest"]["config_hash"]
    assert a["series"]["net"].equals(b["series"]["net"])

def test_manifest_reflects_event_counts():
    cfg = GeneratorConfig(n_days=1095)
    result = generate_dataset(cfg)
    assert result["manifest"]["n_drift_events"] == len(result["drift_events"])
    assert result["manifest"]["n_anomaly_events"] == len(result["anomaly_labels"])