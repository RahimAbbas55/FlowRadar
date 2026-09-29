from flowradar.config import GeneratorConfig
from flowradar.drift.scheduler import apply_drift_events, schedule_drift_events
from flowradar.segments.base_series import generate_base_series


def test_all_events_start_after_baseline():
    cfg = GeneratorConfig(n_days=730)
    events = schedule_drift_events(cfg, seed=1)
    for event in events:
        days_in = (event.start_date - cfg.start_date).days
        assert days_in >= cfg.drift.clean_baseline_days


def test_event_count_matches_config():
    cfg = GeneratorConfig(n_days=730)
    events = schedule_drift_events(cfg, seed=1)
    # 1 abrupt + 1 gradual + 1 volatility per segment by default, 3 segments
    assert len(events) == 3 * 3


def test_determinism_same_seed():
    cfg = GeneratorConfig(n_days=730)
    a = schedule_drift_events(cfg, seed=7)
    b = schedule_drift_events(cfg, seed=7)
    assert [e.start_date for e in a] == [e.start_date for e in b]
    assert [e.magnitude for e in a] == [e.magnitude for e in b]


def test_apply_drift_events_changes_series():
    cfg = GeneratorConfig(n_days=730)
    base = generate_base_series(cfg)
    events = schedule_drift_events(cfg, seed=1)
    drifted = apply_drift_events(base, events, seed=100)
    # some values must differ from the undrifted base series
    assert not base["outflow"].equals(drifted["outflow"]) or not base["inflow"].equals(
        drifted["inflow"]
    )


def test_apply_drift_preserves_row_count():
    cfg = GeneratorConfig(n_days=730)
    base = generate_base_series(cfg)
    events = schedule_drift_events(cfg, seed=1)
    drifted = apply_drift_events(base, events, seed=100)
    assert len(drifted) == len(base)