from flowradar.anomalies.scheduler import schedule_and_apply_anomalies
from flowradar.config import GeneratorConfig
from flowradar.segments.base_series import generate_base_series

def test_row_count_preserved():
    cfg = GeneratorConfig(n_days=730)
    base = generate_base_series(cfg)
    out, events = schedule_and_apply_anomalies(base, cfg, seed=1)
    assert len(out) == len(base)

def test_determinism_same_seed():
    cfg = GeneratorConfig(n_days=730)
    base = generate_base_series(cfg)
    out_a, events_a = schedule_and_apply_anomalies(base, cfg, seed=5)
    out_b, events_b = schedule_and_apply_anomalies(base, cfg, seed=5)
    assert [e.event_date for e in events_a] == [e.event_date for e in events_b]

def test_payroll_delay_only_targets_salaried():
    cfg = GeneratorConfig(n_days=730)
    base = generate_base_series(cfg)
    _, events = schedule_and_apply_anomalies(base, cfg, seed=3)
    payroll_events = [e for e in events if e.anomaly_type == "payroll_delay"]
    assert all(e.segment == "salaried" for e in payroll_events)

def test_produces_some_events():
    cfg = GeneratorConfig(n_days=1095)
    base = generate_base_series(cfg)
    _, events = schedule_and_apply_anomalies(base, cfg, seed=1)
    assert len(events) > 0