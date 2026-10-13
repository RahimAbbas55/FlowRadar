import mlflow
import pytest
from flowradar.tracking.promotion import get_champion_version
from flowradar.demo.drift_demo import default_demo_config, default_demo_events, run_drift_demo

@pytest.fixture(scope="module")
def demo_result(tmp_path_factory):
    # the demo is slow, so it runs once and every test reads the same result
    tmp = tmp_path_factory.mktemp("demo")
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")

    config = default_demo_config()
    result = run_drift_demo(config, default_demo_events(config), num_boost_round=50)
    yield result
    patch.undo()

def test_pre_shift_checkpoint_does_not_retrain(demo_result):
    first = demo_result["timeline"][0]
    assert first["as_of_day"] == 690
    assert first["retrained"] is False

def test_post_shift_checkpoint_retrains_and_promotes(demo_result):
    second = demo_result["timeline"][1]
    assert second["as_of_day"] == 800
    assert second["decision"]["retrain"] is True
    assert second["retrained"] is True
    assert second["challenger_version"] == 2
    assert second["promoted"] is True

def test_drift_is_attributed_to_the_shifted_segment_only(demo_result):
    # the error monitor does not fire for this shift (the champion adapts through its
    # lag features), so attribution is asserted on the drift monitor alone
    decision = demo_result["timeline"][1]["decision"]
    assert decision["drift_fired"] is True
    assert decision["drifted_segments"] == ["small_business"]
    assert "salaried" not in decision["drifted_segments"]

def test_champion_alias_moves_to_the_new_version(demo_result):
    assert int(get_champion_version().version) == 2