import re
import yaml
import pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())

def _job(workflow):
    return workflow["jobs"]["pytest"]

def _steps(workflow):
    return _job(workflow)["steps"]

def test_ci_runs_on_pushes_to_main_and_on_pull_requests(workflow):
    # PyYAML reads the bare key "on" as the boolean True
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["push"]["branches"] == ["main"]
    assert "pull_request" in triggers

def test_ci_uses_the_python_version_the_project_targets(workflow):
    setup = next(s for s in _steps(workflow) if s.get("uses", "").startswith("actions/setup-python"))
    assert setup["with"]["python-version"] == "3.12"
    assert setup["with"]["cache"] == "pip"
    assert "3.12" in (ROOT / "pyproject.toml").read_text()

def test_ci_installs_the_requirements_and_the_package_then_runs_the_whole_suite(workflow):
    commands = "\n".join(s.get("run", "") for s in _steps(workflow))
    assert "pip install -r requirements.txt" in commands
    assert "pip install -e ." in commands
    assert re.search(r"^pytest -q$", commands, re.MULTILINE)

def test_ci_has_a_timeout_and_read_only_permissions(workflow):
    assert _job(workflow)["timeout-minutes"] <= 60
    assert workflow["permissions"] == {"contents": "read"}

def test_every_action_is_pinned_to_a_major_version(workflow):
    uses = [s["uses"] for s in _steps(workflow) if "uses" in s]
    assert uses
    assert all(re.fullmatch(r"[\w./-]+@v\d+", u) for u in uses)

def test_the_license_is_mit_and_names_the_holder_and_year():
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License")
    assert "Rahim Abbas Sindhu" in text
    assert "2026" in text