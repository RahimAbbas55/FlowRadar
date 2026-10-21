import re
import sys
import yaml
import json
import pytest
import subprocess
from pathlib import Path
import flowradar.dashboard as dashboard_package
ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_APP = Path(dashboard_package.__file__).parent / "app.py"

# imports a clean subprocess uses to find what a service really loads and which packages it needs
HELPER = r"""
import ast, importlib, importlib.metadata as md, json, sys

entries, extra, heavy = (json.loads(arg) for arg in sys.argv[1:4])
for name in entries:
    importlib.import_module(name)

files = {m.__file__ for n, m in sys.modules.items() if n.startswith("flowradar") and getattr(m, "__file__", None)}
files |= set(extra)

tops = set()
for path in files:
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
tops = {t for t in tops if t not in sys.stdlib_module_names and t != "flowradar"}

providers = md.packages_distributions()
print(json.dumps({
    "imports": {t: providers.get(t, []) for t in sorted(tops)},
    "loaded": sorted(n for n in heavy if n in sys.modules),
}))
"""

API_ENTRIES = ["flowradar.serving.api"]
DASHBOARD_ENTRIES = [
    "flowradar.dashboard.client",
    "flowradar.dashboard.frames",
    "flowradar.dashboard.charts",
    "flowradar.dashboard.settings",
]
def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

def _requirements(path: str) -> set:
    names = set()
    for line in (ROOT / path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(_normalise(re.split(r"[<>=!~;\[ ]", line)[0]))
    return names

def _inspect(entries, extra=(), heavy=()):
    result = subprocess.run(
        [sys.executable, "-c", HELPER, json.dumps(entries), json.dumps([str(p) for p in extra]), json.dumps(list(heavy))],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )
    return json.loads(result.stdout)

def _missing(imports: dict, requirements: set) -> list:
    # an import is covered when any distribution providing it is listed in the requirements
    return sorted(
        name for name, providers in imports.items()
        if not requirements & {_normalise(p) for p in providers}
    )

@pytest.fixture(scope="module")
def api_report():
    heavy = ["statsforecast", "prefect", "streamlit", "altair"]
    return _inspect(API_ENTRIES, heavy=heavy)

@pytest.fixture(scope="module")
def dashboard_report():
    heavy = ["lightgbm", "mlflow", "evidently", "statsforecast", "prefect", "fastapi"]
    return _inspect(DASHBOARD_ENTRIES, extra=[DASHBOARD_APP], heavy=heavy)

def test_every_package_the_api_imports_is_in_its_requirements(api_report):
    assert _missing(api_report["imports"], _requirements("docker/requirements-api.txt")) == []

def test_every_package_the_dashboard_imports_is_in_its_requirements(dashboard_report):
    assert _missing(dashboard_report["imports"], _requirements("docker/requirements-dashboard.txt")) == []

def test_the_serving_stack_does_not_load_the_heavy_training_or_ui_packages(api_report):
    assert api_report["loaded"] == []

def test_the_dashboard_does_not_load_any_modelling_package(dashboard_report):
    assert dashboard_report["loaded"] == []

def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())

def test_compose_wires_the_dashboard_to_a_healthy_api():
    services = _compose()["services"]
    assert set(services) == {"api", "dashboard"}
    assert services["dashboard"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["dashboard"]["environment"]["FLOWRADAR_API_URL"] == "http://api:8000"
    assert "8000:8000" in services["api"]["ports"]
    assert "8501:8501" in services["dashboard"]["ports"]
    for service in services.values():
        assert "healthcheck" in service
        assert (ROOT / service["build"]["dockerfile"]).exists()

def test_compose_defaults_to_amd64_and_keeps_mlflow_state_in_a_volume():
    compose = _compose()
    for service in compose["services"].values():
        assert "linux/amd64" in service["platform"]
    api = compose["services"]["api"]
    assert api["environment"]["MLFLOW_TRACKING_URI"].startswith("sqlite:////state/")
    assert any(volume.endswith(":/state") for volume in api["volumes"])
    assert "flowradar-state" in compose["volumes"]

def test_dockerignore_keeps_secrets_and_local_state_out_of_the_build():
    lines = (ROOT / ".dockerignore").read_text().split()
    for entry in (".env", ".venv", ".git", "*.db", "mlruns", "*.tfstate"):
        assert entry in lines

@pytest.mark.parametrize("name", ["api", "dashboard"])
def test_dockerfiles_run_as_a_non_root_user_and_never_copy_secrets(name):
    text = (ROOT / "docker" / f"{name}.Dockerfile").read_text()
    assert re.search(r"^USER app$", text, re.MULTILINE)
    assert ".env" not in text.replace(".env.example", "")
    assert "pip install" in text and "--no-cache-dir" in text

def test_the_api_image_installs_the_openmp_runtime_lightgbm_needs():
    assert "libgomp1" in (ROOT / "docker" / "api.Dockerfile").read_text()