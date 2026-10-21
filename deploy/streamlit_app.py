import runpy
import sys
from pathlib import Path

# Community Cloud runs this file from a checkout of the repository, and the package lives under
# src/, so it is put on the path before the dashboard script is run
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "src" / "flowradar" / "dashboard" / "app.py"), run_name="__main__")