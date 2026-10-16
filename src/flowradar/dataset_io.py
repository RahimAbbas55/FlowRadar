import json
import pandas as pd
from pathlib import Path
# table name -> columns to parse as dates when loading
_TABLES = {
    "series": ["date"],
    "counterfactual": ["date"],
    "anomaly_labels": ["event_date", "end_date" , "arrival_date"],
    "drift_events": ["start_date"],
    "calendar": ["date"],
}

def save_dataset(dataset: dict, path) -> Path:
    # writes each table as CSV plus the manifest as JSON, so a dataset can be reloaded
    # later without regenerating it
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    for name in _TABLES:
        dataset[name].to_csv(out / f"{name}.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(dataset["manifest"], indent=2, default=str))
    return out

def _read_table(path: Path, date_cols: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # an empty table (for example no anomalies were injected) is written as an empty file
        return pd.DataFrame()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

def load_dataset(path) -> dict:
    # inverse of save_dataset, returning the same keys generate_dataset produces
    root = Path(path)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"no saved dataset at {root}")
    dataset = {name: _read_table(root / f"{name}.csv", cols) for name, cols in _TABLES.items()}
    dataset["manifest"] = json.loads((root / "manifest.json").read_text())
    return dataset