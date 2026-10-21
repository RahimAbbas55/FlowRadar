import copy
import json
from pathlib import Path
from flowradar.dashboard.client import ApiError
# the choices the dashboard offers, so a snapshot holds a stored result for each of them
FORECAST_ALPHAS = (0.2, 0.1, 0.05)
ANOMALY_ALPHAS = (0.01, 0.05, 0.1)
REVIEW_WINDOWS = (7, 14, 30, 60, 90)
DEFAULT_FORECAST_ALPHA = 0.1
DEFAULT_ANOMALY_ALPHA = 0.01
DEFAULT_REVIEW_DAYS = 30
_SECTIONS = ("health", "drift", "forecast", "anomalies")

def alpha_key(alpha: float) -> str:
    # the JSON key for an alpha, so 0.1 and 0.10 find the same stored result
    return f"{float(alpha):g}"

class SnapshotClient:
    # serves stored API responses with the same methods and JSON shapes as FlowRadarClient
    def __init__(self, path):
        self._path = Path(path)
        self._data = None

    def _load(self) -> dict:
        if self._data is None:
            try:
                data = json.loads(self._path.read_text())
            except FileNotFoundError as exc:
                raise ApiError(
                    f"no snapshot found at {self._path}. Build it with python -m flowradar.dashboard.build_snapshot"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ApiError(f"the snapshot at {self._path} is not valid JSON: {exc}") from exc
            missing = [section for section in _SECTIONS if section not in data]
            if missing:
                raise ApiError(f"the snapshot at {self._path} is missing sections: {missing}")
            self._data = data
        return self._data

    @staticmethod
    def _lookup(node: dict, key: str, label: str):
        if key not in node:
            raise ApiError(
                f"the snapshot has no stored result for {label} {key}, it has: {sorted(node, key=float)}"
            )
        return node[key]

    def _segments(self, segment):
        segments = self._load()["health"]["segments"]
        if segment is None:
            return segments
        if segment not in segments:
            raise ApiError(f"snapshot has no segment {segment!r}, expected one of {segments}")
        return [segment]

    def meta(self) -> dict:
        return copy.deepcopy(self._load().get("meta", {}))

    def health(self) -> dict:
        return copy.deepcopy(self._load()["health"])

    def forecast(self, segment: str | None = None, alpha: float | None = None) -> dict:
        data = self._load()
        key = alpha_key(DEFAULT_FORECAST_ALPHA if alpha is None else alpha)
        payloads = [
            self._lookup(data["forecast"][s], key, "forecast interval alpha") for s in self._segments(segment)
        ]
        merged = copy.deepcopy(payloads[0])
        merged["points"] = [point for payload in payloads for point in copy.deepcopy(payload["points"])]
        return merged

    def anomalies(
        self,
        segment: str | None = None,
        alpha: float | None = None,
        days: int | None = None,
        only_flagged: bool | None = None,
    ) -> dict:
        data = self._load()
        alpha_part = alpha_key(DEFAULT_ANOMALY_ALPHA if alpha is None else alpha)
        days_part = str(int(DEFAULT_REVIEW_DAYS if days is None else days))
        payloads = []
        for s in self._segments(segment):
            by_alpha = self._lookup(data["anomalies"][s], alpha_part, "anomaly sensitivity alpha")
            payloads.append(self._lookup(by_alpha, days_part, "review window of days"))

        merged = copy.deepcopy(payloads[0])
        merged["points"] = [point for payload in payloads for point in copy.deepcopy(payload["points"])]
        merged["flagged"] = sum(payload["flagged"] for payload in payloads)
        if only_flagged:
            merged["points"] = [p for p in merged["points"] if p["status"] == "anomaly"]
        return merged

    def drift(
        self,
        current_days: int | None = None,
        reference_days: int | None = None,
        mode: str | None = None,
    ) -> dict:
        stored = self._load()["drift"]
        asked = {"current_days": current_days, "reference_days": reference_days, "mode": mode}
        for name, value in asked.items():
            if value is not None and value != stored[name]:
                raise ApiError(
                    f"the snapshot only has the default drift windows ({stored['current_days']} days against "
                    f"{stored['reference_days']}, mode {stored['mode']})"
                )
        return copy.deepcopy(stored)