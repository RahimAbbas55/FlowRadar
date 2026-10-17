import httpx

class ApiError(RuntimeError):
    # raised for connection failures and non-200 responses, with a message fit to show a user
    pass

def _detail(response) -> str:
    try:
        return str(response.json().get("detail", response.text))[:300]
    except Exception:
        return str(response.text)[:300]

class FlowRadarClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0, http=None):
        # http can be any object with a get(path, params=...) method, such as FastAPI's TestClient
        self._http = http if http is not None else httpx.Client(base_url=base_url, timeout=timeout)

    def _get(self, path: str, **params) -> dict:
        # parameters left as None are not sent, so the API's own defaults stay the only copy
        params = {key: value for key, value in params.items() if value is not None}
        try:
            response = self._http.get(path, params=params)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the FlowRadar API ({exc.__class__.__name__}): {exc}") from exc
        if response.status_code != 200:
            raise ApiError(f"API returned {response.status_code} for {path}: {_detail(response)}")
        return response.json()

    def health(self) -> dict:
        return self._get("/health")

    def forecast(self, segment: str | None = None, alpha: float | None = None) -> dict:
        return self._get("/forecast", segment=segment, alpha=alpha)

    def anomalies(
        self,
        segment: str | None = None,
        alpha: float | None = None,
        days: int | None = None,
        only_flagged: bool | None = None,
    ) -> dict:
        return self._get("/anomalies", segment=segment, alpha=alpha, days=days, only_flagged=only_flagged)

    def drift(
        self,
        current_days: int | None = None,
        reference_days: int | None = None,
        mode: str | None = None,
    ) -> dict:
        return self._get("/drift", current_days=current_days, reference_days=reference_days, mode=mode)