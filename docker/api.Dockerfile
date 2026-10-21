FROM python:3.12-slim

# lightgbm needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY docker/requirements-api.txt docker/requirements-api.txt
RUN pip install --no-cache-dir -r docker/requirements-api.txt

COPY src src

# the synthetic dataset is seeded and deterministic, so it is generated once at build time
RUN python -c "from flowradar.config import GeneratorConfig; from flowradar.dataset import generate_dataset; from flowradar.dataset_io import save_dataset; save_dataset(generate_dataset(GeneratorConfig()), '/app/data/default')"

# MLflow state lives in a volume. The working directory is /state, so MLflow's relative
# artifact path (./mlruns) always resolves to the same place
RUN useradd --create-home app && mkdir /state && chown app:app /state
USER app
WORKDIR /state

ENV MLFLOW_TRACKING_URI=sqlite:////state/mlflow.db \
    FLOWRADAR_DATA_DIR=/app/data/default

EXPOSE 8000
CMD ["uvicorn", "flowradar.serving.api:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]