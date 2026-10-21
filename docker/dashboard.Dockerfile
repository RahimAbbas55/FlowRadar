FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY docker/requirements-dashboard.txt docker/requirements-dashboard.txt
RUN pip install --no-cache-dir -r docker/requirements-dashboard.txt

COPY src src

RUN useradd --create-home app
USER app

ENV FLOWRADAR_API_URL=http://api:8000

EXPOSE 8501
CMD ["streamlit", "run", "src/flowradar/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]