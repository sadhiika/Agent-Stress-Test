FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
# Cloud Run injects $PORT (defaults to 8080). Bind uvicorn to it.
CMD exec uvicorn harness.api:app --host 0.0.0.0 --port ${PORT:-8080}