FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
# For Cloud Run you would wrap main.py in a small FastAPI app exposing an
# HTTP endpoint; this CMD runs the CLI demo as a placeholder.
CMD ["python", "main.py", "--target-description", "a generic demo assistant"]
