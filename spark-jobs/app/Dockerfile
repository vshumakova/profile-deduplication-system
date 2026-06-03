FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    libgomp1 \
    gcc \
    g++ \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -c "import pandas; import numpy; import sklearn; import lightgbm; import joblib; import pyspark; print('All dependencies installed successfully')"

COPY app ./app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

CMD ["python", "-m", "app.worker"]