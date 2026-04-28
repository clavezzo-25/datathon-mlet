FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

COPY pyproject.toml .
COPY README.md .
COPY src/ src/
COPY data/ data/

RUN pip install --upgrade pip setuptools wheel \
    && pip install .

COPY . .

RUN mkdir -p data/raw models reports mlflow \
    && chown -R appuser:appuser /app \
    && git config --global --add safe.directory /app

USER appuser

CMD ["dvc", "repro"]