FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY artifacts /app/artifacts
COPY scripts /app/scripts

RUN mkdir -p /app/logs /app/mlflow_data /app/mlartifacts

EXPOSE 8000
EXPOSE 8501