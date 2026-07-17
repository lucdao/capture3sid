FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY snapshot.py .
COPY snapshot_api.py .
COPY capture_sync_api.py .
COPY camera_config_store.py .

RUN mkdir -p /data/vehicle_captures /data/config

CMD ["python", "snapshot.py"]
