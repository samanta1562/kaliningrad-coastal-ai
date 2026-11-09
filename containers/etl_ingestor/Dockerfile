FROM python:3.12-slim

# Системные библиотеки для работы с георастрами (GDAL / Rasterio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1002 -s /bin/bash vkr-engineer || true
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Проверка готовности контейнера при старте
CMD ["python", "src/healthcheck.py"]