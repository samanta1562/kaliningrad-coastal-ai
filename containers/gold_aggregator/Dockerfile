FROM python:3.12-slim

RUN useradd -m -u 1002 -s /bin/bash vkr-engineer || true

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

USER 1002:1002
CMD ["python", "src/build_gold_marts.py"]
