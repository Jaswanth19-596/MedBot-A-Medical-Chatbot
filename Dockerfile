FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements-app.txt 

# =========RUNTIME STAGE ================
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app.py .
COPY src ./src
COPY rate_limit.py .
COPY setup.py .
COPY config ./config/


EXPOSE 8501

CMD ["streamlit", "run", "app.py"]