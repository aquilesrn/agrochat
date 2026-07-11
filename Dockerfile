FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Estructura de directorios
RUN mkdir -p /app/data/{raw/avances,raw/coyuntura,raw/docs_pdf,duckdb,chroma} \
    /app/scripts /app/config

EXPOSE 8501 8888
CMD ["bash"]
