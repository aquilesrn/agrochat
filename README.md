# AgroChat - TFM

Sistema inteligente de consulta y análisis de estadísticas agrarias
mediante RAG, aprendizaje automático explicable y visualización geoespacial interactiva.

## Estructura

```
TFM/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── scripts/
│   ├── etl_avances.py            # ETL: Cuadernos avances -> DuckDB
│   ├── renombrar_coyuntura.py    # Normalizar nombres archivos coyuntura
│   ├── etl_coyuntura.py          # ETL: Informes semanales precios -> DuckDB
│   ├── aemet_descarga.py         # Descarga datos climáticos AEMET -> DuckDB
│   ├── cargar_chromadb.py        # Generar fragmentos + cargar en ChromaDB
│   └── explorar_datos.py         # Consultas de exploración para Jupyter
├── data/
│   ├── raw/
│   │   ├── avances/              # Cuadernos mensuales (.xls/.xlsm)
│   │   ├── coyuntura/2022..2026/ # Informes semanales precios (.xlsx)
│   │   └── docs/                 # PDFs documentación MAPA
│   ├── duckdb/                   # agrochat.duckdb
│   └── chroma/                   # ChromaDB
└── .gitignore
```

## Pipeline completo

```bash
# 0. Levantar servicios
docker-compose up --build -d

# 1. Renombrar archivos de coyuntura
MSYS_NO_PATHCONV=1 docker exec -it agrochat python scripts/renombrar_coyuntura.py \
  --input-dir /app/data/raw/coyuntura

# 2. ETL Avances (~145 archivos -> DuckDB)
MSYS_NO_PATHCONV=1 docker exec -it agrochat python scripts/etl_avances.py \
  --input-dir /app/data/raw/avances --db /app/data/duckdb/agrochat.duckdb

# 3. ETL Coyuntura (~236 archivos -> DuckDB)
MSYS_NO_PATHCONV=1 docker exec -it agrochat python scripts/etl_coyuntura.py \
  --input-dir /app/data/raw/coyuntura --db /app/data/duckdb/agrochat.duckdb

# 4. AEMET: descargar inventario de estaciones
MSYS_NO_PATHCONV=1 docker exec -it agrochat python scripts/aemet_descarga.py \
  --api-key AEMET_API_KEY --inventario --db /app/data/duckdb/agrochat.duckdb

# 5. AEMET: descargar datos climáticos mensuales 2014-2025
MSYS_NO_PATHCONV=1 docker exec -it agrochat python scripts/aemet_descarga.py \
  --api-key AEMET_API_KEY --datos --anio-ini 2014 --anio-fin 2025 --db /app/data/duckdb/agrochat.duckdb

# 6. Cargar todo en ChromaDB (fragmentos textuales para RAG)
MSYS_NO_PATHCONV=1 docker exec -it agrochat python scripts/cargar_chromadb.py \
  --db /app/data/duckdb/agrochat.duckdb --chroma-path /app/data/chroma \
  --fuente todas --pdf-dir /app/data/raw/docs

# 7. Jupyter: http://localhost:8888 (token: agrochat)
```

## Tablas DuckDB

| Tabla | Contenido | Registros estimados |
|---|---|---|
| `avances` | Superficie y producción por cultivo/provincia/mes | ~540.000 |
| `coyuntura` | Precios semanales medios nacionales | ~26.000 |
| `aemet_estaciones` | Inventario estaciones meteorológicas | ~900 |
| `clima_mensual` | Series climáticas mensuales por estación | ~120.000 |
| `clima_provincial` (vista) | Medias climáticas agregadas por provincia | ~8.000 |
