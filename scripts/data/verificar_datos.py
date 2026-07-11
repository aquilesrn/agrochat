"""
AgroChat — Verificación completa del estado de los datos
Ejecutar: docker exec agrochat python scripts/verificar_datos.py
"""
import duckdb, os, sys
from pathlib import Path

DB_PATH     = os.environ.get("DUCKDB_PATH", "/app/data/duckdb/agrochat.duckdb")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/app/data/chroma")
PQ_PATH     = "/app/data/processed/dataset_modelado_final.parquet"
MODEL_PATH  = "/app/data/models/xgboost_rendimiento.json"

OK, FAIL, WARN = "✅", "❌", "⚠️ "

def check(label, ok, detail=""):
    mark = OK if ok else FAIL
    print(f"  {mark} {label}" + (f": {detail}" if detail else ""))
    return ok

print("\n" + "="*55)
print("  VERIFICACIÓN DE DATOS — AgroChat")
print("="*55)

# ── 1. DuckDB ──────────────────────────────────────────────────
print("\n[1] DuckDB")
db_exists = Path(DB_PATH).exists()
check("Fichero DB existe", db_exists, DB_PATH)

if db_exists:
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    check("Tablas encontradas", len(tables) > 0, str(tables))

    for tabla, col_check, min_rows in [
        ("avances",          "cultivo",        500_000),
        ("coyuntura",        "producto",         24_000),
        ("clima_mensual",    "t_media",          60_000),
        ("aemet_estaciones", "indicativo",          700),
    ]:
        if tabla in tables:
            n = con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
            check(f"  {tabla}", n >= min_rows, f"{n:,} filas")
        else:
            check(f"  {tabla}", False, "tabla no encontrada")

    # Muestra rápida de datos reales
    print("\n  Muestra avances (trigo blando, nacional, 2023):")
    df = con.execute("""
        SELECT periodo_anio, periodo_mes, regimen,
               ROUND(sup_avance,0) AS sup_ha,
               ROUND(prod_avance,3) AS prod_miles_t
        FROM avances
        WHERE cultivo LIKE '%TRIGO BLANDO%'
          AND nivel='nacional' AND periodo_anio=2023
          AND regimen='total'
        ORDER BY periodo_mes
        LIMIT 5
    """).df()
    if df.empty:
        print(f"  {FAIL} Sin datos de trigo blando 2023")
    else:
        print(df.to_string(index=False))

    # Muestra coyuntura
    print("\n  Muestra coyuntura (trigo, 2023):")
    df2 = con.execute("""
        SELECT anio, semana, producto,
               ROUND(precio_sem_actual,2) AS precio,
               unidad
        FROM coyuntura
        WHERE LOWER(producto) LIKE '%trigo%' AND anio=2023
        LIMIT 3
    """).df()
    print(df2.to_string(index=False) if not df2.empty else f"  {WARN} Sin datos")

    con.close()

# ── 2. ChromaDB ────────────────────────────────────────────────
print("\n[2] ChromaDB")
chroma_exists = Path(CHROMA_PATH).exists()
check("Directorio existe", chroma_exists, CHROMA_PATH)

if chroma_exists:
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection(
            name="agrochat",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        n_col = col.count()
        check("Colección 'agrochat'", n_col > 100_000, f"{n_col:,} fragmentos")

        # Test búsqueda semántica
        r = col.query(query_texts=["ESYRCE encuesta superficies"], n_results=2,
                      include=["documents","metadatas"])
        check("Búsqueda semántica OK",
              len(r["documents"][0]) > 0,
              f"{len(r['documents'][0])} resultados")
        if r["documents"][0]:
            print(f"  Fragmento más relevante: {r['documents'][0][0][:120]}...")
    except Exception as e:
        check("ChromaDB accesible", False, str(e))

# ── 3. Dataset procesado ───────────────────────────────────────
print("\n[3] Dataset de modelado")
pq_exists = Path(PQ_PATH).exists()
check("Parquet final existe", pq_exists, PQ_PATH)
if pq_exists:
    import pandas as pd
    df_pq = pd.read_parquet(PQ_PATH)
    check("Filas en dataset",  len(df_pq) > 100_000, f"{len(df_pq):,}")
    nulls = df_pq[["cultivo_enc","anio","mes","t_media_c","rendimiento"]].isnull().sum().sum()
    check("Sin nulos en features clave", nulls == 0, f"{nulls} nulos")

# ── 4. Modelo XGBoost ──────────────────────────────────────────
print("\n[4] Modelo XGBoost")
model_exists = Path(MODEL_PATH).exists()
check("Modelo entrenado existe", model_exists, MODEL_PATH)
if not model_exists:
    print(f"  {WARN} Modelo perdido — necesita reentrenarse (tarda ~20 segundos)")
    print(f"       Ejecutar: python scripts/entrenar_modelo.py")

# ── Resumen ────────────────────────────────────────────────────
print("\n" + "="*55)
if db_exists and chroma_exists and pq_exists:
    print("  ✅ Datos intactos. Solo falta reentrenar el modelo.")
else:
    print("  ❌ Hay datos perdidos — revisar los puntos marcados.")
print("="*55 + "\n")
