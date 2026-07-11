"""
══════════════════════════════════════════════════════════════════
AgroChat — Procesamiento y limpieza del dataset de modelado
══════════════════════════════════════════════════════════════════
Toma el estado actual de DuckDB (ya validado por validar_datos.py)
y produce un dataset Parquet limpio y listo para XGBoost/SHAP.

Problemas identificados en el diagnóstico y acciones tomadas:

  AVANCES
  -------
  [1] 15.967 filas con periodo_anio=2026 → se excluyen del modelado
      (datos parciales del año en curso, no comparables con histórico)
  [2] Nulos estructurales en sup/prod: se conservan en avances pero
      se excluyen del dataset de modelado (sin rendimiento calculable)

  CLIMA_MENSUAL
  -------------
  [3] t_max_abs / t_min_abs / horas_sol → 100% nulos → columnas descartadas
  [4] dias_precip → 86% nulos → columna descartada
  [5] t_media, t_max_med, t_min_med, precip_mm, hr_media → ~10-15% nulos
      → imputados con mediana por provincia × mes (no por media global:
         el clima es fuertemente estacional y geográfico)

  DATASET DE MODELADO
  -------------------
  [6] 10 provincias sin cobertura climática → se mantienen en el dataset
      con features climáticas a NaN; XGBoost las maneja nativamente,
      pero se añade flag `sin_clima` para análisis posterior
  [7] Target `rendimiento` con valores extremos (outliers de cultivo):
      se aplica winsorización al p99 por cultivo para no distorsionar
      el entrenamiento (los valores reales se conservan en columna aparte)

Uso:
  python scripts/procesar_datos.py --db /app/data/duckdb/agrochat.duckdb

Salidas:
  /app/data/processed/dataset_modelado_limpio.parquet  ← para XGBoost
  /app/data/processed/informe_procesamiento.json
"""

import argparse, duckdb, json, logging, os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("procesar_datos")


def seccion(titulo: str):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {titulo}")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════
# 1. AVANCES — excluir 2026 y validar rango
# ══════════════════════════════════════════════════════════════════

def procesar_avances(con) -> dict:
    seccion("1. AVANCES — exclusión año 2026 y diagnóstico")
    informe = {}

    n_total = con.execute("SELECT COUNT(*) FROM avances").fetchone()[0]
    n_2026  = con.execute(
        "SELECT COUNT(*) FROM avances WHERE periodo_anio = 2026"
    ).fetchone()[0]
    n_otros = con.execute(
        "SELECT COUNT(*) FROM avances WHERE periodo_anio > 2025"
    ).fetchone()[0]

    log.info(f"Filas totales en avances : {n_total:,}")
    log.info(f"Filas con anio=2026      : {n_2026:,}  → se EXCLUIRÁN del modelado")
    log.info(f"Filas con anio>2025      : {n_otros:,}")
    log.info("  (se conservan en DuckDB; se filtran al construir el dataset)")

    # Nulos en columnas clave por nivel
    for nivel in ['nacional', 'ccaa', 'provincia']:
        n, n_sup, n_prod = con.execute(f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN sup_avance  IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN prod_avance IS NULL THEN 1 ELSE 0 END)
            FROM avances WHERE nivel='{nivel}'
        """).fetchone()
        pct_sup  = n_sup  / n * 100 if n else 0
        pct_prod = n_prod / n * 100 if n else 0
        log.info(
            f"  nivel={nivel:<10} {n:>8,} filas | "
            f"sup_avance nulos={pct_sup:.1f}% | prod_avance nulos={pct_prod:.1f}%"
        )

    informe["avances_filas_2026_excluidas"] = int(n_2026)
    return informe


# ══════════════════════════════════════════════════════════════════
# 2. CLIMA — imputación y descarte de columnas inútiles
# ══════════════════════════════════════════════════════════════════

# Columnas con 100% nulos o >85% nulos → descartar
COLS_CLIMA_DESCARTAR = ["t_max_abs", "t_min_abs", "horas_sol", "dias_precip"]

# Columnas con nulos moderados → imputar con mediana por provincia × mes
COLS_CLIMA_IMPUTAR = ["t_media", "t_max_med", "t_min_med", "precip_mm", "hr_media"]

def procesar_clima(con) -> dict:
    seccion("2. CLIMA_MENSUAL — descarte e imputación")
    informe = {}

    n_total = con.execute("SELECT COUNT(*) FROM clima_mensual").fetchone()[0]

    # Diagnóstico previo
    log.info("Porcentaje de nulos por columna climática:")
    for col in COLS_CLIMA_DESCARTAR + COLS_CLIMA_IMPUTAR:
        n_nulos = con.execute(
            f"SELECT SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) FROM clima_mensual"
        ).fetchone()[0]
        pct = n_nulos / n_total * 100
        accion = "DESCARTADA" if col in COLS_CLIMA_DESCARTAR else "imputar con mediana por prov×mes"
        log.info(f"  {col:<15}: {pct:5.1f}% nulos → {accion}")

    # Imputación: mediana por (cod_provincia, mes)
    log.info("")
    log.info("Imputando con mediana por provincia × mes...")
    total_imputados = 0

    for col in COLS_CLIMA_IMPUTAR:
        # Calcular medianas de referencia
        medianas = con.execute(f"""
            SELECT cod_provincia, mes,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}) AS mediana
            FROM clima_mensual
            WHERE {col} IS NOT NULL AND cod_provincia IS NOT NULL
            GROUP BY cod_provincia, mes
        """).df()

        if medianas.empty:
            log.info(f"  {col}: sin datos para calcular mediana — omitida")
            continue

        # Aplicar imputación en Python (más seguro que UPDATE correlacionado en DuckDB)
        df_col = con.execute(f"""
            SELECT rowid, cod_provincia, mes, {col}
            FROM clima_mensual
            WHERE {col} IS NULL AND cod_provincia IS NOT NULL
        """).df()

        if df_col.empty:
            log.info(f"  {col}: sin nulos que imputar ✅")
            continue

        df_merged = df_col.merge(medianas, on=["cod_provincia", "mes"], how="left")
        df_to_update = df_merged[df_merged["mediana"].notna()]

        if df_to_update.empty:
            log.info(f"  {col}: sin medianas de referencia disponibles")
            continue

        # Insertar en una tabla temporal y hacer UPDATE desde ella
        con.execute("DROP TABLE IF EXISTS _tmp_impute")
        con.execute(
            "CREATE TEMP TABLE _tmp_impute (rowid BIGINT, valor DOUBLE)"
        )
        # Registrar el dataframe como relación DuckDB
        tmp_df = df_to_update[["rowid", "mediana"]].rename(columns={"mediana": "valor"})
        con.execute("INSERT INTO _tmp_impute SELECT * FROM tmp_df")
        con.execute(f"""
            UPDATE clima_mensual
            SET {col} = (SELECT valor FROM _tmp_impute WHERE _tmp_impute.rowid = clima_mensual.rowid)
            WHERE rowid IN (SELECT rowid FROM _tmp_impute)
        """)
        n_imp = len(df_to_update)
        total_imputados += n_imp
        log.info(f"  {col}: {n_imp:,} valores imputados con mediana ✅")

    informe["clima_valores_imputados"] = total_imputados

    # Verificar estado final
    log.info("")
    log.info("Estado final tras imputación:")
    for col in COLS_CLIMA_IMPUTAR:
        n_nulos = con.execute(
            f"SELECT SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) FROM clima_mensual"
        ).fetchone()[0]
        pct = n_nulos / n_total * 100
        estado = "✅" if pct < 5 else "⚠️"
        log.info(f"  {estado} {col:<15}: {pct:.1f}% nulos restantes")

    return informe


# ══════════════════════════════════════════════════════════════════
# 3. DATASET LIMPIO — construcción final
# ══════════════════════════════════════════════════════════════════

def construir_dataset_limpio(con, out_dir: str) -> dict:
    seccion("3. CONSTRUCCIÓN DEL DATASET LIMPIO")
    informe = {}

    os.makedirs(out_dir, exist_ok=True)

    log.info("Construyendo cruce avances × clima_provincial (2014-2025, sin 2026)...")

    # Features climáticas disponibles tras el procesamiento
    # Se excluyen las columnas con 100% nulos: t_max_abs_c, t_min_abs_c, horas_sol_med
    df = con.execute("""
        SELECT
            a.cod_provincia,
            a.provincia,
            a.ccaa,
            a.cultivo,
            a.periodo_anio              AS anio,
            a.periodo_mes               AS mes,
            a.sup_avance,
            a.prod_avance,
            ROUND(a.prod_avance / NULLIF(a.sup_avance, 0), 6) AS rendimiento,
            -- Clima: solo columnas con datos reales
            cp.t_media_c,
            cp.t_max_abs_c,     -- mediana imputada en la vista
            cp.t_min_abs_c,
            cp.precip_media_mm,
            cp.hr_media_pct,
            cp.n_estaciones             AS estaciones_clima,
            -- Flag para provincias sin cobertura climática
            CASE WHEN cp.cod_provincia IS NULL THEN TRUE ELSE FALSE END AS sin_clima
        FROM avances a
        LEFT JOIN clima_provincial cp
               ON  a.cod_provincia = cp.cod_provincia
               AND a.periodo_anio  = cp.anio
               AND a.periodo_mes   = cp.mes
        WHERE a.nivel           = 'provincia'
          AND a.regimen         = 'total'
          AND a.sup_avance      > 0
          AND a.prod_avance     IS NOT NULL
          AND a.outlier_flag    = FALSE
          AND a.periodo_anio   BETWEEN 2014 AND 2025
    """).df()

    log.info(f"Filas en cruce (filtrado) : {len(df):,}")

    # Calcular rendimiento y filtrar filas sin él
    df = df[df["rendimiento"].notna() & (df["rendimiento"] > 0)].copy()
    log.info(f"Filas con rendimiento>0  : {len(df):,}")

    # ── Winsorización del target por cultivo al percentil 99 ──────
    log.info("Winsorizando rendimiento al p99 por cultivo...")
    df["rendimiento_raw"] = df["rendimiento"].copy()

    def winsorize_p99(s):
        p99 = s.quantile(0.99)
        return s.clip(upper=p99)

    df["rendimiento"] = df.groupby("cultivo")["rendimiento"].transform(winsorize_p99)
    n_wins = (df["rendimiento"] < df["rendimiento_raw"]).sum()
    log.info(f"  Valores winsorizados    : {n_wins:,} ({n_wins/len(df)*100:.2f}%)")
    informe["rendimiento_winsorizados"] = int(n_wins)

    # ── Diagnóstico de nulos en features climáticas ───────────────
    log.info("")
    log.info("Nulos en features climáticas tras imputación en clima_mensual:")
    cols_clima = ["t_media_c", "t_max_abs_c", "t_min_abs_c", "precip_media_mm", "hr_media_pct"]
    for col in cols_clima:
        if col in df.columns:
            n_nul = df[col].isna().sum()
            pct   = n_nul / len(df) * 100
            estado = "✅" if pct < 5 else ("⚠️" if pct < 30 else "❌")
            log.info(f"  {estado} {col:<20}: {n_nul:,} nulos ({pct:.1f}%)")

    n_sin_clima = df["sin_clima"].sum()
    log.info(f"Filas de provincias sin clima: {n_sin_clima:,} ({n_sin_clima/len(df)*100:.1f}%)")
    informe["filas_sin_clima"] = int(n_sin_clima)

    # ── Label Encoding ────────────────────────────────────────────
    log.info("")
    log.info("Aplicando Label Encoding...")
    le_cultivo   = LabelEncoder()
    le_provincia = LabelEncoder()
    le_ccaa      = LabelEncoder()

    df["cultivo_enc"]   = le_cultivo.fit_transform(df["cultivo"].fillna("DESCONOCIDO"))
    df["provincia_enc"] = le_provincia.fit_transform(df["provincia"].fillna("DESCONOCIDO"))
    df["ccaa_enc"]      = le_ccaa.fit_transform(df["ccaa"].fillna("DESCONOCIDO"))

    # Guardar mappings actualizados
    mappings = {
        "cultivo":   dict(zip(le_cultivo.classes_.tolist(),
                              [int(x) for x in range(len(le_cultivo.classes_))])),
        "provincia": dict(zip(le_provincia.classes_.tolist(),
                              [int(x) for x in range(len(le_provincia.classes_))])),
        "ccaa":      dict(zip(le_ccaa.classes_.tolist(),
                              [int(x) for x in range(len(le_ccaa.classes_))])),
    }
    enc_path = os.path.join(out_dir, "label_encoders.json")
    with open(enc_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    log.info(f"  Label encoders → {enc_path}")

    # ── Selección de features finales ─────────────────────────────
    # Se excluyen horas_sol_med, t_max_abs_c, t_min_abs_c (100% nulos en vista
    # porque t_max_abs y t_min_abs eran 100% nulos en clima_mensual — la vista
    # los agrega pero siguen vacíos)
    FEATURES = [
        "cultivo_enc", "provincia_enc", "ccaa_enc",
        "anio", "mes",
        "t_media_c",
        "precip_media_mm",
        "hr_media_pct",
    ]
    TARGET = "rendimiento"
    META   = ["cultivo", "provincia", "ccaa", "cod_provincia",
              "rendimiento_raw", "sin_clima", "sup_avance", "prod_avance"]

    # Verificación final de nulos en features seleccionadas
    log.info("")
    log.info("Verificación final de features seleccionadas:")
    for col in FEATURES:
        n_nul = df[col].isna().sum()
        pct   = n_nul / len(df) * 100
        estado = "✅" if pct == 0 else ("⚠️" if pct < 10 else "❌")
        log.info(f"  {estado} {col:<20}: {n_nul:,} nulos ({pct:.1f}%)")

    # Guardar dataset limpio
    df_final = df[FEATURES + [TARGET] + META].copy()
    pq_path = os.path.join(out_dir, "dataset_modelado_limpio.parquet")
    df_final.to_parquet(pq_path, index=False)

    log.info("")
    log.info(f"Dataset limpio guardado : {pq_path}")
    log.info(f"  Filas   : {len(df_final):,}")
    log.info(f"  Columnas: {df_final.shape[1]}")
    log.info(f"  Features: {FEATURES}")
    log.info(f"  Target  : {TARGET}")
    log.info(f"  Cultivos: {df['cultivo'].nunique()}")
    log.info(f"  Provincias: {df['cod_provincia'].nunique()}")
    log.info(f"  Periodo : {int(df['anio'].min())} – {int(df['anio'].max())}")

    informe.update({
        "features":                   FEATURES,
        "target":                     TARGET,
        "filas_dataset_limpio":       len(df_final),
        "cultivos_unicos":            int(df["cultivo"].nunique()),
        "provincias_cubiertas":       int(df["cod_provincia"].nunique()),
        "periodo_min":                int(df["anio"].min()),
        "periodo_max":                int(df["anio"].max()),
        "ruta_parquet_limpio":        pq_path,
        "ruta_label_encoders":        enc_path,
    })

    # Estadísticos del target
    rend = df_final[TARGET]
    log.info("")
    log.info("Estadísticos del target 'rendimiento' (miles t/ha):")
    log.info(f"  Min  : {rend.min():.4f}")
    log.info(f"  Media: {rend.mean():.4f}")
    log.info(f"  Mediana: {rend.median():.4f}")
    log.info(f"  Max  : {rend.max():.4f}")
    log.info(f"  Std  : {rend.std():.4f}")

    return informe


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AgroChat — Procesamiento de datos para modelado")
    ap.add_argument("--db",      default=os.environ.get("DUCKDB_PATH", "/app/data/duckdb/agrochat.duckdb"),
                    help="Ruta al DuckDB (default: $DUCKDB_PATH)")
    ap.add_argument("--out-dir", default="/app/data/processed",
                    help="Directorio de salida (default: /app/data/processed)")
    args = ap.parse_args()

    if not Path(args.db).exists():
        log.error(f"DuckDB no encontrado: {args.db}")
        sys.exit(1)

    log.info(f"Conectando a {args.db}")
    con = duckdb.connect(args.db, read_only=False)

    tablas = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    log.info(f"Tablas disponibles: {tablas}")

    # Verificar que validar_datos.py ya fue ejecutado (outlier_flag debe existir)
    cols_av = [r[0] for r in con.execute("DESCRIBE avances").fetchall()]
    if "outlier_flag" not in cols_av:
        log.error("Columna 'outlier_flag' no encontrada en avances.")
        log.error("Ejecuta primero: python scripts/validar_datos.py --db ...")
        sys.exit(1)

    informe = {}

    # 1. Diagnóstico avances + exclusión 2026
    informe.update(procesar_avances(con))

    # 2. Imputación climática
    if "clima_mensual" in tablas:
        informe.update(procesar_clima(con))
    else:
        log.warning("Tabla clima_mensual no encontrada — saltando imputación climática")

    # 3. Dataset limpio
    informe.update(construir_dataset_limpio(con, args.out_dir))

    # Resumen
    seccion("RESUMEN EJECUTIVO")
    log.info(f"Filas 2026 excluidas del modelado : {informe.get('avances_filas_2026_excluidas',0):,}")
    log.info(f"Valores climáticos imputados       : {informe.get('clima_valores_imputados',0):,}")
    log.info(f"Rendimientos winsorizados (p99)    : {informe.get('rendimiento_winsorizados',0):,}")
    log.info(f"Filas dataset limpio               : {informe.get('filas_dataset_limpio',0):,}")
    log.info(f"Features de entrenamiento          : {informe.get('features',[])}")
    log.info(f"Cultivos                           : {informe.get('cultivos_unicos',0)}")
    log.info(f"Provincias                         : {informe.get('provincias_cubiertas',0)}")
    log.info(f"Periodo                            : {informe.get('periodo_min')} – {informe.get('periodo_max')}")
    log.info("")
    log.info("Estado: ✅ Dataset listo para entrenamiento XGBoost/SHAP")

    # Guardar informe
    inf_path = os.path.join(args.out_dir, "informe_procesamiento.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(inf_path, "w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=2)
    log.info(f"Informe guardado: {inf_path}")

    con.close()


if __name__ == "__main__":
    main()
