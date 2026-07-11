"""
══════════════════════════════════════════════════════════════════
AgroChat — Validación, limpieza y transformación de datos
══════════════════════════════════════════════════════════════════
Script ejecutable equivalente a los scripts ETL anteriores.
Realiza todas las operaciones de calidad de datos sobre las
tablas ya cargadas en DuckDB y genera el dataset de modelado.

Uso:
  python scripts/validar_datos.py --db /app/data/duckdb/agrochat.duckdb

Salidas:
  - Tablas avances, coyuntura y clima_mensual limpiadas in-place
  - /app/data/processed/dataset_modelado.parquet
  - /app/data/processed/label_encoders.json
  - /app/data/processed/informe_calidad.json  (resumen legible por el agente)
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
log = logging.getLogger("validar_datos")


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def contar_nulos(con, tabla: str) -> dict:
    cols = [r[0] for r in con.execute(f"DESCRIBE {tabla}").fetchall()]
    null_q = ", ".join([
        f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}"
        for c in cols
    ])
    row = con.execute(f"SELECT {null_q} FROM {tabla}").fetchone()
    return {cols[i]: row[i] for i in range(len(cols)) if row[i] and row[i] > 0}


def seccion(titulo: str):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {titulo}")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════
# 1. AVANCES
# ══════════════════════════════════════════════════════════════════

def validar_avances(con) -> dict:
    seccion("1. TABLA avances")
    informe = {}

    n_total = con.execute("SELECT COUNT(*) FROM avances").fetchone()[0]
    log.info(f"Filas totales       : {n_total:,}")

    # Distribución por nivel
    for nivel, n in con.execute(
        "SELECT nivel, COUNT(*) FROM avances GROUP BY nivel ORDER BY 2 DESC"
    ).fetchall():
        log.info(f"  nivel={nivel:<12} {n:>8,} filas")

    # Rango temporal
    anio_min, anio_max, mes_min, mes_max = con.execute("""
        SELECT MIN(periodo_anio), MAX(periodo_anio),
               MIN(periodo_mes),  MAX(periodo_mes)
        FROM avances
    """).fetchone()
    log.info(f"Rango años          : {anio_min} – {anio_max}")
    log.info(f"Rango meses         : {mes_min} – {mes_max}")

    # ── Nulos ─────────────────────────────────────────────────────
    nulos = contar_nulos(con, "avances")
    if nulos:
        log.info(f"Columnas con nulos  : {list(nulos.keys())}")
    else:
        log.info("Columnas con nulos  : ninguna ✅")
    informe["nulos_avances"] = nulos

    # ── Meses/años fuera de rango ─────────────────────────────────
    n_mes_inv = con.execute(
        "SELECT COUNT(*) FROM avances WHERE periodo_mes < 1 OR periodo_mes > 12"
    ).fetchone()[0]
    n_anio_inv = con.execute(
        "SELECT COUNT(*) FROM avances WHERE periodo_anio < 2014 OR periodo_anio > 2025"
    ).fetchone()[0]
    log.info(f"Meses fuera [1-12]  : {n_mes_inv}")
    log.info(f"Años fuera [2014-2025]: {n_anio_inv}")

    # ── Valores negativos → NULL ──────────────────────────────────
    n_neg_sup = con.execute(
        "SELECT COUNT(*) FROM avances WHERE sup_avance < 0"
    ).fetchone()[0]
    n_neg_prod = con.execute(
        "SELECT COUNT(*) FROM avances WHERE prod_avance < 0"
    ).fetchone()[0]

    if n_neg_sup > 0:
        con.execute("UPDATE avances SET sup_avance = NULL WHERE sup_avance < 0")
        log.info(f"CORR: {n_neg_sup} sup_avance negativas → NULL ✅")
    else:
        log.info("sup_avance negativas: ninguna ✅")

    if n_neg_prod > 0:
        con.execute("UPDATE avances SET prod_avance = NULL WHERE prod_avance < 0")
        log.info(f"CORR: {n_neg_prod} prod_avance negativas → NULL ✅")
    else:
        log.info("prod_avance negativas: ninguna ✅")

    informe["negativos_corregidos"] = n_neg_sup + n_neg_prod

    # ── Outliers por cultivo (IQR × 3.0) → flag ──────────────────
    # Añadir columna flag si no existe
    cols_existentes = [r[0] for r in con.execute("DESCRIBE avances").fetchall()]
    if "outlier_flag" not in cols_existentes:
        con.execute("ALTER TABLE avances ADD COLUMN outlier_flag BOOLEAN DEFAULT FALSE")
        log.info("Columna outlier_flag añadida a avances")
    else:
        # Resetear flags previos para recalcular limpio
        con.execute("UPDATE avances SET outlier_flag = FALSE")

    # Calcular outliers en Python (DuckDB no soporta bien ventanas en UPDATE)
    df_nac = con.execute("""
        SELECT rowid, cultivo, sup_avance
        FROM avances
        WHERE nivel = 'nacional' AND regimen = 'total'
          AND sup_avance IS NOT NULL
    """).df()

    def iqr_outlier_mask(s, factor=3.0):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - factor * iqr) | (s > q3 + factor * iqr)

    df_nac["es_outlier"] = df_nac.groupby("cultivo")["sup_avance"].transform(iqr_outlier_mask)
    outlier_rowids = df_nac[df_nac["es_outlier"]]["rowid"].tolist()

    if outlier_rowids:
        ids_str = ", ".join(str(r) for r in outlier_rowids)
        con.execute(f"UPDATE avances SET outlier_flag = TRUE WHERE rowid IN ({ids_str})")
        log.info(f"Outliers marcados (IQR×3): {len(outlier_rowids):,} filas con outlier_flag=TRUE")
        log.info("  (se conservan; son anomalías con valor analítico)")
    else:
        log.info("Outliers IQR×3      : ninguno detectado ✅")

    informe["outliers_flagged_avances"] = len(outlier_rowids)
    return informe


# ══════════════════════════════════════════════════════════════════
# 2. COYUNTURA
# ══════════════════════════════════════════════════════════════════

def validar_coyuntura(con) -> dict:
    seccion("2. TABLA coyuntura")
    informe = {}

    n_total = con.execute("SELECT COUNT(*) FROM coyuntura").fetchone()[0]
    log.info(f"Filas totales       : {n_total:,}")

    anio_min, anio_max, sem_min, sem_max, n_prod = con.execute("""
        SELECT MIN(anio), MAX(anio), MIN(semana), MAX(semana),
               COUNT(DISTINCT producto)
        FROM coyuntura
    """).fetchone()
    log.info(f"Rango años          : {anio_min} – {anio_max}")
    log.info(f"Rango semanas       : {sem_min} – {sem_max}")
    log.info(f"Productos únicos    : {n_prod}")

    # Nulos
    nulos = contar_nulos(con, "coyuntura")
    if nulos:
        log.info(f"Columnas con nulos  : {nulos}")
    else:
        log.info("Columnas con nulos  : ninguna ✅")
    informe["nulos_coyuntura"] = nulos

    # Semanas inválidas
    n_sem_inv = con.execute(
        "SELECT COUNT(*) FROM coyuntura WHERE semana < 1 OR semana > 53"
    ).fetchone()[0]
    log.info(f"Semanas fuera [1-53]: {n_sem_inv}")

    # Precios inválidos (<= 0)
    n_precio_inv = con.execute(
        "SELECT COUNT(*) FROM coyuntura WHERE precio_sem_actual <= 0"
    ).fetchone()[0]
    if n_precio_inv > 0:
        con.execute(
            "UPDATE coyuntura SET precio_sem_actual = NULL WHERE precio_sem_actual <= 0"
        )
        log.info(f"CORR: {n_precio_inv} precios <=0 → NULL ✅")
    else:
        log.info("Precios inválidos (<=0): ninguno ✅")

    informe["precios_invalidos_corregidos"] = n_precio_inv

    # Recalcular variacion_pct para coherencia interna
    con.execute("""
        UPDATE coyuntura
        SET variacion_pct = ROUND(
            (precio_sem_actual - precio_sem_anterior)
            / NULLIF(precio_sem_anterior, 0) * 100, 2
        )
        WHERE precio_sem_actual IS NOT NULL
          AND precio_sem_anterior IS NOT NULL
    """)
    log.info("variacion_pct recalculada para coherencia interna ✅")

    # Variaciones extremas (diagnóstico, no se corrigen — pueden ser reales)
    n_extremas = con.execute("""
        SELECT COUNT(*) FROM coyuntura WHERE ABS(variacion_pct) > 50
    """).fetchone()[0]
    log.info(f"Variaciones >50%    : {n_extremas} (registradas, no corregidas)")
    informe["variaciones_extremas_pct50"] = n_extremas

    return informe


# ══════════════════════════════════════════════════════════════════
# 3. CLIMA_MENSUAL
# ══════════════════════════════════════════════════════════════════

LIMITES_FISICOS = {
    "t_media":    (-20.0, 45.0),
    "t_max_abs":  (-15.0, 50.0),
    "t_min_abs":  (-30.0, 40.0),
    "t_max_med":  (-20.0, 48.0),
    "t_min_med":  (-25.0, 42.0),
    "precip_mm":  (0.0,   1000.0),
    "dias_precip":(0.0,   31.0),
    "horas_sol":  (0.0,   400.0),
    "hr_media":   (0.0,   100.0),
}

def validar_clima(con) -> dict:
    seccion("3. TABLA clima_mensual")
    informe = {}

    n_total = con.execute("SELECT COUNT(*) FROM clima_mensual").fetchone()[0]
    log.info(f"Filas totales       : {n_total:,}")

    est, prov, anio_min, anio_max = con.execute("""
        SELECT COUNT(DISTINCT indicativo),
               COUNT(DISTINCT cod_provincia),
               MIN(anio), MAX(anio)
        FROM clima_mensual
    """).fetchone()
    log.info(f"Estaciones          : {est}")
    log.info(f"Provincias cubiertas: {prov}")
    log.info(f"Rango años          : {anio_min} – {anio_max}")

    # Nulos por variable climática
    nulos = contar_nulos(con, "clima_mensual")
    log.info("Nulos por columna:")
    for col, n in nulos.items():
        pct = n / n_total * 100
        log.info(f"  {col:<15}: {n:>8,} ({pct:.1f}%)")
    informe["nulos_clima"] = {k: int(v) for k, v in nulos.items()}

    # Validación contra límites físicos → NULL
    total_corr = 0
    for col, (lo, hi) in LIMITES_FISICOS.items():
        n_out = con.execute(f"""
            SELECT COUNT(*) FROM clima_mensual
            WHERE {col} IS NOT NULL AND ({col} < {lo} OR {col} > {hi})
        """).fetchone()[0]
        if n_out > 0:
            con.execute(f"""
                UPDATE clima_mensual
                SET {col} = NULL
                WHERE {col} < {lo} OR {col} > {hi}
            """)
            log.info(f"CORR: {col:<15} {n_out:>5} valores fuera [{lo},{hi}] → NULL ✅")
            total_corr += n_out
        else:
            log.info(f"  {col:<15}: sin valores fuera de rango ✅")

    informe["outliers_fisicos_corregidos_clima"] = total_corr

    # Provincias con avances pero sin clima (cobertura incompleta)
    sin_clima = con.execute("""
        SELECT COUNT(DISTINCT a.cod_provincia)
        FROM avances a
        WHERE a.nivel = 'provincia'
          AND a.cod_provincia IS NOT NULL
          AND a.cod_provincia NOT IN (
              SELECT DISTINCT cod_provincia FROM clima_mensual
              WHERE cod_provincia IS NOT NULL
          )
    """).fetchone()[0]
    log.info(f"Provincias sin clima: {sin_clima} (cruce avances × clima incompleto)")
    informe["provincias_sin_cobertura_climatica"] = sin_clima

    return informe


# ══════════════════════════════════════════════════════════════════
# 4. DATASET DE MODELADO (avances × clima → Parquet)
# ══════════════════════════════════════════════════════════════════

def generar_dataset_modelado(con, out_dir: str) -> dict:
    seccion("4. DATASET DE MODELADO")
    informe = {}

    os.makedirs(out_dir, exist_ok=True)

    # Cruce avances × clima_provincial con rendimiento calculado
    log.info("Construyendo cruce avances × clima_provincial...")
    df = con.execute("""
        SELECT
            a.cod_provincia,
            a.provincia,
            a.ccaa,
            a.cultivo,
            a.periodo_anio          AS anio,
            a.periodo_mes           AS mes,
            a.sup_avance,
            a.prod_avance,
            CASE
                WHEN a.sup_avance > 0 AND a.prod_avance IS NOT NULL
                THEN ROUND(a.prod_avance / a.sup_avance, 6)
                ELSE NULL
            END                     AS rendimiento,
            cp.t_media_c,
            cp.precip_media_mm,
            cp.horas_sol_med,
            cp.hr_media_pct,
            cp.t_max_abs_c,
            cp.t_min_abs_c,
            cp.n_estaciones         AS estaciones_clima
        FROM avances a
        LEFT JOIN clima_provincial cp
               ON  a.cod_provincia = cp.cod_provincia
               AND a.periodo_anio  = cp.anio
               AND a.periodo_mes   = cp.mes
        WHERE a.nivel      = 'provincia'
          AND a.regimen    = 'total'
          AND a.sup_avance > 0
          AND a.outlier_flag = FALSE
    """).df()

    log.info(f"Filas en cruce      : {len(df):,}")
    n_con_rend = df["rendimiento"].notna().sum()
    n_con_clima = df["t_media_c"].notna().sum()
    log.info(f"  Con rendimiento   : {n_con_rend:,} ({n_con_rend/len(df)*100:.1f}%)")
    log.info(f"  Con datos clima   : {n_con_clima:,} ({n_con_clima/len(df)*100:.1f}%)")
    log.info(f"  Cultivos únicos   : {df['cultivo'].nunique()}")
    log.info(f"  Provincias        : {df['cod_provincia'].nunique()}")

    # Label Encoding (XGBoost no necesita one-hot; LabelEncoder es suficiente)
    log.info("Aplicando Label Encoding (cultivo, provincia, ccaa)...")
    df_enc = df[df["rendimiento"].notna()].copy()

    le_cultivo   = LabelEncoder()
    le_provincia = LabelEncoder()
    le_ccaa      = LabelEncoder()

    df_enc["cultivo_enc"]   = le_cultivo.fit_transform(
        df_enc["cultivo"].fillna("DESCONOCIDO")
    )
    df_enc["provincia_enc"] = le_provincia.fit_transform(
        df_enc["provincia"].fillna("DESCONOCIDO")
    )
    df_enc["ccaa_enc"]      = le_ccaa.fit_transform(
        df_enc["ccaa"].fillna("DESCONOCIDO")
    )

    # Guardar mappings para interpretabilidad en el agente y el dashboard
    mappings = {
        "cultivo":   dict(zip(
            le_cultivo.classes_.tolist(),
            [int(x) for x in range(len(le_cultivo.classes_))]
        )),
        "provincia": dict(zip(
            le_provincia.classes_.tolist(),
            [int(x) for x in range(len(le_provincia.classes_))]
        )),
        "ccaa":      dict(zip(
            le_ccaa.classes_.tolist(),
            [int(x) for x in range(len(le_ccaa.classes_))]
        )),
    }
    enc_path = os.path.join(out_dir, "label_encoders.json")
    with open(enc_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    log.info(f"Label encoders guardados en {enc_path}")

    # Features y target
    FEATURES = [
        "cultivo_enc", "provincia_enc", "ccaa_enc",
        "anio", "mes",
        "t_media_c", "precip_media_mm", "horas_sol_med",
        "hr_media_pct", "t_max_abs_c", "t_min_abs_c",
    ]
    TARGET = "rendimiento"
    META   = ["cultivo", "provincia", "ccaa", "cod_provincia"]

    df_final = df_enc[FEATURES + [TARGET] + META]

    # Nulos en features (diagnóstico pre-modelado)
    nulos_features = df_final[FEATURES].isnull().sum()
    nulos_features = nulos_features[nulos_features > 0]
    if not nulos_features.empty:
        log.info("Nulos en features (serán manejados por XGBoost natively):")
        for col, n in nulos_features.items():
            log.info(f"  {col:<20}: {n:,} ({n/len(df_final)*100:.1f}%)")
    else:
        log.info("Nulos en features   : ninguno ✅")

    # Guardar Parquet
    pq_path = os.path.join(out_dir, "dataset_modelado.parquet")
    df_final.to_parquet(pq_path, index=False)
    log.info(f"Dataset guardado    : {pq_path}")
    log.info(f"  Filas: {len(df_final):,}  |  Columnas: {len(df_final.columns)}")

    informe.update({
        "filas_cruce_total":         len(df),
        "filas_con_rendimiento":     int(n_con_rend),
        "filas_con_clima":           int(n_con_clima),
        "filas_dataset_modelado":    len(df_final),
        "cultivos_unicos":           int(df["cultivo"].nunique()),
        "provincias_cubiertas":      int(df["cod_provincia"].nunique()),
        "features":                  FEATURES,
        "target":                    TARGET,
        "ruta_parquet":              pq_path,
        "ruta_label_encoders":       enc_path,
    })
    return informe


# ══════════════════════════════════════════════════════════════════
# 5. INFORME FINAL
# ══════════════════════════════════════════════════════════════════

def imprimir_resumen(informe_total: dict):
    seccion("RESUMEN EJECUTIVO")

    log.info(f"avances — negativos corregidos  : {informe_total.get('negativos_corregidos', 0)}")
    log.info(f"avances — outliers flagged      : {informe_total.get('outliers_flagged_avances', 0)}")
    log.info(f"coyuntura — precios corregidos  : {informe_total.get('precios_invalidos_corregidos', 0)}")
    log.info(f"coyuntura — variaciones >50%    : {informe_total.get('variaciones_extremas_pct50', 0)}")
    log.info(f"clima — fuera límites físicos   : {informe_total.get('outliers_fisicos_corregidos_clima', 0)}")
    log.info(f"clima — provincias sin cobertura: {informe_total.get('provincias_sin_cobertura_climatica', 0)}")
    log.info("")
    log.info(f"Dataset modelado — filas        : {informe_total.get('filas_dataset_modelado', 0):,}")
    log.info(f"Dataset modelado — ruta         : {informe_total.get('ruta_parquet', '-')}")
    log.info("")
    log.info("Estado: ✅ Datos listos para agente RAG y modelo XGBoost/SHAP")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AgroChat — Validación y limpieza de datos")
    ap.add_argument("--db",      required=True, help="Ruta al DuckDB")
    ap.add_argument("--out-dir", default="/app/data/processed",
                    help="Directorio de salida para Parquet y JSON")
    args = ap.parse_args()

    if not Path(args.db).exists():
        log.error(f"DuckDB no encontrado: {args.db}")
        sys.exit(1)

    log.info(f"Conectando a {args.db}")
    con = duckdb.connect(args.db, read_only=False)

    tablas = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    log.info(f"Tablas en BD: {tablas}")

    informe_total = {}

    if "avances" in tablas:
        informe_total.update(validar_avances(con))
    else:
        log.warning("Tabla 'avances' no encontrada — saltando")

    if "coyuntura" in tablas:
        informe_total.update(validar_coyuntura(con))
    else:
        log.warning("Tabla 'coyuntura' no encontrada — saltando")

    if "clima_mensual" in tablas:
        informe_total.update(validar_clima(con))
    else:
        log.warning("Tabla 'clima_mensual' no encontrada — saltando")

    if "avances" in tablas:
        informe_total.update(generar_dataset_modelado(con, args.out_dir))

    imprimir_resumen(informe_total)

    # Guardar informe JSON (legible por el agente RAG)
    informe_path = os.path.join(args.out_dir, "informe_calidad.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(informe_path, "w", encoding="utf-8") as f:
        json.dump(informe_total, f, ensure_ascii=False, indent=2)
    log.info(f"Informe guardado    : {informe_path}")

    con.close()


if __name__ == "__main__":
    main()
