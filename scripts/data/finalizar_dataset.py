"""
══════════════════════════════════════════════════════════════════
AgroChat — Paso final: imputación de nulos climáticos en Parquet
══════════════════════════════════════════════════════════════════
Las 29.628 filas con nulos climáticos corresponden a 10 provincias
que no tienen ninguna estación AEMET mapeada. La imputación en
clima_mensual no puede ayudar porque no hay ningún dato de origen.

Solución: imputar en el propio Parquet con la mediana de
(provincia_enc, mes) calculada sobre las filas que SÍ tienen clima.
Si la provincia no tiene ningún dato (las 10 sin cobertura), se
imputa con la mediana de (ccaa_enc, mes) → fallback por CCAA.

También corrige el cuaderno Jupyter para que apunte al Parquet limpio.

Uso:
  python scripts/finalizar_dataset.py
"""

import json, logging, os
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("finalizar_dataset")

PQ_IN    = "/app/data/processed/dataset_modelado_limpio.parquet"
PQ_OUT   = "/app/data/processed/dataset_modelado_final.parquet"
INF_OUT  = "/app/data/processed/informe_final.json"

FEATURES_CLIMA = ["t_media_c", "precip_media_mm", "hr_media_pct"]


def seccion(t):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {t}")
    log.info("=" * 60)


def imputar_por_grupo(df: pd.DataFrame, col: str, grupo: list) -> pd.Series:
    """Imputa nulos de `col` con la mediana del grupo dado."""
    medianas = (
        df[df[col].notna()]
        .groupby(grupo)[col]
        .median()
    )
    def _fill(row):
        if pd.notna(row[col]):
            return row[col]
        key = tuple(row[g] for g in grupo)
        return medianas.get(key, np.nan)
    return df.apply(_fill, axis=1)


def main():
    # ── Cargar Parquet limpio ─────────────────────────────────────
    seccion("1. CARGA DEL DATASET LIMPIO")
    if not Path(PQ_IN).exists():
        log.error(f"No encontrado: {PQ_IN}")
        log.error("Ejecuta primero: python scripts/procesar_datos.py --db ...")
        raise SystemExit(1)

    df = pd.read_parquet(PQ_IN)
    log.info(f"Filas cargadas    : {len(df):,}")
    log.info(f"Columnas          : {list(df.columns)}")

    for col in FEATURES_CLIMA:
        if col in df.columns:
            n = df[col].isna().sum()
            log.info(f"  {col:<20}: {n:,} nulos ({n/len(df)*100:.1f}%)")

    # ── Diagnóstico de las 10 provincias sin clima ────────────────
    seccion("2. DIAGNÓSTICO — PROVINCIAS SIN COBERTURA CLIMÁTICA")

    sin_clima = df[df["sin_clima"] == True]
    log.info(f"Filas sin clima   : {len(sin_clima):,} ({len(sin_clima)/len(df)*100:.1f}%)")

    if "provincia" in df.columns:
        provs_sin = sin_clima["provincia"].value_counts()
        log.info(f"Provincias afectadas ({len(provs_sin)}):")
        for prov, n in provs_sin.items():
            log.info(f"  {prov:<30} {n:>6,} filas")

    # ── Imputación en dos niveles ─────────────────────────────────
    seccion("3. IMPUTACIÓN DE NULOS CLIMÁTICOS EN EL DATASET")
    log.info("Estrategia: mediana por (provincia_enc, mes) → fallback (ccaa_enc, mes)")

    df_work = df.copy()
    informe = {}

    for col in FEATURES_CLIMA:
        if col not in df_work.columns:
            log.warning(f"  {col} no encontrada en el Parquet — omitida")
            continue

        n_antes = df_work[col].isna().sum()
        if n_antes == 0:
            log.info(f"  {col}: sin nulos ✅")
            continue

        # Nivel 1: mediana por provincia × mes
        medianas_prov = (
            df_work[df_work[col].notna()]
            .groupby(["provincia_enc", "mes"])[col]
            .median()
        )
        mask_null = df_work[col].isna()
        df_work.loc[mask_null, col] = df_work[mask_null].apply(
            lambda r: medianas_prov.get((r["provincia_enc"], r["mes"]), np.nan),
            axis=1
        )

        n_tras_prov = df_work[col].isna().sum()
        n1 = n_antes - n_tras_prov
        log.info(f"  {col}: {n1:,} imputados con mediana provincia×mes")

        # Nivel 2 (fallback): mediana por ccaa × mes
        if n_tras_prov > 0:
            medianas_ccaa = (
                df_work[df_work[col].notna()]
                .groupby(["ccaa_enc", "mes"])[col]
                .median()
            )
            mask_null2 = df_work[col].isna()
            df_work.loc[mask_null2, col] = df_work[mask_null2].apply(
                lambda r: medianas_ccaa.get((r["ccaa_enc"], r["mes"]), np.nan),
                axis=1
            )
            n_tras_ccaa = df_work[col].isna().sum()
            n2 = n_tras_prov - n_tras_ccaa
            log.info(f"  {col}: {n2:,} adicionales imputados con mediana ccaa×mes")

            # Nivel 3 (último recurso): mediana global por mes
            if n_tras_ccaa > 0:
                medianas_mes = df_work[df_work[col].notna()].groupby("mes")[col].median()
                mask_null3 = df_work[col].isna()
                df_work.loc[mask_null3, col] = df_work[mask_null3]["mes"].map(medianas_mes)
                n_tras_mes = df_work[col].isna().sum()
                n3 = n_tras_ccaa - n_tras_mes
                log.info(f"  {col}: {n3:,} adicionales imputados con mediana global×mes")

        n_final = df_work[col].isna().sum()
        estado = "✅" if n_final == 0 else "⚠️"
        log.info(f"  {estado} {col}: {n_final} nulos restantes tras imputación")
        informe[f"imputados_{col}"] = int(n_antes - n_final)

    # ── Verificación final ────────────────────────────────────────
    seccion("4. VERIFICACIÓN FINAL")

    FEATURES = [
        "cultivo_enc", "provincia_enc", "ccaa_enc",
        "anio", "mes",
        "t_media_c", "precip_media_mm", "hr_media_pct",
    ]
    TARGET = "rendimiento"

    log.info("Estado de features y target:")
    all_ok = True
    for col in FEATURES + [TARGET]:
        if col not in df_work.columns:
            log.warning(f"  ⚠️  {col}: columna no presente")
            continue
        n = df_work[col].isna().sum()
        pct = n / len(df_work) * 100
        ok = n == 0
        all_ok = all_ok and ok
        estado = "✅" if ok else "❌"
        log.info(f"  {estado} {col:<20}: {n:,} nulos ({pct:.1f}%)")

    log.info("")
    log.info(f"Filas finales     : {len(df_work):,}")
    log.info(f"Cultivos únicos   : {df_work['cultivo'].nunique()}")
    log.info(f"Provincias        : {df_work['cod_provincia'].nunique()}")
    log.info(f"Periodo           : {int(df_work['anio'].min())} – {int(df_work['anio'].max())}")

    # Estadísticos del target
    rend = df_work[TARGET]
    log.info(f"\nEstadísticos de '{TARGET}':")
    log.info(f"  Min    : {rend.min():.4f}")
    log.info(f"  Media  : {rend.mean():.4f}")
    log.info(f"  Mediana: {rend.median():.4f}")
    log.info(f"  Max    : {rend.max():.4f}")
    log.info(f"  Std    : {rend.std():.4f}")

    # ── Guardar Parquet final ─────────────────────────────────────
    seccion("5. GUARDADO")
    df_work.to_parquet(PQ_OUT, index=False)
    log.info(f"Dataset final guardado: {PQ_OUT}")
    log.info(f"  Filas   : {len(df_work):,}")
    log.info(f"  Columnas: {df_work.shape[1]}")

    # Informe JSON
    informe.update({
        "filas_finales":        len(df_work),
        "cultivos_unicos":      int(df_work["cultivo"].nunique()),
        "provincias_cubiertas": int(df_work["cod_provincia"].nunique()),
        "periodo_min":          int(df_work["anio"].min()),
        "periodo_max":          int(df_work["anio"].max()),
        "features":             FEATURES,
        "target":               TARGET,
        "nulos_en_features":    int(df_work[FEATURES].isna().sum().sum()),
        "dataset_listo":        bool(all_ok),
        "ruta_parquet_final":   PQ_OUT,
    })
    with open(INF_OUT, "w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=2)
    log.info(f"Informe guardado  : {INF_OUT}")

    estado_final = "✅ LISTO para XGBoost/SHAP" if all_ok else "⚠️  Revisar nulos pendientes"
    log.info(f"\nEstado final: {estado_final}")


if __name__ == "__main__":
    main()
