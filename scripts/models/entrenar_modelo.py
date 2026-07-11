"""
══════════════════════════════════════════════════════════════════
AgroChat — Entrenamiento XGBoost + cálculo de SHAP values
══════════════════════════════════════════════════════════════════
Entrena un modelo XGBoost para predecir el rendimiento agrícola
(miles de t/ha) a partir de variables climáticas y de cultivo,
calcula los valores SHAP para explicabilidad y guarda los artefactos
necesarios para el dashboard y el agente.

Entradas:
  /app/data/processed/dataset_modelado_final.parquet
  /app/data/processed/label_encoders.json

Salidas:
  /app/data/models/xgboost_rendimiento.json     ← modelo entrenado
  /app/data/models/shap_values.parquet          ← SHAP values (una fila por instancia)
  /app/data/models/shap_summary.parquet         ← importancia media |SHAP| por feature
  /app/data/models/metricas_modelo.json         ← R², RMSE, MAE (train/val/test)
  /app/data/models/feature_names.json           ← lista de features en orden

Uso:
  python scripts/entrenar_modelo.py
  python scripts/entrenar_modelo.py --test-size 0.15 --n-estimators 500
"""

import argparse, json, logging, os, time
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("entrenar_modelo")

PQ_PATH   = "/app/data/processed/dataset_modelado_final.parquet"
ENC_PATH  = "/app/data/processed/label_encoders.json"
OUT_DIR   = "/app/data/models"

FEATURES = [
    "cultivo_enc", "provincia_enc", "ccaa_enc",
    "anio", "mes",
    "t_media_c", "precip_media_mm", "hr_media_pct",
]
TARGET = "rendimiento"


def seccion(t):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {t}")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════
# 1. CARGA Y SPLITS
# ══════════════════════════════════════════════════════════════════

def cargar_datos(test_size: float, val_size: float, seed: int, pq_path: str = PQ_PATH):
    seccion("1. CARGA DEL DATASET")

    if not Path(pq_path).exists():
        raise FileNotFoundError(f"Parquet no encontrado: {pq_path}\n"
                                "Ejecuta primero: python scripts/finalizar_dataset.py")

    df = pd.read_parquet(pq_path)
    log.info(f"Filas cargadas    : {len(df):,}")
    log.info(f"Features          : {FEATURES}")
    log.info(f"Target            : {TARGET}")

    # Verificar 0 nulos
    n_nulos = df[FEATURES + [TARGET]].isnull().sum().sum()
    if n_nulos > 0:
        raise ValueError(f"Dataset con {n_nulos} nulos — ejecuta finalizar_dataset.py")
    log.info("Nulos en features : 0 ✅")

    X = df[FEATURES].values
    y = df[TARGET].values

    # Split estratificado por cultivo para representación uniforme
    # Train / Val / Test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_relative, random_state=seed
    )

    log.info(f"Split train/val/test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    log.info(f"  Train: {len(X_train)/len(X)*100:.1f}%  "
             f"Val: {len(X_val)/len(X)*100:.1f}%  "
             f"Test: {len(X_test)/len(X)*100:.1f}%")
    return X_train, X_val, X_test, y_train, y_val, y_test, df


# ══════════════════════════════════════════════════════════════════
# 2. ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════

def entrenar(X_train, X_val, y_train, y_val,
             n_estimators: int, max_depth: int,
             learning_rate: float, seed: int) -> xgb.XGBRegressor:
    seccion("2. ENTRENAMIENTO XGBoost")

    # Parámetros optimizados para series temporales agrarias:
    # - max_depth moderado (5) para evitar sobreajuste en cultivos con pocos datos
    # - subsample y colsample_bytree para regularización
    # - enable_categorical=False porque ya usamos Label Encoding
    params = {
        "n_estimators":       n_estimators,
        "max_depth":          max_depth,
        "learning_rate":      learning_rate,
        "subsample":          0.8,
        "colsample_bytree":   0.8,
        "min_child_weight":   5,
        "reg_alpha":          0.1,    # L1
        "reg_lambda":         1.0,    # L2
        "objective":          "reg:squarederror",
        "tree_method":        "hist", # rápido en CPU
        "random_state":       seed,
        "n_jobs":             -1,
        "early_stopping_rounds": 30,
        "eval_metric":        "rmse",
    }

    log.info("Parámetros:")
    for k, v in params.items():
        log.info(f"  {k:<25}: {v}")

    model = xgb.XGBRegressor(**params)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    elapsed = time.time() - t0

    log.info(f"\nEntrenamiento completado en {elapsed:.1f}s")
    log.info(f"Mejor iteración   : {model.best_iteration}")
    log.info(f"Mejor RMSE (val)  : {model.best_score:.6f}")

    return model


# ══════════════════════════════════════════════════════════════════
# 3. EVALUACIÓN
# ══════════════════════════════════════════════════════════════════

def evaluar(model, X_train, X_val, X_test,
            y_train, y_val, y_test) -> dict:
    seccion("3. EVALUACIÓN DEL MODELO")

    metricas = {}
    for nombre, X, y in [("train", X_train, y_train),
                          ("val",   X_val,   y_val),
                          ("test",  X_test,  y_test)]:
        y_pred = model.predict(X)
        r2   = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae  = mean_absolute_error(y, y_pred)

        log.info(f"  {nombre.upper():<6}  R²={r2:.4f}  RMSE={rmse:.6f}  MAE={mae:.6f}")
        metricas[nombre] = {
            "r2":   round(float(r2),   4),
            "rmse": round(float(rmse), 6),
            "mae":  round(float(mae),  6),
            "n":    int(len(y)),
        }

    # Overfitting check
    gap_r2 = metricas["train"]["r2"] - metricas["test"]["r2"]
    log.info(f"\n  Gap R² (train-test): {gap_r2:.4f} "
             f"{'⚠️ posible sobreajuste' if gap_r2 > 0.15 else '✅ aceptable'}")

    return metricas


# ══════════════════════════════════════════════════════════════════
# 4. SHAP VALUES
# ══════════════════════════════════════════════════════════════════

def calcular_shap(model, X_test, df, test_idx) -> tuple:
    seccion("4. CÁLCULO DE SHAP VALUES")

    log.info(f"Calculando SHAP sobre {len(X_test):,} instancias de test...")
    t0 = time.time()

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    elapsed = time.time() - t0
    log.info(f"SHAP calculado en {elapsed:.1f}s")

    # DataFrame de SHAP values con nombres de feature
    df_shap = pd.DataFrame(
        shap_values,
        columns=[f"shap_{f}" for f in FEATURES]
    )

    # Añadir metadatos para el dashboard
    meta_cols = ["cultivo", "provincia", "ccaa", "cod_provincia",
                 "anio", "mes", "rendimiento", "rendimiento_raw"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    df_meta = df.iloc[test_idx][meta_cols].reset_index(drop=True)
    df_shap = pd.concat([df_meta, df_shap], axis=1)

    # Importancia media por feature: mean(|SHAP|)
    shap_abs_mean = pd.DataFrame({
        "feature":    FEATURES,
        "shap_mean_abs": np.abs(shap_values).mean(axis=0),
        "shap_mean":     shap_values.mean(axis=0),
        "shap_std":      shap_values.std(axis=0),
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)

    log.info("\nImportancia de features (mean |SHAP|):")
    for _, row in shap_abs_mean.iterrows():
        bar = "█" * int(row["shap_mean_abs"] / shap_abs_mean["shap_mean_abs"].max() * 20)
        log.info(f"  {row['feature']:<20} {row['shap_mean_abs']:.6f}  {bar}")

    return df_shap, shap_abs_mean


# ══════════════════════════════════════════════════════════════════
# 5. GUARDADO DE ARTEFACTOS
# ══════════════════════════════════════════════════════════════════

def guardar(model, df_shap, shap_summary, metricas, out_dir: str):
    seccion("5. GUARDADO DE ARTEFACTOS")

    os.makedirs(out_dir, exist_ok=True)

    # Modelo
    model_path = os.path.join(out_dir, "xgboost_rendimiento.json")
    model.save_model(model_path)
    log.info(f"Modelo            : {model_path}")

    # SHAP values completos (para el dashboard por provincia/cultivo)
    shap_path = os.path.join(out_dir, "shap_values.parquet")
    df_shap.to_parquet(shap_path, index=False)
    log.info(f"SHAP values       : {shap_path}  ({len(df_shap):,} filas)")

    # SHAP summary (para el panel global de importancia)
    summary_path = os.path.join(out_dir, "shap_summary.parquet")
    shap_summary.to_parquet(summary_path, index=False)
    log.info(f"SHAP summary      : {summary_path}")

    # Métricas
    metricas_path = os.path.join(out_dir, "metricas_modelo.json")
    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    log.info(f"Métricas          : {metricas_path}")

    # Feature names (para reconstruir predicciones desde el agente)
    features_path = os.path.join(out_dir, "feature_names.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "target": TARGET}, f, indent=2)
    log.info(f"Feature names     : {features_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AgroChat — Entrenamiento XGBoost + SHAP")
    ap.add_argument("--parquet",       default=PQ_PATH)
    ap.add_argument("--out-dir",       default=OUT_DIR)
    ap.add_argument("--test-size",     type=float, default=0.15)
    ap.add_argument("--val-size",      type=float, default=0.15)
    ap.add_argument("--n-estimators",  type=int,   default=800)
    ap.add_argument("--max-depth",     type=int,   default=5)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    pq_path = args.parquet
    t_total = time.time()

    # 1. Datos
    X_train, X_val, X_test, y_train, y_val, y_test, df = cargar_datos(
        args.test_size, args.val_size, args.seed, pq_path
    )

    # Índices de test en el df original (para añadir metadatos al SHAP)
    n = len(df)
    idx_all = np.arange(n)
    idx_tmp, idx_test = train_test_split(idx_all, test_size=args.test_size,
                                          random_state=args.seed)
    val_relative = args.val_size / (1 - args.test_size)
    _, idx_val   = train_test_split(idx_tmp, test_size=val_relative,
                                    random_state=args.seed)

    # 2. Entrenamiento
    model = entrenar(
        X_train, X_val, y_train, y_val,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    # 3. Evaluación
    metricas = evaluar(model, X_train, X_val, X_test, y_train, y_val, y_test)

    # 4. SHAP
    df_shap, shap_summary = calcular_shap(model, X_test, df, idx_test)

    # 5. Guardar
    guardar(model, df_shap, shap_summary, metricas, args.out_dir)

    # Resumen final
    seccion("RESUMEN EJECUTIVO")
    log.info(f"R² test           : {metricas['test']['r2']:.4f}")
    log.info(f"RMSE test         : {metricas['test']['rmse']:.6f} miles t/ha")
    log.info(f"MAE test          : {metricas['test']['mae']:.6f} miles t/ha")
    log.info(f"Mejor iteración   : {model.best_iteration}")
    log.info(f"Feature más import: {shap_summary.iloc[0]['feature']}")
    log.info(f"Tiempo total      : {time.time()-t_total:.1f}s")
    log.info("")
    log.info("Artefactos en /app/data/models/:")
    log.info("  xgboost_rendimiento.json  ← modelo")
    log.info("  shap_values.parquet       ← SHAP por instancia")
    log.info("  shap_summary.parquet      ← importancia global")
    log.info("  metricas_modelo.json      ← R², RMSE, MAE")
    log.info("")
    log.info("Estado: ✅ Modelo listo para el dashboard y el agente")


if __name__ == "__main__":
    main()
