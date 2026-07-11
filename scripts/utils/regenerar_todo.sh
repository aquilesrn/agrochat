#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# AgroChat — Regenera processed/ y models/ desde DuckDB
# Los datos en DuckDB y ChromaDB están intactos.
# Recorre los 3 pasos de procesamiento/entrenamiento.
#
# Uso: MSYS_NO_PATHCONV=1 bash scripts/utils/regenerar_todo.sh
#
# Estructura de carpetas esperada:
#   scripts/data/    → procesar_datos.py, finalizar_dataset.py
#   scripts/models/  → entrenar_modelo.py
#   scripts/utils/   → este script
# ══════════════════════════════════════════════════════════════════
set -e

C="agrochat"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  AgroChat — Regeneración de processed/ y models/"
echo "══════════════════════════════════════════════════════"

echo ""
echo "▶ [1/3] procesar_datos.py  (limpieza + imputación climática)..."
docker exec "$C" python scripts/data/procesar_datos.py
echo "  ✅ Paso 1 completado"

echo ""
echo "▶ [2/3] finalizar_dataset.py  (imputación CCAA + Parquet final)..."
docker exec "$C" python scripts/data/finalizar_dataset.py
echo "  ✅ Paso 2 completado"

echo ""
echo "▶ [3/3] entrenar_modelo.py  (XGBoost + SHAP ~20s)..."
docker exec "$C" python scripts/models/entrenar_modelo.py
echo "  ✅ Paso 3 completado"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Verificando resultados..."
echo "══════════════════════════════════════════════════════"
docker exec "$C" python scripts/data/verificar_datos.py

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Siguiente: prueba el agente con:"
echo "  MSYS_NO_PATHCONV=1 docker exec -it agrochat python \\"
echo "    scripts/agent/agente.py --provider claude \\"
echo "    --model claude-sonnet-4-6 --test --verbose"
echo "══════════════════════════════════════════════════════"
