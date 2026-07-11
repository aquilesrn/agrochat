"""
══════════════════════════════════════════════════════════════════
ETL: Informes Semanales de Coyuntura -> DuckDB
══════════════════════════════════════════════════════════════════
Procesa las hojas de precios medios nacionales (Pág. 4, 5, 7) de
los informes semanales del MAPA y carga en DuckDB.

Datos extraídos:
  - Pág. 4: Cereales, arroz, oleaginosas, tortas, proteicos, vinos, aceites
  - Pág. 5: Frutas y hortalizas (precios en origen)
  - Pág. 7: Productos ganaderos (vacuno, cordero, porcino, pollo, huevos, etc.)

Uso:
  python etl_coyuntura.py --input-dir ./data/raw/coyuntura --db ./data/duckdb/agrochat.duckdb
  python etl_coyuntura.py --file coyuntura_S18_2026.xlsx --db ./data/duckdb/agrochat.duckdb
  python etl_coyuntura.py --input-dir ./data/raw/coyuntura --db ... --dry-run

Requisito: los archivos deben estar ya renombrados con renombrar_coyuntura.py
           o bien se puede usar --file para un archivo cualquiera.
"""

import argparse, duckdb, glob, logging, os, re, sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("etl_coyuntura")

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════

HOJAS_PRECIOS = {
    "Pág. 4": "agricola",       # Cereales, arroz, oleaginosas, vinos, aceites
    "Pág. 5": "frutihort",      # Frutas y hortalizas
    "Pág. 7": "ganadero",       # Vacuno, cordero, porcino, pollo, huevos, leche
}

# ══════════════════════════════════════════════════════════════════
# DETECCIÓN DE METADATOS
# ══════════════════════════════════════════════════════════════════

def detect_metadata(filepath: str) -> dict:
    """
    Extrae año, semana actual, semana anterior y rango de fechas
    del contenido de Pág. 4.
    """
    xls = pd.ExcelFile(filepath, engine="openpyxl")
    df = pd.read_excel(xls, sheet_name="Pág. 4", header=None)

    # Semana actual: R6/C4 "Semana XX"
    week = None
    week_prev = None
    for c in [4, 3]:
        v = df.iloc[6, c] if df.shape[1] > c else None
        if pd.notna(v):
            m = re.search(r'Semana\s+(\d+)', str(v))
            if m:
                if week is None:
                    week = int(m.group(1))
                else:
                    week_prev = int(m.group(1))

    # Si solo encontramos en C4, buscar C3 para semana anterior
    if week and not week_prev:
        v3 = df.iloc[6, 3] if df.shape[1] > 3 else None
        if pd.notna(v3):
            m = re.search(r'Semana\s+(\d+)', str(v3))
            if m:
                week_prev = int(m.group(1))

    # Año
    year = None
    for r in [8, 7]:
        for c in [4, 3]:
            if r < len(df) and c < df.shape[1]:
                v = df.iloc[r, c]
                if pd.notna(v):
                    m = re.search(r'(20\d{2})', str(v).strip())
                    if m:
                        year = int(m.group(1))
                        break
        if year:
            break

    # Fechas: R7/C3 y R7/C4 (formato variable: "05-11/12", "02/01-08/01 2023", etc.)
    fecha_prev = str(df.iloc[7, 3]).strip() if df.shape[1] > 3 and pd.notna(df.iloc[7, 3]) else None
    fecha_actual = str(df.iloc[7, 4]).strip() if df.shape[1] > 4 and pd.notna(df.iloc[7, 4]) else None

    # Limpiar fechas (quitar "(especificaciones)" y similar)
    if fecha_prev and "especificaciones" in fecha_prev.lower():
        fecha_prev = None
    if fecha_actual and "especificaciones" in fecha_actual.lower():
        fecha_actual = None

    return {
        "anio": year,
        "semana": week,
        "semana_anterior": week_prev,
        "fecha_rango_anterior": fecha_prev,
        "fecha_rango_actual": fecha_actual,
    }


# ══════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE PRODUCTOS Y PRECIOS
# ══════════════════════════════════════════════════════════════════

def safe_float(val):
    if pd.isna(val): return None
    s = str(val).strip()
    if s in ("-", "—", "–", ""): return None
    # Manejar comas como separador decimal
    s = s.replace(",", ".")
    try: return float(s)
    except (ValueError, TypeError): return None


def extract_product_name_and_unit(raw: str) -> tuple[str, str]:
    """
    Separa nombre de producto y unidad.
    Ej: "Trigo blando panificable (€/t)" -> ("Trigo blando panificable", "€/t")
        "Limón (€/100 kg)"               -> ("Limón", "€/100 kg")
        "Huevos tipo jaula... (€/docena)" -> ("Huevos tipo jaula...", "€/docena")
    """
    raw = raw.strip()
    # Buscar unidad entre paréntesis al final
    m = re.search(r'\((€[^)]+)\)\s*\*?\s*$', raw)
    if m:
        unidad = m.group(1).strip()
        nombre = raw[:m.start()].strip().rstrip("*").strip()
        return nombre, unidad
    return raw, ""


def extract_sheet_prices(df: pd.DataFrame, seccion: str, meta: dict) -> list[dict]:
    """
    Extrae todos los productos y precios de una hoja de precios medios nacionales.
    """
    records = []
    current_category = ""

    # Las filas de datos empiezan después de las cabeceras
    # El patrón: si C1 tiene "(X)" o C2 tiene "€", es un producto
    # Si C2 tiene texto sin "€", es una categoría
    start_row = 9 if seccion == "agricola" else (7 if seccion == "frutihort" else 7)

    for i in range(start_row, len(df)):
        c1 = df.iloc[i, 1] if df.shape[1] > 1 else None
        c2 = df.iloc[i, 2] if df.shape[1] > 2 else None
        c3 = df.iloc[i, 3] if df.shape[1] > 3 else None
        c4 = df.iloc[i, 4] if df.shape[1] > 4 else None
        c5 = df.iloc[i, 5] if df.shape[1] > 5 else None
        c6 = df.iloc[i, 6] if df.shape[1] > 6 else None

        if pd.isna(c2) or not str(c2).strip():
            continue

        c2_str = str(c2).strip()

        # ¿Es una categoría? (texto sin "€", sin nota "(X)" en C1)
        c1_str = str(c1).strip() if pd.notna(c1) else ""
        is_note = bool(re.match(r'^\(\d+\)', c1_str))

        if not is_note and "€" not in c2_str:
            # Es una categoría
            current_category = c2_str.upper().strip()
            continue

        if "€" not in c2_str:
            continue

        # Es un producto con precio
        nombre, unidad = extract_product_name_and_unit(c2_str)

        # Limpiar sangrado (indentación) del nombre
        nombre_limpio = nombre.strip()
        es_subtipo = nombre_limpio.startswith(" ") or c2_str.startswith("     ")

        records.append({
            "anio": meta["anio"],
            "semana": meta["semana"],
            "seccion": seccion,
            "categoria": current_category,
            "producto": nombre_limpio.strip(),
            "unidad": unidad,
            "precio_sem_anterior": safe_float(c3),
            "precio_sem_actual": safe_float(c4),
            "variacion_euros": safe_float(c5),
            "variacion_pct": safe_float(c6),
            "es_subtipo": es_subtipo,
        })

    return records


# ══════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════

def process_file(filepath: str) -> pd.DataFrame:
    meta = detect_metadata(filepath)
    log.info(f"Procesando: {Path(filepath).name} (S{meta['semana']:02d} {meta['anio']})")

    xls = pd.ExcelFile(filepath, engine="openpyxl")
    all_records = []

    for sheet_name, seccion in HOJAS_PRECIOS.items():
        if sheet_name not in xls.sheet_names:
            log.warning(f"  Hoja '{sheet_name}' no encontrada")
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        records = extract_sheet_prices(df, seccion, meta)
        all_records.extend(records)

    log.info(f"  -> {len(all_records)} productos extraídos")
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# BASE DE DATOS
# ══════════════════════════════════════════════════════════════════

DDL = """
CREATE TABLE IF NOT EXISTS coyuntura (
    anio                INTEGER NOT NULL,    -- Año del informe
    semana              INTEGER NOT NULL,    -- Semana del informe (1-52)
    seccion             VARCHAR NOT NULL,    -- agricola | frutihort | ganadero
    categoria           VARCHAR,             -- CEREALES, ARROZ, FRUTAS, VACUNO, etc.
    producto            VARCHAR NOT NULL,    -- Nombre del producto
    unidad              VARCHAR,             -- €/t, €/100 kg, €/docena, etc.
    precio_sem_anterior DOUBLE,              -- Precio semana anterior
    precio_sem_actual   DOUBLE,              -- Precio semana actual
    variacion_euros     DOUBLE,              -- Variación semanal en €
    variacion_pct       DOUBLE,              -- Variación semanal en %
    es_subtipo          BOOLEAN DEFAULT FALSE, -- Si es un subtipo (indentado)
    fecha_carga         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_coy_producto
    ON coyuntura(producto, anio, semana);
CREATE INDEX IF NOT EXISTS idx_coy_periodo
    ON coyuntura(anio, semana);
CREATE INDEX IF NOT EXISTS idx_coy_seccion
    ON coyuntura(seccion, categoria, anio, semana);
"""

COLUMNS_ORDERED = [
    "anio", "semana", "seccion", "categoria", "producto", "unidad",
    "precio_sem_anterior", "precio_sem_actual",
    "variacion_euros", "variacion_pct", "es_subtipo",
]


def init_db(db_path: str):
    os.makedirs(Path(db_path).parent, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(DDL)
    return con


def load_to_duckdb(con, df: pd.DataFrame, meta: dict):
    if df.empty:
        return

    df = df[COLUMNS_ORDERED].copy()

    # Idempotencia
    con.execute("DELETE FROM coyuntura WHERE anio=? AND semana=?",
                [meta["anio"], meta["semana"]])
    con.execute("INSERT INTO coyuntura SELECT *, CURRENT_TIMESTAMP FROM df")

    n = con.execute("SELECT COUNT(*) FROM coyuntura WHERE anio=? AND semana=?",
                    [meta["anio"], meta["semana"]]).fetchone()[0]
    log.info(f"  -> {n} registros cargados en DuckDB")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="ETL: Informes de Coyuntura -> DuckDB")
    ap.add_argument("--input-dir", help="Directorio raíz (recorre subcarpetas)")
    ap.add_argument("--file", help="Archivo individual")
    ap.add_argument("--db", required=True, help="Ruta al .duckdb")
    ap.add_argument("--dry-run", action="store_true", help="Solo procesar, no cargar")
    args = ap.parse_args()

    files = []
    if args.file:
        files = [args.file]
    elif args.input_dir:
        for ext in ("*.xlsx", "*.xlsm"):
            files.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
        files = sorted(set(files))
    else:
        log.error("Especifica --input-dir o --file"); sys.exit(1)

    if not files:
        log.error("No se encontraron archivos"); sys.exit(1)

    log.info(f"Archivos encontrados: {len(files)}")
    con = init_db(args.db) if not args.dry_run else None

    total_records = 0
    errors = []

    for fp in tqdm(files, desc="Procesando"):
        try:
            meta = detect_metadata(fp)
            df = process_file(fp)
            total_records += len(df)
            if con is not None and not df.empty:
                load_to_duckdb(con, df, meta)
        except Exception as e:
            log.error(f"Error en {fp}: {e}")
            errors.append((fp, str(e)))

    log.info(f"\n{'='*60}")
    log.info(f"ETL COMPLETADO: {len(files)-len(errors)}/{len(files)} archivos, {total_records:,} registros")
    if errors:
        for f, e in errors:
            log.warning(f"  ERROR: {Path(f).name}: {e}")

    if con:
        r = con.execute("""
            SELECT COUNT(*) total,
                   COUNT(DISTINCT producto) productos,
                   COUNT(DISTINCT anio || '-S' || semana) semanas,
                   MIN(anio) anio_min, MAX(anio) anio_max
            FROM coyuntura
        """).fetchone()
        log.info(f"\n  BD: {r[0]:,} registros | {r[1]} productos | {r[2]} semanas | {r[3]}-{r[4]}")

        # Desglose por sección
        for row in con.execute("""
            SELECT seccion, COUNT(DISTINCT producto), COUNT(*)
            FROM coyuntura GROUP BY seccion ORDER BY seccion
        """).fetchall():
            log.info(f"      {row[0]}: {row[1]} productos, {row[2]} registros")
        con.close()


if __name__ == "__main__":
    main()
