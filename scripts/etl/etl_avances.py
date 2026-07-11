"""
══════════════════════════════════════════════════════════════════
ETL: Cuadernos de Avances de Superficies y Producciones -> DuckDB
══════════════════════════════════════════════════════════════════
Procesa archivos Excel mensuales del MAPA (2014-2026) y carga
los datos en DuckDB.

Soporta dos formatos:
  - Formato A (.xls):  2014-2025, ~55 hojas, solo totales
  - Formato B (.xlsm): 2026+, ~75 hojas + SRT (Secano/Regadío/Total)

Uso:
  python etl_avances.py --input-dir ./data/raw/avances --db ./data/duckdb/agrochat.duckdb
  python etl_avances.py --file cuaderno_abril2025.xls --db ./data/duckdb/agrochat.duckdb
  python etl_avances.py --input-dir ./data/raw/avances --db ... --dry-run
"""

import argparse, duckdb, glob, logging, os, re, sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("etl_avances")

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════

SKIP_SHEETS = {"portada", "índice", "índice ", "resumen nacional", "Hoja_del_programa"}

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

# ── Mapeo estático provincia (cod INE) -> CCAA ───────────────────
# Invariante entre archivos. Resuelve el problema de las
# provincias uniprovinciales (Asturias, Cantabria, Navarra,
# La Rioja, Baleares, Madrid, Murcia) que aparecen con código
# INE pero son también CCAA, y cuyo current_ccaa se propagaba
# erróneamente a las provincias del siguiente bloque.
PROV_TO_CCAA = {
    15: "GALICIA", 27: "GALICIA", 32: "GALICIA", 36: "GALICIA",
    33: "P. DE ASTURIAS",
    39: "CANTABRIA",
    1: "PAIS VASCO", 20: "PAIS VASCO", 48: "PAIS VASCO",
    31: "NAVARRA",
    26: "LA RIOJA",
    22: "ARAGÓN", 44: "ARAGÓN", 50: "ARAGÓN",
    8: "CATALUÑA", 17: "CATALUÑA", 25: "CATALUÑA", 43: "CATALUÑA",
    7: "BALEARES",
    5: "CASTILLA Y LEÓN", 9: "CASTILLA Y LEÓN", 24: "CASTILLA Y LEÓN",
    34: "CASTILLA Y LEÓN", 37: "CASTILLA Y LEÓN", 40: "CASTILLA Y LEÓN",
    42: "CASTILLA Y LEÓN", 47: "CASTILLA Y LEÓN", 49: "CASTILLA Y LEÓN",
    28: "MADRID",
    2: "CASTILLA-MANCHA", 13: "CASTILLA-MANCHA", 16: "CASTILLA-MANCHA",
    19: "CASTILLA-MANCHA", 45: "CASTILLA-MANCHA",
    3: "C. VALENCIANA", 12: "C. VALENCIANA", 46: "C. VALENCIANA",
    30: "R. DE MURCIA",
    6: "EXTREMADURA", 10: "EXTREMADURA",
    4: "ANDALUCÍA", 11: "ANDALUCÍA", 14: "ANDALUCÍA", 18: "ANDALUCÍA",
    21: "ANDALUCÍA", 23: "ANDALUCÍA", 29: "ANDALUCÍA", 41: "ANDALUCÍA",
    35: "CANARIAS", 38: "CANARIAS",
}

# Códigos INE de provincias uniprovinciales
UNIPROV_CODES = {33, 39, 31, 26, 7, 28, 30}

# CCAA multiprovinciales (aparecen como texto sin código INE)
CCAA_NAMES = {
    "GALICIA", "PAIS VASCO", "PAÍS VASCO", "ARAGÓN", "ARAGON",
    "CATALUÑA", "CASTILLA Y LEÓN", "CASTILLA Y LEON",
    "CASTILLA-MANCHA", "CASTILLA-LA MANCHA",
    "C. VALENCIANA", "COMUNIDAD VALENCIANA",
    "EXTREMADURA", "ANDALUCÍA", "ANDALUCIA", "CANARIAS",
}

# ══════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════

def parse_filename(filepath: str) -> dict:
    """Extrae mes y año del nombre: cuaderno_abril2025.xls -> {mes:4, anio:2025}"""
    fname = Path(filepath).stem.lower()
    m = re.search(r'cuaderno_(\w+?)(\d{4})', fname)
    if not m:
        raise ValueError(f"Nombre no reconocido: {filepath}")
    mes_str = m.group(1).lower()
    if mes_str not in MESES:
        raise ValueError(f"Mes no reconocido: '{mes_str}' en {filepath}")
    return {"mes": MESES[mes_str], "anio": int(m.group(2)), "mes_nombre": mes_str}


def safe_float(val):
    """Convierte a float o None."""
    if pd.isna(val): return None
    try: return float(val)
    except (ValueError, TypeError): return None


def safe_int(val):
    """Convierte a int o None."""
    f = safe_float(val)
    return int(f) if f is not None else None


def classify_row(cell_value):
    """
    Clasifica una fila por el valor de columna 0.
    Retorna dict con tipo, cod (INE), nombre.
    """
    if pd.isna(cell_value):
        return {"tipo": "skip"}
    s = str(cell_value).strip()
    if not s or s.startswith("(") or s.startswith("*"):
        return {"tipo": "skip"}
    if "ESPAÑA" in s.upper():
        return {"tipo": "total", "cod": None, "nombre": "ESPAÑA"}
    if "SUMA PROV" in s.upper():
        return {"tipo": "skip"}

    # Provincia: empieza con código INE
    m = re.match(r'^(\d{1,2})\s+(.+)', s)
    if m:
        cod = int(m.group(1))
        nombre = m.group(2).strip()
        if cod in UNIPROV_CODES:
            return {"tipo": "uniprov", "cod": cod, "nombre": nombre}
        return {"tipo": "provincia", "cod": cod, "nombre": nombre}

    # CCAA multiprovincial
    if s.upper().strip() in CCAA_NAMES:
        return {"tipo": "ccaa", "cod": None, "nombre": s.strip()}

    return {"tipo": "skip"}


# ══════════════════════════════════════════════════════════════════
# EXTRACCIÓN: FORMATO A (.xls, 2014-2025)
# 11 columnas: C2-C4 superficies, C5 índice, C7-C9 producciones, C10 índice
# ══════════════════════════════════════════════════════════════════

def extract_sheet_xls(df: pd.DataFrame, file_meta: dict) -> list[dict]:
    cultivo = str(df.iloc[1, 0]).strip() if pd.notna(df.iloc[1, 0]) else "DESCONOCIDO"

    anio_def = safe_int(df.iloc[5, 2])
    anio_prov = safe_int(df.iloc[5, 3])
    anio_avance = safe_int(df.iloc[5, 4])
    mes_avance = safe_int(df.iloc[6, 4]) or file_meta["mes"]

    records = []

    for i in range(8, len(df)):
        rc = classify_row(df.iloc[i, 0])
        if rc["tipo"] == "skip":
            continue

        # Valores numéricos (iguales para todos los tipos de fila)
        vals = {
            "sup_definitivo": safe_float(df.iloc[i, 2]),
            "sup_provisional": safe_float(df.iloc[i, 3]),
            "sup_avance": safe_float(df.iloc[i, 4]),
            "sup_indice": safe_float(df.iloc[i, 5]),
            "prod_definitivo": safe_float(df.iloc[i, 7]),
            "prod_provisional": safe_float(df.iloc[i, 8]),
            "prod_avance": safe_float(df.iloc[i, 9]),
            "prod_indice": safe_float(df.iloc[i, 10]) if df.shape[1] > 10 else None,
        }

        base = {
            "periodo_anio": file_meta["anio"],
            "periodo_mes": file_meta["mes"],
            "cultivo": cultivo,
            "regimen": "total",
            "anio_definitivo": anio_def,
            "anio_provisional": anio_prov,
            "anio_avance": anio_avance,
            "mes_avance": mes_avance,
        }

        if rc["tipo"] == "provincia":
            ccaa = PROV_TO_CCAA.get(rc["cod"], "DESCONOCIDA")
            records.append({**base, **vals,
                "nivel": "provincia", "cod_provincia": rc["cod"],
                "provincia": rc["nombre"], "ccaa": ccaa})

        elif rc["tipo"] == "uniprov":
            ccaa = PROV_TO_CCAA.get(rc["cod"], "DESCONOCIDA")
            records.append({**base, **vals,
                "nivel": "provincia", "cod_provincia": rc["cod"],
                "provincia": rc["nombre"], "ccaa": ccaa})

        elif rc["tipo"] == "ccaa":
            records.append({**base, **vals,
                "nivel": "ccaa", "cod_provincia": None,
                "provincia": None, "ccaa": rc["nombre"]})

        elif rc["tipo"] == "total":
            records.append({**base, **vals,
                "nivel": "nacional", "cod_provincia": None,
                "provincia": None, "ccaa": "ESPAÑA"})

    return records


# ══════════════════════════════════════════════════════════════════
# EXTRACCIÓN: FORMATO B (.xlsm, 2026+, hojas SRT, 20 columnas)
# Secano: C1-C3 sup, C11-C13 prod
# Regadío: C4-C6 sup, C14-C16 prod
# Total: C7-C9 sup, C17-C19 prod
# ══════════════════════════════════════════════════════════════════

def extract_sheet_srt(df: pd.DataFrame, file_meta: dict) -> list[dict]:
    cultivo = str(df.iloc[1, 0]).strip() if pd.notna(df.iloc[1, 0]) else "DESCONOCIDO"
    cultivo = re.sub(r'\s+S/R/T\s*$', '', cultivo, flags=re.IGNORECASE).strip()

    anio_def = safe_int(df.iloc[5, 1])
    anio_prov = safe_int(df.iloc[5, 2])
    anio_avance = safe_int(df.iloc[5, 3])

    records = []

    regimenes = {
        "secano":  {"sup": (1, 2, 3),   "prod": (11, 12, 13)},
        "regadio": {"sup": (4, 5, 6),   "prod": (14, 15, 16)},
        "total":   {"sup": (7, 8, 9),   "prod": (17, 18, 19)},
    }

    for i in range(8, len(df)):
        rc = classify_row(df.iloc[i, 0])
        if rc["tipo"] == "skip":
            continue

        for regimen, cols in regimenes.items():
            sc, pc = cols["sup"], cols["prod"]
            v = {
                "sup_definitivo":  safe_float(df.iloc[i, sc[0]]) if df.shape[1] > sc[0] else None,
                "sup_provisional": safe_float(df.iloc[i, sc[1]]) if df.shape[1] > sc[1] else None,
                "sup_avance":      safe_float(df.iloc[i, sc[2]]) if df.shape[1] > sc[2] else None,
                "sup_indice":      None,
                "prod_definitivo":  safe_float(df.iloc[i, pc[0]]) if df.shape[1] > pc[0] else None,
                "prod_provisional": safe_float(df.iloc[i, pc[1]]) if df.shape[1] > pc[1] else None,
                "prod_avance":      safe_float(df.iloc[i, pc[2]]) if df.shape[1] > pc[2] else None,
                "prod_indice":      None,
            }
            if all(x is None for x in v.values()):
                continue

            if rc["tipo"] in ("provincia", "uniprov"):
                ccaa = PROV_TO_CCAA.get(rc["cod"], "DESCONOCIDA")
                records.append({
                    "periodo_anio": file_meta["anio"], "periodo_mes": file_meta["mes"],
                    "cultivo": cultivo, "regimen": regimen,
                    "nivel": "provincia", "cod_provincia": rc["cod"],
                    "provincia": rc["nombre"], "ccaa": ccaa,
                    **v, "anio_definitivo": anio_def, "anio_provisional": anio_prov,
                    "anio_avance": anio_avance, "mes_avance": file_meta["mes"],
                })
            elif rc["tipo"] == "ccaa":
                records.append({
                    "periodo_anio": file_meta["anio"], "periodo_mes": file_meta["mes"],
                    "cultivo": cultivo, "regimen": regimen,
                    "nivel": "ccaa", "cod_provincia": None,
                    "provincia": None, "ccaa": rc["nombre"],
                    **v, "anio_definitivo": anio_def, "anio_provisional": anio_prov,
                    "anio_avance": anio_avance, "mes_avance": file_meta["mes"],
                })
            elif rc["tipo"] == "total":
                records.append({
                    "periodo_anio": file_meta["anio"], "periodo_mes": file_meta["mes"],
                    "cultivo": cultivo, "regimen": regimen,
                    "nivel": "nacional", "cod_provincia": None,
                    "provincia": None, "ccaa": "ESPAÑA",
                    **v, "anio_definitivo": anio_def, "anio_provisional": anio_prov,
                    "anio_avance": anio_avance, "mes_avance": file_meta["mes"],
                })

    return records


# ══════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════

def process_file(filepath: str) -> pd.DataFrame:
    file_meta = parse_filename(filepath)
    is_xlsm = Path(filepath).suffix.lower() == ".xlsm"

    log.info(f"Procesando: {Path(filepath).name} ({file_meta['mes_nombre']} {file_meta['anio']})")

    xls = pd.ExcelFile(filepath)
    all_records = []
    sheets_ok = sheets_err = 0

    for sn in xls.sheet_names:
        if sn.strip() in SKIP_SHEETS:
            continue

        if is_xlsm:
            if "-SRT" not in sn:
                continue
            try:
                df = pd.read_excel(xls, sheet_name=sn, header=None)
                all_records.extend(extract_sheet_srt(df, file_meta))
                sheets_ok += 1
            except Exception as e:
                log.warning(f"  Error en '{sn}': {e}")
                sheets_err += 1
        else:
            try:
                df = pd.read_excel(xls, sheet_name=sn, header=None)
                all_records.extend(extract_sheet_xls(df, file_meta))
                sheets_ok += 1
            except Exception as e:
                log.warning(f"  Error en '{sn}': {e}")
                sheets_err += 1

    log.info(f"  -> {sheets_ok} hojas OK, {sheets_err} errores, {len(all_records)} registros")
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# BASE DE DATOS
# ══════════════════════════════════════════════════════════════════

DDL = """
CREATE TABLE IF NOT EXISTS avances (
    periodo_anio     INTEGER NOT NULL,   -- Año del archivo
    periodo_mes      INTEGER NOT NULL,   -- Mes del archivo
    cultivo          VARCHAR NOT NULL,   -- Nombre del cultivo
    regimen          VARCHAR DEFAULT 'total',  -- secano | regadio | total
    nivel            VARCHAR NOT NULL,   -- provincia | ccaa | nacional
    cod_provincia    INTEGER,            -- Código INE de provincia
    provincia        VARCHAR,            -- Nombre de la provincia
    ccaa             VARCHAR,            -- Comunidad Autónoma
    sup_definitivo   DOUBLE,             -- Superficie definitiva (ha)
    sup_provisional  DOUBLE,             -- Superficie provisional (ha)
    sup_avance       DOUBLE,             -- Superficie avance (ha)
    sup_indice       DOUBLE,             -- Índice superficie (año anterior=100)
    prod_definitivo  DOUBLE,             -- Producción definitiva (miles Tm)
    prod_provisional DOUBLE,             -- Producción provisional (miles Tm)
    prod_avance      DOUBLE,             -- Producción avance (miles Tm)
    prod_indice      DOUBLE,             -- Índice producción (año anterior=100)
    anio_definitivo  INTEGER,            -- Año del dato definitivo (año-2)
    anio_provisional INTEGER,            -- Año del dato provisional (año-1)
    anio_avance      INTEGER,            -- Año del avance (año actual)
    mes_avance       INTEGER,            -- Mes del avance
    fecha_carga      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_av_cult_prov
    ON avances(cultivo, cod_provincia, periodo_anio, periodo_mes);
CREATE INDEX IF NOT EXISTS idx_av_periodo
    ON avances(periodo_anio, periodo_mes);
CREATE INDEX IF NOT EXISTS idx_av_ccaa
    ON avances(ccaa, cultivo, periodo_anio);
"""

COLUMNS_ORDERED = [
    "periodo_anio", "periodo_mes", "cultivo", "regimen", "nivel",
    "cod_provincia", "provincia", "ccaa",
    "sup_definitivo", "sup_provisional", "sup_avance", "sup_indice",
    "prod_definitivo", "prod_provisional", "prod_avance", "prod_indice",
    "anio_definitivo", "anio_provisional", "anio_avance", "mes_avance",
]


def init_db(db_path: str):
    os.makedirs(Path(db_path).parent, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(DDL)
    log.info(f"BD inicializada: {db_path}")
    return con


def load_to_duckdb(con, df: pd.DataFrame, filepath: str):
    if df.empty:
        return
    meta = parse_filename(filepath)

    # Asegurar orden de columnas y tipos
    df = df[COLUMNS_ORDERED].copy()
    for col in ("cod_provincia", "anio_definitivo", "anio_provisional", "anio_avance", "mes_avance"):
        df[col] = pd.array(df[col], dtype=pd.Int32Dtype())

    # Idempotencia: borrar datos previos del mismo periodo
    con.execute("DELETE FROM avances WHERE periodo_anio=? AND periodo_mes=?",
                [meta["anio"], meta["mes"]])
    con.execute("INSERT INTO avances SELECT *, CURRENT_TIMESTAMP FROM df")

    n = con.execute("SELECT COUNT(*) FROM avances WHERE periodo_anio=? AND periodo_mes=?",
                    [meta["anio"], meta["mes"]]).fetchone()[0]
    log.info(f"  -> {n:,} registros cargados en DuckDB")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="ETL: Cuadernos de Avances -> DuckDB")
    ap.add_argument("--input-dir", help="Directorio con archivos Excel")
    ap.add_argument("--file", help="Archivo Excel individual")
    ap.add_argument("--db", required=True, help="Ruta al .duckdb")
    ap.add_argument("--dry-run", action="store_true", help="Solo procesar, no cargar en BD")
    args = ap.parse_args()

    files = []
    if args.file:
        files = [args.file]
    elif args.input_dir:
        for ext in ("*.xls", "*.xlsm", "*.xlsx"):
            files.extend(glob.glob(os.path.join(args.input_dir, f"cuaderno_{ext}")))
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
            df = process_file(fp)
            total_records += len(df)
            if con is not None and not df.empty:
                load_to_duckdb(con, df, fp)
        except Exception as e:
            log.error(f"Error en {fp}: {e}")
            errors.append((fp, str(e)))

    # ── Resumen ──
    log.info(f"\n{'='*60}")
    log.info(f"ETL COMPLETADO")
    log.info(f"  Archivos OK: {len(files)-len(errors)}/{len(files)}")
    log.info(f"  Registros totales: {total_records:,}")
    if errors:
        for f, e in errors:
            log.warning(f"  ERROR: {Path(f).name}: {e}")

    if con:
        r = con.execute("""
            SELECT COUNT(*) total,
                   COUNT(DISTINCT cultivo) cultivos,
                   COUNT(DISTINCT cod_provincia) provincias,
                   COUNT(DISTINCT ccaa) ccaa,
                   MIN(periodo_anio) anio_min,
                   MAX(periodo_anio) anio_max
            FROM avances WHERE nivel='provincia'
        """).fetchone()
        log.info(f"\n  BD: {r[0]:,} reg. provinciales | {r[1]} cultivos | {r[2]} prov. | {r[3]} CCAA")
        log.info(f"      Periodo: {r[4]}-{r[5]}")
        con.close()


if __name__ == "__main__":
    main()
