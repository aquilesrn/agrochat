"""
══════════════════════════════════════════════════════════════════
Descarga de datos climáticos mensuales de AEMET OpenData -> DuckDB
══════════════════════════════════════════════════════════════════
Descarga series climáticas mensuales por estación, almacena en
DuckDB y genera una vista agregada a nivel provincial para cruce
con los datos de avances de cultivos.

Uso:
  # 1. Descargar inventario de estaciones
  python aemet_descarga.py --api-key TU_KEY --inventario --db agrochat.duckdb

  # 2. Descargar datos mensuales 2014-2025
  python aemet_descarga.py --api-key TU_KEY --datos --anio-ini 2014 --anio-fin 2025 --db agrochat.duckdb

  # Prueba rápida (10 estaciones)
  python aemet_descarga.py --api-key TU_KEY --datos --max-estaciones 10 --db agrochat.duckdb

CORRECCIONES v2:
  - Descarga año a año (la API AEMET rechaza silenciosamente rangos > 1 año
    en muchos endpoints; así se evitan los 0 registros sin error visible)
  - Logging de errores HTTP/API para diagnosticar fallos por estación
  - Eliminada la línea con DDL.split() que causaba ParserException en DuckDB
    (la vista se crea correctamente en la ejecución inicial del DDL completo)
  - Pausa entre peticiones ajustada para respetar el rate-limit de AEMET
"""

import argparse, duckdb, logging, os, sys, time
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("aemet")

BASE_URL = "https://opendata.aemet.es/opendata/api"

# Mapeo nombre provincia AEMET -> código INE
PROV_AEMET_TO_INE = {
    "A CORUÑA": 15, "CORUÑA, A": 15, "LUGO": 27, "OURENSE": 32, "PONTEVEDRA": 36,
    "ASTURIAS": 33, "CANTABRIA": 39,
    "ARABA/ÁLAVA": 1, "ÁLAVA": 1, "BIZKAIA": 48, "VIZCAYA": 48,
    "GIPUZKOA": 20, "GUIPÚZCOA": 20,
    "NAVARRA": 31, "LA RIOJA": 26,
    "HUESCA": 22, "TERUEL": 44, "ZARAGOZA": 50,
    "BARCELONA": 8, "GIRONA": 17, "LLEIDA": 25, "TARRAGONA": 43,
    "ILLES BALEARS": 7, "BALEARS, ILLES": 7,
    "ÁVILA": 5, "BURGOS": 9, "LEÓN": 24, "PALENCIA": 34, "SALAMANCA": 37,
    "SEGOVIA": 40, "SORIA": 42, "VALLADOLID": 47, "ZAMORA": 49,
    "MADRID": 28,
    "ALBACETE": 2, "CIUDAD REAL": 13, "CUENCA": 16, "GUADALAJARA": 19, "TOLEDO": 45,
    "ALICANTE": 3, "ALACANT": 3, "CASTELLÓN": 12, "CASTELLÓ": 12,
    "VALENCIA": 46, "VALÈNCIA": 46,
    "MURCIA": 30,
    "BADAJOZ": 6, "CÁCERES": 10,
    "ALMERÍA": 4, "CÁDIZ": 11, "CÓRDOBA": 14, "GRANADA": 18,
    "HUELVA": 21, "JAÉN": 23, "MÁLAGA": 29, "SEVILLA": 41,
    "LAS PALMAS": 35, "STA. CRUZ DE TENERIFE": 38, "SANTA CRUZ DE TENERIFE": 38,
    "CEUTA": 51, "MELILLA": 52,
}


def aemet_request(url: str, api_key: str):
    """
    Petición AEMET OpenData (2 pasos: metadatos -> datos reales).
    AEMET devuelve primero un JSON con una URL de datos; hay que
    hacer una segunda petición a esa URL para obtener los registros.
    Lanza excepción con mensaje descriptivo en caso de error.
    """
    headers = {"api_key": api_key}
    r1 = requests.get(url, headers=headers, timeout=30)
    r1.raise_for_status()
    meta = r1.json()

    estado = meta.get("estado")
    if estado == 404:
        raise ValueError(f"Sin datos (404): {meta.get('descripcion', '')}")
    if estado != 200:
        raise ValueError(f"AEMET estado={estado}: {meta.get('descripcion', meta)}")

    data_url = meta.get("datos")
    if not data_url:
        raise ValueError("Sin URL de datos en la respuesta AEMET")

    r2 = requests.get(data_url, timeout=60)
    r2.raise_for_status()
    return r2.json()


def parse_float_es(val):
    """Convierte valor con coma decimal (formato AEMET) a float."""
    if pd.isna(val) or str(val).strip() in ("", "Ip", "Varias", "Acum"):
        return None
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def descargar_inventario(api_key: str) -> pd.DataFrame:
    url = f"{BASE_URL}/valores/climatologicos/inventarioestaciones/todasestaciones"
    data = aemet_request(url, api_key)
    df = pd.DataFrame(data)
    df["cod_provincia"] = df["provincia"].str.upper().str.strip().map(PROV_AEMET_TO_INE)
    log.info(f"Inventario: {len(df)} estaciones")
    return df


def descargar_mensuales_anio(api_key: str, estacion: str, anio: int) -> list:
    """
    Descarga datos mensuales de UNA estación para UN año concreto.

    FIX: La API AEMET rechaza silenciosamente rangos de años largos
    en el endpoint de climatologías mensuales (devuelve estado 200
    pero 0 registros, o estado 404 sin levantar excepción en la
    versión original). Descargando año a año se evita este problema.
    """
    url = (
        f"{BASE_URL}/valores/climatologicos/mensualesanuales/datos/"
        f"anioini/{anio}/aniofin/{anio}/estacion/{estacion}"
    )
    try:
        return aemet_request(url, api_key)
    except ValueError as e:
        # 404 = estación sin datos ese año → normal, no es un error crítico
        msg = str(e)
        if "404" not in msg:
            log.debug(f"  {estacion}/{anio}: {msg}")
        return []
    except Exception as e:
        log.debug(f"  {estacion}/{anio}: {e}")
        return []


def procesar_mensuales(raw: list) -> pd.DataFrame:
    rows = []
    for r in raw:
        fecha = r.get("fecha", "")
        parts = fecha.split("-")
        if len(parts) != 2:
            continue
        try:
            anio, mes = int(parts[0]), int(parts[1])
        except ValueError:
            continue

        rows.append({
            "indicativo":   r.get("indicativo", ""),
            "anio":         anio,
            "mes":          mes,
            "t_media":      parse_float_es(r.get("tm_mes")),
            "t_max_abs":    parse_float_es(r.get("ta_max")),
            "t_min_abs":    parse_float_es(r.get("ta_min")),
            "t_max_med":    parse_float_es(r.get("tm_max")),
            "t_min_med":    parse_float_es(r.get("tm_min")),
            "precip_mm":    parse_float_es(r.get("p_mes")),
            "dias_precip":  parse_float_es(r.get("n_llu")),
            "horas_sol":    parse_float_es(r.get("sol")),
            "hr_media":     parse_float_es(r.get("hr")),
        })
    return pd.DataFrame(rows)


# ── DDL ───────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS aemet_estaciones (
    indicativo    VARCHAR PRIMARY KEY,
    nombre        VARCHAR,
    provincia     VARCHAR,
    cod_provincia INTEGER,
    altitud       DOUBLE,
    latitud       VARCHAR,
    longitud      VARCHAR
);

CREATE TABLE IF NOT EXISTS clima_mensual (
    indicativo    VARCHAR NOT NULL,
    anio          INTEGER NOT NULL,
    mes           INTEGER NOT NULL,
    t_media       DOUBLE,
    t_max_abs     DOUBLE,
    t_min_abs     DOUBLE,
    t_max_med     DOUBLE,
    t_min_med     DOUBLE,
    precip_mm     DOUBLE,
    dias_precip   DOUBLE,
    horas_sol     DOUBLE,
    hr_media      DOUBLE,
    cod_provincia INTEGER,
    fecha_carga   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicativo, anio, mes)
);

CREATE INDEX IF NOT EXISTS idx_clima_prov ON clima_mensual(cod_provincia, anio, mes);

CREATE OR REPLACE VIEW clima_provincial AS
SELECT
    cod_provincia,
    anio,
    mes,
    ROUND(AVG(t_media), 1)   AS t_media_c,
    ROUND(MAX(t_max_abs), 1) AS t_max_abs_c,
    ROUND(MIN(t_min_abs), 1) AS t_min_abs_c,
    ROUND(AVG(precip_mm), 1) AS precip_media_mm,
    ROUND(AVG(horas_sol), 1) AS horas_sol_med,
    ROUND(AVG(hr_media), 1)  AS hr_media_pct,
    COUNT(*)                 AS n_estaciones
FROM clima_mensual
WHERE cod_provincia IS NOT NULL
GROUP BY cod_provincia, anio, mes;
"""

COLUMNS = [
    "indicativo", "anio", "mes",
    "t_media", "t_max_abs", "t_min_abs", "t_max_med", "t_min_med",
    "precip_mm", "dias_precip", "horas_sol", "hr_media", "cod_provincia",
]


def main():
    ap = argparse.ArgumentParser(description="AEMET OpenData -> DuckDB")
    ap.add_argument("--api-key", default=os.environ.get("AEMET_API_KEY"))
    ap.add_argument("--db", required=True)
    ap.add_argument("--inventario", action="store_true")
    ap.add_argument("--datos",      action="store_true")
    ap.add_argument("--anio-ini",   type=int, default=2014)
    ap.add_argument("--anio-fin",   type=int, default=2025)
    ap.add_argument("--max-estaciones", type=int, default=0)
    ap.add_argument("--pausa", type=float, default=1.0,
                    help="Segundos entre peticiones (default: 1.0)")
    args = ap.parse_args()

    if not args.api_key:
        log.error("Especifica --api-key o la variable de entorno AEMET_API_KEY")
        sys.exit(1)

    os.makedirs(Path(args.db).parent, exist_ok=True)
    con = duckdb.connect(args.db)
    con.execute(DDL)   # Crea tablas + vista; idempotente por IF NOT EXISTS / OR REPLACE

    # ── Inventario ──────────────────────────────────────────────
    if args.inventario:
        log.info("Descargando inventario...")
        df = descargar_inventario(args.api_key)
        df_db = df[["indicativo", "nombre", "provincia", "cod_provincia",
                     "altitud", "latitud", "longitud"]].copy()
        df_db["altitud"] = pd.to_numeric(df_db["altitud"], errors="coerce")
        con.execute("DELETE FROM aemet_estaciones")
        con.execute("INSERT INTO aemet_estaciones SELECT * FROM df_db")
        n = con.execute("SELECT COUNT(*) FROM aemet_estaciones").fetchone()[0]
        m = con.execute(
            "SELECT COUNT(*) FROM aemet_estaciones WHERE cod_provincia IS NOT NULL"
        ).fetchone()[0]
        log.info(f"  {n} estaciones ({m} mapeadas a provincia)")

    # ── Datos mensuales ─────────────────────────────────────────
    if args.datos:
        estaciones = con.execute("""
            SELECT indicativo, nombre, cod_provincia
            FROM aemet_estaciones
            WHERE cod_provincia IS NOT NULL
            ORDER BY cod_provincia
        """).fetchall()

        if not estaciones:
            log.error("No hay estaciones en la BD. Ejecuta primero --inventario.")
            sys.exit(1)

        if args.max_estaciones > 0:
            estaciones = estaciones[:args.max_estaciones]

        anios = list(range(args.anio_ini, args.anio_fin + 1))
        total_peticiones = len(estaciones) * len(anios)
        log.info(
            f"Descargando {len(estaciones)} estaciones × {len(anios)} años "
            f"= {total_peticiones} peticiones ({args.anio_ini}-{args.anio_fin})..."
        )

        total_registros = 0
        errs = 0
        peticion = 0

        for i, (indic, nombre, cprov) in enumerate(estaciones):
            registros_estacion = 0

            for anio in anios:
                peticion += 1
                raw = descargar_mensuales_anio(args.api_key, indic, anio)

                if raw:
                    df_m = procesar_mensuales(raw)
                    if not df_m.empty:
                        df_m["cod_provincia"] = cprov
                        df_m = df_m[COLUMNS]
                        # Upsert: borra y reinserta solo los registros de este año
                        con.execute(
                            "DELETE FROM clima_mensual WHERE indicativo=? AND anio=?",
                            [indic, anio]
                        )
                        con.execute(
                            "INSERT INTO clima_mensual SELECT *, CURRENT_TIMESTAMP FROM df_m"
                        )
                        registros_estacion += len(df_m)
                        total_registros += len(df_m)
                else:
                    errs += 1

                time.sleep(args.pausa)

            # Progreso cada 50 estaciones
            if (i + 1) % 50 == 0 or (i + 1) == len(estaciones):
                log.info(
                    f"  [{i+1}/{len(estaciones)}] "
                    f"{total_registros:,} registros acumulados "
                    f"(errores/sin datos: {errs})"
                )

        # ── Resumen final ────────────────────────────────────────
        log.info(f"\n{'='*60}")
        log.info(f"COMPLETADO:")
        log.info(f"  Estaciones procesadas : {len(estaciones)}")
        log.info(f"  Peticiones sin datos  : {errs} ({errs/total_peticiones*100:.1f}%)")
        log.info(f"  Registros insertados  : {total_registros:,}")

        r = con.execute(
            "SELECT COUNT(DISTINCT cod_provincia), MIN(anio), MAX(anio) "
            "FROM clima_mensual"
        ).fetchone()
        if r and r[0]:
            log.info(f"  Provincias cubiertas  : {r[0]}")
            log.info(f"  Periodo               : {r[1]}-{r[2]}")
        else:
            log.warning("  La tabla clima_mensual sigue vacía. Revisa la API key o conectividad.")

    con.close()


if __name__ == "__main__":
    main()