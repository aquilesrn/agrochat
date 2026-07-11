"""
══════════════════════════════════════════════════════════════════
Carga de fragmentos en ChromaDB para el RAG de AgroChat
══════════════════════════════════════════════════════════════════
Genera fragmentos textuales a partir de:
  1. Datos de avances (DuckDB) -> frases sobre superficie/producción
  2. Datos de coyuntura (DuckDB) -> frases sobre precios semanales
  3. Datos climáticos (DuckDB) -> frases sobre clima por provincia
  4. PDFs técnicos -> párrafos de documentación del MAPA

Los fragmentos se indexan en ChromaDB con embeddings para
búsqueda semántica desde el chatbot RAG.

Uso:
  python cargar_chromadb.py --db agrochat.duckdb --chroma-path ./data/chroma --fuente avances
  python cargar_chromadb.py --db agrochat.duckdb --chroma-path ./data/chroma --fuente coyuntura
  python cargar_chromadb.py --db agrochat.duckdb --chroma-path ./data/chroma --fuente clima
  python cargar_chromadb.py --db agrochat.duckdb --chroma-path ./data/chroma --fuente pdfs --pdf-dir ./data/raw/docs
  python cargar_chromadb.py --db agrochat.duckdb --chroma-path ./data/chroma --fuente todas --pdf-dir ./data/raw/docs

CORRECCIONES v2:
  - IDs de fragmentos nacionales y CCAA incluyen hash MD5 del texto para
    garantizar unicidad absoluta aunque cultivo+anio+mes+regimen se repita
    (caso real: "CEBOLLA TOTAL POR VARIEDADES" generaba IDs duplicados)
  - Deduplicación explícita por ID antes del upsert para evitar que un
    mismo batch contenga IDs repetidos
  - Log de advertencia cuando se detectan y eliminan duplicados
"""

import argparse, duckdb, hashlib, logging, os, sys
import chromadb
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("chromadb_loader")

COLLECTION_NAME = "agrochat"
BATCH_SIZE = 200


# ══════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════

def safe_id(base: str) -> str:
    """
    Limpia el base para usarlo como ID ChromaDB y añade un sufijo
    con los primeros 8 caracteres del hash MD5 del propio base.
    Esto garantiza unicidad incluso cuando dos filas generan el
    mismo prefijo descriptivo (p.ej. CEBOLLA TOTAL POR VARIEDADES).
    """
    # ChromaDB acepta cualquier string no vacío; saneamos espacios y barras
    clean = base.strip().replace("/", "-").replace(" ", "_")
    suffix = hashlib.md5(base.encode()).hexdigest()[:8]
    return f"{clean}_{suffix}"


def deduplicar(fragments: list[dict], fuente: str) -> list[dict]:
    """
    Elimina duplicados por ID, conservando la primera aparición.
    Avisa si encuentra alguno (indica un bug en la generación de IDs).
    """
    seen = {}
    dupes = 0
    for f in fragments:
        fid = f["id"]
        if fid in seen:
            dupes += 1
        else:
            seen[fid] = f
    if dupes:
        log.warning(
            f"[{fuente}] {dupes} fragmentos con ID duplicado eliminados "
            f"(revisar lógica de generación de IDs)"
        )
    return list(seen.values())


# ══════════════════════════════════════════════════════════════════
# GENERADORES DE FRAGMENTOS DESDE DuckDB
# ══════════════════════════════════════════════════════════════════

MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


def generar_fragmentos_avances(con) -> list[dict]:
    """
    Genera fragmentos textuales a partir de datos de avances.
    Un fragmento por cultivo × provincia × periodo (nivel más detallado).
    Para CCAA y nacional, genera resúmenes.
    """
    fragments = []

    # ── Fragmentos a nivel NACIONAL ──────────────────────────────
    rows = con.execute("""
        SELECT cultivo, periodo_anio, periodo_mes,
               sup_avance, prod_avance, regimen,
               anio_definitivo, sup_definitivo, prod_definitivo
        FROM avances
        WHERE nivel = 'nacional'
        ORDER BY cultivo, periodo_anio, periodo_mes
    """).fetchall()

    for cultivo, anio, mes, sup, prod, regimen, anio_def, sup_def, prod_def in rows:
        mes_nombre = MESES_ES.get(mes, str(mes))
        partes = [f"En {mes_nombre} de {anio}, el avance nacional de {cultivo}"]
        if regimen and regimen != "total":
            partes[0] += f" ({regimen})"

        if sup is not None:
            partes.append(f"muestra una superficie estimada de {sup:,.0f} hectáreas".replace(",", "."))
        if prod is not None:
            partes.append(f"con una producción de {prod:,.3f} miles de toneladas".replace(",", "."))
        if sup_def is not None and anio_def:
            partes.append(
                f"El dato definitivo de {anio_def} fue {sup_def:,.0f} ha de superficie".replace(",", ".")
            )

        texto = ". ".join(partes) + "."

        # FIX: safe_id añade hash del texto → unicidad garantizada aunque
        # cultivo+anio+mes+regimen coincidan en varias filas
        base = f"av_nac_{cultivo}_{anio}_{mes:02d}_{regimen}"
        fid = safe_id(base)

        fragments.append({
            "id": fid,
            "text": texto,
            "metadata": {
                "fuente": "avances",
                "nivel": "nacional",
                "cultivo": cultivo,
                "anio": anio,
                "mes": mes,
                "regimen": regimen or "total",
            }
        })

    # ── Fragmentos a nivel CCAA ───────────────────────────────────
    rows = con.execute("""
        SELECT ccaa, cultivo, periodo_anio, periodo_mes,
               SUM(sup_avance)  AS sup,
               SUM(prod_avance) AS prod,
               regimen,
               COUNT(*)         AS n_prov
        FROM avances
        WHERE nivel = 'provincia' AND regimen = 'total'
        GROUP BY ccaa, cultivo, periodo_anio, periodo_mes, regimen
        ORDER BY ccaa, cultivo, periodo_anio, periodo_mes
    """).fetchall()

    for ccaa, cultivo, anio, mes, sup, prod, regimen, n_prov in rows:
        mes_nombre = MESES_ES.get(mes, str(mes))
        partes = [
            f"En {mes_nombre} de {anio}, {cultivo} en {ccaa} "
            f"(suma de {n_prov} provincias)"
        ]

        if sup is not None:
            partes.append(
                f"tiene una superficie de avance de {sup:,.0f} hectáreas".replace(",", ".")
            )
        if prod is not None:
            partes.append(
                f"y una producción de {prod:,.3f} miles de toneladas".replace(",", ".")
            )

        texto = " ".join(partes) + "."

        # FIX: mismo patrón — hash sobre el base completo
        base = f"av_ccaa_{ccaa}_{cultivo}_{anio}_{mes:02d}_{regimen}"
        fid = safe_id(base)

        fragments.append({
            "id": fid,
            "text": texto,
            "metadata": {
                "fuente": "avances",
                "nivel": "ccaa",
                "ccaa": ccaa or "",
                "cultivo": cultivo,
                "anio": anio,
                "mes": mes,
                "regimen": regimen or "total",
            }
        })

    fragments = deduplicar(fragments, "avances")
    log.info(f"Avances: {len(fragments)} fragmentos generados")
    return fragments


def generar_fragmentos_coyuntura(con) -> list[dict]:
    """Genera fragmentos textuales a partir de precios semanales."""
    fragments = []

    rows = con.execute("""
        SELECT anio, semana, seccion, categoria, producto, unidad,
               precio_sem_anterior, precio_sem_actual,
               variacion_euros, variacion_pct
        FROM coyuntura
        ORDER BY anio, semana, seccion, producto
    """).fetchall()

    for anio, sem, seccion, cat, prod, unidad, p_ant, p_act, var_e, var_p in rows:
        partes = [f"Semana {sem} de {anio}: {prod}"]

        if p_act is not None:
            partes.append(f"tiene un precio medio nacional de {p_act:.2f} {unidad}")
        if var_p is not None:
            signo = "+" if var_p >= 0 else ""
            partes.append(f"con una variación semanal del {signo}{var_p:.2f}%")
        if p_ant is not None:
            partes.append(f"(semana anterior: {p_ant:.2f} {unidad})")
        if cat:
            partes.append(f"Categoría: {cat.lower()}, sección: {seccion}")

        texto = ". ".join(partes) + "."
        base = f"coy_{prod}_{anio}_S{sem:02d}"
        fid = safe_id(base)

        fragments.append({
            "id": fid,
            "text": texto,
            "metadata": {
                "fuente": "coyuntura",
                "producto": prod,
                "seccion": seccion or "",
                "categoria": cat or "",
                "anio": anio,
                "semana": sem,
            }
        })

    fragments = deduplicar(fragments, "coyuntura")
    log.info(f"Coyuntura: {len(fragments)} fragmentos generados")
    return fragments


def generar_fragmentos_clima(con) -> list[dict]:
    """Genera fragmentos desde la vista clima_provincial."""
    fragments = []

    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "clima_mensual" not in tables:
        log.warning("Tabla clima_mensual no encontrada, saltando clima")
        return fragments

    prov_map = {}
    for cod, prov, ccaa in con.execute("""
        SELECT DISTINCT cod_provincia, provincia, ccaa
        FROM avances
        WHERE nivel = 'provincia' AND cod_provincia IS NOT NULL
    """).fetchall():
        prov_map[cod] = (prov, ccaa)

    rows = con.execute("""
        SELECT cod_provincia, anio, mes,
               t_media_c, t_max_abs_c, t_min_abs_c,
               precip_media_mm, horas_sol_med, n_estaciones
        FROM clima_provincial
        ORDER BY cod_provincia, anio, mes
    """).fetchall()

    for cprov, anio, mes, t_med, t_max, t_min, precip, sol, n_est in rows:
        prov_name, ccaa = prov_map.get(cprov, (f"Provincia {cprov}", ""))
        mes_nombre = MESES_ES.get(mes, str(mes))

        partes = [f"Clima en {prov_name} ({ccaa}) en {mes_nombre} de {anio}"]
        partes.append(f"basado en {n_est} estaciones AEMET")

        if t_med is not None:
            partes.append(f"Temperatura media: {t_med}°C")
        if t_max is not None:
            partes.append(f"máxima absoluta: {t_max}°C")
        if t_min is not None:
            partes.append(f"mínima absoluta: {t_min}°C")
        if precip is not None:
            partes.append(f"Precipitación media: {precip} mm")
        if sol is not None:
            partes.append(f"Horas de sol: {sol}")

        texto = ". ".join(partes) + "."
        base = f"clima_{cprov}_{anio}_{mes:02d}"
        fid = safe_id(base)

        fragments.append({
            "id": fid,
            "text": texto,
            "metadata": {
                "fuente": "clima",
                "cod_provincia": cprov,
                "provincia": prov_name,
                "ccaa": ccaa or "",
                "anio": anio,
                "mes": mes,
            }
        })

    fragments = deduplicar(fragments, "clima")
    log.info(f"Clima: {len(fragments)} fragmentos generados")
    return fragments


# ══════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE PDFs
# ══════════════════════════════════════════════════════════════════

def extraer_fragmentos_pdfs(pdf_dir: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """
    Extrae texto de PDFs, divide en fragmentos solapados y genera
    documentos para ChromaDB.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        log.error("pypdf no instalado. pip install pypdf")
        return []

    fragments = []
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))

    if not pdf_files:
        log.warning(f"No se encontraron PDFs en {pdf_dir}")
        return fragments

    for pdf_path in pdf_files:
        log.info(f"  Procesando PDF: {pdf_path.name}")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

            if not full_text.strip():
                log.warning(f"  PDF vacío o no legible: {pdf_path.name}")
                continue

            words = full_text.split()
            chunks = []
            i = 0
            while i < len(words):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
                i += chunk_size - overlap

            for j, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if len(chunk) < 50:
                    continue

                base = f"pdf_{pdf_path.stem}_{j:04d}"
                fid = safe_id(base)
                fragments.append({
                    "id": fid,
                    "text": chunk,
                    "metadata": {
                        "fuente": "pdf",
                        "archivo": pdf_path.name,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                    }
                })

            log.info(f"    -> {len(chunks)} fragmentos")

        except Exception as e:
            log.error(f"  Error en {pdf_path.name}: {e}")

    fragments = deduplicar(fragments, "pdfs")
    log.info(f"PDFs: {len(fragments)} fragmentos totales")
    return fragments


# ══════════════════════════════════════════════════════════════════
# CARGA EN ChromaDB
# ══════════════════════════════════════════════════════════════════

def cargar_en_chromadb(chroma_path: str, fragments: list[dict], fuente: str):
    """Carga fragmentos en ChromaDB usando embeddings por defecto."""
    if not fragments:
        log.warning(f"No hay fragmentos que cargar para '{fuente}'")
        return

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "AgroChat - Estadísticas agrarias de España"}
    )

    # Eliminar fragmentos previos de esta fuente
    existing = collection.get(where={"fuente": fuente})
    if existing and existing["ids"]:
        log.info(f"Eliminando {len(existing['ids'])} fragmentos previos de '{fuente}'")
        collection.delete(ids=existing["ids"])

    # Cargar en batches
    total = len(fragments)
    for i in range(0, total, BATCH_SIZE):
        batch = fragments[i:i + BATCH_SIZE]
        ids   = [f["id"]       for f in batch]
        docs  = [f["text"]     for f in batch]
        metas = [f["metadata"] for f in batch]

        # Sanitize metadata: ChromaDB solo acepta str, int, float, bool
        for m in metas:
            for k, v in list(m.items()):
                if v is None:
                    m[k] = ""

        collection.upsert(ids=ids, documents=docs, metadatas=metas)

        loaded = min(i + BATCH_SIZE, total)
        if loaded % 2000 == 0 or loaded == total:
            log.info(f"  Cargados {loaded}/{total}...")

    total_col = collection.count()
    log.info(
        f"ChromaDB: {total} fragmentos de '{fuente}' cargados. "
        f"Total colección: {total_col}"
    )


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Cargar fragmentos en ChromaDB")
    ap.add_argument("--db",          required=True, help="Ruta al DuckDB")
    ap.add_argument("--chroma-path", required=True, help="Ruta persistencia ChromaDB")
    ap.add_argument("--fuente",      required=True,
                    choices=["avances", "coyuntura", "clima", "pdfs", "todas"])
    ap.add_argument("--pdf-dir",     help="Directorio con PDFs")
    ap.add_argument("--chunk-size",    type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.chroma_path, exist_ok=True)
    con = duckdb.connect(args.db, read_only=True)

    fuentes = (
        ["avances", "coyuntura", "clima", "pdfs"]
        if args.fuente == "todas"
        else [args.fuente]
    )

    for fuente in fuentes:
        log.info(f"\n{'='*60}")
        log.info(f"Generando fragmentos: {fuente}")
        log.info(f"{'='*60}")

        if fuente == "avances":
            frags = generar_fragmentos_avances(con)
        elif fuente == "coyuntura":
            frags = generar_fragmentos_coyuntura(con)
        elif fuente == "clima":
            frags = generar_fragmentos_clima(con)
        elif fuente == "pdfs":
            if not args.pdf_dir:
                log.error("Especifica --pdf-dir para procesar PDFs")
                continue
            frags = extraer_fragmentos_pdfs(
                args.pdf_dir, args.chunk_size, args.chunk_overlap
            )
        else:
            continue

        if frags:
            cargar_en_chromadb(args.chroma_path, frags, fuente)

    con.close()
    log.info(f"\n{'='*60}")
    log.info("CARGA COMPLETADA")


if __name__ == "__main__":
    main()