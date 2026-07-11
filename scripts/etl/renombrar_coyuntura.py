"""
══════════════════════════════════════════════════════════════════
Renombrar archivos de Informes Semanales de Coyuntura
══════════════════════════════════════════════════════════════════
Lee cada archivo Excel, extrae año y semana del contenido (Pág. 4),
y renombra al formato estándar: coyuntura_S{semana:02d}_{año}.xlsx

Uso:
  python renombrar_coyuntura.py --input-dir ./data/raw/coyuntura --dry-run
  python renombrar_coyuntura.py --input-dir ./data/raw/coyuntura

Si se pasa --input-dir, recorre las subcarpetas (2022/, 2023/, etc.).
Si se pasa --input-file, renombra un solo archivo.
"""

import argparse, glob, logging, os, re, sys, shutil
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("renombrar")


def detect_year_week(filepath: str) -> tuple[int, int]:
    """Extrae año y número de semana del contenido de Pág. 4."""
    xls = pd.ExcelFile(filepath, engine="openpyxl")
    if "Pág. 4" not in xls.sheet_names:
        raise ValueError(f"No se encontró la hoja 'Pág. 4'")

    df = pd.read_excel(xls, sheet_name="Pág. 4", header=None)

    # Semana: R6/C4 contiene "Semana XX" (semana actual del informe)
    week = None
    for c in [4, 3]:
        v = df.iloc[6, c] if df.shape[1] > c else None
        if pd.notna(v):
            m = re.search(r'Semana\s+(\d+)', str(v))
            if m:
                week = int(m.group(1))
                break
    if week is None:
        raise ValueError("No se pudo detectar el número de semana")

    # Año: buscar en R8/C4, R8/C3, R7/C4, R7/C3
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
    if year is None:
        raise ValueError("No se pudo detectar el año")

    return year, week


def main():
    ap = argparse.ArgumentParser(description="Renombrar archivos de coyuntura")
    ap.add_argument("--input-dir", help="Directorio raíz (con subcarpetas por año)")
    ap.add_argument("--input-file", help="Archivo individual")
    ap.add_argument("--dry-run", action="store_true", help="Solo mostrar cambios")
    args = ap.parse_args()

    files = []
    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        for ext in ("*.xlsx", "*.xlsm", "*.xls"):
            files.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
        files = sorted(set(files))
    else:
        log.error("Especifica --input-dir o --input-file"); sys.exit(1)

    if not files:
        log.error("No se encontraron archivos"); sys.exit(1)

    log.info(f"Archivos encontrados: {len(files)}")

    renamed = 0
    skipped = 0
    errors = []

    for filepath in files:
        fname = Path(filepath).name
        parent = Path(filepath).parent

        # Ignorar archivos ya renombrados
        if re.match(r'^coyuntura_S\d{2}_\d{4}\.xlsx$', fname):
            log.info(f"  SKIP (ya renombrado): {fname}")
            skipped += 1
            continue

        try:
            year, week = detect_year_week(filepath)
            new_name = f"coyuntura_S{week:02d}_{year}.xlsx"
            new_path = parent / new_name

            if new_path.exists() and str(new_path) != str(filepath):
                log.warning(f"  CONFLICTO: {fname} -> {new_name} (ya existe)")
                errors.append((filepath, f"destino ya existe: {new_name}"))
                continue

            if args.dry_run:
                log.info(f"  DRY-RUN: {fname}  ->  {new_name}")
            else:
                os.rename(filepath, new_path)
                log.info(f"  OK: {fname}  ->  {new_name}")
            renamed += 1

        except Exception as e:
            log.error(f"  ERROR: {fname}: {e}")
            errors.append((filepath, str(e)))

    log.info(f"\n{'='*60}")
    log.info(f"RESUMEN: {renamed} renombrados, {skipped} ya correctos, {len(errors)} errores")
    if errors:
        for f, e in errors:
            log.warning(f"  {Path(f).name}: {e}")


if __name__ == "__main__":
    main()
