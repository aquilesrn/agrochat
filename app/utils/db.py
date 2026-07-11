"""
Conexión a DuckDB — singleton por sesión Streamlit.
"""
import os
import duckdb
import streamlit as st

DB_PATH = os.environ.get("DUCKDB_PATH", "/app/data/duckdb/agrochat.duckdb")


@st.cache_resource
def get_connection():
    """Conexión de solo lectura compartida entre componentes."""
    return duckdb.connect(DB_PATH, read_only=True)


def query_df(sql: str, params: list = None):
    """Ejecuta SQL y devuelve DataFrame. Maneja errores con mensaje claro."""
    try:
        con = get_connection()
        if params:
            return con.execute(sql, params).df()
        return con.execute(sql).df()
    except Exception as e:
        st.error(f"Error en consulta: {e}")
        return None


# ── Consultas reutilizables ───────────────────────────────────────

def cultivos_disponibles() -> list:
    df = query_df("""
        SELECT DISTINCT cultivo FROM avances
        WHERE nivel = 'nacional' AND sup_avance IS NOT NULL
        ORDER BY cultivo
    """)
    return df["cultivo"].tolist() if df is not None else []


def provincias_disponibles() -> list:
    df = query_df("""
        SELECT DISTINCT provincia, cod_provincia FROM avances
        WHERE nivel = 'provincia' AND provincia IS NOT NULL
        ORDER BY provincia
    """)
    return df.to_dict("records") if df is not None else []


def anios_disponibles() -> list:
    df = query_df("""
        SELECT DISTINCT periodo_anio FROM avances
        WHERE periodo_anio BETWEEN 2014 AND 2025
        ORDER BY periodo_anio
    """)
    return df["periodo_anio"].tolist() if df is not None else []
