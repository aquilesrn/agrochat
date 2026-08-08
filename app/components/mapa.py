"""
Mapa coroplético de España por provincia.
Usa Plotly con GeoJSON de provincias INE.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests, json

# GeoJSON de provincias españolas (códigos INE, fuente pública)
GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforgermany/click_that_hood/"
    "main/public/data/spain-provinces.geojson"
)

@st.cache_data(ttl=86400)
def cargar_geojson():
    """Carga el GeoJSON de provincias. Cachea 24h."""
    try:
        r = requests.get(GEOJSON_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"No se pudo cargar el GeoJSON de provincias: {e}")
        return None


def mapa_coropletico(
    df: pd.DataFrame,
    col_provincia: str = "provincia",
    col_valor: str = "valor",
    col_id: str = "cod_provincia",
    titulo: str = "Rendimiento por provincia",
    etiqueta_color: str = "Valor",
    escala: str = "Greens",
    cod_resaltar: str | list | None = None,
) -> go.Figure | None:
    """
    Genera un mapa coroplético de España.

    df debe tener al menos:
      - col_provincia: nombre de la provincia
      - col_valor: valor numérico a representar
      - col_id: código INE (se normaliza internamente a cadena de 2 dígitos)

    cod_resaltar: código(s) INE a resaltar con borde grueso. Puede ser un
    string ('41') o una lista de strings (['01','22','24']) para rankings.
    """
    geojson = cargar_geojson()
    if geojson is None or df is None or df.empty:
        return None

    df = df.copy()

    # ── FIX CLAVE ──────────────────────────────────────────────────
    # El GeoJSON guarda cod_prov como cadena con cero a la izquierda ('07', '41')
    # mientras que DuckDB devuelve cod_provincia como entero (7, 41). Plotly casa
    # locations con featureidkey por igualdad EXACTA, así que 7 != '07' y el mapa
    # salía vacío/incompleto. Se normaliza a cadena de 2 dígitos.
    df["_cod_str"] = (
        df[col_id].astype("Int64").astype(str).str.zfill(2)
    )

    # Normalizar cod_resaltar a lista de cadenas de 2 dígitos
    if cod_resaltar is None:
        resaltar = []
    elif isinstance(cod_resaltar, (list, tuple, set)):
        resaltar = [str(c).zfill(2) for c in cod_resaltar]
    else:
        resaltar = [str(cod_resaltar).zfill(2)]

    try:
        fig = px.choropleth(
            df,
            geojson=geojson,
            locations="_cod_str",
            featureidkey="properties.cod_prov",
            color=col_valor,
            hover_name=col_provincia,
            hover_data={"_cod_str": False, col_valor: ":.3f"},
            color_continuous_scale=escala,
            title=titulo,
            labels={col_valor: etiqueta_color},
        )
    except Exception as e:
        st.error(f"Error generando mapa: {e}")
        return None

    # Capa de resaltado: borde grueso sobre la(s) provincia(s) objetivo.
    if resaltar:
        feats_sel = {
            "type": "FeatureCollection",
            "features": [
                f for f in geojson["features"]
                if f["properties"].get("cod_prov") in resaltar
            ],
        }
        if feats_sel["features"]:
            fig.add_trace(go.Choropleth(
                geojson=feats_sel,
                locations=[f["properties"]["cod_prov"] for f in feats_sel["features"]],
                featureidkey="properties.cod_prov",
                z=[1] * len(feats_sel["features"]),
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_color="#c0392b",
                marker_line_width=2.5,
                hoverinfo="skip",
            ))

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcoastlines=True,
        coastlinecolor="lightgray",
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=420,
        coloraxis_colorbar=dict(title=etiqueta_color, thickness=14, len=0.6),
        title_font_size=14,
    )
    return fig


def mapa_desde_respuesta_agente(texto_respuesta: str, df_datos: pd.DataFrame | None) -> None:
    """
    Intenta mostrar un mapa si la respuesta del agente contiene datos provinciales.
    df_datos: DataFrame con columnas [provincia, cod_provincia, valor_numerico]
    """
    if df_datos is None or df_datos.empty:
        return

    # Detectar columna numérica principal
    cols_num = df_datos.select_dtypes(include="number").columns.tolist()
    cols_num = [c for c in cols_num if c not in ("cod_provincia", "anio", "mes", "semana")]

    if not cols_num or "provincia" not in df_datos.columns or "cod_provincia" not in df_datos.columns:
        return

    col_val = cols_num[0]
    df_mapa = df_datos[["provincia", "cod_provincia", col_val]].dropna()
    if df_mapa.empty:
        return

    fig = mapa_coropletico(
        df_mapa,
        col_provincia="provincia",
        col_valor=col_val,
        col_id="cod_provincia",
        titulo=f"Distribución provincial — {col_val}",
        etiqueta_color=col_val,
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
