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
) -> go.Figure | None:
    """
    Genera un mapa coroplético de España.

    df debe tener al menos:
      - col_provincia: nombre de la provincia
      - col_valor: valor numérico a representar
      - col_id: código INE (entero)
    """
    geojson = cargar_geojson()
    if geojson is None or df is None or df.empty:
        return None

    # El GeoJSON usa el campo "cod_prov" o similar — inspeccionamos las propiedades
    # y mapeamos cod_provincia INE al featureId
    try:
        fig = px.choropleth(
            df,
            geojson=geojson,
            locations=col_id,
            featureidkey="properties.cod_prov",
            color=col_valor,
            hover_name=col_provincia,
            hover_data={col_id: False, col_valor: ":.3f"},
            color_continuous_scale=escala,
            title=titulo,
            labels={col_valor: etiqueta_color},
        )
    except Exception:
        # Fallback: intentar con 'id' como featureidkey
        try:
            fig = px.choropleth(
                df,
                geojson=geojson,
                locations=col_id,
                featureidkey="id",
                color=col_valor,
                hover_name=col_provincia,
                color_continuous_scale=escala,
                title=titulo,
                labels={col_valor: etiqueta_color},
            )
        except Exception as e:
            st.error(f"Error generando mapa: {e}")
            return None

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
