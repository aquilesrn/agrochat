# -*- coding: utf-8 -*-
"""
Pagina 1 - Chat conversacional + Mapa coropletico + datos climaticos
Layout: chat arriba (ancho completo), mapa + tabla debajo.
El st.rerun() en chat_ui garantiza que el mapa se renderiza con datos actualizados.
"""
import sys
import streamlit as st

sys.path.insert(0, "/app/app")

from components.chat import chat_ui, boton_limpiar, get_agente
from components.mapa import mapa_coropletico
from components.exportar import boton_descarga

st.set_page_config(
    page_title="Chat y Mapa - AgroChat",
    page_icon="chat",
    layout="wide",
)

st.title("Chat y Mapa")
st.caption("Consulta en lenguaje natural - MAPA & AEMET 2014-2025")

provider = st.session_state.get("llm_provider", "claude")
model    = st.session_state.get("llm_model",    "claude-sonnet-4-6")

# ── ZONA 1: Chat (ancho completo) ─────────────────────────────────
agente = get_agente(provider, model)
if agente is None:
    st.error(
        f"No se pudo inicializar el agente ({provider}/{model}). "
        "Verifica las API keys en .env y reinicia el contenedor."
    )
    st.stop()

df_resultado = chat_ui(agente)

col_b1, col_b2 = st.columns(2)
with col_b1:
    boton_limpiar()
with col_b2:
    if df_resultado is not None:
        boton_descarga(df_resultado, "Resultado_Chat", "Exportar Excel")

st.divider()

# ── ZONA 2: Mapa + Tabla (debajo del chat, despues del rerun) ─────
st.markdown("#### Distribucion geografica")

# Debug info si existe
if st.session_state.get("mapa_debug"):
    dbg = st.session_state["mapa_debug"]
    st.caption(
        f"Datos para: {dbg['cultivo']} | "
        f"{dbg['anio_ini']}/{dbg['mes_ini']:02d} - {dbg['anio_fin']}/{dbg['mes_fin']:02d} | "
        f"{dbg['filas']} provincias encontradas"
    )

if st.session_state.get("mapa_error"):
    st.error(f"Error en consulta: {st.session_state['mapa_error']}")

df_mapa = st.session_state.get("ultimo_df")

if df_mapa is not None and not df_mapa.empty and "cod_provincia" in df_mapa.columns:

    VARS_MAPA = {
        "rend_kg_ha":       "Rendimiento (kg/ha)",
        "sup_media_ha":     "Superficie media (ha)",
        "temp_media_c":     "Temperatura media (C)",
        "precip_media_mm":  "Precipitacion media (mm)",
        "hr_media_pct":     "Humedad relativa (%)",
    }
    vars_disponibles = {
        k: v for k, v in VARS_MAPA.items()
        if k in df_mapa.columns and df_mapa[k].notna().any()
    }

    if vars_disponibles:
        col_mapa, col_tabla = st.columns([3, 2], gap="large")

        with col_mapa:
            col_var = st.selectbox(
                "Variable a visualizar:",
                options=list(vars_disponibles.keys()),
                format_func=lambda x: vars_disponibles[x],
                key="mapa_var_sel",
            )
            escalas = {
                "rend_kg_ha":      "Greens",
                "sup_media_ha":    "Blues",
                "temp_media_c":    "Reds",
                "precip_media_mm": "Blues",
                "hr_media_pct":    "Teal",
            }
            fig = mapa_coropletico(
                df_mapa,
                col_provincia="provincia",
                col_valor=col_var,
                col_id="cod_provincia",
                titulo=vars_disponibles[col_var],
                etiqueta_color=vars_disponibles[col_var],
                escala=escalas.get(col_var, "Greens"),
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "El mapa no pudo generarse (requiere conexion a internet "
                    "para cargar el GeoJSON de provincias). "
                    "Mostrando tabla como alternativa:"
                )
                st.dataframe(
                    df_mapa[["provincia", "rend_kg_ha"]].dropna(),
                    use_container_width=True, hide_index=True
                )

        with col_tabla:
            st.markdown("**Datos por provincia**")
            rename = {
                "provincia":       "Provincia",
                "rend_kg_ha":      "Rend. (kg/ha)",
                "sup_media_ha":    "Sup. (ha)",
                "temp_media_c":    "Temp. (C)",
                "precip_media_mm": "Precip. (mm)",
                "hr_media_pct":    "HR (%)",
                "n_periodos":      "Meses",
            }
            cols_show = [c for c in rename if c in df_mapa.columns]
            df_show = df_mapa[cols_show].rename(columns=rename)
            st.dataframe(df_show, use_container_width=True, hide_index=True)
            boton_descarga(df_mapa, "Datos_provinciales")

            if "temp_media_c" in df_mapa.columns:
                n_sin_clima = int(df_mapa["temp_media_c"].isna().sum())
                if n_sin_clima > 0:
                    st.caption(
                        f"{n_sin_clima} provincia(s) sin datos climaticos AEMET "
                        "(imputadas con mediana de su CCAA)."
                    )
    else:
        st.info("No hay columnas numericas mapeables en los datos.")

else:
    st.info(
        "El mapa aparecera aqui cuando tu pregunta incluya datos por provincia. "
        "Ejemplo: 'Que rendimiento tuvo la cebada en Cordoba en 2022?'"
    )

# ── Ejemplos ──────────────────────────────────────────────────────
with st.expander("Ejemplos de preguntas", expanded=False):
    ejemplos = [
        "Cual fue la produccion nacional de trigo blando en 2023?",
        "Que rendimiento tuvo la cebada en Cordoba entre febrero y octubre de 2022?",
        "Muestra la superficie de olivar por provincia en 2022",
        "Hubo algun ano anomalo en el rendimiento del girasol en Sevilla?",
        "Que es la ESYRCE y que informacion recoge?",
        "Cual fue el precio del trigo blando panificable en la semana 20 de 2023?",
    ]
    for ej in ejemplos:
        st.markdown(f"- {ej}")
