# -*- coding: utf-8 -*-
"""
Pagina 2 - Dashboard de analisis
Tabs: Evolucion temporal | Anomalias | Explicabilidad SHAP | Coyuntura
"""
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, "/app/app")

from utils.db        import query_df, cultivos_disponibles, anios_disponibles
from utils.modelo    import get_shap_summary, get_shap_values, get_metricas
from components.graficos  import (
    grafico_anomalias, grafico_shap_global,
    grafico_shap_cultivo, grafico_precios,
)
from components.exportar import boton_descarga

st.set_page_config(
    page_title="Analisis - AgroChat",
    page_icon="chart",
    layout="wide",
)

st.title("Dashboard de analisis")
st.caption("Evolucion temporal - Deteccion de anomalias - Explicabilidad SHAP - Coyuntura")

MESES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
         7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

tab1, tab2, tab3, tab4 = st.tabs([
    "Evolucion temporal",
    "Deteccion de anomalias",
    "Explicabilidad SHAP",
    "Precios de coyuntura",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 - EVOLUCION TEMPORAL con filtros en cascada
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Evolucion de superficies y producciones")

    # Fila 1: Cultivo | Desde | Hasta | Metrica
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    cultivos = cultivos_disponibles()
    with c1:
        cultivo_sel = st.selectbox(
            "Cultivo", cultivos,
            index=cultivos.index("TRIGO BLANDO") if "TRIGO BLANDO" in cultivos else 0,
            key="ev_cultivo",
        )
    anios = anios_disponibles()
    with c2:
        anio_ini = st.selectbox("Desde", anios, index=0, key="ev_ini")
    with c3:
        anio_fin = st.selectbox("Hasta", anios, index=len(anios)-1, key="ev_fin")
    with c4:
        metrica = st.selectbox(
            "Metrica",
            ["sup_avance", "prod_avance"],
            format_func=lambda x: "Superficie (ha)" if x == "sup_avance"
                                  else "Produccion (miles t)",
            key="ev_metrica",
        )

    # Fila 2: Nivel (solo Nacional y CCAA — Provincia se gestiona desde CCAA)
    nivel = st.radio(
        "Nivel",
        ["Nacional", "CCAA"],
        horizontal=True,
        key="ev_nivel",
    )

    # Fila 3: Filtros geograficos en cascada
    nivel_sql    = "nacional"
    col_zona_sql = "'Espana'"
    where_extra  = []
    params_extra = []
    titulo_geo   = "Espana (nacional)"
    provincias_sel = []   # lista de provincias seleccionadas (multiselect)

    if nivel == "Nacional":
        nivel_sql    = "nacional"
        col_zona_sql = "'Espana'"
        titulo_geo   = "Espana (nacional)"

    else:  # CCAA
        df_ccaas = query_df("""
            SELECT DISTINCT ccaa FROM avances
            WHERE nivel = 'ccaa' AND ccaa IS NOT NULL
              AND UPPER(cultivo) = UPPER(?)
            ORDER BY ccaa
        """, [cultivo_sel])
        ccaas = df_ccaas["ccaa"].tolist() if df_ccaas is not None else []

        col_ccaa, col_prov = st.columns([1, 2])
        with col_ccaa:
            ccaa_sel = st.selectbox("Comunidad autonoma", ccaas, key="ev_ccaa")

        # Provincias de esa CCAA para ese cultivo
        df_prov_ccaa = query_df("""
            SELECT DISTINCT provincia FROM avances
            WHERE nivel = 'provincia' AND UPPER(ccaa) = UPPER(?)
              AND UPPER(cultivo) = UPPER(?) AND provincia IS NOT NULL
            ORDER BY provincia
        """, [ccaa_sel, cultivo_sel]) if ccaa_sel else None
        provs_ccaa = df_prov_ccaa["provincia"].tolist() if df_prov_ccaa is not None else []

        with col_prov:
            # Multiselect: sin seleccion = mostrar CCAA agregada; con seleccion = esas provincias
            provincias_sel = st.multiselect(
                "Provincias (vacio = toda la CCAA agregada)",
                options=provs_ccaa,
                default=[],
                key="ev_prov_multi",
                placeholder="Selecciona una o varias provincias...",
            )

        if not provincias_sel:
            # Sin provincias seleccionadas: datos agregados a nivel CCAA
            nivel_sql    = "ccaa"
            col_zona_sql = "ccaa"
            where_extra  = ["UPPER(ccaa) = UPPER(?)"]
            params_extra = [ccaa_sel]
            titulo_geo   = ccaa_sel
        else:
            # Una o varias provincias: comparativa
            nivel_sql    = "provincia"
            col_zona_sql = "provincia"
            placeholders = ", ".join(["?"] * len(provincias_sel))
            where_extra  = [f"provincia IN ({placeholders})"]
            params_extra = provincias_sel
            if len(provincias_sel) == 1:
                titulo_geo = f"{provincias_sel[0]} ({ccaa_sel})"
            else:
                titulo_geo = f"{len(provincias_sel)} provincias de {ccaa_sel}"

    # Construir y ejecutar la consulta
    where_base = [
        "UPPER(cultivo) = UPPER(?)",
        f"nivel = '{nivel_sql}'",
        "regimen = 'total'",
        "periodo_anio BETWEEN ? AND ?",
        f"{metrica} IS NOT NULL",
    ]
    params = [cultivo_sel, anio_ini, anio_fin] + params_extra
    where_all = " AND ".join(where_base + where_extra)

    sql_ev = f"""
        SELECT periodo_anio, periodo_mes, {metrica}, {col_zona_sql} AS zona
        FROM avances
        WHERE {where_all}
        ORDER BY zona, periodo_anio, periodo_mes
    """
    df_ev = query_df(sql_ev, params)

    if df_ev is not None and not df_ev.empty:
        df_ev["periodo"] = (
            df_ev["periodo_anio"].astype(str) + "-" +
            df_ev["periodo_mes"].map(lambda m: MESES.get(m, str(m)))
        )
        etiqueta = "Superficie (ha)" if metrica == "sup_avance" else "Produccion (miles t)"
        n_zonas  = df_ev["zona"].nunique()

        fig = px.line(
            df_ev,
            x="periodo",
            y=metrica,
            color="zona" if n_zonas > 1 else None,
            title=f"{cultivo_sel} - {etiqueta} | {titulo_geo} ({anio_ini}-{anio_fin})",
            labels={"periodo": "Periodo", metrica: etiqueta, "zona": ""},
            markers=True,
        )
        fig.update_layout(
            height=440,
            margin={"t": 60, "b": 70},
            xaxis_tickangle=-45,
            xaxis=dict(tickmode="auto", nticks=20),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        if n_zonas > 1:
            st.caption(
                f"Mostrando {n_zonas} series. "
                "Haz clic en la leyenda para mostrar/ocultar cada una."
            )

        with st.expander("Ver datos"):
            st.dataframe(df_ev, use_container_width=True, hide_index=True)
            boton_descarga(df_ev, "Evolucion_temporal")
    else:
        st.info("Sin datos para los filtros seleccionados.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 - DETECCION DE ANOMALIAS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Deteccion de anomalias en rendimientos historicos")
    st.caption(
        "Metodo: Z-score (umbral 2.5 sigma) e IQR (factor 2.0). "
        "Las anomalias son meses con rendimientos inusuales respecto al historico."
    )

    ca1, ca2, ca3 = st.columns([2, 2, 1])
    cultivos_an = cultivos_disponibles()
    with ca1:
        cultivo_an = st.selectbox(
            "Cultivo", cultivos_an, key="an_cultivo",
            index=cultivos_an.index("TRIGO BLANDO") if "TRIGO BLANDO" in cultivos_an else 0,
        )
    with ca2:
        df_provs = query_df("""
            SELECT DISTINCT provincia FROM avances
            WHERE nivel='provincia' AND provincia IS NOT NULL
            ORDER BY provincia
        """)
        provs = df_provs["provincia"].tolist() if df_provs is not None else []
        prov_sel = st.selectbox(
            "Provincia", provs, key="an_prov",
            index=provs.index("Toledo") if "Toledo" in provs else 0,
        )
    with ca3:
        metodo_an = st.selectbox("Metodo", ["ambos", "zscore", "iqr"], key="an_metodo")

    df_serie = query_df("""
        SELECT periodo_anio AS anio, periodo_mes AS mes,
               ROUND(prod_avance / NULLIF(sup_avance, 0), 6) AS rendimiento
        FROM avances
        WHERE UPPER(cultivo) = UPPER(?)
          AND nivel = 'provincia'
          AND LOWER(provincia) = LOWER(?)
          AND regimen = 'total'
          AND sup_avance > 0
          AND prod_avance IS NOT NULL
          AND periodo_anio BETWEEN 2014 AND 2025
        ORDER BY anio, mes
    """, [cultivo_an, prov_sel])

    if df_serie is not None and not df_serie.empty:
        df_serie["periodo"] = (
            df_serie["anio"].astype(str) + "-" +
            df_serie["mes"].map(lambda m: MESES.get(m, str(m)))
        )
        r = df_serie["rendimiento"]
        media, std = r.mean(), r.std()
        q1, q3 = r.quantile(0.25), r.quantile(0.75)
        iqr = q3 - q1

        mask = pd.Series(False, index=r.index)
        if metodo_an in ("zscore", "ambos") and std > 0:
            mask |= ((r - media) / std).abs() > 2.5
        if metodo_an in ("iqr", "ambos") and iqr > 0:
            mask |= (r < q1 - 2 * iqr) | (r > q3 + 2 * iqr)

        df_anom = df_serie[mask].copy()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Observaciones", f"{len(df_serie)}")
        col_m2.metric("Anomalias", f"{len(df_anom)}",
                      delta=f"{len(df_anom)/len(df_serie)*100:.1f}%")
        col_m3.metric("Rendimiento medio", f"{media:.4f} miles t/ha")
        col_m4.metric("Desv. estandar", f"{std:.4f}")

        fig_an = grafico_anomalias(df_serie, df_anom, cultivo_an, prov_sel)
        st.plotly_chart(fig_an, use_container_width=True)

        if not df_anom.empty:
            st.markdown("**Anomalias detectadas:**")
            df_anom_show = df_anom[["anio", "mes", "rendimiento"]].copy()
            df_anom_show["z_score"] = ((df_anom["rendimiento"] - media) / std).round(2)
            df_anom_show["direccion"] = df_anom_show["z_score"].apply(
                lambda z: "Alto" if z > 0 else "Bajo"
            )
            st.dataframe(df_anom_show, use_container_width=True, hide_index=True)
            boton_descarga(df_anom_show, "Anomalias")
        else:
            st.success(f"No se detectaron anomalias para {cultivo_an} en {prov_sel}.")
    else:
        st.info("Sin datos para los filtros seleccionados.")


# ══════════════════════════════════════════════════════════════════
# TAB 3 - EXPLICABILIDAD SHAP
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Explicabilidad del modelo - SHAP values")

    metricas = get_metricas()
    if metricas:
        cs1, cs2, cs3 = st.columns(3)
        cs1.metric("R2 train", metricas.get("train", {}).get("r2", "-"))
        cs2.metric("R2 val",   metricas.get("val",   {}).get("r2", "-"))
        cs3.metric("R2 test",  metricas.get("test",  {}).get("r2", "-"))

    st.divider()

    df_summary = get_shap_summary()
    df_shap    = get_shap_values()

    if df_summary.empty:
        st.warning("Artefactos SHAP no encontrados. Ejecuta scripts/models/entrenar_modelo.py")
    else:
        col_sh1, col_sh2 = st.columns(2, gap="large")

        with col_sh1:
            st.markdown("##### Importancia global de variables")
            fig_shap = grafico_shap_global(df_summary)
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption(
                "El tipo de cultivo domina con diferencia (SHAP medio aprox. 0.017). "
                "El efecto climatico es mas visible cuando se filtra por un cultivo concreto."
            )

        with col_sh2:
            st.markdown("##### SHAP por cultivo")
            if not df_shap.empty and "cultivo" in df_shap.columns:
                cultivos_shap = sorted(df_shap["cultivo"].unique().tolist())
                cultivo_shap = st.selectbox(
                    "Cultivo", cultivos_shap,
                    index=cultivos_shap.index("TRIGO BLANDO")
                          if "TRIGO BLANDO" in cultivos_shap else 0,
                    key="shap_cultivo",
                )
                fig_shap_c = grafico_shap_cultivo(df_shap, cultivo_shap)
                st.plotly_chart(fig_shap_c, use_container_width=True)
                st.caption(
                    "Barras positivas: la variable aumenta el rendimiento predicho. "
                    "Barras negativas: la variable lo reduce."
                )
            else:
                st.info("SHAP por cultivo no disponible.")

        with st.expander("Ver tabla completa de importancia"):
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            boton_descarga(df_summary, "SHAP_summary")


# ══════════════════════════════════════════════════════════════════
# TAB 4 - COYUNTURA / PRECIOS
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Evolucion de precios semanales - Coyuntura agraria (MAPA)")

    cp1, cp2, cp3 = st.columns([2, 1, 1])

    df_prods = query_df("""
        SELECT DISTINCT producto, seccion FROM coyuntura
        ORDER BY seccion, producto
    """)
    productos = df_prods["producto"].tolist() if df_prods is not None else []

    with cp1:
        prod_sel = st.selectbox(
            "Producto", productos,
            index=productos.index("Trigo blando panificable")
                  if "Trigo blando panificable" in productos else 0,
            key="coy_prod",
        )
    anios_coy = query_df("SELECT DISTINCT anio FROM coyuntura ORDER BY anio")
    anios_coy = anios_coy["anio"].tolist() if anios_coy is not None else [2022, 2025]
    with cp2:
        anio_coy_ini = st.selectbox("Desde", anios_coy, index=0, key="coy_ini")
    with cp3:
        anio_coy_fin = st.selectbox(
            "Hasta", anios_coy, index=len(anios_coy)-1, key="coy_fin"
        )

    df_coy = query_df("""
        SELECT anio, semana, producto, unidad,
               precio_sem_anterior, precio_sem_actual, variacion_pct
        FROM coyuntura
        WHERE producto = ?
          AND anio BETWEEN ? AND ?
        ORDER BY anio, semana
    """, [prod_sel, anio_coy_ini, anio_coy_fin])

    if df_coy is not None and not df_coy.empty:
        cp_m1, cp_m2, cp_m3 = st.columns(3)
        ultimo = df_coy.iloc[-1]
        cp_m1.metric(
            "Ultimo precio",
            f"{ultimo['precio_sem_actual']:.2f} {ultimo['unidad']}",
            delta=f"{ultimo['variacion_pct']:+.1f}% vs sem. anterior"
                  if pd.notna(ultimo["variacion_pct"]) else None,
        )
        cp_m2.metric("Precio minimo",
                     f"{df_coy['precio_sem_actual'].min():.2f} {ultimo['unidad']}")
        cp_m3.metric("Precio maximo",
                     f"{df_coy['precio_sem_actual'].max():.2f} {ultimo['unidad']}")

        fig_coy = grafico_precios(df_coy, prod_sel)
        st.plotly_chart(fig_coy, use_container_width=True)

        with st.expander("Ver datos de precios"):
            st.dataframe(df_coy, use_container_width=True, hide_index=True)
            boton_descarga(df_coy, f"Precios_{prod_sel[:20]}")
    else:
        st.info("Sin datos de coyuntura para los filtros seleccionados.")
