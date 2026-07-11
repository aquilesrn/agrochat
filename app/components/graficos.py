"""
Componentes de visualización reutilizables para el dashboard.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MESES = {
    1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
    7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic",
}


# ── Evolución temporal ────────────────────────────────────────────

def grafico_evolucion(df: pd.DataFrame, cultivo: str, metrica: str = "sup_avance") -> go.Figure:
    """
    Serie temporal de sup_avance o prod_avance para un cultivo.
    df debe tener: periodo_anio, periodo_mes, metrica, [nivel/provincia/ccaa]
    """
    if df is None or df.empty:
        return go.Figure()

    df = df.copy()
    df["periodo"] = df["periodo_anio"].astype(str) + "-" + df["periodo_mes"].map(
        lambda m: MESES.get(m, str(m))
    )

    etiqueta = {
        "sup_avance":   "Superficie (ha)",
        "prod_avance":  "Producción (miles t)",
        "rendimiento":  "Rendimiento (miles t/ha)",
    }.get(metrica, metrica)

    color_col = None
    for c in ["provincia", "ccaa", "nivel"]:
        if c in df.columns and df[c].nunique() > 1:
            color_col = c
            break

    fig = px.line(
        df, x="periodo", y=metrica, color=color_col,
        title=f"{cultivo} — {etiqueta}",
        labels={"periodo": "Periodo", metrica: etiqueta},
        markers=True,
    )
    fig.update_layout(
        height=380, margin={"t": 50, "b": 40},
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Detección de anomalías ────────────────────────────────────────

def grafico_anomalias(df_serie: pd.DataFrame, df_anomalias: pd.DataFrame,
                      cultivo: str, provincia: str) -> go.Figure:
    """
    Serie temporal con puntos de anomalía marcados en rojo.
    """
    if df_serie is None or df_serie.empty:
        return go.Figure()

    fig = go.Figure()

    # Serie completa
    fig.add_trace(go.Scatter(
        x=df_serie["periodo"],
        y=df_serie["rendimiento"],
        mode="lines+markers",
        name="Rendimiento",
        line=dict(color="#1f6b3a", width=2),
        marker=dict(size=5),
    ))

    # Banda ±2σ
    media = df_serie["rendimiento"].mean()
    std   = df_serie["rendimiento"].std()
    fig.add_hrect(
        y0=media - 2.5 * std, y1=media + 2.5 * std,
        fillcolor="rgba(31,107,58,0.07)", line_width=0,
        annotation_text="±2.5σ", annotation_position="top left",
    )
    fig.add_hline(y=media, line_dash="dash", line_color="gray",
                  annotation_text=f"Media: {media:.4f}", annotation_position="right")

    # Anomalías
    if df_anomalias is not None and not df_anomalias.empty:
        fig.add_trace(go.Scatter(
            x=df_anomalias["periodo"],
            y=df_anomalias["rendimiento"],
            mode="markers",
            name="Anomalía",
            marker=dict(color="red", size=12, symbol="diamond",
                        line=dict(width=1.5, color="darkred")),
        ))

    fig.update_layout(
        title=f"Anomalías — {cultivo} en {provincia}",
        xaxis_title="Periodo",
        yaxis_title="Rendimiento (miles t/ha)",
        height=400,
        margin={"t": 50, "b": 40},
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Panel SHAP ────────────────────────────────────────────────────

def grafico_shap_global(df_summary: pd.DataFrame) -> go.Figure:
    """Barplot horizontal de importancia global (mean |SHAP|)."""
    if df_summary is None or df_summary.empty:
        return go.Figure()

    nombres = {
        "cultivo_enc":     "Tipo de cultivo",
        "provincia_enc":   "Provincia",
        "ccaa_enc":        "Comunidad autónoma",
        "anio":            "Año",
        "mes":             "Mes",
        "t_media_c":       "Temperatura media",
        "precip_media_mm": "Precipitación",
        "hr_media_pct":    "Humedad relativa",
    }
    df = df_summary.copy()
    df["feature_label"] = df["feature"].map(lambda x: nombres.get(x, x))
    df = df.sort_values("shap_mean_abs", ascending=True)

    fig = px.bar(
        df, x="shap_mean_abs", y="feature_label",
        orientation="h",
        title="Importancia de variables — media |SHAP|",
        labels={"shap_mean_abs": "Importancia media |SHAP|", "feature_label": "Variable"},
        color="shap_mean_abs",
        color_continuous_scale="Greens",
    )
    fig.update_layout(
        height=350, margin={"t": 50, "b": 20},
        coloraxis_showscale=False,
        showlegend=False,
    )
    return fig


def grafico_shap_cultivo(df_shap: pd.DataFrame, cultivo: str) -> go.Figure:
    """SHAP values por provincia para un cultivo concreto."""
    if df_shap is None or df_shap.empty or "cultivo" not in df_shap.columns:
        return go.Figure()

    df = df_shap[df_shap["cultivo"] == cultivo].copy()
    if df.empty:
        return go.Figure()

    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    if not shap_cols:
        return go.Figure()

    nombres = {
        "shap_cultivo_enc":     "Tipo de cultivo",
        "shap_provincia_enc":   "Provincia",
        "shap_ccaa_enc":        "CCAA",
        "shap_anio":            "Año",
        "shap_mes":             "Mes",
        "shap_t_media_c":       "Temperatura",
        "shap_precip_media_mm": "Precipitación",
        "shap_hr_media_pct":    "Humedad",
    }

    medias = df[shap_cols].mean().reset_index()
    medias.columns = ["feature", "shap_medio"]
    medias["feature_label"] = medias["feature"].map(lambda x: nombres.get(x, x))
    medias = medias.sort_values("shap_medio")

    colors = ["#d73027" if v < 0 else "#1f6b3a" for v in medias["shap_medio"]]

    fig = go.Figure(go.Bar(
        x=medias["shap_medio"],
        y=medias["feature_label"],
        orientation="h",
        marker_color=colors,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"SHAP medio — {cultivo}",
        xaxis_title="Contribución SHAP media",
        height=350, margin={"t": 50, "b": 20},
    )
    return fig


# ── Coyuntura / Precios ───────────────────────────────────────────

def grafico_precios(df: pd.DataFrame, producto: str) -> go.Figure:
    """Serie temporal de precio semanal con banda de variación."""
    if df is None or df.empty:
        return go.Figure()

    df = df.copy().sort_values(["anio", "semana"])
    df["periodo"] = df["anio"].astype(str) + "-S" + df["semana"].astype(str).str.zfill(2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["periodo"], y=df["precio_sem_actual"],
        mode="lines+markers",
        name="Precio",
        line=dict(color="#1f6b3a", width=2),
        marker=dict(size=4),
    ))
    if "precio_sem_anterior" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["periodo"], y=df["precio_sem_anterior"],
            mode="lines",
            name="Semana anterior",
            line=dict(color="lightgray", width=1, dash="dot"),
        ))

    unidad = df["unidad"].iloc[0] if "unidad" in df.columns else "€"
    fig.update_layout(
        title=f"{producto} — Evolución de precios",
        xaxis_title="Semana",
        yaxis_title=f"Precio ({unidad})",
        height=380, margin={"t": 50, "b": 40},
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
