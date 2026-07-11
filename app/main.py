"""
AgroChat — Dashboard principal
Punto de entrada de la aplicación Streamlit.
"""
import os
import streamlit as st

st.set_page_config(
    page_title="AgroChat",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global mínimo ─────────────────────────────────────────────
st.markdown("""
<style>
/* Reducir padding superior */
.block-container { padding-top: 1.5rem; }
/* Chat input fijo al fondo */
.stChatInputContainer { border-top: 1px solid #e0e0e0; padding-top: 0.5rem; }
/* Encabezados de sección */
.section-header {
    font-size: 0.8rem; font-weight: 600; color: #666;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar: configuración global ────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/"
             "Barley_field.jpg/320px-Barley_field.jpg",
             width=260)
    st.title("🌾 AgroChat")
    st.caption("Estadísticas agrarias · MAPA & AEMET · 2014–2025")
    st.divider()

    st.markdown('<p class="section-header">Proveedor LLM</p>', unsafe_allow_html=True)
    provider = st.selectbox(
        "Proveedor",
        options=["claude", "gemini", "openai", "ollama"],
        index=0,
        label_visibility="collapsed",
    )

    modelos = {
        "claude":  ["claude-sonnet-4-6", "claude-haiku-3-5"],
        "gemini":  ["gemini-1.5-flash",  "gemini-1.5-pro"],
        "openai":  ["gpt-4o",            "gpt-4o-mini"],
        "ollama":  ["llama3.1:8b",       "qwen2.5:3b", "mistral:7b"],
    }
    model = st.selectbox(
        "Modelo",
        options=modelos.get(provider, ["llama3.1:8b"]),
        label_visibility="collapsed",
    )

    # Guardar en session_state para que los componentes lo lean
    st.session_state["llm_provider"] = provider
    st.session_state["llm_model"]    = model

    st.divider()
    st.caption("TFM · Máster Big Data y Ciencia de Datos · VIU 2025–2026")

# ── Página de bienvenida (home) ───────────────────────────────────
st.title("🌾 AgroChat")
st.markdown("""
Bienvenido al sistema inteligente de consulta y análisis de estadísticas agrarias españolas.

Selecciona una sección en el menú lateral:

| Sección | Descripción |
|---|---|
| 💬 **Chat y Mapa** | Consulta en lenguaje natural + mapa coroplético |
| 📊 **Análisis** | Evolución temporal, anomalías, SHAP y coyuntura |
""")

import sys
sys.path.insert(0, "/app/app")
from utils.modelo import get_metricas

_metricas = get_metricas()
_r2_test  = _metricas.get("test",  {}).get("r2", "0.827")
_r2_train = _metricas.get("train", {}).get("r2", "0.887")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Fragmentos ChromaDB", "175.881", help="Indexados para búsqueda semántica")
with col2:
    st.metric("Registros DuckDB", "536.485", help="Avances + coyuntura + clima AEMET")
with col3:
    st.metric(
        "XGBoost R² (test)",
        f"{_r2_test}",
        delta=f"train: {_r2_train}",
        help="R² en conjunto de test (15% datos, nunca vistos por el modelo). "
             "El de entrenamiento es siempre mayor — el de test es el relevante.",
    )

st.info("💡 Configura el proveedor LLM en el menú lateral antes de usar el chat.")
