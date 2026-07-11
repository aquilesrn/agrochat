"""
Carga de artefactos del modelo XGBoost + SHAP.
"""
import json
import os
import pandas as pd
import streamlit as st
import xgboost as xgb

MODELS_DIR  = "/app/data/models"
PROCESS_DIR = "/app/data/processed"


@st.cache_resource
def get_model():
    path = os.path.join(MODELS_DIR, "xgboost_rendimiento.json")
    if not os.path.exists(path):
        return None
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m


@st.cache_data
def get_shap_summary() -> pd.DataFrame:
    path = os.path.join(MODELS_DIR, "shap_summary.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def get_shap_values() -> pd.DataFrame:
    path = os.path.join(MODELS_DIR, "shap_values.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def get_metricas() -> dict:
    path = os.path.join(MODELS_DIR, "metricas_modelo.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def get_label_encoders() -> dict:
    path = os.path.join(PROCESS_DIR, "label_encoders.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
