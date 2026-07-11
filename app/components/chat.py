"""
Componente de chat — integra el agente con el historial de Streamlit.
"""
import os, re, sys
import duckdb
import pandas as pd
import streamlit as st

DB_PATH = os.environ.get("DUCKDB_PATH", "/app/data/duckdb/agrochat.duckdb")

MESES_ES = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"octubre":10,"noviembre":11,"diciembre":12,
}

# Lista de provincias para detección (normalizada sin tildes)
PROVINCIAS_ES = [
    "alava","albacete","alicante","almeria","avila","badajoz","barcelona",
    "burgos","caceres","cadiz","cantabria","castellon","ciudad real","cordoba",
    "cuenca","girona","granada","guadalajara","guipuzcoa","huelva","huesca",
    "jaen","leon","lleida","lugo","madrid","malaga","murcia","navarra",
    "ourense","palencia","pontevedra","rioja","salamanca","segovia","sevilla",
    "soria","tarragona","teruel","toledo","valencia","valladolid","vizcaya",
    "zamora","zaragoza","asturias","balears","coruña","tenerife","palmas",
    # con tildes también
    "álava","ávila","cáceres","cádiz","castellón","córdoba","jaén","león",
    "málaga","almería","cantabria","navarra","aragón","cataluña",
]


def _normalizar(s: str) -> str:
    """Quita tildes para comparación robusta."""
    t = {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u","ñ":"n",
         "Á":"a","É":"e","Í":"i","Ó":"o","Ú":"u","Ü":"u","Ñ":"n"}
    return "".join(t.get(c, c) for c in s).lower()


# ── Inicialización del agente ─────────────────────────────────────

@st.cache_resource
def get_agente(provider: str, model: str):
    agent_dir = "/app/scripts/agent"
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    try:
        from agente import crear_agente
        return crear_agente(provider=provider, model=model, verbose=False)
    except Exception as e:
        st.error(f"Error cargando el agente: {e}")
        return None


def inicializar_historial():
    if "mensajes"    not in st.session_state: st.session_state.mensajes    = []
    if "ultimo_df"   not in st.session_state: st.session_state.ultimo_df   = None
    if "ultima_preg" not in st.session_state: st.session_state.ultima_preg = ""


# ── Extracción de contexto de la pregunta ────────────────────────

def extraer_contexto(pregunta: str) -> dict | None:
    """
    Extrae cultivo, rango temporal, CCAA y si la pregunta es geográfica.
    Devuelve None si no hay suficiente contexto para el mapa.
    """
    p = _normalizar(pregunta)

    # Mapa de CCAA: nombre normalizado → valor en DuckDB
    CCAA_MAP = {
        "andalucia":        "Andalucía",
        "aragon":           "Aragón",
        "asturias":         "Asturias",
        "baleares":         "Islas Baleares",
        "canarias":         "Canarias",
        "cantabria":        "Cantabria",
        "castilla la mancha": "Castilla-La Mancha",
        "castilla leon":    "Castilla y León",
        "cataluna":         "Cataluña",
        "extremadura":      "Extremadura",
        "galicia":          "Galicia",
        "madrid":           "Comunidad de Madrid",
        "murcia":           "Región de Murcia",
        "navarra":          "Navarra",
        "pais vasco":       "País Vasco",
        "euskadi":          "País Vasco",
        "rioja":            "La Rioja",
        "valencia":         "Comunitat Valenciana",
        "valenciana":       "Comunitat Valenciana",
    }

    # Detectar CCAA mencionada
    ccaa_filtro = None
    for key, val in CCAA_MAP.items():
        if key in p:
            ccaa_filtro = val
            break

    # ¿Menciona algo geográfico (provincia o CCAA)?
    es_provincial = (
        "provincia" in p or
        "provincias" in p or
        ccaa_filtro is not None or
        any(_normalizar(prov) in p for prov in PROVINCIAS_ES)
    )
    if not es_provincial:
        return None

    # Cultivo — nombre más largo primero para evitar matches parciales
    cultivos = [
        "trigo blando","trigo duro","trigo","cebada","maiz","girasol",
        "olivar","aceituna","vid","vinedo","arroz","avena","centeno",
        "colza","algodon","remolacha","patata","tomate","pimiento",
        "lechuga","naranja","limon","almendro","algarrobo","garbanzo",
        "lenteja","guisante","jijona","nabo",
    ]
    cultivo = None
    for c in sorted(cultivos, key=len, reverse=True):
        if _normalizar(c) in p:
            cultivo = c.upper()
            break
    if not cultivo:
        return None

    # Años
    anios = [int(a) for a in re.findall(r"20\d{2}", pregunta)]
    anio_ini = min(anios) if anios else 2023
    anio_fin = max(anios) if anios else anio_ini

    # Meses — detectar en ORDEN DE APARICIÓN en la frase, no por valor numérico
    # "Marzo 2022 a Febrero 2023" → ini=Marzo(3), fin=Febrero(2)
    meses_ordenados = []
    for mes, num in MESES_ES.items():
        idx = p.find(mes)
        if idx >= 0:
            meses_ordenados.append((idx, num))
    meses_ordenados.sort(key=lambda x: x[0])  # orden de aparición en el texto

    if len(meses_ordenados) >= 2:
        mes_ini = meses_ordenados[0][1]
        mes_fin = meses_ordenados[-1][1]
    elif len(meses_ordenados) == 1:
        mes_ini = mes_fin = meses_ordenados[0][1]
    else:
        mes_ini, mes_fin = 1, 12

    # Periodo que cruza año: mes_ini > mes_fin Y hay dos años distintos
    cruza_anio = (mes_ini > mes_fin and anio_ini < anio_fin)

    return {
        "cultivo":    cultivo,
        "anio_ini":   anio_ini,
        "anio_fin":   anio_fin,
        "mes_ini":    mes_ini,
        "mes_fin":    mes_fin,
        "ccaa":       ccaa_filtro,     # None = todas las CCAA
        "cruza_anio": cruza_anio,
    }


# ── Consulta DuckDB para el mapa ─────────────────────────────────

def consultar_datos_mapa(pregunta: str) -> pd.DataFrame | None:
    """
    Consulta directa a DuckDB cruzando avances × clima_provincial.
    Filtra por CCAA si se menciona en la pregunta.
    Maneja periodos que cruzan año (ej: marzo 2022 - febrero 2023).
    """
    ctx = extraer_contexto(pregunta)
    if ctx is None:
        return None

    cultivo    = ctx["cultivo"]
    anio_ini   = ctx["anio_ini"]
    anio_fin   = ctx["anio_fin"]
    mes_ini    = ctx["mes_ini"]
    mes_fin    = ctx["mes_fin"]
    ccaa       = ctx["ccaa"]
    cruza_anio = ctx["cruza_anio"]

    # Condición temporal: periodo simple vs. periodo que cruza año
    if cruza_anio:
        # Ej: marzo 2022 - febrero 2023
        # Incluir: (anio=2022 AND mes>=3) OR (anio=2023 AND mes<=2)
        cond_periodo = (
            f"((a.periodo_anio = {anio_ini} AND a.periodo_mes >= {mes_ini}) "
            f"OR (a.periodo_anio = {anio_fin} AND a.periodo_mes <= {mes_fin}))"
        )
    else:
        cond_periodo = (
            f"a.periodo_anio BETWEEN {anio_ini} AND {anio_fin} "
            f"AND a.periodo_mes BETWEEN {mes_ini} AND {mes_fin}"
        )

    # Filtro de CCAA — usa LIKE para ser robusto ante variaciones del valor en DB
    # Ej: "Andalucía" puede estar como "Andalucía" o "ANDALUCIA" según el año de carga
    if ccaa:
        # Extraer palabra clave de la CCAA (la más distintiva, sin artículos)
        ccaa_keyword = ccaa.replace("Comunitat ", "").replace("Comunidad de ", "") \
                           .replace("Región de ", "").replace("Islas ", "") \
                           .replace("La ", "").split()[0]
        cond_ccaa = f"AND UPPER(a.ccaa) LIKE UPPER('%{ccaa_keyword}%')"
    else:
        cond_ccaa = ""

    sql = f"""
        SELECT
            a.provincia,
            a.cod_provincia,
            COUNT(DISTINCT a.periodo_anio || '-' || a.periodo_mes) AS n_periodos,
            ROUND(AVG(a.sup_avance), 0)  AS sup_media_ha,
            ROUND(SUM(a.prod_avance) / NULLIF(SUM(a.sup_avance), 0) * 1000, 1) AS rend_kg_ha,
            ROUND(AVG(cp.t_media_c), 1)   AS temp_media_c,
            ROUND(AVG(cp.precip_media_mm),1) AS precip_media_mm,
            ROUND(AVG(cp.hr_media_pct), 1)   AS hr_media_pct
        FROM avances a
        LEFT JOIN clima_provincial cp
               ON  a.cod_provincia = cp.cod_provincia
               AND a.periodo_anio  = cp.anio
               AND a.periodo_mes   = cp.mes
        WHERE UPPER(a.cultivo) LIKE UPPER(?)
          AND a.nivel = 'provincia'
          AND a.cod_provincia IS NOT NULL
          AND a.sup_avance > 0
          AND a.prod_avance IS NOT NULL
          {cond_ccaa}
          AND {cond_periodo}
        GROUP BY a.provincia, a.cod_provincia
        ORDER BY rend_kg_ha DESC NULLS LAST
    """
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df  = con.execute(sql, [f"%{cultivo}%"]).df()
        con.close()

        if df.empty:
            return None

        st.session_state["mapa_debug"] = {
            "cultivo": cultivo, "anio_ini": anio_ini, "anio_fin": anio_fin,
            "mes_ini": mes_ini, "mes_fin": mes_fin, "ccaa": ccaa or "todas",
            "ccaa_keyword": ccaa_keyword if ccaa else "N/A",
            "cruza_anio": cruza_anio,
            "filas": len(df), "provincias": df["provincia"].tolist()[:8],
            "sql_preview": sql[:300],
        }
        return df

    except Exception as e:
        st.session_state["mapa_error"] = str(e)
        return None


# ── Renderizado del chat ──────────────────────────────────────────

def renderizar_historial():
    for msg in st.session_state.mensajes:
        role = msg["role"]
        with st.chat_message(role, avatar="🌾" if role == "assistant" else "👤"):
            st.markdown(msg["content"])


def procesar_input(pregunta: str, agente) -> tuple[str, pd.DataFrame | None]:
    if agente is None:
        return "El agente no está disponible.", None
    try:
        resultado = agente({"input": pregunta})
        respuesta = resultado.get("output", "Sin respuesta.")
    except Exception as e:
        return f"Error: {e}", None

    df = consultar_datos_mapa(pregunta)
    return respuesta, df


def chat_ui(agente):
    inicializar_historial()
    renderizar_historial()

    if pregunta := st.chat_input("Escribe tu pregunta sobre estadísticas agrarias..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(pregunta)
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        st.session_state.ultima_preg = pregunta

        with st.chat_message("assistant", avatar="🌾"):
            with st.spinner("Consultando datos..."):
                respuesta, df = procesar_input(pregunta, agente)
            st.markdown(respuesta)

        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})

        # Actualizar df ANTES del rerun para que el mapa lo vea
        if df is not None and not df.empty:
            st.session_state.ultimo_df = df
        else:
            # Consulta independiente como fallback
            df_fallback = consultar_datos_mapa(pregunta)
            if df_fallback is not None and not df_fallback.empty:
                st.session_state.ultimo_df = df_fallback

        # Forzar re-render para que la zona del mapa vea el estado actualizado
        st.rerun()

    return st.session_state.get("ultimo_df")


def boton_limpiar():
    if st.button("🗑 Limpiar conversación", use_container_width=True):
        for k in ["mensajes", "ultimo_df", "ultima_preg", "mapa_debug", "mapa_error"]:
            st.session_state.pop(k, None)
        st.rerun()
