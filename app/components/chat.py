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


# Nombre de provincia normalizado → código INE (2 dígitos, con cero a la izquierda).
# Se usa para filtrar el mapa por una provincia concreta y para el resaltado.
PROVINCIA_A_COD = {
    "alava":"01","araba":"01","albacete":"02","alicante":"03","alacant":"03",
    "almeria":"04","avila":"05","badajoz":"06","baleares":"07","balears":"07",
    "barcelona":"08","burgos":"09","caceres":"10","cadiz":"11","castellon":"12",
    "ciudad real":"13","cordoba":"14","coruna":"15","a coruna":"15","cuenca":"16",
    "girona":"17","granada":"18","guadalajara":"19","guipuzcoa":"20","gipuzkoa":"20",
    "huelva":"21","huesca":"22","jaen":"23","leon":"24","lleida":"25","lerida":"25",
    "rioja":"26","la rioja":"26","lugo":"27","madrid":"28","malaga":"29","murcia":"30",
    "navarra":"31","ourense":"32","orense":"32","asturias":"33","palencia":"34",
    "palmas":"35","las palmas":"35","pontevedra":"36","salamanca":"37","tenerife":"38",
    "cantabria":"39","segovia":"40","sevilla":"41","soria":"42","tarragona":"43",
    "teruel":"44","toledo":"45","valencia":"46","valladolid":"47","vizcaya":"48",
    "bizkaia":"48","zamora":"49","zaragoza":"50","ceuta":"51","melilla":"52",
}

# Palabras que indican una pregunta de tipo ranking ("qué provincia tuvo el mayor…").
# En estos casos el mapa debe resaltar el top-N aunque no se nombre una provincia.
PALABRAS_RANKING = [
    "que provincia", "cual provincia", "mayor", "menor", "mas alto", "mas bajo",
    "maximo", "minimo", "ranking", "top", "lidera", "encabeza", "principales",
    "que comunidad", "cuales provincias", "que provincias",
]


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

    # Detectar provincia CONCRETA mencionada (además de la CCAA).
    # Se recorre el diccionario por clave más larga primero ("ciudad real"
    # antes que cualquier subcadena) para evitar falsos positivos.
    provincia_filtro = None      # nombre normalizado tal como aparece
    cod_provincia    = None      # código INE de 2 dígitos
    for nombre in sorted(PROVINCIA_A_COD, key=len, reverse=True):
        if re.search(r"\b" + re.escape(nombre) + r"\b", p):
            provincia_filtro = nombre
            cod_provincia    = PROVINCIA_A_COD[nombre]
            break

    # ¿Es una pregunta de tipo ranking? (top de provincias)
    es_ranking = any(kw in p for kw in PALABRAS_RANKING)

    # ¿Menciona algo geográfico (provincia, CCAA o ranking)?
    es_provincial = (
        "provincia" in p or
        "provincias" in p or
        ccaa_filtro is not None or
        cod_provincia is not None or
        es_ranking
    )
    if not es_provincial:
        return None

    # Cultivo — nombre más largo primero para evitar matches parciales.
    # Se añaden agregados frecuentes ("cereales de invierno") que no son un
    # cultivo único pero sí una consulta provincial legítima.
    cultivos = [
        "cereales de invierno","cereal de invierno","cereales","cereal",
        "trigo blando","trigo duro","trigo","cebada","maiz","girasol",
        "olivar","aceituna","vid","vinedo","arroz","avena","centeno",
        "colza","algodon","remolacha","patata","tomate","pimiento",
        "lechuga","naranja","limon","almendro","algarrobo","garbanzo",
        "lenteja","guisante","jijona","nabo","citricos","citrico",
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
        "cultivo":       cultivo,
        "anio_ini":      anio_ini,
        "anio_fin":      anio_fin,
        "mes_ini":       mes_ini,
        "mes_fin":       mes_fin,
        "ccaa":          ccaa_filtro,      # None = todas las CCAA
        "provincia":     provincia_filtro, # None = sin filtro de provincia
        "cod_provincia": cod_provincia,    # código INE de la provincia (o None)
        "es_ranking":    es_ranking,       # True si es pregunta de tipo top-N
        "cruza_anio":    cruza_anio,
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

    cultivo       = ctx["cultivo"]
    anio_ini      = ctx["anio_ini"]
    anio_fin      = ctx["anio_fin"]
    mes_ini       = ctx["mes_ini"]
    mes_fin       = ctx["mes_fin"]
    ccaa          = ctx["ccaa"]
    provincia     = ctx["provincia"]
    cod_provincia = ctx["cod_provincia"]
    es_ranking    = ctx["es_ranking"]
    cruza_anio    = ctx["cruza_anio"]

    # Condición temporal: periodo simple vs. periodo que cruza año.
    # Se genera dos veces: para la tabla avances (alias a.) y para clima (alias cp.),
    # porque las columnas se llaman distinto (periodo_anio/periodo_mes vs anio/mes).
    if cruza_anio:
        # Ej: marzo 2022 - febrero 2023
        # Incluir: (anio=2022 AND mes>=3) OR (anio=2023 AND mes<=2)
        cond_periodo = (
            f"((a.periodo_anio = {anio_ini} AND a.periodo_mes >= {mes_ini}) "
            f"OR (a.periodo_anio = {anio_fin} AND a.periodo_mes <= {mes_fin}))"
        )
        cond_periodo_clima = (
            f"((cp.anio = {anio_ini} AND cp.mes >= {mes_ini}) "
            f"OR (cp.anio = {anio_fin} AND cp.mes <= {mes_fin}))"
        )
    else:
        cond_periodo = (
            f"a.periodo_anio BETWEEN {anio_ini} AND {anio_fin} "
            f"AND a.periodo_mes BETWEEN {mes_ini} AND {mes_fin}"
        )
        cond_periodo_clima = (
            f"cp.anio BETWEEN {anio_ini} AND {anio_fin} "
            f"AND cp.mes BETWEEN {mes_ini} AND {mes_fin}"
        )

    # Si se nombró una provincia concreta pero NO una CCAA, derivamos la CCAA de
    # esa provincia para que el mapa muestre la región completa con la provincia
    # objetivo resaltada (en lugar de un mapa de una sola provincia, poco legible).
    # El resaltado lo aplica la capa de mapa a partir de cod_provincia.
    if cod_provincia and not ccaa:
        try:
            con_tmp = duckdb.connect(DB_PATH, read_only=True)
            fila = con_tmp.execute(
                "SELECT ccaa FROM avances WHERE cod_provincia = ? "
                "AND ccaa IS NOT NULL LIMIT 1", [int(cod_provincia)]
            ).fetchone()
            con_tmp.close()
            if fila and fila[0]:
                ccaa = fila[0]
        except Exception:
            pass  # si falla, se deja sin filtro y se muestra todo el país

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

    # ── FIX DEL BUG DE AGREGACIÓN Y DESGLOSE DE CULTIVOS ──────────
    # Los avances del MAPA son ACUMULATIVOS: cada mes repite la superficie y la
    # producción estimadas hasta la fecha. Además, para un mismo cultivo coexisten
    # la fila agregada y sus desgloses (p. ej. CEBADA TOTAL junto a CEBADA DE DOS
    # CARRERAS y CEBADA DE SEIS CARRERAS), todas con regimen='total'. Sumar todo
    # inflaba las cifras por partida doble.
    #
    # HEURÍSTICA (nomenclatura del MAPA irregular; ver limitaciones en la memoria):
    #   1) Se excluyen los agregados redundantes ("totales de totales" y resúmenes):
    #      POR VARIEDADES / POR TIPOS / POR ÉPOCAS / RESUMEN GENERAL, y las variantes
    #      marcadas con (*) que duplican columnas de control.
    #   2) Si para la familia consultada existe una fila con TOTAL, se usa SOLO esa
    #      (flag es_total) y se descartan sus componentes. Si no hay TOTAL, se usan
    #      las filas base que cumplen el LIKE.
    #   3) Cierre de campaña = último periodo_mes CON producción no nula.
    #   4) Rendimiento = SUM(prod)/SUM(sup)*1e6 (media ponderada por superficie).
    #      prod_avance está en MILES de toneladas: kg/ha = (miles_t * 1e6) / ha.
    #   El clima se une DESPUÉS de agregar, para no multiplicar filas.
    sql = f"""
        WITH filtrado AS (
            SELECT
                a.provincia, a.cod_provincia, a.cultivo,
                a.periodo_anio, a.periodo_mes,
                a.sup_avance, a.prod_avance,
                CASE WHEN UPPER(a.cultivo) LIKE '%TOTAL%' THEN 1 ELSE 0 END AS es_total
            FROM avances a
            WHERE UPPER(a.cultivo) LIKE UPPER(?)
              AND a.nivel = 'provincia'
              AND a.cod_provincia IS NOT NULL
              AND a.sup_avance > 0
              AND a.prod_avance IS NOT NULL
              -- Excluir agregados redundantes y variantes de control:
              AND UPPER(a.cultivo) NOT LIKE '%POR VARIEDADES%'
              AND UPPER(a.cultivo) NOT LIKE '%POR TIPOS%'
              AND UPPER(a.cultivo) NOT LIKE '%POR ÉPOCAS%'
              AND UPPER(a.cultivo) NOT LIKE '%POR EPOCAS%'
              AND UPPER(a.cultivo) NOT LIKE '%RESUMEN GENERAL%'
              AND a.cultivo NOT LIKE '%(*)%'
              {cond_ccaa}
              AND {cond_periodo}
        ),
        marca_total AS (
            -- ¿Existe alguna fila TOTAL para el conjunto filtrado? Si la hay,
            -- nos quedaremos solo con las TOTAL; si no, con todas las base.
            SELECT *, MAX(es_total) OVER () AS hay_total FROM filtrado
        ),
        seleccion AS (
            SELECT * FROM marca_total
            WHERE (hay_total = 1 AND es_total = 1)
               OR (hay_total = 0)
        ),
        cierre AS (
            -- Una sola fila por cultivo/provincia/año: el último mes con producción.
            SELECT s.*
            FROM seleccion s
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY s.cod_provincia, s.cultivo, s.periodo_anio
                ORDER BY s.periodo_mes DESC
            ) = 1
        ),
        clima AS (
            SELECT cp.cod_provincia,
                   AVG(cp.t_media_c)      AS temp_media_c,
                   AVG(cp.precip_media_mm) AS precip_media_mm,
                   AVG(cp.hr_media_pct)   AS hr_media_pct
            FROM clima_provincial cp
            WHERE {cond_periodo_clima}
              AND cp.mes BETWEEN 1 AND 12
            GROUP BY cp.cod_provincia
        )
        SELECT
            c.provincia,
            c.cod_provincia,
            COUNT(DISTINCT c.cultivo)                     AS n_cultivos,
            COUNT(DISTINCT c.periodo_anio)                AS n_periodos,
            ROUND(SUM(c.sup_avance), 0)                   AS sup_media_ha,
            ROUND(SUM(c.prod_avance) / NULLIF(SUM(c.sup_avance), 0) * 1000000, 1) AS rend_kg_ha,
            ROUND(cl.temp_media_c, 1)                     AS temp_media_c,
            ROUND(cl.precip_media_mm, 1)                  AS precip_media_mm,
            ROUND(cl.hr_media_pct, 1)                     AS hr_media_pct
        FROM cierre c
        LEFT JOIN clima cl ON c.cod_provincia = cl.cod_provincia
        GROUP BY c.provincia, c.cod_provincia,
                 cl.temp_media_c, cl.precip_media_mm, cl.hr_media_pct
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

        # Información para el resaltado del mapa (la consume 1_Chat_y_Mapa.py):
        #  - cod_resaltar: código INE de la provincia objetivo (o None)
        #  - es_ranking: si True, la página resaltará el top-5 por rend_kg_ha
        st.session_state["mapa_resaltado"] = {
            "cod_resaltar": cod_provincia,
            "es_ranking":   es_ranking,
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
        for k in ["mensajes", "ultimo_df", "ultima_preg", "mapa_debug",
                  "mapa_error", "mapa_resaltado"]:
            st.session_state.pop(k, None)
        st.rerun()
