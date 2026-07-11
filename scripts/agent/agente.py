"""
══════════════════════════════════════════════════════════════════
AgroChat — Agente multi-proveedor (LangChain 1.x)
══════════════════════════════════════════════════════════════════
Soporta cuatro proveedores de LLM seleccionables por argumento:

  --provider ollama  → Ollama local (llama3.1, qwen2.5, mistral...)
  --provider claude  → Anthropic Claude API (claude-sonnet-4-5...)
  --provider gemini  → Google Gemini API (gemini-1.5-flash...)
  --provider openai  → OpenAI API (gpt-4o...)

Herramientas:
  1. buscar_documentos  — RAG semántico sobre ChromaDB
  2. consultar_datos    — NL→SQL sobre DuckDB
  3. detectar_anomalias — Z-score + IQR sobre rendimientos históricos

Uso:
  python scripts/agente.py --provider ollama --model llama3.1:8b --verbose
  python scripts/agente.py --provider gemini --model gemini-1.5-flash
  python scripts/agente.py --provider claude --model claude-sonnet-4-5
  python scripts/agente.py --provider ollama --test
  python scripts/agente.py --provider ollama --pregunta "Produccion trigo 2023"
"""

import argparse, json, logging, os, re, sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("agente")

DB_PATH     = os.environ.get("DUCKDB_PATH", "/app/data/duckdb/agrochat.duckdb")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/app/data/chroma")
COLLECTION  = "agrochat"

DEFAULT_MODELS = {
    "ollama": "llama3.1:8b",
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-1.5-flash",
    "openai": "gpt-4o",
}

# ══════════════════════════════════════════════════════════════════
# CARGA DE LLM POR PROVEEDOR
# ══════════════════════════════════════════════════════════════════

def cargar_llm(provider: str, model: str, temperature: float = 0):
    p = provider.lower()
    if p == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
        try:
            from langchain_ollama import ChatOllama
            log.info(f"Ollama @ {host} | modelo: {model}")
            return ChatOllama(model=model, base_url=host, temperature=temperature)
        except ImportError:
            log.error("Instala: langchain-ollama>=0.2.0  (añade al requirements.txt y reconstruye)")
            sys.exit(1)

    elif p == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            log.error("ANTHROPIC_API_KEY no definida en .env")
            log.error("Obtén tu key en https://console.anthropic.com/settings/keys")
            log.error("NOTA: El plan Pro de claude.ai NO incluye acceso a la API.")
            log.error("La API tiene facturacion por tokens independiente del plan Pro.")
            sys.exit(1)
        try:
            from langchain_anthropic import ChatAnthropic
            log.info(f"Claude API | modelo: {model}")
            return ChatAnthropic(model=model, api_key=key,
                                 temperature=temperature, max_tokens=4096)
        except ImportError:
            log.error("Instala: langchain-anthropic>=0.3.0  (añade al requirements.txt y reconstruye)")
            sys.exit(1)

    elif p == "gemini":
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            log.error("GOOGLE_API_KEY no definida en .env")
            log.error("Obtén tu key GRATUITA en https://aistudio.google.com/apikey")
            log.error("Capa gratuita: 15 RPM, 1M tokens/dia con gemini-1.5-flash")
            sys.exit(1)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            log.info(f"Gemini API | modelo: {model}")
            return ChatGoogleGenerativeAI(model=model, google_api_key=key,
                                          temperature=temperature)
        except ImportError:
            log.error("Instala: langchain-google-genai>=2.0.0  (añade al requirements.txt y reconstruye)")
            sys.exit(1)

    elif p == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            log.error("OPENAI_API_KEY no definida en .env")
            sys.exit(1)
        try:
            from langchain_openai import ChatOpenAI
            log.info(f"OpenAI API | modelo: {model}")
            return ChatOpenAI(model=model, temperature=temperature)
        except ImportError:
            log.error("Instala: langchain-openai>=0.2.0")
            sys.exit(1)
    else:
        log.error(f"Proveedor desconocido: '{p}'. Opciones: ollama, claude, gemini, openai")
        sys.exit(1)


def cargar_llm_sql(provider: str):
    """LLM ligero para traduccion NL→SQL."""
    sql_model = {
        "ollama": os.environ.get("OLLAMA_MODEL", DEFAULT_MODELS["ollama"]),
        "claude": "claude-sonnet-4-6",   # mismo modelo: haiku-3-5 da 404
        "gemini": "gemini-1.5-flash",
        "openai": "gpt-4o-mini",
    }.get(provider, DEFAULT_MODELS.get(provider, "llama3.1:8b"))
    return cargar_llm(provider, sql_model, temperature=0)


# ══════════════════════════════════════════════════════════════════
# HERRAMIENTA 1: RAG
# ══════════════════════════════════════════════════════════════════

import chromadb
from chromadb.utils import embedding_functions

_chroma_col = None

def _init_chroma():
    global _chroma_col
    if _chroma_col is None:
        ef = embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_col = client.get_collection(name=COLLECTION, embedding_function=ef)
    return _chroma_col


@tool
def buscar_documentos(query: str) -> str:
    """Busca informacion cualitativa sobre estadisticas agrarias: avances MAPA,
    precios de coyuntura, datos climaticos y documentacion tecnica (ESYRCE,
    calendarios de siembra, glosarios). Usar para definiciones, metodologia
    o contexto general sobre un cultivo o producto.
    Input: texto de busqueda en lenguaje natural."""
    try:
        col = _init_chroma()
        r = col.query(query_texts=[query], n_results=5,
                      include=["documents", "metadatas", "distances"])
        docs, metas, dists = r["documents"][0], r["metadatas"][0], r["distances"][0]
        if not docs:
            return "Sin resultados relevantes."
        partes = []
        for doc, meta, d in zip(docs, metas, dists):
            ctx = f"[{meta.get('fuente','?')}"
            for k in ["cultivo","provincia","anio"]:
                if meta.get(k): ctx += f"|{meta[k]}"
            ctx += f"|rel:{round((1-d)*100,1)}%]"
            partes.append(f"{ctx}\n{doc}")
        return "\n---\n".join(partes)
    except Exception as e:
        return f"Error ChromaDB: {e}"


# ══════════════════════════════════════════════════════════════════
# HERRAMIENTA 2: SQL
# ══════════════════════════════════════════════════════════════════

SCHEMA = """
TABLAS DuckDB (agrochat.duckdb):
avances: cultivo, nivel('nacional'/'ccaa'/'provincia'), ccaa, provincia,
  cod_provincia, periodo_anio(2014-2026), periodo_mes(1-12),
  regimen('secano'/'regadio'/'total'), sup_avance(ha), prod_avance(miles t)
coyuntura: anio, semana, seccion, producto, unidad,
  precio_sem_anterior, precio_sem_actual, variacion_pct
clima_provincial(vista): cod_provincia, anio, mes,
  t_media_c, precip_media_mm, hr_media_pct
NOTAS: nivel='nacional' para total España. prod_avance en MILES de toneladas.
Filtrar: AND sup_avance IS NOT NULL AND prod_avance IS NOT NULL. LIMIT 20.
"""

_llm_sql = None  # se inicializa en crear_agente()


@tool
def consultar_datos(consulta: str) -> str:
    """Consulta datos cuantitativos: superficies, producciones, rendimientos,
    precios. Usar para numeros concretos, rankings, comparativas entre provincias
    o anios. Input: descripcion en lenguaje natural, NO SQL directo.
    Ejemplo: 'produccion nacional trigo blando 2023'"""
    global _llm_sql
    if _llm_sql is None:
        return "Error: LLM SQL no inicializado."
    prompt = (f"Solo devuelve SQL DuckDB valido, sin markdown.\n"
              f"ESQUEMA:\n{SCHEMA}\nCONSULTA: {consulta}")
    try:
        raw = _llm_sql.invoke(prompt).content.strip()
        sql = re.sub(r"```(?:sql)?", "", raw).strip().strip("`").strip()
        log.info(f"SQL: {sql[:120]}")
    except Exception as e:
        return f"Error generando SQL: {e}"
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df  = con.execute(sql).df()
        con.close()
    except Exception as e:
        return f"Error SQL: {e}\nSQL intentado: {sql}"
    if df.empty:
        return f"Sin resultados.\nSQL: {sql}"
    return f"Resultado ({len(df)} filas):\n{df.to_string(index=False, max_rows=20)}\nSQL: {sql}"


# ══════════════════════════════════════════════════════════════════
# HERRAMIENTA 3: ANOMALIAS
# ══════════════════════════════════════════════════════════════════

@tool
def detectar_anomalias(parametros: str) -> str:
    """Detecta rendimientos agricolas anomalos (muy altos o bajos) para un
    cultivo y/o provincia usando Z-score e IQR. Usar para anos excepcionales,
    malas cosechas o rendimientos inusuales.
    Input JSON: cultivo(str), provincia(str), anio_ini(int), anio_fin(int),
    metodo('zscore'/'iqr'/'ambos'). Ej: {"cultivo":"TRIGO BLANDO","provincia":"Toledo"}"""
    try:
        params = json.loads(parametros)
    except (json.JSONDecodeError, TypeError):
        params = {}
        for f in ["cultivo","provincia"]:
            m = re.search(rf'{f}["\s:=]+([A-ZÁÉÍÓÚÜÑ\s]+)',
                          str(parametros), re.IGNORECASE)
            if m:
                v = m.group(1).strip()
                params[f] = v.upper() if f=="cultivo" else v.title()

    cultivo   = params.get("cultivo","").upper().strip()
    provincia = params.get("provincia","").strip().title()
    a0, a1    = int(params.get("anio_ini",2014)), int(params.get("anio_fin",2025))
    metodo    = params.get("metodo","ambos")

    where = ["nivel='provincia'","regimen='total'","sup_avance>0",
             "prod_avance IS NOT NULL", f"periodo_anio BETWEEN {a0} AND {a1}"]
    if cultivo:   where.append(f"UPPER(cultivo) LIKE '%{cultivo}%'")
    if provincia: where.append(f"LOWER(provincia) LIKE '%{provincia.lower()}%'")

    sql = (f"SELECT cultivo,provincia,periodo_anio anio,periodo_mes mes,"
           f"ROUND(prod_avance/sup_avance,6) rendimiento "
           f"FROM avances WHERE {' AND '.join(where)} ORDER BY cultivo,provincia,anio,mes")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df  = con.execute(sql).df()
        con.close()
    except Exception as e:
        return f"Error consulta: {e}"

    if df.empty:
        return f"Sin datos: cultivo={cultivo or 'todos'}, provincia={provincia or 'todas'}"

    lines, n = [], 0
    for (c,p), g in df.groupby(["cultivo","provincia"]):
        r = g["rendimiento"].dropna()
        if len(r) < 5: continue
        mask = pd.Series(False, index=r.index)
        if metodo in ("zscore","ambos") and r.std()>0:
            mask |= ((r-r.mean())/r.std()).abs() > 2.5
        if metodo in ("iqr","ambos"):
            q1,q3 = r.quantile(.25), r.quantile(.75)
            iq = q3-q1
            if iq>0: mask |= (r < q1-2*iq)|(r > q3+2*iq)
        filas = g[mask]; n += len(filas)
        if not filas.empty:
            lines.append(f"\n{c} | {p}  media={r.mean():.4f} std={r.std():.4f}")
            for _,row in filas.iterrows():
                z = (row.rendimiento-r.mean())/r.std() if r.std()>0 else 0
                lines.append(f"  • {int(row.anio)}-{int(row.mes):02d}: "
                             f"{row.rendimiento:.4f} ({'alto' if z>0 else 'bajo'} z={z:.2f})")
    if not lines:
        return f"Sin anomalias. n={len(df)}, metodo={metodo}"
    return f"ANOMALIAS | {cultivo or 'todos'} | {provincia or 'todas'} | {a0}-{a1} | n={len(df)} | detectadas={n}" + "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# AGENTE LCEL
# ══════════════════════════════════════════════════════════════════

SYSTEM_MSG = ("Eres AgroChat, experto en estadisticas agrarias espanolas. "
              "Datos reales MAPA y AEMET 2014-2026. "
              "Responde siempre en espanol. No inventes datos. "
              "Cita fuente y ano. Indica si son avances (estimaciones).")

TOOLS = [buscar_documentos, consultar_datos, detectar_anomalias]


def crear_agente(provider: str, model: str, temperature: float = 0, verbose: bool = False):
    global _llm_sql
    llm      = cargar_llm(provider, model, temperature)
    _llm_sql = cargar_llm_sql(provider)
    tool_map = {t.name: t for t in TOOLS}

    # Intentar tool-calling nativo
    use_tc = False
    try:
        llm_tc = llm.bind_tools(TOOLS)
        # Test rapido: si bind_tools no lanza, asumimos soporte
        use_tc = True
        log.info("Tool-calling nativo activado")
    except Exception as e:
        log.warning(f"Tool-calling no disponible ({e}) — modo ReAct texto")
        llm_tc = llm

    def _run_tool_calling(input_dict):
        pregunta = input_dict.get("input","")
        msgs = [{"role":"system","content":SYSTEM_MSG},
                {"role":"user","content":pregunta}]
        for i in range(8):
            resp = llm_tc.invoke(msgs)
            tcs  = getattr(resp, "tool_calls", None) or []
            if verbose:
                log.info(f"[{i+1}] tools={len(tcs)} content={str(resp.content)[:60]}")
            if not tcs:
                return {"output": resp.content, "input": pregunta}
            msgs.append(resp)
            for tc in tcs:
                fn, args, tid = tc["name"], tc["args"], tc["id"]
                if verbose: log.info(f"  -> {fn}({list(args.values())[:1]})")
                result = tool_map[fn].invoke(args) if fn in tool_map else f"Herramienta desconocida: {fn}"
                if verbose: log.info(f"  <- {len(str(result))} chars")
                msgs.append(ToolMessage(content=str(result), tool_call_id=tid))
        return {"output":"Limite de iteraciones.", "input": pregunta}

    def _run_react_texto(input_dict):
        pregunta = input_dict.get("input","")
        sys_react = (SYSTEM_MSG +
            "\n\nHerramientas disponibles:\n"
            "- buscar_documentos(query): busqueda documental\n"
            "- consultar_datos(consulta): datos cuantitativos SQL\n"
            "- detectar_anomalias(parametros): anomalias en rendimientos\n\n"
            "Formato obligatorio en cada paso:\n"
            "Pensamiento: <razonamiento>\nAccion: <herramienta>\nEntrada: <argumento>\n"
            "Tras recibir el resultado, continua o escribe:\nRespuesta Final: <respuesta>")
        hist = f"Sistema: {sys_react}\n\nUsuario: {pregunta}\n"
        for i in range(8):
            resp  = llm.invoke(hist)
            texto = resp.content if hasattr(resp,"content") else str(resp)
            if verbose: log.info(f"[{i+1}] {texto[:80]}")
            if "Respuesta Final:" in texto:
                return {"output": texto.split("Respuesta Final:")[-1].strip(), "input": pregunta}
            ma = re.search(r"Accion:\s*(\w+)", texto)
            me = re.search(r"Entrada:\s*(.+?)(?:\n|$)", texto, re.DOTALL)
            if ma and me:
                fn   = ma.group(1).strip()
                inp  = me.group(1).strip()
                if verbose: log.info(f"  -> {fn}({inp[:50]})")
                if fn in tool_map:
                    first_key = list(tool_map[fn].args.keys())[0]
                    res = tool_map[fn].invoke({first_key: inp})
                else:
                    res = f"Herramienta desconocida: {fn}"
                if verbose: log.info(f"  <- {len(str(res))} chars")
                hist += f"\n{texto}\nObservacion: {str(res)[:1000]}\n"
            else:
                return {"output": texto.strip(), "input": pregunta}
        return {"output": "Limite de iteraciones.", "input": pregunta}

    return _run_tool_calling if use_tc else _run_react_texto


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def modo_interactivo(agente, provider, model):
    print(f"\n{'='*60}\n  AgroChat [{provider} / {model}]\n  Escribe 'salir' para terminar\n{'='*60}\n")
    while True:
        try:
            q = input("Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego."); break
        if not q: continue
        if q.lower() in ("salir","exit","quit"): print("Hasta luego."); break
        try:
            r = agente({"input": q})
            print(f"\nAgroChat: {r.get('output','Sin respuesta')}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")


def main():
    ap = argparse.ArgumentParser(description="AgroChat agente multi-proveedor")
    ap.add_argument("--provider", default="ollama",
                    choices=["ollama","claude","gemini","openai"])
    ap.add_argument("--model",       default=None)
    ap.add_argument("--pregunta",    default=None)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--verbose",     action="store_true")
    ap.add_argument("--test",        action="store_true")
    ap.add_argument("--api-key",     default=None,
                    help="API key del proveedor (alternativa al .env)")
    args = ap.parse_args()

    if args.api_key:
        env_key = {"openai":"OPENAI_API_KEY","claude":"ANTHROPIC_API_KEY",
                   "gemini":"GOOGLE_API_KEY"}.get(args.provider)
        if env_key:
            os.environ[env_key] = args.api_key

    model = args.model or DEFAULT_MODELS.get(args.provider, "llama3.1:8b")

    import langchain
    log.info(f"LangChain {langchain.__version__} | provider={args.provider} | model={model}")

    agente = crear_agente(args.provider, model, args.temperature, args.verbose)
    log.info("Agente listo\n")

    TESTS = [
        ("RAG",       "Que es la ESYRCE y que informacion recoge?"),
        ("SQL",       "Cual fue la produccion nacional de trigo blando en 2023?"),
        ("Anomalias", '{"cultivo":"TRIGO BLANDO","provincia":"Toledo"}'),
    ]

    if args.test:
        print(f"\n{'='*60}\n  BATERIA DE PRUEBA [{args.provider}/{model}]\n{'='*60}")
        for tool_name, q in TESTS:
            print(f"\n[{tool_name}] {q}\n{'-'*60}")
            try:
                if tool_name == "Anomalias":
                    print(detectar_anomalias.invoke({"parametros": q})[:500])
                else:
                    r = agente({"input": q})
                    print(f"AgroChat: {r.get('output','Sin respuesta')}")
            except Exception as e:
                print(f"[Error] {e}")
        print(f"\n{'='*60}")
    elif args.pregunta:
        r = agente({"input": args.pregunta})
        print(f"\nAgroChat: {r.get('output','Sin respuesta')}")
    else:
        modo_interactivo(agente, args.provider, model)


if __name__ == "__main__":
    main()
