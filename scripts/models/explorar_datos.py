"""
══════════════════════════════════════════════════════════════
Exploración de datos AgroChat - DuckDB
══════════════════════════════════════════════════════════════
Copiar celdas en un notebook Jupyter (http://localhost:8888)
"""

# %% Celda 1: Conexión
import duckdb
con = duckdb.connect("/app/data/duckdb/agrochat.duckdb", read_only=True)

# %% Celda 2: Resumen general de AVANCES
con.sql("""
    SELECT 'AVANCES' as tabla,
           COUNT(*) total,
           COUNT(DISTINCT cultivo) cultivos,
           COUNT(DISTINCT cod_provincia) provincias,
           COUNT(DISTINCT ccaa) ccaa,
           MIN(periodo_anio) desde,
           MAX(periodo_anio) hasta
    FROM avances
    UNION ALL
    SELECT 'COYUNTURA',
           COUNT(*), COUNT(DISTINCT producto), NULL, NULL,
           MIN(anio), MAX(anio)
    FROM coyuntura
""").show()

# %% Celda 3: Cultivos disponibles (avances)
con.sql("""
    SELECT DISTINCT cultivo
    FROM avances
    ORDER BY cultivo
""").show(max_rows=50)

# %% Celda 4: Serie temporal de un cultivo en una provincia
con.sql("""
    SELECT periodo_anio, periodo_mes,
           sup_avance AS superficie_ha,
           prod_avance AS produccion_kT
    FROM avances
    WHERE cultivo = 'TRIGO BLANDO'
      AND cod_provincia = 41        -- Sevilla
      AND nivel = 'provincia'
    ORDER BY periodo_anio, periodo_mes
""").show(max_rows=50)

# %% Celda 5: Top 10 provincias productoras de aceite de oliva (último año)
con.sql("""
    SELECT provincia, ccaa,
           MAX(prod_avance) AS produccion_kT
    FROM avances
    WHERE cultivo LIKE '%OLIV%'
      AND nivel = 'provincia'
      AND periodo_anio = 2025
      AND periodo_mes = (SELECT MAX(periodo_mes) FROM avances WHERE periodo_anio=2025)
    GROUP BY provincia, ccaa
    ORDER BY produccion_kT DESC NULLS LAST
    LIMIT 10
""").show()

# %% Celda 6: Precios del trigo blando (coyuntura)
con.sql("""
    SELECT anio, semana,
           precio_sem_actual,
           variacion_pct,
           unidad
    FROM coyuntura
    WHERE producto LIKE 'Trigo blando%'
    ORDER BY anio, semana
""").show(max_rows=30)

# %% Celda 7: Productos con mayor subida de precio (última semana disponible)
con.sql("""
    WITH ultima AS (
        SELECT anio, semana FROM coyuntura
        ORDER BY anio DESC, semana DESC LIMIT 1
    )
    SELECT c.producto, c.seccion, c.precio_sem_actual,
           c.variacion_pct, c.unidad
    FROM coyuntura c
    JOIN ultima u ON c.anio = u.anio AND c.semana = u.semana
    WHERE c.variacion_pct IS NOT NULL
    ORDER BY c.variacion_pct DESC
    LIMIT 10
""").show()

# %% Celda 8: Verificar CCAA - provincias asignadas correctamente
con.sql("""
    SELECT ccaa,
           COUNT(DISTINCT cod_provincia) n_provincias,
           STRING_AGG(DISTINCT provincia, ', ' ORDER BY provincia) provincias
    FROM avances
    WHERE nivel = 'provincia' AND periodo_anio = 2025
    GROUP BY ccaa
    ORDER BY ccaa
""").show(max_rows=20)

# %% Celda 9: Cruce avances × coyuntura (ejemplo)
con.sql("""
    SELECT a.periodo_anio, a.periodo_mes,
           a.sup_avance AS sup_ha,
           a.prod_avance AS prod_kT,
           c.precio_sem_actual AS precio_eur_t
    FROM avances a
    LEFT JOIN coyuntura c
      ON c.anio = a.periodo_anio
      AND c.semana = a.periodo_mes * 4   -- aprox: mes → semana
      AND c.producto LIKE 'Trigo blando%'
    WHERE a.cultivo = 'TRIGO BLANDO'
      AND a.nivel = 'nacional'
      AND a.periodo_anio >= 2022
    ORDER BY a.periodo_anio, a.periodo_mes
""").show()
