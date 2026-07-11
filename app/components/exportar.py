"""
Exportación a Excel de resultados del dashboard.
"""
import io
from datetime import datetime
import pandas as pd
import xlsxwriter


def exportar_excel(dataframes: dict, nombre_base: str = "agrochat") -> bytes:
    """
    Recibe un dict {nombre_hoja: DataFrame} y genera un Excel en memoria.
    Devuelve los bytes listos para st.download_button.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        wb  = writer.book
        fmt_header = wb.add_format({
            "bold": True, "bg_color": "#1f6b3a", "font_color": "white",
            "border": 1, "align": "center",
        })
        fmt_num = wb.add_format({"num_format": "#,##0.000", "border": 1})
        fmt_int = wb.add_format({"num_format": "#,##0",     "border": 1})
        fmt_txt = wb.add_format({"border": 1})

        for hoja, df in dataframes.items():
            if df is None or df.empty:
                continue
            sheet_name = hoja[:31]  # Excel limita a 31 chars
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, header=False)
            ws = writer.sheets[sheet_name]

            # Encabezados con formato
            for col_num, col_name in enumerate(df.columns):
                ws.write(0, col_num, col_name, fmt_header)

            # Ancho de columna automático
            for col_num, col_name in enumerate(df.columns):
                max_len = max(len(str(col_name)),
                              df[col_name].astype(str).str.len().max() if not df.empty else 0)
                ws.set_column(col_num, col_num, min(max_len + 2, 40))

        # Hoja de metadatos
        ws_meta = wb.add_worksheet("Info")
        ws_meta.write("A1", "Exportado por",  fmt_header)
        ws_meta.write("B1", "AgroChat — TFM VIU 2025-2026", fmt_txt)
        ws_meta.write("A2", "Fecha",           fmt_header)
        ws_meta.write("B2", datetime.now().strftime("%Y-%m-%d %H:%M"), fmt_txt)
        ws_meta.write("A3", "Fuente",          fmt_header)
        ws_meta.write("B3", "MAPA / AEMET — Datos reales 2014-2025", fmt_txt)
        ws_meta.set_column("A:A", 18)
        ws_meta.set_column("B:B", 40)

    return buffer.getvalue()


def boton_descarga(df: pd.DataFrame, nombre_hoja: str = "Datos",
                   label: str = "⬇ Descargar Excel"):
    """Wrapper de st.download_button para exportar un DataFrame."""
    import streamlit as st
    if df is None or df.empty:
        st.info("Sin datos que exportar.")
        return
    datos = exportar_excel({nombre_hoja: df})
    fname = f"agrochat_{nombre_hoja.lower().replace(' ','_')}_{datetime.now():%Y%m%d}.xlsx"
    st.download_button(
        label=label,
        data=datos,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
