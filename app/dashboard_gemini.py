"""
Interfaz Streamlit con selector de tipo de resultado (Por Valor o Por Cantidad).
Por defecto muestra consultas por Valor ($).
"""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from app.dashboard_data_gemini import DashboardDataOrchestrator

st.set_page_config(
    page_title="Dashboard de Pronósticos & Métricas WMAPE",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_orchestrator() -> DashboardDataOrchestrator:
    return DashboardDataOrchestrator()


orchestrator = get_orchestrator()

# ----- SIDEBAR / FILTROS -----
st.sidebar.title("Filtros de Control")

# Selector del tipo de consulta (Valor por defecto)
metric_option = st.sidebar.radio(
    "Consultar Resultados Por:",
    options=["Por Valor ($)", "Por Cantidad (Unidades)"],
    index=0,
)
metric_mode = "value" if metric_option == "Por Valor ($)" else "quantity"

sections = orchestrator.get_sections()
if not sections:
    st.error(
        "No se encontraron datos precalculados. Verifica que hayas ejecutado el pipeline."
    )
    st.stop()

selected_section = st.sidebar.selectbox("Sección", options=sections, index=0)

available_stores = ["Todas"] + orchestrator.get_stores_by_section(selected_section)
selected_store_raw = st.sidebar.selectbox("Tienda", options=available_stores, index=0)
selected_store = None if selected_store_raw == "Todas" else selected_store_raw

available_skus = ["Todos"] + orchestrator.get_skus_by_section(
    selected_section, store=selected_store
)
selected_sku_raw = st.sidebar.selectbox("SKU", options=available_skus, index=0)
selected_sku = None if selected_sku_raw == "Todos" else selected_sku_raw

unit_label = "Monto ($)" if metric_mode == "value" else "Unidades"
st.title(f"Sección: {selected_section} — [{metric_option}]")

# ----- SECCIÓN SUPERIOR: GRÁFICO DE SERIE TEMPORAL -----
st.subheader(f"Serie Temporal ({unit_label}): Real vs. Pronosticado")

df_ts = orchestrator.get_timeseries(
    seccion=selected_section,
    store=selected_store,
    sku=selected_sku,
    metric_mode=metric_mode,
)

if not df_ts.is_empty():
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_ts["ds"].to_list(),
            y=df_ts["real"].to_list(),
            mode="lines",
            name=f"Real ({unit_label})",
            line=dict(color="#1f77b4", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_ts["ds"].to_list(),
            y=df_ts["pred"].to_list(),
            mode="lines",
            name=f"Pronóstico ({unit_label})",
            # line=dict(color="#ff7f0e", width=2, dash="dash"),
            line=dict(color="#ff7f0e", width=2),
        )
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Fecha",
        yaxis_title=unit_label,
        hovermode="x unified",
        template="plotly_white",
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay datos disponibles para la combinación seleccionada.")

# ----- SECCIÓN INFERIOR: TABLAS DE RANKING PRECALCULADAS -----
col_left, col_right = st.columns(2)

wmape_col_name = "wmape_val" if metric_mode == "value" else "wmape_qty"
sum_col_name = "sum_val" if metric_mode == "value" else "sum_y"
sum_label = "Total Valor ($)" if metric_mode == "value" else "Total Cantidad (Unid)"

with col_left:
    st.subheader("Ranking por Tiendas (Top Mayor WMAPE)")
    df_store_rank = orchestrator.get_ranking_table(
        seccion=selected_section,
        axis="store",
        fixed_peer=selected_sku,
        metric_mode=metric_mode,
        top_n=10,
    )
    if not df_store_rank.is_empty():
        display_store = df_store_rank.select(
            [
                pl.col("store").alias("Tienda"),
                pl.col(wmape_col_name).round(2).alias("WMAPE (%)"),
                pl.col(sum_col_name)
                .round(2 if metric_mode == "value" else 0)
                .alias(sum_label),
                pl.col("n_points").alias("Puntos Evaluación"),
            ]
        ).to_pandas()
        st.dataframe(display_store, use_container_width=True, hide_index=True)
    else:
        st.write("Sin datos de ranking de tiendas.")

with col_right:
    st.subheader("Ranking por SKUs (Top Mayor WMAPE)")
    df_sku_rank = orchestrator.get_ranking_table(
        seccion=selected_section,
        axis="sku",
        fixed_peer=selected_store,
        metric_mode=metric_mode,
        top_n=10,
    )
    if not df_sku_rank.is_empty():
        display_sku = df_sku_rank.select(
            [
                pl.col("sku").alias("SKU"),
                pl.col(wmape_col_name).round(2).alias("WMAPE (%)"),
                pl.col(sum_col_name)
                .round(2 if metric_mode == "value" else 0)
                .alias(sum_label),
                pl.col("n_points").alias("Puntos Evaluación"),
            ]
        ).to_pandas()
        st.dataframe(display_sku, use_container_width=True, hide_index=True)
    else:
        st.write("Sin datos de ranking de SKUs.")
