"""
Forecast Explorer – Streamlit (solo visualización)
=================================================
Cálculos: app.backend + app.dashboard_data
Este módulo solo: controles st, dataframes y Plotly.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Forecast Explorer", layout="wide")
st.title("📈 Forecast Explorer · Secciones 1 & 23")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings

try:
    from app import backend
    from app.dashboard_data import prepare_dashboard_state
except ImportError:  # pragma: no cover
    import backend  # type: ignore
    from dashboard_data import prepare_dashboard_state  # type: ignore

_RANK_HEIGHT = 35 + 5 * 35  # ~5 filas visibles

# ─────────────────────────────────────────────────────────────────────────────
# Carga (I/O de UI; parseo delegado al backend)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando forecasts…", ttl=3600)
def _load_parquet(path_str: str, mtime: float):
    return backend.load_forecast_parquet(path_str)


@st.cache_data(show_spinner=False)
def _load_bytes(data: bytes, name: str):
    return backend.load_forecast_bytes(data, name)


uploaded = st.sidebar.file_uploader(
    "Cargar forecasts (CSV o Parquet)", type=["csv", "parquet"]
)
if uploaded is not None:
    res_df = _load_bytes(uploaded.getvalue(), uploaded.name)
    st.sidebar.success(f"Cargado: {uploaded.name}")
else:
    default_path = Path(settings.FORECAST_PATH)
    if default_path.exists():
        res_df = _load_parquet(str(default_path), default_path.stat().st_mtime)
        st.sidebar.success(f"Cargado: {default_path.name}")
    else:
        st.info(
            "Sube un archivo CSV o Parquet, "
            "o genera `data/output/forecast.parquet` con el pipeline."
        )
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Controles
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Visualización")
unidad = st.sidebar.radio(
    "Unidad",
    ["Valor ($)", "Unidades"],
    index=0,
    horizontal=True,
    key="unidad",
)
freq = st.sidebar.radio(
    "Agregación temporal",
    ["Diario", "Semanal", "Mensual"],
    horizontal=True,
    key="freq",
)

all_ids = sorted(res_df["unique_id"].unique().to_list())
label_map, desc_map = backend.build_label_maps(res_df)
NOMBRES = settings.NOMBRES_NIVELES


def label_for(uid: str) -> str:
    return label_map.get(uid, settings.display_label(uid))


st.sidebar.markdown("### Selección de nivel")
if st.sidebar.button("🔄 Reiniciar niveles"):
    for k in list(st.session_state.keys()):
        if k.startswith("nivel_") or k.startswith("sel_"):
            del st.session_state[k]
    st.rerun()

if "sel_from_table" in st.session_state:
    target = st.session_state.pop("sel_from_table")
    parts = target.split("||")
    st.session_state["nivel_0"] = parts[0]
    if len(parts) >= 2:
        st.session_state[f"nivel_1__{parts[0]}"] = "||".join(parts[:2])
    if len(parts) >= 3:
        st.session_state[f"nivel_2__{'||'.join(parts[:2])}"] = target

opciones_raiz = backend.opciones_nivel(all_ids, None)
default_idx = opciones_raiz.index("1") if "1" in opciones_raiz else 0
selected_id = st.sidebar.selectbox(
    NOMBRES[0],
    opciones_raiz,
    index=default_idx if opciones_raiz else 0,
    key="nivel_0",
    format_func=lambda x: f"Sección {x}",
)

niveles_recorridos: list[tuple[str, list[str]]] = [(NOMBRES[0], opciones_raiz)]
nivel = 1
while True:
    hijos = backend.opciones_nivel(all_ids, selected_id)
    if not hijos:
        break
    nombre_nivel = NOMBRES[nivel] if nivel < len(NOMBRES) else f"Nivel {nivel}"
    niveles_recorridos.append((nombre_nivel, hijos))
    opcion_mantener = f"— Quedarse en «{label_for(selected_id)}» —"
    key = f"nivel_{nivel}__{selected_id}"
    eleccion = st.sidebar.selectbox(
        nombre_nivel,
        [opcion_mantener] + hijos,
        index=0,
        key=key,
        format_func=lambda x: x if x.startswith("—") else label_for(x),
    )
    if eleccion == opcion_mantener:
        break
    selected_id = eleccion
    nivel += 1

nombre_nivel_actual, candidatos = niveles_recorridos[-1]

# Corte in/out = train_end de la sección (sin control en sidebar)
_probe = backend.filter_series(
    backend.prepare_unit_df(
        res_df,
        unidad,
        "value" in res_df.columns and "valuehat" in res_df.columns,
    ),
    selected_id,
)
_hz = backend.resolve_horizons(
    _probe, selected_id.split("||")[0], set(res_df.columns)
)
cutoff_date = _hz.get("train_end") or backend.ds_range(_probe)[0] or dt.date.today()

# ─────────────────────────────────────────────────────────────────────────────
# ViewModel (todos los cálculos fuera de este módulo)
# ─────────────────────────────────────────────────────────────────────────────
view = prepare_dashboard_state(
    res_df,
    unidad=unidad,
    freq=freq,
    selected_id=selected_id,
    cutoff_date=cutoff_date,
    candidatos=candidatos,
    nombre_nivel=nombre_nivel_actual,
    label_map=label_map,
    desc_map=desc_map,
)

st.subheader(f"NIVEL: **{view.label}**")

# ─────────────────────────────────────────────────────────────────────────────
# Rankings
# ─────────────────────────────────────────────────────────────────────────────
def _show_ranking(display, key: str) -> None:
    if display.height == 0:
        st.caption("Sin datos suficientes.")
        return
    visible = [c for c in display.columns if c != "unique_id"]
    event = st.dataframe(
        display.select(visible),
        width="content",
        hide_index=True,
        height=_RANK_HEIGHT,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
        column_config={
            "Código": st.column_config.TextColumn("Código"),
            "Descripción": st.column_config.TextColumn("Descripción"),
            "N puntos": st.column_config.NumberColumn("N puntos"),
        },
    )
    if event and event.selection and event.selection.rows:
        st.session_state["sel_from_table"] = display["unique_id"][
            event.selection.rows[0]
        ]
        st.rerun()


tab_wmape, tab_rot = st.tabs(["📉 Mejor wMAPE", "🔄 Mayor rotación"])
with tab_wmape:
    st.markdown("#### Ranking wMAPE Total")
    st.caption(
        f"Nivel: **{view.nombre_nivel}** · Unidad: **{view.unidad}** · "
        f"{view.ranking_n_series} series (~5 filas visibles, scroll)."
    )
    _show_ranking(view.ranking_wmape, f"tabla_wmape_{view.selected_id}")

with tab_rot:
    st.markdown("#### Ranking por rotación")
    st.caption(f"Nivel: **{view.nombre_nivel}** · Unidad: **{view.unidad}**")
    _show_ranking(view.ranking_rotacion, f"tabla_rot_{view.selected_id}")

# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────
hz = view.horizons
st.markdown("#### Métricas")
st.caption(
    f"Agregación: **{view.freq}** · Unidad: **{view.unidad}** · "
    f"Sección **{view.seccion}** · Train → {hz.get('train_end')} · "
    f"OOS [{hz.get('test_start')} → {hz.get('test_end')}] · "
    f"Solo-forecast [{hz.get('forecast_start')} → {hz.get('forecast_end')}]. "
    "WMAPE = Σ|y−ŷ|/Σ|y| (excl. y=0)."
)

c1, c2, c3 = st.columns(3)
for col, title, key in (
    (c1, "In-sample", "in"),
    (c2, "Out-sample", "out"),
    (c3, "Total (hasta test_end)", "total"),
):
    m = view.metrics[key]
    with col:
        st.markdown(f"**{title}**")
        st.metric("wMAPE", f"{m['wmape']:.2%}")
        st.metric("BIAS", f"{m['bias']:+.2%}")
        st.caption(f"{m['n']} períodos")

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico (solo mapeo de series ya calculadas)
# ─────────────────────────────────────────────────────────────────────────────
unidad_label = "Valor ($)" if view.unidad.startswith("Valor") else "Unidades"
freq_label = {"Diario": "día", "Semanal": "semana", "Mensual": "mes"}[view.freq]
ch = view.chart

fig = go.Figure()
if ch["hist_ds"]:
    fig.add_trace(
        go.Scatter(
            x=ch["hist_ds"],
            y=ch["hist_y"],
            name="y (real)",
            fill="tozeroy",
            mode="lines",
            line=dict(color="rgba(31, 119, 180, 1)"),
            fillcolor="rgba(31, 119, 180, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ch["hist_ds"],
            y=ch["hist_yhat"],
            name="yhat (predicción)",
            fill="tozeroy",
            mode="lines",
            line=dict(color="rgba(255, 127, 14, 1)"),
            fillcolor="rgba(255, 127, 14, 0.25)",
        )
    )
if ch["fcst_ds"]:
    fig.add_trace(
        go.Scatter(
            x=ch["fcst_ds"],
            y=ch["fcst_yhat"],
            name="yhat (solo forecast)",
            mode="lines",
            line=dict(color="rgba(255, 127, 14, 1)", dash="dot", width=2),
        )
    )

cutoff_x = ch["cutoff"]
if isinstance(cutoff_x, dt.date) and not isinstance(cutoff_x, dt.datetime):
    cutoff_x = dt.datetime.combine(cutoff_x, dt.time.min)
fig.add_vline(
    x=cutoff_x,
    line_dash="dash",
    line_color="rgba(220, 50, 50, 0.8)",
    annotation_text="Corte train/OOS",
    annotation_position="top left",
)
te = ch["test_end"]
if isinstance(te, dt.date):
    fig.add_vline(
        x=dt.datetime.combine(te, dt.time.min),
        line_dash="dot",
        line_color="rgba(100, 100, 100, 0.6)",
        annotation_text="Fin OOS",
        annotation_position="top right",
    )
fig.update_layout(
    title=f"y vs yhat — {view.label} ({view.freq.lower()}, {unidad_label})",
    xaxis_title="Fecha",
    yaxis_title=f"{unidad_label} / {freq_label}",
    legend_title_text="",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=50, b=40),
)
st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Detalle
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Ver datos detallados"):
    st.caption(f"Agregación **{view.freq}** · Unidad **{view.unidad}**")
    _num_fmt = st.column_config.NumberColumn(format="#,##0.###")
    _detail_cfg = {
        c: _num_fmt
        for c in ("y", "yhat", "value", "valuehat", "abs_error", "price", "pricehat")
        if c in view.detail.columns
    }
    st.dataframe(
        view.detail,
        width="stretch",
        hide_index=True,
        height=_RANK_HEIGHT,
        column_config=_detail_cfg,
    )
