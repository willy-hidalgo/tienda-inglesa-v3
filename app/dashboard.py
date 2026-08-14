"""
Forecast Explorer – Streamlit (solo visualización)
=================================================
Cálculos: app.backend + app.dashboard_data
Este módulo solo: controles st, dataframes y Plotly.

Modelo de filtros: Sección es obligatoria; Tienda y SKU son dos filtros
INDEPENDIENTES (no jerárquicos) al mismo nivel — cualquiera puede estar
vacío, uno solo, o ambos a la vez. El orden de selección importa: el filtro
tocado más recientemente "ancla" y acota las opciones del otro (si elegís
Tienda, el select de SKU se acota a los SKU de esa tienda; si elegís SKU
primero, el select de Tienda se acota a las tiendas donde existe ese SKU).

Nota de performance: `prepare_dashboard_state` se envuelve en
`st.cache_data`. Streamlit re-ejecuta todo este script en cada interacción
de UI; sin cache, cada cambio de selección recalculaba agregación temporal +
rankings sobre el DataFrame completo aunque los datos de origen (el
parquet) no hubieran cambiado.
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

st.set_page_config(page_title="Forecast Explorer", layout="wide")
st.title("📈 Forecast Explorer · Secciones 1 & 23")

logger = logging.getLogger(__name__)

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
_SENTINEL_TIENDA = "— Todas las tiendas —"
_SENTINEL_SKU = "— Todos los SKU —"

# ─────────────────────────────────────────────────────────────────────────────
# Carga (I/O de UI; parseo delegado al backend)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando forecasts…", ttl=3600)
def _load_parquet(path_str: str, mtime: float):
    return backend.load_forecast_parquet(path_str)


@st.cache_data(show_spinner=False)
def _load_bytes(data: bytes, name: str):
    return backend.load_forecast_bytes(data, name)


@st.cache_data(show_spinner=False)
def _cached_label_maps(_res_df, mtime_key: float):
    """Label/desc maps solo dependen del parquet cargado."""
    return backend.build_label_maps(_res_df)


@st.cache_data(show_spinner=False)
def _cached_all_ids(_res_df, mtime_key: float) -> tuple[str, ...]:
    return tuple(sorted(_res_df["unique_id"].unique().to_list()))


@st.cache_data(show_spinner=False)
def _cached_section_metrics(
    _res_df,
    unidad: str,
    seccion: str,
    all_ids_tuple: tuple[str, ...],
    mtime_key: float,
):
    """WMAPE por unique_id a nivel sección — independiente de tienda/SKU.

    Es el coste dominante del dashboard (filtra ~23k unique_ids sobre el
    panel completo de ~27M filas y tarda 1-3 min). Se cachea doblemente:

      1. en memoria (`st.cache_data`) por (unidad, sección) — reuso dentro
         de una sesión en cada cambio de filtro fino;
      2. en disco (`data/output/.dashcache/tabla_base_*`) por (mtime del
         parquet, unidad, sección) — reuso entre sesiones / cold-start y al
         cambiar de sección sin volver a calcular en caliente.
    """
    cache_dir = settings.OUT_DIR / ".dashcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _unit_slug = "valor" if unidad.startswith("Valor") else "unidades"
    tb_path = cache_dir / f"tabla_base_{mtime_key:.3f}_{_unit_slug}_{seccion}.parquet"
    nsp_path = cache_dir / f"tabla_base_{mtime_key:.3f}_{_unit_slug}_{seccion}.nsp"
    if tb_path.exists() and nsp_path.exists():
        try:
            return pl.read_parquet(tb_path), int(nsp_path.read_text().strip())
        except Exception:  # pragma: no cover
            logger.warning("Caché tabla_base corrupta; se recalcula: %s", tb_path)

    cols = set(_res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    unit_df = backend.prepare_unit_df(_res_df, unidad, has_value)
    hz_spine = settings.section_horizons(seccion)
    n_spine = (hz_spine["forecast_end"] - hz_spine["train_start"]).days + 1
    candidatos = [
        uid for uid in all_ids_tuple if uid == seccion or uid.startswith(f"{seccion}||")
    ]
    n_data = backend.spine_n_fechas(unit_df, candidatos)
    if n_data > n_spine:
        n_spine = n_data
    tabla_base = backend.wmape_por_id(candidatos, unit_df, n_fechas_spine=n_spine)
    try:
        tabla_base.write_parquet(tb_path)
        nsp_path.write_text(str(n_spine))
        logger.info("✓ tabla_base cacheada en disco: %s", tb_path.name)
    except Exception:  # pragma: no cover
        logger.warning("No se pudo escribir caché tabla_base: %s", tb_path)
    return tabla_base, n_spine


@st.cache_data(show_spinner=False)
def _cached_dashboard_state(
    _res_df,
    unidad: str,
    freq: str,
    seccion: str,
    store: str | None,
    sku: str | None,
    cutoff_date: dt.date,
    all_ids_tuple: tuple[str, ...],
    _label_map: dict[str, str],
    _desc_map: dict[str, str],
    _tabla_base,
    n_spine: int,
):
    """Envoltorio cacheado de `prepare_dashboard_state`. Los argumentos
    prefijados con `_` no se hashean (DataFrame/dicts grandes, estables
    mientras no cambie el archivo cargado); la clave de cache es el resto:
    exactamente lo que identifica una selección de filtros."""
    return prepare_dashboard_state(
        _res_df,
        unidad=unidad,
        freq=freq,
        seccion=seccion,
        store=store,
        sku=sku,
        cutoff_date=cutoff_date,
        all_ids=list(all_ids_tuple),
        label_map=_label_map,
        desc_map=_desc_map,
        tabla_base=_tabla_base,
        n_spine=n_spine,
    )


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

# Etiquetas según unidad (mapeo interno siempre yhat / yhat28)
_is_valor = unidad.startswith("Valor")
_opt_yhat = "valuehat" if _is_valor else "yhat"
_opt_yhat28 = "valuehat28" if _is_valor else "yhat28"

# El rolling28 es opcional (settings.COMPUTE_ROLLING_28) y, desde el modelo
# jerárquico RLS-sección, SOLO existe para nodos de sección. Si el dataset
# cargado no lo tiene en absoluto, el control no tiene sentido (1 sola
# opción) y no se muestra.
dataset_has_rolling28 = (
    "yhat28" in res_df.columns and res_df.select("yhat28").drop_nulls().height > 0
) or (
    "valuehat28" in res_df.columns
    and res_df.select("valuehat28").drop_nulls().height > 0
)

if dataset_has_rolling28:
    _forecast_opts = [_opt_yhat, _opt_yhat28]
    series_forecast = st.sidebar.multiselect(
        "Series de forecast visibles",
        options=_forecast_opts,
        default=_forecast_opts,
        key="series_forecast",
        help="Al menos una serie debe permanecer visible.",
    )
    if not series_forecast:
        series_forecast = [_opt_yhat]
        st.sidebar.warning(f"Debe quedar al menos una serie; se mantiene «{_opt_yhat}».")
    show_yhat = _opt_yhat in series_forecast
    show_yhat28 = _opt_yhat28 in series_forecast
else:
    show_yhat = True
    show_yhat28 = False

# Clave de invalidación de caches ligadas al parquet (mtime o nombre upload).
_mtime_key = (
    float(Path(settings.FORECAST_PATH).stat().st_mtime)
    if uploaded is None and Path(settings.FORECAST_PATH).exists()
    else hash(uploaded.name if uploaded is not None else "none")
)
all_ids = list(_cached_all_ids(res_df, _mtime_key))
label_map, desc_map = _cached_label_maps(res_df, _mtime_key)


def label_for(uid: str) -> str:
    return label_map.get(uid, settings.display_label(uid))


# ─────────────────────────────────────────────────────────────────────────────
# Filtros: Sección (obligatoria) + Tienda / SKU (independientes, opcionales)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filtros")

if st.sidebar.button("🔄 Reiniciar filtros"):
    for k in ("sel_tienda", "sel_sku", "_last_touched", "_prev_seccion"):
        st.session_state.pop(k, None)
    st.rerun()

secciones_disp = backend.secciones_disponibles(all_ids)
default_sec_idx = secciones_disp.index("1") if "1" in secciones_disp else 0
seccion = st.sidebar.selectbox(
    "Sección",
    secciones_disp,
    index=default_sec_idx if secciones_disp else 0,
    key="sel_seccion",
    format_func=lambda x: f"Sección {x}",
)

# Cambiar de sección invalida cualquier tienda/sku elegido previamente.
if st.session_state.get("_prev_seccion") != seccion:
    st.session_state["sel_tienda"] = _SENTINEL_TIENDA
    st.session_state["sel_sku"] = _SENTINEL_SKU
    st.session_state["_last_touched"] = None
    st.session_state["_prev_seccion"] = seccion

# Click en una fila de ranking: aplica el filtro correspondiente y ancla
# ese eje (ver más abajo, tablas de ranking).
if "_pending_tienda" in st.session_state:
    st.session_state["sel_tienda"] = st.session_state.pop("_pending_tienda")
    st.session_state["_last_touched"] = "tienda"
if "_pending_sku" in st.session_state:
    st.session_state["sel_sku"] = st.session_state.pop("_pending_sku")
    st.session_state["_last_touched"] = "sku"


def _touch_tienda() -> None:
    st.session_state["_last_touched"] = "tienda"


def _touch_sku() -> None:
    st.session_state["_last_touched"] = "sku"


last_touched = st.session_state.get("_last_touched")
_raw_tienda = st.session_state.get("sel_tienda", _SENTINEL_TIENDA)
_raw_sku = st.session_state.get("sel_sku", _SENTINEL_SKU)
_sku_val = None if _raw_sku == _SENTINEL_SKU else _raw_sku
_store_val = None if _raw_tienda == _SENTINEL_TIENDA else _raw_tienda

# Tienda: si el ancla es SKU (y hay un SKU concreto elegido), acotar a las
# tiendas donde ese SKU existe; si no, todas las tiendas de la sección.
if last_touched == "sku" and _sku_val is not None:
    tienda_opts = [_SENTINEL_TIENDA] + backend.stores_for_sku(all_ids, seccion, _sku_val)
else:
    tienda_opts = [_SENTINEL_TIENDA] + backend.all_stores_in_section(all_ids, seccion)
if _raw_tienda not in tienda_opts:
    st.session_state["sel_tienda"] = _SENTINEL_TIENDA
    _raw_tienda = _SENTINEL_TIENDA

store_sel = st.sidebar.selectbox(
    "Tienda",
    tienda_opts,
    key="sel_tienda",
    on_change=_touch_tienda,
    format_func=lambda x: x if x == _SENTINEL_TIENDA else label_for(
        settings.make_unique_id(seccion, store=x)
    ),
)
store_sel_val = None if store_sel == _SENTINEL_TIENDA else store_sel

# SKU: si el ancla es Tienda (y hay una tienda concreta elegida), acotar a
# los SKU de esa tienda; si no, todos los SKU de la sección.
if last_touched == "tienda" and store_sel_val is not None:
    sku_opts = [_SENTINEL_SKU] + backend.skus_for_store(all_ids, seccion, store_sel_val)
else:
    sku_opts = [_SENTINEL_SKU] + backend.all_skus_in_section(all_ids, seccion)
if _raw_sku not in sku_opts:
    st.session_state["sel_sku"] = _SENTINEL_SKU
    _raw_sku = _SENTINEL_SKU

sku_sel = st.sidebar.selectbox(
    "SKU",
    sku_opts,
    key="sel_sku",
    on_change=_touch_sku,
    format_func=lambda x: x if x == _SENTINEL_SKU else label_for(
        settings.make_unique_id(seccion, sku=x)
    ),
)
sku_sel_val = None if sku_sel == _SENTINEL_SKU else sku_sel

# Corte in/out = train_end de la sección (settings; sin re-escanear el DF).
# Si el nodo seleccionado no existe aún, section_horizons sigue siendo válido.
_hz_sec = settings.section_horizons(seccion)
cutoff_date = _hz_sec.get("train_end") or dt.date.today()

# Métricas de ranking a nivel sección (cacheadas; no se recalculan al
# cambiar tienda/SKU dentro de la misma sección+unidad).
_tabla_base, _n_spine = _cached_section_metrics(
    res_df, unidad, seccion, tuple(all_ids), _mtime_key
)

# ─────────────────────────────────────────────────────────────────────────────
# ViewModel (todos los cálculos fuera de este módulo; cacheado por selección)
# ─────────────────────────────────────────────────────────────────────────────
view = _cached_dashboard_state(
    res_df,
    unidad,
    freq,
    seccion,
    store_sel_val,
    sku_sel_val,
    cutoff_date,
    tuple(all_ids),
    label_map,
    desc_map,
    _tabla_base,
    _n_spine,
)

# Contexto de tienda (solo si Tienda Y SKU están ambos activos) va ARRIBA
# del encabezado del nodo, para que sea lo primero que se lee.
if view.store_context:
    st.caption(f"📍 Tienda: {view.store_context}")

_header_label = {
    "seccion": "SECCIÓN",
    "tienda": "TIENDA",
    "sku": "SKU",
    "tienda_sku": "SKU",
}[view.node_kind]
st.subheader(f"{_header_label}: **{view.label}**")

# ─────────────────────────────────────────────────────────────────────────────
# Ranking: dos tablas (Tiendas / SKU), cada una respeta el filtro cruzado
# activo (ver dashboard_data.prepare_dashboard_state y backend.ranking_table)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### Ranking")
st.caption(
    f"Sección **{view.seccion}** · Unidad: **{view.unidad}** · "
    f"N puntos totales (spine): **{view.n_spine}** (~5 filas visibles, scroll)."
)


def _show_ranking(display, key: str, pending_key: str, extract_field: str) -> None:
    if display.height == 0:
        st.caption(
            "Sin datos suficientes para ranking "
            "(no hay hojas sku+tienda con rotación > 0 en esta selección, "
            "o el parquet no incluye ese nivel)."
        )
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
            "N puntos": st.column_config.NumberColumn("N puntos (≠0)"),
            "% ≠0": st.column_config.TextColumn("% ≠0"),
        },
    )
    if event and event.selection and event.selection.rows:
        clicked_uid = display["unique_id"][event.selection.rows[0]]
        p = settings.split_unique_id(clicked_uid)
        st.session_state[pending_key] = p[extract_field]
        st.rerun()


col_t, col_s = st.columns(2)
with col_t:
    st.markdown("**Tiendas**")
    _show_ranking(view.ranking_tiendas, f"tabla_tiendas_{view.selected_id}", "_pending_tienda", "store")
with col_s:
    st.markdown("**SKU**")
    _show_ranking(view.ranking_skus, f"tabla_skus_{view.selected_id}", "_pending_sku", "sku")

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

# Rolling28: solo si este nodo tiene datos calculados (desde el modelo
# jerárquico, solo el nodo de SECCIÓN los tiene — ver settings.COMPUTE_ROLLING_28).
if view.has_rolling28:
    st.markdown("#### Métricas Rolling 28d (`yhat28`)")
    st.caption(
        "Walk-forward por bloques de 28 días. Solo disponible a nivel sección "
        "(único nivel con RLS real). WMAPE₂₈ = Σ|y−ŷ₂₈|/Σ|y| (excl. y=0)."
    )
    m28 = view.metrics_28
    c28a, c28b, c28c = st.columns(3)
    with c28a:
        st.metric("wMAPE 28", f"{m28['wmape_28']:.2%}")
    with c28b:
        st.metric("BIAS 28", f"{m28['bias_28']:+.2%}")
    with c28c:
        st.metric("N períodos", f"{m28['n']}")

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
    if show_yhat:
        fig.add_trace(
            go.Scatter(
                x=ch["hist_ds"],
                y=ch["hist_yhat"],
                name=f"{_opt_yhat} (predicción)",
                fill="tozeroy",
                mode="lines",
                line=dict(color="rgba(255, 127, 14, 1)"),
                fillcolor="rgba(255, 127, 14, 0.25)",
            )
        )
if ch["fcst_ds"] and show_yhat:
    fig.add_trace(
        go.Scatter(
            x=ch["fcst_ds"],
            y=ch["fcst_yhat"],
            name=f"{_opt_yhat} (solo forecast)",
            mode="lines",
            line=dict(color="rgba(255, 127, 14, 1)", dash="dot", width=2),
        )
    )
# yhat28: solo si (a) el checkbox lo pide Y (b) este nodo tiene rolling28.
if view.has_rolling28 and show_yhat28 and ch.get("hist_yhat28"):
    fig.add_trace(
        go.Scatter(
            x=ch["hist_ds"],
            y=ch["hist_yhat28"],
            name=f"{_opt_yhat28} (rolling 28d)",
            mode="lines",
            line=dict(color="rgba(44, 160, 44, 1)", width=2),
        )
    )
if view.has_rolling28 and show_yhat28 and ch.get("fcst_yhat28"):
    fig.add_trace(
        go.Scatter(
            x=ch["fcst_ds"],
            y=ch["fcst_yhat28"],
            name=f"{_opt_yhat28} (forecast)",
            mode="lines",
            line=dict(color="rgba(44, 160, 44, 0.8)", dash="dot", width=2),
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
_series_caption = [_opt_yhat] if show_yhat else []
if view.has_rolling28 and show_yhat28:
    _series_caption.append(_opt_yhat28)
if _series_caption:
    st.caption("Mostrando forecast: **" + ", ".join(_series_caption) + "**")
st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Detalle
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Ver datos detallados"):
    st.caption(f"Agregación **{view.freq}** · Unidad **{view.unidad}**")
    st.dataframe(
        view.detail,
        width="stretch",
        hide_index=True,
        height=_RANK_HEIGHT,
    )
