"""
Forecast Explorer – Streamlit (solo visualización)
=================================================
Path rápido: artefactos precalculados (index + metrics + series por unique_id).
Sin artefactos: fallback legacy (lento) sobre forecast.parquet completo.

Modelo de filtros: Sección obligatoria; Tienda y SKU independientes.
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
    from app import dashboard_artifacts as artifacts
    from app.dashboard_data import (
        prepare_dashboard_state,
        prepare_dashboard_state_fast,
    )
except ImportError:  # pragma: no cover
    import backend  # type: ignore
    import dashboard_artifacts as artifacts  # type: ignore
    from dashboard_data import (  # type: ignore
        prepare_dashboard_state,
        prepare_dashboard_state_fast,
    )

_RANK_HEIGHT = 35 + 5 * 35  # ~5 filas visibles
_SENTINEL_TIENDA = "— Todas las tiendas —"
_SENTINEL_SKU = "— Todos los SKU —"


# ─────────────────────────────────────────────────────────────────────────────
# Carga de artefactos (rápido) o parquet completo (legacy)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando índice dashboard…", ttl=3600)
def _load_index(mtime_key: float, forecast_path: str):
    return artifacts.load_index(Path(forecast_path) if forecast_path else None)


@st.cache_data(show_spinner="Cargando métricas…", ttl=3600)
def _load_metrics(mtime_key: float, forecast_path: str):
    return artifacts.load_metrics(Path(forecast_path) if forecast_path else None)


@st.cache_data(show_spinner="Cargando forecasts…", ttl=3600)
def _load_parquet(path_str: str, mtime: float):
    return backend.load_forecast_parquet(path_str)


@st.cache_data(show_spinner=False)
def _load_bytes(data: bytes, name: str):
    return backend.load_forecast_bytes(data, name)


@st.cache_data(show_spinner=False)
def _cached_label_maps(_res_df, mtime_key: float):
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
    cols = set(_res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    unit_df = backend.prepare_unit_df(_res_df, unidad, has_value)
    hz_spine = settings.section_horizons(seccion)
    n_spine = (hz_spine["forecast_end"] - hz_spine["train_start"]).days + 1
    candidatos = [
        uid for uid in all_ids_tuple if uid == seccion or uid.startswith(f"{seccion}||")
    ]
    n_data = backend.spine_n_fechas(unit_df, candidatos)
    n_spine = max(n_spine, n_data)
    # Rankings: wMAPE out-of-sample
    tabla_base = backend.wmape_por_id(
        candidatos,
        unit_df,
        n_fechas_spine=n_spine,
        period_types=["out_sample"],
    )
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


@st.cache_data(show_spinner=False)
def _cached_series(
    mtime_key: float,
    forecast_path: str,
    unique_id: str,
    unidad: str,
    has_value: bool,
):
    """Cache por unique_id: evita re-scan del parquet al volver a la misma serie."""
    return artifacts.load_series(
        unique_id,
        Path(forecast_path) if forecast_path else None,
        unidad=unidad,
        has_value=has_value,
    )


@st.cache_data(show_spinner=False)
def _cached_fast_state(
    mtime_key: float,
    forecast_path: str,
    unidad: str,
    freq: str,
    seccion: str,
    store: str | None,
    sku: str | None,
    cutoff_date: dt.date,
):
    """Clave de cache = filtros. Index/metrics + 1 serie cacheada por unique_id."""
    index = _load_index(mtime_key, forecast_path)
    metrics = _load_metrics(mtime_key, forecast_path)
    selected_id = settings.make_unique_id(seccion, store=store, sku=sku)
    has_value = bool(index.get("has_value"))
    df_daily = _cached_series(mtime_key, forecast_path, selected_id, unidad, has_value)
    return prepare_dashboard_state_fast(
        index=index,
        metrics=metrics,
        unidad=unidad,
        freq=freq,
        seccion=seccion,
        store=store,
        sku=sku,
        cutoff_date=cutoff_date,
        forecast_path=forecast_path or None,
        df_daily=df_daily,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fuente de datos
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader(
    "Cargar forecasts (CSV o Parquet)", type=["csv", "parquet"]
)

use_fast = False
res_df = None
index: dict = {}
forecast_path_str = ""
_mtime_key = 0.0

if uploaded is not None:
    # Upload → siempre legacy (no hay artefactos para un archivo arbitrario)
    res_df = _load_bytes(uploaded.getvalue(), uploaded.name)
    st.sidebar.success(f"Cargado (legacy): {uploaded.name}")
    _mtime_key = float(hash(uploaded.name))
    use_fast = False
else:
    default_path = Path(settings.FORECAST_PATH)
    forecast_path_str = str(default_path)
    if default_path.exists():
        _mtime_key = float(default_path.stat().st_mtime)
        if artifacts.artifacts_exist(default_path):
            # Invalidar si el forecast es más nuevo que el index
            try:
                index = _load_index(_mtime_key, forecast_path_str)
                art_mtime = float(index.get("forecast_mtime") or 0)
                if art_mtime and abs(art_mtime - _mtime_key) > 1.0:
                    st.sidebar.warning(
                        "Artefactos desactualizados respecto a forecast.parquet. "
                        "Reconstruir: `python -m app.dashboard_artifacts`"
                    )
                    use_fast = False
                    res_df = _load_parquet(forecast_path_str, _mtime_key)
                else:
                    use_fast = True
                    st.sidebar.success(f"Artefactos listos · {default_path.name}")
            except Exception as exc:  # noqa: BLE001
                st.sidebar.warning(f"No se pudieron cargar artefactos: {exc}")
                use_fast = False
                res_df = _load_parquet(forecast_path_str, _mtime_key)
        else:
            st.sidebar.warning(
                "⚠️ Sin artefactos → modo LENTO. "
                "Ejecutá: `python -m app.dashboard_artifacts` "
                "(o menú opción 4) y recargá."
            )
            use_fast = False
            res_df = _load_parquet(forecast_path_str, _mtime_key)
            st.sidebar.success(f"Cargado (legacy): {default_path.name}")
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

_is_valor = unidad.startswith("Valor")
_opt_yhat = "valuehat" if _is_valor else "yhat"
_opt_yhat28 = "valuehat28" if _is_valor else "yhat28"

if use_fast:
    dataset_has_rolling28 = bool(index.get("has_rolling28"))
else:
    assert res_df is not None
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
        st.sidebar.warning(
            f"Debe quedar al menos una serie; se mantiene «{_opt_yhat}»."
        )
    show_yhat = _opt_yhat in series_forecast
    show_yhat28 = _opt_yhat28 in series_forecast
else:
    show_yhat = True
    show_yhat28 = False

# Label / ids source
if use_fast:
    label_map: dict[str, str] = index.get("label_map") or {}
    secciones_disp = list(index.get("secciones") or [])
else:
    assert res_df is not None
    all_ids = list(_cached_all_ids(res_df, _mtime_key))
    label_map, desc_map = _cached_label_maps(res_df, _mtime_key)
    secciones_disp = backend.secciones_disponibles(all_ids)


def label_for(uid: str) -> str:
    return label_map.get(uid, settings.display_label(uid))


# ─────────────────────────────────────────────────────────────────────────────
# Filtros
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filtros")

if st.sidebar.button("🔄 Reiniciar filtros"):
    for k in ("sel_tienda", "sel_sku", "_last_touched", "_prev_seccion"):
        st.session_state.pop(k, None)
    st.rerun()

default_sec_idx = secciones_disp.index("1") if "1" in secciones_disp else 0
seccion = st.sidebar.selectbox(
    "Sección",
    secciones_disp,
    index=default_sec_idx if secciones_disp else 0,
    key="sel_seccion",
    format_func=lambda x: f"Sección {x}",
)

if st.session_state.get("_prev_seccion") != seccion:
    st.session_state["sel_tienda"] = _SENTINEL_TIENDA
    st.session_state["sel_sku"] = _SENTINEL_SKU
    st.session_state["_last_touched"] = None
    st.session_state["_prev_seccion"] = seccion

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

if use_fast:
    if last_touched == "sku" and _sku_val is not None:
        tienda_opts = [_SENTINEL_TIENDA] + list(
            (index.get("stores_for_sku") or {}).get(seccion, {}).get(_sku_val, [])
        )
    else:
        tienda_opts = [_SENTINEL_TIENDA] + list(
            (index.get("stores_by_sec") or {}).get(seccion, [])
        )
else:
    if last_touched == "sku" and _sku_val is not None:
        tienda_opts = [_SENTINEL_TIENDA] + backend.stores_for_sku(
            all_ids, seccion, _sku_val
        )
    else:
        tienda_opts = [_SENTINEL_TIENDA] + backend.all_stores_in_section(
            all_ids, seccion
        )

if _raw_tienda not in tienda_opts:
    st.session_state["sel_tienda"] = _SENTINEL_TIENDA
    _raw_tienda = _SENTINEL_TIENDA

store_sel = st.sidebar.selectbox(
    "Tienda",
    tienda_opts,
    key="sel_tienda",
    on_change=_touch_tienda,
    format_func=lambda x: (
        x
        if x == _SENTINEL_TIENDA
        else label_for(settings.make_unique_id(seccion, store=x))
    ),
)
store_sel_val = None if store_sel == _SENTINEL_TIENDA else store_sel

if use_fast:
    if last_touched == "tienda" and store_sel_val is not None:
        sku_opts = [_SENTINEL_SKU] + list(
            (index.get("skus_for_store") or {}).get(seccion, {}).get(store_sel_val, [])
        )
    else:
        sku_opts = [_SENTINEL_SKU] + list(
            (index.get("skus_by_sec") or {}).get(seccion, [])
        )
else:
    if last_touched == "tienda" and store_sel_val is not None:
        sku_opts = [_SENTINEL_SKU] + backend.skus_for_store(
            all_ids, seccion, store_sel_val
        )
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
    format_func=lambda x: (
        x if x == _SENTINEL_SKU else label_for(settings.make_unique_id(seccion, sku=x))
    ),
)
sku_sel_val = None if sku_sel == _SENTINEL_SKU else sku_sel

_hz_sec = settings.section_horizons(seccion)
cutoff_date = _hz_sec.get("train_end") or dt.date.today()

# ─────────────────────────────────────────────────────────────────────────────
# ViewModel
# ─────────────────────────────────────────────────────────────────────────────
if use_fast:
    view = _cached_fast_state(
        _mtime_key,
        forecast_path_str,
        unidad,
        freq,
        seccion,
        store_sel_val,
        sku_sel_val,
        cutoff_date,
    )
else:
    assert res_df is not None
    _tabla_base, _n_spine = _cached_section_metrics(
        res_df, unidad, seccion, tuple(all_ids), _mtime_key
    )
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

# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────
if view.store_context:
    st.caption(f"📍 Tienda: {view.store_context}")

_header_label = {
    "seccion": "SECCIÓN",
    "tienda": "TIENDA",
    "sku": "SKU",
    "tienda_sku": "SKU",
}[view.node_kind]
st.subheader(f"{_header_label}: **{view.label}**")

st.markdown("#### Ranking")
st.caption(
    f"Sección **{view.seccion}** · Unidad: **{view.unidad}** · "
    f"N puntos totales (spine): **{view.n_spine}** (~5 filas visibles, scroll). "
    "wMAPE de ranking = **out-of-sample** (bottom-up hojas sku+tienda)."
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
            "N puntos": st.column_config.NumberColumn("N puntos (días ≠0)"),
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
    _show_ranking(
        view.ranking_tiendas,
        f"tabla_tiendas_{view.selected_id}",
        "_pending_tienda",
        "store",
    )
with col_s:
    st.markdown("**SKU**")
    _show_ranking(
        view.ranking_skus, f"tabla_skus_{view.selected_id}", "_pending_sku", "sku"
    )

# hz = view.horizons
# st.markdown("#### Métricas")
# st.caption(
#     f"Agregación: **{view.freq}** · Unidad: **{view.unidad}** · "
#     f"Sección **{view.seccion}** · Train → {hz.get('train_end')} · "
#     f"OOS [{hz.get('test_start')} → {hz.get('test_end')}] · "
#     f"Solo-forecast [{hz.get('forecast_start')} → {hz.get('forecast_end')}]. "
#     "WMAPE bottom-up = Σ|y−ŷ|/Σ|y| sobre hojas sku+tienda; "
#     "OOS = period_type out_sample (misma definición que el ranking)."
# )

# c1, c2, c3 = st.columns(3)
# for col, title, key in (
#     (c1, "In-sample", "in"),
#     (c2, "Out-sample", "out"),
#     (c3, "Total (hasta test_end)", "total"),
# ):
#     m = view.metrics[key]
#     with col:
#         st.markdown(f"**{title}**")
#         st.metric("wMAPE", f"{m['wmape']:.2%}")
#         st.metric("BIAS", f"{m['bias']:+.2%}")
#         st.caption(f"{m['n']} períodos")

hz = view.horizons

st.markdown("#### Métricas Out-of-Sample")
st.caption(
    f"Agregación: **{view.freq}** · "
    f"Unidad: **{view.unidad}** · "
    f"Sección **{view.seccion}** · "
    f"OOS [{hz.get('test_start')} → {hz.get('test_end')}] · "
    "wMAPE = Σ|y−ŷ| / Σ|y|"
)

m = view.metrics["out"]

st.metric("wMAPE", f"{m['wmape']:.2%}", f"{m['bias']:+.2%}")

st.caption(f"{m['n']} períodos")

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

with st.expander("Ver datos detallados"):
    st.caption(f"Agregación **{view.freq}** · Unidad **{view.unidad}**")
    st.dataframe(
        view.detail,
        width="stretch",
        hide_index=True,
        height=_RANK_HEIGHT,
    )
