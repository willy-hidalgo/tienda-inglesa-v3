"""
Forecast Explorer – Streamlit (solo visualización)
=================================================
Path rápido: artefactos precalculados (index + metrics + series por unique_id).
Sin artefactos: fallback legacy (lento) sobre forecast.parquet completo.

Modelo de filtros: Sección obligatoria; Tienda y SKU independientes.
"""
# Legacy test labels retained as comments: Ranking global SKU+Tienda | global_leaf_ranking_unit

from __future__ import annotations

import datetime as dt
import io
import math
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st
import xlsxwriter

st.set_page_config(page_title="Forecast Explorer", layout="wide")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings

st.title(f"📈 Forecast Explorer · v{settings.APP_VERSION} · Secciones 1 & 23")

try:
    from app import backend
    from app import dashboard_artifacts as artifacts
    from app.dashboard_data import (
        global_leaf_ranking_from_metrics,
        load_model_compare_fast,
        prepare_dashboard_state,
        prepare_dashboard_state_fast,
    )
except ImportError:  # pragma: no cover
    import backend  # type: ignore
    import dashboard_artifacts as artifacts  # type: ignore
    from dashboard_data import (  # type: ignore
        global_leaf_ranking_from_metrics,
        load_model_compare_fast,
        prepare_dashboard_state,
        prepare_dashboard_state_fast,
    )

_RANK_HEIGHT = 35 + 5 * 35  # ~5 filas visibles
_SENTINEL_TIENDA = "— Todas las tiendas —"
_SENTINEL_SKU = "— Todos los SKU —"


def _ranking_styler(df: pl.DataFrame, *, yellow: bool = False):
    """Styler liviano: mantiene tipos numéricos y evita estilos por celda costosos."""
    pdf = df.to_pandas()
    numeric_formats: dict[str, str] = {}
    percent_cols = {"wMAPE (%)", "BIAS (%)", "% ≠0", "Impacto error (%)"}
    integer_cols = {"Rank", "N puntos", "Días con venta"}
    for c in pdf.columns:
        # Las columnas porcentuales conservan su formato específico mediante
        # column_config. Todas las demás columnas numéricas usan coma de miles.
        if c in percent_cols or "%" in str(c):
            continue
        if c in integer_cols:
            numeric_formats[c] = "{:,.0f}"
            continue
        try:
            if pdf[c].dtype.kind in "fiu":
                numeric_formats[c] = "{:,.2f}"
        except Exception:
            pass
    styler = pdf.style.format(numeric_formats, na_rep="")
    # Evitar estilos de fondo por celda: en rankings grandes aumentan mucho el
    # payload del Styler y degradan el sort interactivo del grid.
    if yellow:
        styler = styler.set_properties(**{"background-color": "#fffde7"})
    return styler.set_table_styles([
        {"selector": "td", "props": [("font-size", "13px")]},
        {"selector": "th", "props": [("font-size", "13px")]},
    ])


def _ranking_excel_bytes(df: pl.DataFrame) -> bytes:
    """Excel específico del ranking global, generado en memoria."""
    buf = io.BytesIO()
    with xlsxwriter.Workbook(buf, {"in_memory": True}) as wb:
        ws = wb.add_worksheet("Ranking SKU+Tienda")
        fmt_header = wb.add_format({"bold": True, "border": 1})
        fmt_pct = wb.add_format({"num_format": '0.00"%"'})
        fmt_num = wb.add_format({"num_format": "#,##0.00"})
        for j, c in enumerate(df.columns):
            ws.write(0, j, c, fmt_header)
        for i, row in enumerate(df.iter_rows(), start=1):
            for j, value in enumerate(row):
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    value = None
                fmt = None
                if df.columns[j] in {"wMAPE (%)", "BIAS (%)", "% ≠0"}:
                    fmt = fmt_pct
                elif df.columns[j] in {"Volumen OOS", "Pronóstico OOS", "Error abs. OOS"}:
                    fmt = fmt_num
                ws.write(i, j, value, fmt)
        ws.freeze_panes(1, 0)
        if df.columns:
            ws.autofilter(0, 0, max(df.height, 1), len(df.columns) - 1)
        widths = {
            "Rank": 8, "Unidad": 12, "Sección": 9, "Tienda": 10,
            "Tienda descripción": 24, "SKU": 12, "SKU descripción": 36,
            "wMAPE (%)": 12, "BIAS (%)": 12, "Rotación": 14, "N puntos": 10,
            "Días con venta": 14, "% ≠0": 10, "Cohort": 14,
            "Volumen OOS": 16, "Pronóstico OOS": 16, "Error abs. OOS": 16, "unique_id": 30,
        }
        for j, c in enumerate(df.columns):
            ws.set_column(j, j, widths.get(c, 14))
    return buf.getvalue()




def _bias_badge(label: str, value: float | None) -> None:
    """Badge tipo píldora: verde/↑ positivo, rojo/↓ negativo, como el control solicitado."""
    if value is None:
        st.markdown(
            f"<div style='margin-top:-0.20rem;margin-bottom:0.55rem'>"
            f"<span style='font-size:0.78rem;color:#64748b;font-weight:400'>{label}</span><br>"
            "<span style='display:inline-flex;align-items:center;padding:0.22rem 0.55rem;"
            "border-radius:999px;background:#f1f5f9;color:#64748b;font-weight:400'>N/A</span></div>",
            unsafe_allow_html=True,
        )
        return
    v = float(value)
    if v > 0:
        fg, bg, arrow = "#15803d", "#dcfce7", "↑"
    elif v < 0:
        fg, bg, arrow = "#dc2626", "#fee2e2", "↓"
    else:
        fg, bg, arrow = "#64748b", "#f1f5f9", "→"
    st.markdown(
        f"<div style='margin-top:-0.20rem;margin-bottom:0.55rem'>"
        f"<span style='font-size:0.78rem;color:#64748b;font-weight:400'>{label}</span><br>"
        f"<span style='display:inline-flex;align-items:center;gap:0.30rem;padding:0.22rem 0.58rem;"
        f"border-radius:999px;background:{bg};color:{fg};font-weight:400;line-height:1.15'>"
        f"<span style='font-size:1.05rem'>{arrow}</span><span>{v:+.2%}</span></span></div>",
        unsafe_allow_html=True,
    )


def _robust_outliers(xs: list, ys: list, z: float = 4.5) -> tuple[list, list]:
    """Outliers robustos por MAD; no altera ninguna métrica."""
    vals = [float(v) for v in ys if v is not None and math.isfinite(float(v))]
    if len(vals) < 7:
        return [], []
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    med = vals_sorted[n // 2] if n % 2 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
    dev = sorted(abs(v - med) for v in vals_sorted)
    mad = dev[n // 2] if n % 2 else (dev[n // 2 - 1] + dev[n // 2]) / 2
    if mad <= 1e-12:
        return [], []
    ox, oy = [], []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        fv = float(y)
        if not math.isfinite(fv):
            continue
        rz = 0.67448975 * abs(fv - med) / mad
        if rz >= z:
            ox.append(x); oy.append(fv)
    return ox, oy


def _compact_global_ranking(df: pl.DataFrame) -> pl.DataFrame:
    """Ordena primero código/descripcion/wMAPE/BIAS para alinear las 3 tablas."""
    if df.height == 0:
        return df
    x = df.with_columns(
        (pl.col("Tienda").cast(pl.Utf8) + pl.lit(" · ") + pl.col("SKU").cast(pl.Utf8)).alias("Código"),
        (
            pl.col("Tienda descripción").fill_null("").cast(pl.Utf8)
            + pl.when(pl.col("Tienda descripción").fill_null("") != "").then(pl.lit(" · ")).otherwise(pl.lit(""))
            + pl.col("SKU descripción").fill_null("").cast(pl.Utf8)
        ).alias("Descripción"),
    )
    first = ["Código", "Descripción", "wMAPE (%)", "BIAS (%)"]
    rest = [c for c in x.columns if c not in first and c != "unique_id"]
    return x.select(first + rest + (["unique_id"] if "unique_id" in x.columns else []))

# ─────────────────────────────────────────────────────────────────────────────
# Carga de artefactos (rápido) o parquet completo (legacy)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando índice dashboard…", ttl=3600)
def _load_index(mtime_key: float, forecast_path: str):
    return artifacts.load_index(Path(forecast_path) if forecast_path else None)


@st.cache_data(show_spinner="Cargando métricas…", ttl=3600)
def _load_metrics(mtime_key: float, forecast_path: str):
    return artifacts.load_metrics(Path(forecast_path) if forecast_path else None)


@st.cache_data(show_spinner="Cargando métricas in-sample…", ttl=3600)
def _load_metrics_in_sample(mtime_key: float, forecast_path: str):
    return artifacts.load_metrics_in_sample(Path(forecast_path) if forecast_path else None)


@st.cache_data(show_spinner=False, ttl=3600)
def _load_metrics_scope(mtime_key: float, forecast_path: str, period: str, seccion: str, unidad: str):
    return artifacts.load_metrics_scope(
        Path(forecast_path) if forecast_path else None,
        seccion=seccion, unidad=unidad, period=period,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_global_leaf_ranking(
    mtime_key: float, forecast_path: str, period: str = "oos",
    seccion: str | None = None, unidad: str | None = None,
):
    idx = _load_index(mtime_key, forecast_path)
    if seccion is not None and unidad is not None:
        metrics = _load_metrics_scope(mtime_key, forecast_path, period, str(seccion), str(unidad))
    else:
        metrics = _load_metrics_in_sample(mtime_key, forecast_path) if period == "in_sample" else _load_metrics(mtime_key, forecast_path)
    return global_leaf_ranking_from_metrics(
        metrics, label_map=idx.get("label_map") or {}, desc_map=idx.get("desc_map") or {}, period=period,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_global_ranking_excel(mtime_key: float, forecast_path: str, period: str = "oos") -> bytes:
    return _ranking_excel_bytes(_cached_global_leaf_ranking(mtime_key, forecast_path, period))


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
    ranking_days = (
        (hz_spine["test_end"] - hz_spine["test_start"]).days + 1
    )
    candidatos = [
        uid for uid in all_ids_tuple if uid == seccion or uid.startswith(f"{seccion}||")
    ]
    n_data = backend.spine_n_fechas(unit_df, candidatos)
    if n_data > n_spine:
        n_spine = n_data
    # Rankings: wMAPE out-of-sample
    tabla_base = backend.wmape_por_id(
        candidatos,
        unit_df,
        n_fechas_spine=ranking_days,
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


@st.cache_data(show_spinner="Cargando diagnóstico v11/v12…", ttl=3600)
def _cached_model_compare(
    mtime_key: float,
    forecast_path: str,
    unidad: str,
    seccion: str,
    store: str | None,
    sku: str | None,
    has_value: bool,
):
    return load_model_compare_fast(
        seccion=seccion,
        store=store,
        sku=sku,
        unidad=unidad,
        forecast_path=forecast_path or None,
        has_value=has_value,
    )


@st.cache_data(show_spinner=False)
def _cached_fast_state(
    mtime_key: float,
    forecast_path: str,
    ranking_period: str,
    unidad: str,
    freq: str,
    seccion: str,
    store: str | None,
    sku: str | None,
    cutoff_date: dt.date,
):
    """Clave de cache = filtros. Index/metrics + 1 serie cacheada por unique_id."""
    index = _load_index(mtime_key, forecast_path)
    metrics = _load_metrics_scope(mtime_key, forecast_path, "oos", seccion, unidad)
    ranking_metrics = _load_metrics_scope(mtime_key, forecast_path, ranking_period, seccion, unidad) if ranking_period == "in_sample" else metrics
    selected_id = settings.make_unique_id(seccion, store=store, sku=sku)
    has_value = bool(index.get("has_value"))
    df_daily = _cached_series(
        mtime_key, forecast_path, selected_id, unidad, has_value
    )
    return prepare_dashboard_state_fast(
        index=index,
        metrics=metrics,
        ranking_metrics=ranking_metrics,
        unidad=unidad,
        freq=freq,
        seccion=seccion,
        store=store,
        sku=sku,
        cutoff_date=cutoff_date,
        forecast_path=forecast_path or None,
        df_daily=df_daily,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_cadence_summary(paths_key: tuple[tuple[int, str, int], ...], seccion: str, unidad: str):
    rows = []
    for days, path_str, _mtime in paths_key:
        fp = Path(path_str)
        if not artifacts.artifacts_exist(fp):
            continue
        m = artifacts.load_metrics(fp).filter(
            (pl.col("unique_id") == seccion)
            & (pl.col("unidad") == unidad)
        )
        if m.height:
            row = m.row(0, named=True)
            rows.append({
                "Actualización": f"{days}d",
                "wMAPE OOS (%)": None if row.get("wmape") is None else 100.0 * float(row["wmape"]),
                "BIAS OOS (%)": None if row.get("bias") is None else 100.0 * float(row["bias"]),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Fuente de datos
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader(
    "Cargar forecasts (CSV o Parquet)", type=["csv", "parquet"]
)

use_fast = False
res_df = None
index: dict = {}
scenario_paths: dict[int, Path] = {}
forecast_path_str = ""
_mtime_key = 0.0

if uploaded is not None:
    # Upload → siempre legacy (no hay artefactos para un archivo arbitrario)
    res_df = _load_bytes(uploaded.getvalue(), uploaded.name)
    st.sidebar.success(f"Cargado (legacy): {uploaded.name}")
    _mtime_key = float(hash(uploaded.name))
    use_fast = False
else:
    # v12.8.6: escenarios precalculados. El selector solo cambia archivos/artefactos;
    # nunca entrena ni recalcula modelos dentro de Streamlit.
    scenario_paths = {
        int(days): settings.update_block_forecast_path(int(days))
        for days in getattr(settings, "UPDATE_BLOCK_OPTIONS", (1, 7, 14, 28))
        if settings.update_block_forecast_path(int(days)).exists()
    }
    selected_update_days: int | None = None
    if scenario_paths:
        available = sorted(scenario_paths)
        default_idx = available.index(28) if 28 in available else len(available) - 1
        selected_update_days = st.sidebar.select_slider(
            "Bloque de actualización",
            options=available,
            value=available[default_idx],
            format_func=lambda d: f"{d}d",
            help=(
                "Resultados precalculados. OOS y forecast-only siempre cubren 28 días; "
                "solo cambia cada cuántos días se cierra un bloque y se actualiza el modelo."
            ),
            key="update_block_days",
        )
        default_path = scenario_paths[int(selected_update_days)]
    else:
        default_path = Path(settings.FORECAST_PATH)
        st.sidebar.info(
            "No hay escenarios multi-bloque precalculados; mostrando forecast baseline."
        )
    forecast_path_str = str(default_path)
    if default_path.exists():
        _mtime_key = int(default_path.stat().st_mtime_ns)
        if artifacts.artifacts_exist(default_path):
            try:
                index = _load_index(_mtime_key, forecast_path_str)
                use_fast = True
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudieron cargar artefactos consistentes: {exc}")
                st.stop()
        else:
            index_path = artifacts.artifacts_dir(default_path) / "index.json"
            # Para escenarios multi-cadencia NO cargar el parquet completo como
            # fallback: hace lento el arranque y puede agotar RAM. Mostrar el
            # comando exacto que repara solamente los artefactos del escenario.
            if selected_update_days is not None:
                state = "desactualizados" if index_path.exists() else "ausentes"
                st.sidebar.warning(
                    f"⚠️ Artefactos {state} para {selected_update_days}d. "
                    "No se cargará `forecast.parquet` completo. Ejecuta: "
                    f"`uv run python -m app.dashboard_artifacts --update-block-days {selected_update_days}`"
                )
                st.info(
                    f"El forecast de {selected_update_days}d ya existe; solo faltan sus artefactos rápidos. "
                    "No es necesario recalcular modelos."
                )
                st.stop()
            if index_path.exists():
                st.sidebar.warning(
                    "⚠️ Artefactos desactualizados: usando `forecast.parquet` directamente "
                    "(modo legacy). Regenera con `uv run python -m app.dashboard_artifacts`."
                )
            else:
                st.sidebar.warning(
                    "⚠️ Artefactos ausentes → modo legacy. Para producción genera: "
                    "`uv run python -m app.dashboard_artifacts`."
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

# Unidad y frecuencia gobiernan TODO el ViewModel. Limpiamos únicamente
# eventos pendientes de tablas de un estado anterior; tienda/SKU permanecen
# seleccionados para que la comparación entre unidades sea directa.
_view_mode = (unidad, freq)
if st.session_state.get("_prev_view_mode") != _view_mode:
    st.session_state.pop("_pending_tienda", None)
    st.session_state.pop("_pending_sku", None)
    st.session_state["_prev_view_mode"] = _view_mode

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
        st.sidebar.warning(f"Debe quedar al menos una serie; se mantiene «{_opt_yhat}».")
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
    for k in (
        "sel_tienda", "sel_sku", "_last_touched", "_prev_seccion",
        "_prev_view_mode", "_pending_tienda", "_pending_sku",
    ):
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

# Coherencia bidireccional:
# - si hay SKU seleccionado, la lista de tiendas contiene SOLO tiendas donde existe;
# - si hay tienda seleccionada, la lista SKU contiene SOLO SKU de esa tienda.
# Esto aplica igual si la selección vino del sidebar o de una tabla de ranking.
if use_fast:
    all_stores = list((index.get("stores_by_sec") or {}).get(seccion, []))
    all_skus = list((index.get("skus_by_sec") or {}).get(seccion, []))
    if _sku_val is not None:
        allowed_stores = list(
            (index.get("stores_for_sku") or {}).get(seccion, {}).get(_sku_val, [])
        )
    else:
        allowed_stores = all_stores
else:
    all_stores = backend.all_stores_in_section(all_ids, seccion)
    all_skus = backend.all_skus_in_section(all_ids, seccion)
    allowed_stores = (
        backend.stores_for_sku(all_ids, seccion, _sku_val)
        if _sku_val is not None
        else all_stores
    )

tienda_opts = [_SENTINEL_TIENDA] + list(allowed_stores)
if _raw_tienda not in tienda_opts:
    st.session_state["sel_tienda"] = _SENTINEL_TIENDA
    _raw_tienda = _SENTINEL_TIENDA
    _store_val = None

store_sel = st.sidebar.selectbox(
    "Tienda",
    tienda_opts,
    key="sel_tienda",
    on_change=_touch_tienda,
    format_func=lambda x: x
    if x == _SENTINEL_TIENDA
    else label_for(settings.make_unique_id(seccion, store=x)),
)
store_sel_val = None if store_sel == _SENTINEL_TIENDA else store_sel

if use_fast:
    if store_sel_val is not None:
        allowed_skus = list(
            (index.get("skus_for_store") or {})
            .get(seccion, {})
            .get(store_sel_val, [])
        )
    else:
        allowed_skus = all_skus
else:
    allowed_skus = (
        backend.skus_for_store(all_ids, seccion, store_sel_val)
        if store_sel_val is not None
        else all_skus
    )

sku_opts = [_SENTINEL_SKU] + list(allowed_skus)
if _raw_sku not in sku_opts:
    st.session_state["sel_sku"] = _SENTINEL_SKU
    _raw_sku = _SENTINEL_SKU

sku_sel = st.sidebar.selectbox(
    "SKU",
    sku_opts,
    key="sel_sku",
    on_change=_touch_sku,
    format_func=lambda x: x
    if x == _SENTINEL_SKU
    else label_for(settings.make_unique_id(seccion, sku=x)),
)
sku_sel_val = None if sku_sel == _SENTINEL_SKU else sku_sel

_hz_sec = settings.section_horizons(seccion)
cutoff_date = _hz_sec.get("train_end") or dt.date.today()

# Período de métricas para TODAS las tablas de ranking. El control se renderiza
# en el cuerpo, inmediatamente sobre las tablas; aquí solo leemos su estado para
# construir el ViewModel sin añadir controles al sidebar.
if "ranking_metric_period_selector" not in st.session_state:
    st.session_state["ranking_metric_period_selector"] = "OOS"
_ranking_period_label = str(st.session_state["ranking_metric_period_selector"])
_ranking_is_insample = _ranking_period_label == "In-sample"
_ranking_period = "in_sample" if _ranking_is_insample else "oos"

# ─────────────────────────────────────────────────────────────────────────────
# ViewModel
# ─────────────────────────────────────────────────────────────────────────────
if use_fast:
    view = _cached_fast_state(
        _mtime_key,
        forecast_path_str,
        _ranking_period,
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
_metrics_mode = str(getattr(settings, "METRICS_MODE", "rolling_28")).lower()

_view_signature = "|".join(
    [
        str(view.unidad),
        str(view.freq),
        str(view.seccion),
        str(view.store or "*"),
        str(view.sku or "*"),
    ]
)

_header_label = {
    "seccion": "SECCIÓN",
    "tienda": "TIENDA",
    "sku": "SKU",
    "tienda_sku": "SKU",
}[view.node_kind]
st.subheader(f"{_header_label}: **{view.label}**")
if use_fast and index.get("update_block_days"):
    _ud = int(index["update_block_days"])
    st.caption(
        f"Cadencia: **cada {_ud} día{'s' if _ud != 1 else ''}** · "
        f"OOS **{int(index.get('metric_horizon_days') or 28)} días** · forecast-only **28 días**"
    )

if view.consistency_warnings:
    for _warning in view.consistency_warnings:
        st.error("⚠️ " + _warning)
    st.error(
        "Dashboard detenido: falta la métrica OOS correspondiente en los artefactos "
        "o éstos no pertenecen al forecast actual. Regenera con "
        "`python -m app.dashboard_artifacts`."
    )
    st.stop()

st.info(
    "**Lectura del dashboard:** los resultados operativos se muestran a nivel **SKU+Tienda** y "
    "los **wMAPE oficiales son bottom-up**, calculados desde las hojas SKU+Tienda (días con venta ≠ 0)."
)

# Comparación de cadencias: métrica del nodo SECCIÓN, no de la hoja seleccionada.
# Se ubica inmediatamente debajo de la lectura del dashboard para que su nivel quede claro.
if use_fast and len(scenario_paths) >= 2:
    st.markdown("#### Cadencias precalculadas · nivel Sección")
    _show_cadence_compare = st.toggle(
        "Mostrar comparación de cadencias",
        value=False,
        key="show_cadence_compare",
    )
    if _show_cadence_compare:
        _paths_key = tuple(
            (d, str(fp), int(fp.stat().st_mtime_ns))
            for d, fp in sorted(scenario_paths.items())
            if fp.exists()
        )
        _cadence_rows = _cached_cadence_summary(_paths_key, view.seccion, view.unidad)
        if _cadence_rows:
            st.caption(f"Sección {view.seccion} · {view.unidad} · OOS 28 días")
            st.dataframe(
                _cadence_rows,
                hide_index=True,
                width="content",
                column_config={
                    "wMAPE OOS (%)": st.column_config.NumberColumn("wMAPE OOS (%)", format="%.2f%%"),
                    "BIAS OOS (%)": st.column_config.NumberColumn("BIAS OOS (%)", format="%.2f%%"),
                },
            )

st.markdown("#### Rankings")
# Control compacto y alineado. Segmented control evita la desalineación visual
# del switch entre dos etiquetas y deja explícito cuál período está activo.
_ranking_period_label = st.segmented_control(
    "Período de métricas de ranking",
    options=["OOS", "In-sample"],
    selection_mode="single",
    key="ranking_metric_period_selector",
    label_visibility="collapsed",
    help="Selecciona el período cuyas métricas se muestran en los tres rankings.",
)
if _ranking_period_label is None:
    _ranking_period_label = "OOS"
_ranking_is_insample = _ranking_period_label == "In-sample"
_ranking_period = "in_sample" if _ranking_is_insample else "oos"
st.markdown(f"##### Mostrando métricas: **{_ranking_period_label.upper()}**")
_ranking_scope = []
if view.store:
    _ranking_scope.append(f"SKU en tienda **{view.store}**")
if view.sku:
    _ranking_scope.append(f"tiendas con SKU **{view.sku}**")
_scope_txt = " · ".join(_ranking_scope) if _ranking_scope else "sin filtro cruzado tienda/SKU"
st.caption(
    f"Sección **{view.seccion}** · **{view.unidad}** · {_scope_txt} · "
    "wMAPE bottom-up desde hojas SKU+Tienda Active."
)


def _show_ranking(display, key: str, pending_key: str, extract_field: str) -> None:
    if display.height == 0:
        st.caption(
            "Sin datos suficientes para ranking "
            "(no hay hojas sku+tienda con rotación > 0 en esta selección, "
            "o el parquet no incluye ese nivel)."
        )
        return
    preferred = ["Código", "Descripción", "wMAPE (%)", "BIAS (%)"]
    visible = [c for c in preferred if c in display.columns] + [
        c for c in display.columns if c not in preferred and c != "unique_id"
    ]
    _rank_view = display.select(visible)
    event = st.dataframe(
        _ranking_styler(_rank_view),
        width="stretch",
        hide_index=True,
        height=_RANK_HEIGHT,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
        column_config={
            "Código": st.column_config.TextColumn("Código"),
            "Descripción": st.column_config.TextColumn("Descripción"),
            "wMAPE (%)": st.column_config.NumberColumn("wMAPE (%)", format="%.2f%%"),
            "BIAS (%)": st.column_config.NumberColumn("BIAS (%)", format="%.2f%%"),
            "% ≠0": st.column_config.NumberColumn("% ≠0", format="%.1f%%"),
            "Impacto error (%)": st.column_config.NumberColumn("Impacto error (%)", format="%.1f%%"),
            "Estado": st.column_config.TextColumn("Estado"),
            "Sel.": st.column_config.TextColumn("Sel.", width="small"),
        },
    )
    if event and event.selection and event.selection.rows:
        clicked_uid = display["unique_id"][event.selection.rows[0]]
        p = settings.split_unique_id(clicked_uid)
        st.session_state[pending_key] = p[extract_field]
        st.rerun()


col_t, col_s, col_g = st.columns(3, gap="small")
_global_rank_all = _cached_global_leaf_ranking(
    _mtime_key, forecast_path_str, _ranking_period, str(view.seccion), str(view.unidad)
) if use_fast else pl.DataFrame()
# La tabla visible debe estar en el MISMO contexto que el KPI: sección + unidad.
# El artefacto global conserva todas las secciones/unidades para exportación, pero
# mostrar ambas sin la columna Unidad hace ambiguo un mismo SKU+Tienda.
if use_fast and _global_rank_all.height:
    _global_rank_visible = _global_rank_all.filter(
        (pl.col("Unidad") == view.unidad)
        & (pl.col("Sección").cast(pl.Utf8) == str(view.seccion))
        & (pl.col("Cohort") == "active")
    )
else:
    _global_rank_visible = pl.DataFrame()
if use_fast and _global_rank_visible.height:
    _vol_col = "Volumen in-sample" if _ranking_period == "in_sample" else "Volumen OOS"
    _fc_col = "Pronóstico in-sample" if _ranking_period == "in_sample" else "Pronóstico OOS"
    _err_col = "Error abs. in-sample" if _ranking_period == "in_sample" else "Error abs. OOS"
    _global_rank_display = (
        _global_rank_visible
        .sort(["wMAPE (%)", "Tienda", "SKU"], nulls_last=True, maintain_order=True)
        .drop("Rank")
        .with_row_index("Rank", offset=1)
        .select([
            "Tienda", "SKU", "SKU descripción", "wMAPE (%)", "BIAS (%)", "Rotación", "Rank",
            "Unidad", "N puntos", "Días con venta", "% ≠0", "Cohort",
            _vol_col, _fc_col, _err_col, "unique_id",
        ])
    )
else:
    _global_rank_display = pl.DataFrame()

with col_t:
    st.markdown(f"**Tiendas · {_ranking_period_label.upper()}{' · SKU ' + str(view.sku) if view.sku else ''}**")
    _show_ranking(
        view.ranking_tiendas,
        f"tabla_tiendas_{_view_signature}",
        "_pending_tienda",
        "store",
    )
with col_s:
    st.markdown(f"**SKU · {_ranking_period_label.upper()}{' · tienda ' + str(view.store) if view.store else ''}**")
    _show_ranking(
        view.ranking_skus, f"tabla_skus_{_view_signature}", "_pending_sku", "sku"
    )
with col_g:
    st.markdown(f"**SKU+Tienda · Active · {_ranking_period_label.upper()}**")
    if use_fast and _global_rank_display.height:
        _global_cols = [c for c in _global_rank_display.columns if c != "unique_id"]
        _global_table_view = _global_rank_display.select(_global_cols)
        _global_event = st.dataframe(
            _ranking_styler(_global_table_view, yellow=False),
            width="stretch",
            hide_index=True,
            height=_RANK_HEIGHT,
            on_select="rerun",
            selection_mode="single-row",
            key=f"tabla_global_{_view_signature}_{_ranking_period}",
            column_config={
                # Sin ancho fijo: Streamlit autoajusta códigos al contenido.
                "Tienda": st.column_config.TextColumn("Tienda"),
                "SKU": st.column_config.TextColumn("SKU"),
                "SKU descripción": st.column_config.TextColumn("SKU descripción", width="large"),
                "wMAPE (%)": st.column_config.NumberColumn("wMAPE (%)", format="%.2f%%"),
                "BIAS (%)": st.column_config.NumberColumn("BIAS (%)", format="%.2f%%"),
                "Unidad": st.column_config.TextColumn("Unidad"),
                "% ≠0": st.column_config.NumberColumn("% ≠0", format="%.1f%%"),
                "Cohort": st.column_config.TextColumn("Cohort"),
            },
        )
        if _global_event and _global_event.selection and _global_event.selection.rows:
            _row_ix = int(_global_event.selection.rows[0])
            if 0 <= _row_ix < _global_rank_display.height:
                _uid = _global_rank_display["unique_id"][_row_ix]
                _parts = settings.split_unique_id(_uid)
                st.session_state["_pending_tienda"] = _parts.get("store")
                st.session_state["_pending_sku"] = _parts.get("sku")
                st.session_state["_last_touched"] = "sku_tienda"
                st.rerun()
        _n_pairs = _global_rank_display.height
        _n_skus = int(_global_rank_display.select(pl.col("SKU").n_unique()).item())
        st.caption(f"{_n_pairs:,} SKU+Tienda Active · {_n_skus:,} SKU únicos")
    else:
        st.caption("Ranking SKU+Tienda Active disponible en modo artefactos.")

if use_fast:
    _export_state_key = f"_rank_xlsx::{forecast_path_str}::{_ranking_period}"
    if st.button("Preparar Excel del ranking global SKU+Tienda", key=f"prepare_rank_xlsx_{_ranking_period}"):
        st.session_state[_export_state_key] = _cached_global_ranking_excel(_mtime_key, forecast_path_str, _ranking_period)
    if _export_state_key in st.session_state:
        st.download_button(
            "⬇️ Exportar ranking global SKU+Tienda a Excel",
            data=st.session_state[_export_state_key],
            file_name=f"ranking_sku_tienda_{_ranking_period}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"export_global_rank_{_ranking_period}",
        )

hz = view.horizons


st.markdown("#### Métricas · selección actual")
st.caption(
    f"Sección **{view.seccion}** · **{view.unidad}** · {view.freq} · "
    f"Train hasta **{hz.get('train_end')}** · "
    f"OOS **{hz.get('test_start')} → {hz.get('test_end')}** · "
    f"forecast-only **{hz.get('forecast_start')} → {hz.get('forecast_end')}** · "
    f"Active: ≥{int(getattr(settings, 'OOS_ACTIVE_MIN_NONZERO_DAYS', 7))} días con venta."
)

m = view.metrics.get("out", {})
_coh = view.metrics.get("cohorts", {})
_zero = view.metrics.get("zero_demand", {})

# OOS e in-sample oficiales deben venir de los artefactos de métricas.
# IMPORTANTE: desde que los gráficos de SKU/Tienda/Sección muestran series
# agregadas bottom-up, recalcular in-sample directamente desde la curva visible
# ya no reproduce la definición oficial (error hoja por hoja, y!=0). Además, los
# agregados visuales no tienen una única rls_metric_eligible representativa de
# todas las hojas; usar esa columna podía vaciar el tramo y mostrar falsamente
# 0.00%.  Por eso el KPI in-sample usa metrics_in_sample.parquet igual que OOS
# usa metrics.parquet.
_selected_metric_id = settings.make_unique_id(
    view.seccion, store=view.store, sku=view.sku
)
if use_fast:
    _metrics_in_scope = _load_metrics_scope(
        _mtime_key, forecast_path_str, "in_sample", str(view.seccion), view.unidad
    )
    _m_in_row = _metrics_in_scope.filter(
        pl.col("unique_id") == _selected_metric_id
    )
    _m_in = _m_in_row.row(0, named=True) if _m_in_row.height else {}
else:
    _m_in = backend.metrics_in_out_bottom_up(
        backend.prepare_unit_df(
            res_df, view.unidad, ("value" in res_df.columns and "valuehat" in res_df.columns)
        ),
        seccion=str(view.seccion),
        store=view.store,
        sku=view.sku,
        cutoff=hz.get("test_start") or cutoff_date,
        test_end=hz.get("test_end") or cutoff_date,
    ).get("in", {})

mins, moos, c3, c4 = st.columns([1.15, 1.15, 1, 1])
with mins:
    st.markdown("**In-sample**")
    _wi = _m_in.get("wmape")
    st.metric("wMAPE in-sample", f"{float(_wi):.2%}" if _wi is not None else "N/A")
    _bias_badge("BIAS in-sample", _m_in.get("bias"))
with moos:
    st.markdown("**Out-of-sample**")
    _w = m.get("wmape")
    st.metric("wMAPE OOS", f"{float(_w):.2%}" if _w is not None else "N/A")
    _bias_badge("BIAS OOS", m.get("bias"))
    _b = m.get("bias")
    if _w is not None and _b is not None and abs(float(_w) - abs(float(_b))) <= 5e-6:
        st.caption("wMAPE y |BIAS| coinciden: es válido cuando el error OOS tiene esencialmente un solo signo.")
with c3:
    st.metric("Hojas Active", f"{int(_coh.get('active', 0)):,}")
    st.caption(f"Sparse: {int(_coh.get('sparse', 0)):,}")
with c4:
    st.metric("Hojas zero-demand", f"{int(_coh.get('zero', 0)):,}")
    _zf = float(_zero.get("forecast_sum", 0.0) or 0.0)
    _prefix = "$ " if view.unidad.startswith("Valor") else ""
    st.caption(f"Forecast sin demanda: {_prefix}{_zf:,.2f}")

# Diagnóstico pesado v11/v12: bajo demanda para no cargar millones de hojas
# durante el arranque normal del dashboard.
_cmp = view.model_compare or {}
if use_fast:
    _show_model_diag = st.toggle(
        "Cargar diagnóstico avanzado v11/v12",
        value=False,
        help=(
            "Carga las hojas SKU×tienda del alcance para comparar Final/v11/v12/Oracle. "
            "Puede tardar algunos segundos a nivel sección."
        ),
        key=f"diag_{_view_signature}",
    )
    if _show_model_diag:
        _cmp = _cached_model_compare(
            _mtime_key,
            forecast_path_str,
            unidad,
            seccion,
            store_sel_val,
            sku_sel_val,
            bool(index.get("has_value")),
        ) or {}
if _cmp.get("scenarios"):
    st.markdown("#### Diagnóstico de modelo v12.9.12 · OOS bottom-up Active")
    st.caption(
        "Compara el forecast final contra los dos candidatos ya calculados. "
        "El Oracle elige v11/v12 usando el OOS real y es **solo una cota diagnóstica, no causal**."
    )
    _rows = []
    for _name, _vals in _cmp["scenarios"].items():
        _rows.append({
            "Escenario": _name,
            "wMAPE (%)": f"{float(_vals['wmape']) * 100:,.2f}",
            "BIAS (%)": f"{float(_vals['bias']) * 100:+,.2f}",
        })
    st.dataframe(_rows, width="content", hide_index=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Hojas Active", f"{int(_cmp.get('n_active', 0)):,}")
    with mc2:
        _sel = int(_cmp.get("selected_v12", 0))
        _n = max(int(_cmp.get("n_active", 0)), 1)
        st.metric("Seleccionadas v12", f"{_sel:,} ({_sel/_n:.1%})")
    with mc3:
        _cal = _cmp.get("cal_factor_median")
        st.metric("Calibración nivel mediana", f"{float(_cal):.3f}×" if _cal is not None else "N/A")
    with mc4:
        _gap = float(_cmp.get("selector_gap", 0.0) or 0.0)
        st.metric("Gap selector vs oracle", f"{_gap * 100:+.2f} pp")

    mp1, mp2, mp3, mp4 = st.columns(4)
    with mp1:
        _p = _cmp.get("meta_probability_median")
        st.metric("P(v12 gana) mediana", f"{float(_p):.1%}" if _p is not None else "N/A")
    with mp2:
        _ps = _cmp.get("meta_probability_selected_median")
        st.metric("P(v12 gana) · seleccionadas", f"{float(_ps):.1%}" if _ps is not None else "N/A")
    with mp3:
        st.metric("Cobertura meta-selector", f"{float(_cmp.get('meta_available_pct', 0.0) or 0.0):.1%}")
    with mp4:
        st.metric("Calibración aplicada", f"{float(_cmp.get('cal_applied_pct', 0.0) or 0.0):.1%}")
        st.caption("challenger de nivel apagado por defecto")

    _td = _cmp.get("meta_top_driver")
    _th = float(_cmp.get("meta_threshold", 0.55) or 0.55)
    _mode = _cmp.get("meta_portfolio_mode") or "N/A"
    _policy_gain = _cmp.get("meta_policy_utility_gain")
    ma1, ma2, ma3 = st.columns(3)
    with ma1:
        st.metric("Modo causal portfolio", str(_mode))
    with ma2:
        st.metric("Umbral aprendido", f"{_th:.0%}")
    with ma3:
        st.metric(
            "Gain utilidad política",
            f"{float(_policy_gain) * 100:+.2f} pp" if _policy_gain is not None else "N/A",
        )
    st.caption(
        f"Meta-selector causal adaptativo: modo `{_mode}`, umbral P(v12 gana)={_th:.0%}; "
        f"guard BIAS pasa en {float(_cmp.get('meta_bias_guard_pass_pct', 0.0) or 0.0):.1%} de hojas Active. "
        + (f"Driver dominante entre hojas seleccionadas: `{_td}`." if _td else "Sin driver dominante disponible.")
    )

    if view.unidad.startswith("Valor") and _cmp.get("value_safety_reason") is not None:
        st.markdown("##### v12.9.7 · Value Portfolio Safety (legacy diagnostic)")
        vs1, vs2, vs3, vs4 = st.columns(4)
        with vs1:
            st.metric("Dominancia v12_all", "Sí" if bool(_cmp.get("value_safety_dominance_pass")) else "No")
        with vs2:
            _rc = int(_cmp.get("value_safety_recent_confirmations", 0) or 0)
            _rb = int(_cmp.get("value_safety_recent_blocks", 0) or 0)
            st.metric("Confirmaciones recientes", f"{_rc}/{_rb}")
        with vs3:
            _cov = _cmp.get("value_safety_bias_coverage")
            _cov_th = _cmp.get("value_safety_bias_coverage_threshold")
            st.metric(
                "Cobertura guard BIAS",
                f"{float(_cov):.1%}" if _cov is not None else "N/A",
                delta=(f"mín. {float(_cov_th):.0%}" if _cov_th is not None else None),
            )
        with vs4:
            st.metric("Mejor all-mode seguro", str(_cmp.get("value_safety_best_all_mode") or "N/A"))
        st.caption(
            f"Safety reason: `{_cmp.get('value_safety_reason')}` · "
            f"coverage-pass={'sí' if bool(_cmp.get('value_safety_bias_coverage_pass')) else 'no'} · "
            f"meta-margin={'sí' if bool(_cmp.get('value_safety_meta_margin_pass')) else 'no'}."
        )

    if view.unidad.startswith("Valor") and _cmp.get("v129_wf_reason") is not None:
        st.markdown("##### v12.9.2 · Walk-Forward Portfolio Selector")
        wf1, wf2, wf3, wf4 = st.columns(4)
        with wf1:
            st.metric("Folds walk-forward", str(int(_cmp.get("v129_wf_folds", 0) or 0)))
        with wf2:
            _wr = _cmp.get("v129_wf_win_rate")
            st.metric("Win-rate v12_all", f"{float(_wr):.0%}" if _wr is not None else "N/A")
        with wf3:
            _wg = _cmp.get("v129_wf_weighted_gain")
            st.metric("Gain wMAPE ponderado", f"{float(_wg) * 100:+.2f} pp" if _wg is not None else "N/A")
        with wf4:
            _worst = _cmp.get("v129_wf_worst_gain")
            st.metric("Peor fold", f"{float(_worst) * 100:+.2f} pp" if _worst is not None else "N/A")
        st.caption(
            f"Policy reason: `{_cmp.get('v129_wf_reason')}` · "
            f"meta-folds={int(_cmp.get('v129_wf_meta_folds', 0) or 0)} · "
            f"median-gain={float(_cmp.get('v129_wf_median_gain') or 0.0) * 100:+.2f} pp · "
            f"max-bias-worsen={float(_cmp.get('v129_wf_bias_worsen_max') or 0.0) * 100:+.2f} pp."
        )

    _daily = _cmp.get("daily") or {}
    if _daily.get("ds"):
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=_daily["ds"], y=_daily["actual"], name="Actual Active", mode="lines"))
        fig_cmp.add_trace(go.Scatter(x=_daily["ds"], y=_daily["final"], name="Final seleccionado", mode="lines"))
        fig_cmp.add_trace(go.Scatter(x=_daily["ds"], y=_daily["v11"], name="v11 incumbent", mode="lines", line=dict(dash="dot")))
        fig_cmp.add_trace(go.Scatter(x=_daily["ds"], y=_daily["v12"], name="v12 all", mode="lines", line=dict(dash="dash")))
        fig_cmp.update_layout(
            title="Comparación OOS bottom-up · hojas Active",
            xaxis_title="Fecha", yaxis_title=view.unidad, hovermode="x unified",
            legend_title_text="", margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_cmp, width="stretch")

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
_mark_outliers = st.toggle(
    "Marcar outliers robustos", value=True, key=f"outliers_{_view_signature}",
    help="Marca observaciones alejadas de la mediana según MAD (umbral robusto 4.5). Es solo visual; no cambia métricas ni forecasts.",
)

fig = go.Figure()
# Períodos visualmente distintos + contraste área/línea dentro de cada período.
# In-sample = azul; OOS = naranja; forecast-only = violeta punteado.
_in_actual_line = "rgba(59,130,246,0.58)"
_in_actual_fill = "rgba(147,197,253,0.20)"
_in_forecast_line = "#1d4ed8"
_oos_actual_line = "rgba(249,115,22,0.65)"
_oos_actual_fill = "rgba(254,215,170,0.28)"
_oos_forecast_line = "#ea580c"
_forecast_only_line = "#7c3aed"
if ch.get("in_ds"):
    fig.add_trace(go.Scatter(
        x=ch["in_ds"], y=ch["in_y"], name="Actual · in-sample", mode="lines", fill="tozeroy", fillcolor=_in_actual_fill, line=dict(width=1.0, color=_in_actual_line),
    ))
    if show_yhat:
        fig.add_trace(go.Scatter(
            x=ch["in_ds"], y=ch["in_yhat"], name=f"{_opt_yhat} · in-sample",
            mode="lines", line=dict(width=2.8, color=_in_forecast_line),
        ))
if ch.get("oos_ds"):
    fig.add_trace(go.Scatter(
        x=ch["oos_ds"], y=ch["oos_y"], name="Actual · OOS", mode="lines", fill="tozeroy", fillcolor=_oos_actual_fill, line=dict(width=1.0, color=_oos_actual_line),
    ))
    if show_yhat:
        fig.add_trace(go.Scatter(
            x=ch["oos_ds"], y=ch["oos_yhat"], name=f"{_opt_yhat} · OOS",
            mode="lines", line=dict(width=3.6, color=_oos_forecast_line),
        ))
if ch.get("fcst_ds") and show_yhat:
    fig.add_trace(go.Scatter(
        x=ch["fcst_ds"], y=ch["fcst_yhat"], name=f"{_opt_yhat} · forecast-only",
        mode="lines", line=dict(dash="dot", width=3.2, color=_forecast_only_line),
    ))

if _mark_outliers:
    _ax = list(ch.get("in_ds", [])) + list(ch.get("oos_ds", []))
    _ay = list(ch.get("in_y", [])) + list(ch.get("oos_y", []))
    _ox, _oy = _robust_outliers(_ax, _ay)
    if _ox:
        fig.add_trace(go.Scatter(x=_ox, y=_oy, name="Outlier · actual", mode="markers", marker=dict(size=9, symbol="circle-open")))
    _fx = list(ch.get("in_ds", [])) + list(ch.get("oos_ds", [])) + list(ch.get("fcst_ds", []))
    _fy = list(ch.get("in_yhat", [])) + list(ch.get("oos_yhat", [])) + list(ch.get("fcst_yhat", []))
    _fox, _foy = _robust_outliers(_fx, _fy)
    if _fox:
        fig.add_trace(go.Scatter(x=_fox, y=_foy, name="Outlier · forecast", mode="markers", marker=dict(size=9, symbol="x")))
if view.has_rolling28 and show_yhat28:
    if ch.get("in_yhat28"):
        fig.add_trace(go.Scatter(x=ch["in_ds"], y=ch["in_yhat28"], name=f"{_opt_yhat28} · in-sample", mode="lines", line=dict(width=2.0, dash="dash", color=_in_forecast_line)))
    if ch.get("oos_yhat28"):
        fig.add_trace(go.Scatter(x=ch["oos_ds"], y=ch["oos_yhat28"], name=f"{_opt_yhat28} · OOS", mode="lines", line=dict(width=2.2, dash="dash", color=_oos_forecast_line)))
    if ch.get("fcst_yhat28"):
        fig.add_trace(go.Scatter(x=ch["fcst_ds"], y=ch["fcst_yhat28"], name=f"{_opt_yhat28} · forecast-only", mode="lines", line=dict(width=2.2, dash="dashdot", color=_forecast_only_line)))

# La línea visible marca la PRIMERA fecha OOS de la sección, no el último día train.
cutoff_x = hz.get("test_start") or ch["cutoff"]
if isinstance(cutoff_x, dt.date) and not isinstance(cutoff_x, dt.datetime):
    cutoff_x = dt.datetime.combine(cutoff_x, dt.time.min)
fig.add_vline(
    x=cutoff_x,
    line_dash="dash",
    line_color="rgba(220, 50, 50, 0.8)",
    annotation_text="Inicio OOS",
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
_chart_source = (
    " · forecast agregado bottom-up desde SKU+Tienda"
    if view.node_kind in {"seccion", "tienda", "sku"}
    else ""
)
fig.update_layout(
    title=f"v{settings.APP_VERSION} · actual + in-sample + OOS + forecast-only — {view.label} ({view.freq.lower()}, {unidad_label}){_chart_source}",
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

# Explicación visual opcional de la métrica oficial.  La curva principal
# muestra sumas de actual/forecast; este panel muestra la suma de errores
# absolutos calculados hoja por hoja, por lo que no existe compensación entre
# hojas y coincide con el numerador del wMAPE bottom-up actual.
if view.node_kind in {"seccion", "tienda", "sku"} and (ch.get("in_bu_abs_error") or ch.get("oos_bu_abs_error")):
    with st.expander("Cómo se forma el wMAPE bottom-up", expanded=False):
        st.caption(
            "El forecast del gráfico es la suma diaria de forecasts finales SKU+Tienda. "
            "El wMAPE oficial no cambia: su error absoluto se calcula hoja por hoja "
            "para y ≠ 0 y luego se suma, evitando compensación entre hojas."
        )
        _err_fig = go.Figure()
        if ch.get("in_ds") and ch.get("in_bu_abs_error"):
            _err_fig.add_trace(go.Bar(
                x=ch["in_ds"], y=ch["in_bu_abs_error"], name="Error absoluto BU · in-sample"
            ))
        if ch.get("oos_ds") and ch.get("oos_bu_abs_error"):
            _err_fig.add_trace(go.Bar(
                x=ch["oos_ds"], y=ch["oos_bu_abs_error"], name="Error absoluto BU · OOS"
            ))
        _err_fig.update_layout(
            xaxis_title="Fecha", yaxis_title=f"Error absoluto · {unidad_label}",
            legend_title_text="", hovermode="x unified",
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(_err_fig, width="stretch")


# ── SKU+Tienda no activos: listado + serie completa bajo demanda ─────────────
if use_fast:
    with st.expander("SKU+Tienda no activos · historia + forecast", expanded=False):
        st.caption(
            "Hojas de la sección/unidad actual cuyo cohort OOS no es `active` (sparse o zero). "
            "La serie conserva toda la historia disponible y todo el horizonte forecast-only."
        )
        _na = _compact_global_ranking(_global_rank_all).filter(
            (pl.col("Sección").cast(pl.Utf8) == str(view.seccion))
            & (pl.col("Unidad") == view.unidad)
            & (pl.col("Cohort") != "active")
        ) if _global_rank_all.height else pl.DataFrame()
        if _na.height:
            _na_visible = [c for c in ["Código", "Descripción", "wMAPE (%)", "BIAS (%)", "Pronóstico OOS", "Cohort", "Días con venta", "% ≠0", "unique_id"] if c in _na.columns]
            _na_event = st.dataframe(
                _ranking_styler(_na.select([c for c in _na_visible if c != "unique_id"])),
                width="stretch", height=250, hide_index=True,
                on_select="rerun", selection_mode="single-row", key=f"nonactive_table_{view.seccion}_{view.unidad}",
                column_config={
                    "wMAPE (%)": st.column_config.NumberColumn("wMAPE (%)", format="%.2f%%"),
                    "BIAS (%)": st.column_config.NumberColumn("BIAS (%)", format="%.2f%%"),
                    "% ≠0": st.column_config.NumberColumn("% ≠0", format="%.1f%%"),
                    },
            )
            _na_uid = st.session_state.get("_nonactive_uid")
            if _na_event and _na_event.selection and _na_event.selection.rows:
                _na_uid = _na["unique_id"][_na_event.selection.rows[0]]
                st.session_state["_nonactive_uid"] = _na_uid
            if _na_uid not in set(_na["unique_id"].to_list()):
                _na_uid = _na["unique_id"][0]
                st.session_state["_nonactive_uid"] = _na_uid
            _na_series = _cached_series(_mtime_key, forecast_path_str, _na_uid, view.unidad, bool(index.get("has_value")))
            _na_hz = (index.get("horizons_by_sec") or {}).get(str(view.seccion), {})
            _na_fig = go.Figure()
            if _na_series.height:
                if "period_type" in _na_series.columns:
                    _na_in = _na_series.filter(pl.col("period_type") == "in_sample")
                    _na_oos = _na_series.filter(pl.col("period_type") == "out_sample")
                    _na_fc = _na_series.filter(pl.col("period_type") == "forecast_only")
                else:
                    _na_in, _na_oos, _na_fc = _na_series, pl.DataFrame(), pl.DataFrame()
                if _na_in.height:
                    _na_fig.add_trace(go.Scatter(
                        x=_na_in["ds"].to_list(), y=_na_in["y"].to_list(), name="Actual · in-sample", mode="lines",
                        fill="tozeroy", fillcolor=_in_actual_fill,
                        line=dict(width=1.0, color=_in_actual_line),
                    ))
                    _na_fig.add_trace(go.Scatter(
                        x=_na_in["ds"].to_list(), y=_na_in["yhat"].to_list(), name="Forecast · in-sample",
                        mode="lines", line=dict(width=2.8, color=_in_forecast_line),
                    ))
                if _na_oos.height:
                    _na_fig.add_trace(go.Scatter(
                        x=_na_oos["ds"].to_list(), y=_na_oos["y"].to_list(), name="Actual · OOS", mode="lines",
                        fill="tozeroy", fillcolor=_oos_actual_fill,
                        line=dict(width=1.0, color=_oos_actual_line),
                    ))
                    _na_fig.add_trace(go.Scatter(
                        x=_na_oos["ds"].to_list(), y=_na_oos["yhat"].to_list(), name="Forecast · OOS",
                        mode="lines", line=dict(width=3.4, color=_oos_forecast_line),
                    ))
                if _na_fc.height:
                    _na_fig.add_trace(go.Scatter(
                        x=_na_fc["ds"].to_list(), y=_na_fc["yhat"].to_list(), name="Forecast-only",
                        mode="lines", line=dict(width=3.2, dash="dot", color=_forecast_only_line),
                    ))
                _na_test_start = _na_hz.get("test_start")
                _na_fc_start = _na_hz.get("forecast_start")
                if _na_test_start is not None:
                    _na_fig.add_vline(
                        x=_na_test_start, line_width=1.5, line_dash="dash",
                        annotation_text="Inicio OOS", annotation_position="top left",
                    )
                if _na_fc_start is not None:
                    _na_fig.add_vline(
                        x=_na_fc_start, line_width=1.5, line_dash="dot",
                        annotation_text="Inicio forecast-only", annotation_position="top right",
                    )
                _na_fig.update_layout(title=f"No activo · {_na_uid}", hovermode="x unified", xaxis_title="Fecha", yaxis_title=view.unidad, margin=dict(l=40,r=40,t=50,b=40))
                st.plotly_chart(_na_fig, width="stretch")
        else:
            st.caption("No hay hojas no activas para esta sección/unidad.")

with st.expander("Ver datos detallados"):
    st.caption(f"Agregación **{view.freq}** · Unidad **{view.unidad}**")
    st.dataframe(
        _ranking_styler(view.detail),
        width="stretch",
        hide_index=True,
        height=_RANK_HEIGHT,
    )
