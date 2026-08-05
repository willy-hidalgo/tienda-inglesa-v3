"""
Forecast Explorer – Streamlit
=============================
Carga optimizada: cache de parquet, mapas de etiquetas O(1),
proyección de columnas, rankings paginados.
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

st.set_page_config(page_title="Forecast Explorer", layout="wide")
st.title("📈 Forecast Explorer · Secciones 1 & 23")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings

PAGE_SIZE = 5

# Columnas mínimas necesarias en el dashboard
_DASHBOARD_COLS = [
    "unique_id",
    "ds",
    "y",
    "yhat",
    "price",
    "pricehat",
    "period_type",
    "sku_desc",
    "store_name",
    "seccion",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "forecast_start",
    "forecast_end",
]


# ─────────────────────────────────────────────────────────────────────────────
# Carga cacheada
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando forecasts…", ttl=3600)
def load_forecast_parquet(path_str: str, mtime: float) -> pl.DataFrame:
    """Lee solo columnas usadas; mtime invalida la cache si el archivo cambia."""
    path = Path(path_str)
    schema = pl.scan_parquet(path).collect_schema()
    cols = [c for c in _DASHBOARD_COLS if c in schema.names()]
    df = (
        pl.scan_parquet(path)
        .select(cols)
        .with_columns(pl.col("ds").cast(pl.Date))
        .collect()
    )
    return df


@st.cache_data(show_spinner=False)
def load_forecast_bytes(data: bytes, name: str) -> pl.DataFrame:
    """Carga desde upload (CSV o Parquet) en memoria."""
    import io

    if name.endswith(".csv"):
        df = pl.read_csv(io.BytesIO(data), try_parse_dates=True)
    else:
        df = pl.read_parquet(io.BytesIO(data))
    if "ds" in df.columns and df.schema.get("ds") not in (pl.Date, pl.Datetime):
        df = df.with_columns(pl.col("ds").str.to_date(strict=False))
    elif "ds" in df.columns:
        df = df.with_columns(pl.col("ds").cast(pl.Date))
    return df


@st.cache_data(show_spinner=False)
def build_label_maps(
    unique_ids: tuple[str, ...],
    sku_descs: tuple[str, ...],
    store_names: tuple[str, ...],
) -> dict[str, str]:
    """Mapa unique_id → etiqueta visible (una sola pasada)."""
    out: dict[str, str] = {}
    for uid, desc, sname in zip(unique_ids, sku_descs, store_names):
        out[uid] = settings.display_label(
            uid,
            desc or None,
            sname or None,
        )
    return out


def wmape_por_id(ids: list[str], df: pl.DataFrame) -> pl.DataFrame:
    """WMAPE + n_points sobre ids candidatos; excluye forecast_only si existe."""
    if not ids:
        return pl.DataFrame(
            schema={
                "unique_id": pl.Utf8,
                "wmape": pl.Float64,
                "sum_y": pl.Float64,
                "n_points": pl.UInt32,
            }
        )
    scored = df.filter(pl.col("unique_id").is_in(ids))
    if "period_type" in scored.columns:
        scored = scored.filter(pl.col("period_type") != "forecast_only")
    # solo filas con actuals
    scored = scored.filter(pl.col("y").is_not_null() & (pl.col("y") != 0))
    return (
        scored.group_by("unique_id")
        .agg(
            pl.col("y").sum().alias("sum_y"),
            (pl.col("y") - pl.col("yhat")).abs().sum().alias("sum_abs_error"),
            pl.len().alias("n_points"),
        )
        .filter(pl.col("sum_y") != 0)
        .with_columns((pl.col("sum_abs_error") / pl.col("sum_y")).alias("wmape"))
        .select(["unique_id", "wmape", "sum_y", "n_points"])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entrada de datos
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader(
    "Cargar forecasts (CSV o Parquet)", type=["csv", "parquet"]
)

if uploaded_file is not None:
    res_df = load_forecast_bytes(uploaded_file.getvalue(), uploaded_file.name)
    st.sidebar.success(f"Cargado: {uploaded_file.name}")
else:
    default_path = Path(settings.FORECAST_PATH)
    if default_path.exists():
        mtime = default_path.stat().st_mtime
        res_df = load_forecast_parquet(str(default_path), mtime)
        st.sidebar.success(f"Cargado: {default_path.name}")
    else:
        st.info(
            "Sube un archivo CSV o Parquet con res_df, "
            "o genera `data/output/forecast.parquet` con el pipeline."
        )
        st.stop()

_cols = set(res_df.columns)
HAS_PRICE = "price" in _cols and "pricehat" in _cols
HAS_PERIOD = "period_type" in _cols
HAS_DESC = "sku_desc" in _cols
HAS_STORE_NAME = "store_name" in _cols

# Mapa de etiquetas O(1) — una sola construcción cacheada
_uids = res_df["unique_id"].unique().to_list()
_meta = res_df.select(
    [c for c in ["unique_id", "sku_desc", "store_name"] if c in res_df.columns]
).unique(subset=["unique_id"])
_uid_list = _meta["unique_id"].to_list()
_desc_list = (
    _meta["sku_desc"].to_list()
    if "sku_desc" in _meta.columns
    else [""] * len(_uid_list)
)
_sname_list = (
    _meta["store_name"].to_list()
    if "store_name" in _meta.columns
    else [""] * len(_uid_list)
)
LABEL_MAP = build_label_maps(
    tuple(_uid_list),
    tuple(d if d is not None else "" for d in _desc_list),
    tuple(s if s is not None else "" for s in _sname_list),
)
DESC_MAP = {
    uid: settings.ranking_description(
        uid,
        desc if desc else None,
        sname if sname else None,
    )
    for uid, desc, sname in zip(
        _uid_list,
        (d if d is not None else "" for d in _desc_list),
        (s if s is not None else "" for s in _sname_list),
    )
}


def label_for(uid: str) -> str:
    return LABEL_MAP.get(uid, settings.display_label(uid))


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


def prepare_unit_df(df: pl.DataFrame, unidad: str) -> pl.DataFrame:
    if unidad == "Unidades":
        return df
    if not HAS_PRICE:
        st.sidebar.warning("Sin columnas price/pricehat → se muestran Unidades.")
        return df
    return df.with_columns(
        pl.col("price").alias("y"),
        pl.col("pricehat").alias("yhat"),
    )


res_df_unit = prepare_unit_df(res_df, unidad)


# Índice por unique_id para filtros rápidos (partition una vez)
@st.cache_data(show_spinner=False)
def index_by_uid(uids: tuple[str, ...], _df_height: int) -> dict[str, int]:
    """Solo para invalidar cache; el partition real se hace abajo con el df vivo."""
    return {u: i for i, u in enumerate(uids)}


all_ids = sorted(res_df_unit["unique_id"].unique().to_list())
NOMBRES = settings.NOMBRES_NIVELES


def opciones_nivel(prefix: str | None) -> list[str]:
    if prefix is None:
        return [i for i in all_ids if "||" not in i]
    depth = prefix.count("||") + 1
    pref = prefix + "||"
    return [i for i in all_ids if i.startswith(pref) and i.count("||") == depth]


st.sidebar.markdown("### Selección de nivel")

if st.sidebar.button("🔄 Reiniciar niveles"):
    for k in list(st.session_state.keys()):
        if k.startswith("nivel_") or k.startswith("sel_") or k.startswith("page_"):
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

opciones_raiz = opciones_nivel(None)
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
    hijos = opciones_nivel(selected_id)
    if not hijos:
        break
    nombre_nivel = NOMBRES[nivel] if nivel < len(NOMBRES) else f"Nivel {nivel}"
    niveles_recorridos.append((nombre_nivel, hijos))
    etiqueta = label_for(selected_id)
    opcion_mantener = f"— Quedarse en «{etiqueta}» —"
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

# Filtro por unique_id: expresión polars (rápido, columnar)
df_daily = res_df_unit.filter(pl.col("unique_id") == selected_id).sort("ds")
st.subheader(f"NIVEL: **{label_for(selected_id)}**")

seccion_actual = selected_id.split("||")[0]


def _meta_date(col: str, fallback):
    if col in _cols and df_daily.height:
        vals = df_daily[col].drop_nulls()
        if vals.len():
            v = vals[0]
            return v.date() if isinstance(v, dt.datetime) else v
    return fallback


first_d = df_daily["ds"].min() if df_daily.height else settings.FECHAS_TRAIN[0]
if isinstance(first_d, dt.datetime):
    first_d = first_d.date()
hz = settings.section_horizons(seccion_actual, first_d)
TRAIN_END = _meta_date("train_end", hz["train_end"])
TEST_START_M = _meta_date("test_start", hz["test_start"])
TEST_END_M = _meta_date("test_end", hz["test_end"])
FORECAST_START_M = _meta_date(
    "forecast_start",
    hz.get("forecast_start", hz["test_end"] + dt.timedelta(days=1)),
)
if isinstance(FORECAST_START_M, dt.datetime):
    FORECAST_START_M = FORECAST_START_M.date()
if FORECAST_START_M is None or (
    isinstance(TEST_END_M, dt.date) and FORECAST_START_M <= TEST_END_M
):
    FORECAST_START_M = TEST_END_M + dt.timedelta(days=1)
FORECAST_END_M = _meta_date("forecast_end", hz["forecast_end"])
if isinstance(FORECAST_END_M, dt.datetime):
    FORECAST_END_M = FORECAST_END_M.date()


# ─────────────────────────────────────────────────────────────────────────────
# Agregación temporal
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_df(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    if freq == "Diario" or df.height == 0:
        return df.with_columns((pl.col("y") - pl.col("yhat")).abs().alias("abs_error"))
    period = "1w" if freq == "Semanal" else "1mo"
    aggs = [pl.col("y").sum(), pl.col("yhat").sum()]
    if "period_type" in df.columns:
        aggs.append(pl.col("period_type").max())
    return (
        df.with_columns(pl.col("ds").dt.truncate(period).alias("ds"))
        .group_by("ds")
        .agg(aggs)
        .sort("ds")
        .with_columns((pl.col("y") - pl.col("yhat")).abs().alias("abs_error"))
    )


df_view = aggregate_df(df_daily, freq)


# ─────────────────────────────────────────────────────────────────────────────
# Rankings paginados (cache de wMAPE sobre el frame completo de unidad)
# ─────────────────────────────────────────────────────────────────────────────
def fmt_number(x: float, is_pct: bool = False) -> str:
    if is_pct:
        x = x * 100
    return f"{x:,.2f}"


nombre_nivel_actual, candidatos = niveles_recorridos[-1]

tabla_base = wmape_por_id(candidatos, res_df_unit)


def _desc_for(uid: str) -> str:
    return DESC_MAP.get(uid, "")


def render_paginated_table(
    tabla: pl.DataFrame,
    value_col: str,
    key_prefix: str,
    selected_id: str,
) -> None:
    """
    Columnas: Código | Descripción | métrica | N puntos.
    """
    n = tabla.height
    if n == 0:
        st.caption("Sin datos suficientes.")
        return

    n_pages = max(1, math.ceil(n / PAGE_SIZE))
    page_key = f"page_{key_prefix}_{selected_id}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    col_prev, col_info, col_next, col_sel = st.columns([1, 2, 1, 2])
    with col_prev:
        if st.button(
            "◀ Anterior",
            key=f"prev_{page_key}",
            disabled=st.session_state[page_key] <= 1,
        ):
            st.session_state[page_key] = max(1, st.session_state[page_key] - 1)
            st.rerun()
    with col_info:
        st.caption(f"Página **{st.session_state[page_key]}** / {n_pages} · {n} filas")
    with col_next:
        if st.button(
            "Siguiente ▶",
            key=f"next_{page_key}",
            disabled=st.session_state[page_key] >= n_pages,
        ):
            st.session_state[page_key] = min(n_pages, st.session_state[page_key] + 1)
            st.rerun()
    with col_sel:
        new_page = st.number_input(
            "Ir a página",
            min_value=1,
            max_value=n_pages,
            value=st.session_state[page_key],
            key=f"selpage_{page_key}",
        )
        if new_page != st.session_state[page_key]:
            st.session_state[page_key] = int(new_page)
            st.rerun()

    page = st.session_state[page_key]
    start = (page - 1) * PAGE_SIZE
    page_df = tabla.slice(start, PAGE_SIZE)

    uids = page_df["unique_id"].to_list()
    codes = [settings.ranking_code(u) for u in uids]
    descs = [_desc_for(u) for u in uids]
    n_pts = (
        page_df["n_points"].to_list()
        if "n_points" in page_df.columns
        else [0] * len(uids)
    )
    metrics = page_df[value_col].to_list()

    display = pl.DataFrame(
        {
            "Código": codes,
            "Descripción": descs,
            value_col: metrics,
            "N puntos": n_pts,
            "unique_id": uids,
        }
    )

    event = st.dataframe(
        display.select(["Código", "Descripción", value_col, "N puntos"]),
        width="content",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"tabla_{key_prefix}_{selected_id}_{page}",
        column_config={
            "Código": st.column_config.TextColumn("Código", width="medium"),
            "Descripción": st.column_config.TextColumn("Descripción", width="large"),
            value_col: st.column_config.TextColumn(value_col, width="small"),
            "N puntos": st.column_config.NumberColumn("N puntos", width="small"),
        },
    )
    if event and event.selection and event.selection.rows:
        row_idx = event.selection.rows[0]
        chosen_uid = display["unique_id"][row_idx]
        st.session_state["sel_from_table"] = chosen_uid
        st.rerun()


tab_wmape, tab_rot = st.tabs(["📉 Mejor wMAPE", "🔄 Mayor rotación"])

with tab_wmape:
    st.markdown("#### Ranking wMAPE Total")
    st.caption(
        f"Nivel: **{nombre_nivel_actual}** · Unidad: **{unidad}** · "
        "Todas las filas (5 por página)."
    )
    tabla = (
        tabla_base.filter((pl.col("wmape") != 0) & (pl.col("unique_id") != selected_id))
        .sort("wmape")
        .with_columns(
            pl.col("wmape")
            .map_elements(lambda x: fmt_number(x, is_pct=True), return_dtype=pl.Utf8)
            .alias("wMAPE (%)")
        )
    )
    render_paginated_table(tabla, "wMAPE (%)", "wmape", selected_id)

with tab_rot:
    st.markdown("#### Ranking por rotación")
    st.caption(f"Nivel: **{nombre_nivel_actual}** · Unidad: **{unidad}**")
    label_rot = "Rotación ($)" if unidad.startswith("Valor") else "Rotación (unid.)"
    tabla_r = (
        tabla_base.filter(pl.col("unique_id") != selected_id)
        .sort("sum_y", descending=True)
        .with_columns(
            pl.col("sum_y")
            .map_elements(lambda x: fmt_number(x), return_dtype=pl.Utf8)
            .alias(label_rot)
        )
    )
    render_paginated_table(tabla_r, label_rot, "rot", selected_id)

# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────
min_ds_raw = df_daily["ds"].min() if df_daily.height else None
max_ds_raw = df_daily["ds"].max() if df_daily.height else None


def to_date(d) -> dt.date | None:
    if d is None:
        return None
    if isinstance(d, dt.datetime):
        return d.date()
    return d


min_ds = to_date(min_ds_raw)
max_ds = to_date(max_ds_raw)
cut_ds = TRAIN_END
if min_ds is not None and max_ds is not None and not (min_ds <= cut_ds <= max_ds):
    cut_ds = min_ds or cut_ds

cutoff_date = st.sidebar.date_input(
    "Fecha de corte (inicio out-sample)",
    value=cut_ds,
    min_value=min_ds or cut_ds,
    max_value=max_ds or cut_ds,
    key=f"cutoff__{selected_id}",
)

df_in = df_view.filter(pl.col("ds").cast(pl.Date) < cutoff_date)
df_out = df_view.filter(
    (pl.col("ds").cast(pl.Date) >= cutoff_date)
    & (pl.col("ds").cast(pl.Date) <= TEST_END_M)
    & (pl.col("y").is_not_null())
    & (pl.col("y") != 0)
)


def calcular_metricas(df: pl.DataFrame) -> tuple[float, float]:
    """
    WMAPE = Σ|y−ŷ| / Σ|y|  (excluye y==0).
    BIAS  = Σ(ŷ−y) / Σ y   (excluye y==0).
    """
    if df.height == 0:
        return 0.0, 0.0
    scored = df.filter(pl.col("y").is_not_null() & (pl.col("y") != 0))
    if scored.height == 0:
        return 0.0, 0.0
    if "abs_error" not in scored.columns:
        scored = scored.with_columns(
            (pl.col("y") - pl.col("yhat")).abs().alias("abs_error")
        )
    sum_y = float(scored["y"].sum())
    if sum_y == 0:
        return 0.0, 0.0
    wmape = float(scored["abs_error"].sum()) / abs(sum_y)
    bias = float((scored["yhat"] - scored["y"]).sum()) / sum_y
    return wmape, bias


wmape_in, bias_in = calcular_metricas(df_in)
wmape_out, bias_out = calcular_metricas(df_out)
wmape_total, bias_total = calcular_metricas(
    df_view.filter(
        (pl.col("ds").cast(pl.Date) <= TEST_END_M)
        & (pl.col("y").is_not_null())
        & (pl.col("y") != 0)
    )
)

st.markdown("#### Métricas")
st.caption(
    f"Agregación: **{freq}** · Unidad: **{unidad}** · Sección **{seccion_actual}** · "
    f"Train → {TRAIN_END} · OOS [{TEST_START_M} → {TEST_END_M}] · "
    f"Solo-forecast [{FORECAST_START_M} → {FORECAST_END_M}]."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**In-sample**")
    st.metric("wMAPE", f"{wmape_in:.2%}")
    st.metric("BIAS", f"{bias_in:+.2%}")
    st.caption(f"{df_in.height} períodos")
with col2:
    st.markdown("**Out-sample**")
    st.metric("wMAPE", f"{wmape_out:.2%}")
    st.metric("BIAS", f"{bias_out:+.2%}")
    st.caption(f"{df_out.height} períodos")
with col3:
    st.markdown("**Total (hasta test_end)**")
    st.metric("wMAPE", f"{wmape_total:.2%}")
    st.metric("BIAS", f"{bias_total:+.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico
# ─────────────────────────────────────────────────────────────────────────────
unidad_label = "Valor ($)" if unidad.startswith("Valor") else "Unidades"
freq_label = {"Diario": "día", "Semanal": "semana", "Mensual": "mes"}[freq]

if HAS_PERIOD:
    df_hist = df_view.filter(pl.col("period_type") != "forecast_only")
    df_fcst_only = df_view.filter(pl.col("period_type") == "forecast_only")
else:
    df_hist = df_view.filter(
        (pl.col("ds").cast(pl.Date) <= TEST_END_M)
        & (pl.col("y").is_not_null())
        & (pl.col("y") != 0)
    )
    df_fcst_only = df_view.filter(
        (pl.col("ds").cast(pl.Date) >= FORECAST_START_M)
        & (pl.col("ds").cast(pl.Date) <= FORECAST_END_M)
    )

fig = go.Figure()
if df_hist.height:
    fig.add_trace(
        go.Scatter(
            x=df_hist["ds"].to_list(),
            y=df_hist["y"].to_list(),
            name="y (real)",
            fill="tozeroy",
            mode="lines",
            line=dict(color="rgba(31, 119, 180, 1)"),
            fillcolor="rgba(31, 119, 180, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["ds"].to_list(),
            y=df_hist["yhat"].to_list(),
            name="yhat (predicción)",
            fill="tozeroy",
            mode="lines",
            line=dict(color="rgba(255, 127, 14, 1)"),
            fillcolor="rgba(255, 127, 14, 0.25)",
        )
    )
if df_fcst_only.height:
    fig.add_trace(
        go.Scatter(
            x=df_fcst_only["ds"].to_list(),
            y=df_fcst_only["yhat"].to_list(),
            name="yhat (solo forecast)",
            mode="lines",
            line=dict(color="rgba(255, 127, 14, 1)", dash="dot", width=2),
        )
    )

cutoff_x = (
    dt.datetime.combine(cutoff_date, dt.time.min)
    if isinstance(cutoff_date, dt.date) and not isinstance(cutoff_date, dt.datetime)
    else cutoff_date
)
fig.add_vline(
    x=cutoff_x,
    line_dash="dash",
    line_color="rgba(220, 50, 50, 0.8)",
    annotation_text="Corte train/OOS",
    annotation_position="top left",
)
fig.add_vline(
    x=dt.datetime.combine(TEST_END_M, dt.time.min),
    line_dash="dot",
    line_color="rgba(100, 100, 100, 0.6)",
    annotation_text="Fin OOS",
    annotation_position="top right",
)
fig.update_layout(
    title=f"y vs yhat — {label_for(selected_id)} ({freq.lower()}, {unidad_label})",
    xaxis_title="Fecha",
    yaxis_title=f"{unidad_label} / {freq_label}",
    legend_title_text="",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=50, b=40),
)
st.plotly_chart(fig, width="stretch")

with st.expander("Ver datos detallados"):
    st.caption(f"Datos en agregación **{freq}** y unidad **{unidad}**")
    st.dataframe(df_view, width="content")
