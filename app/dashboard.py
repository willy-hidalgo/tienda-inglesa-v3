"""
Forecast Explorer – Streamlit (solo visualización)
==================================================
Cálculos: app.backend. El pipeline precalcula los parquet una sola vez; el
dashboard construye un contexto pre-agregado por (archivo, unidad) —
`backend.build_dashboard_context` — que se reutiliza en cada interacción.
Cualquier cambio de selección solo recorre subconjuntos (rankings = tabla
pre-agregada; gráfico/métricas = serie elegida) → respuesta inmediata y
correcta, sin re-procesar el panel completo.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import plotly.graph_objects as go
import polars as pl
import streamlit as st

st.set_page_config(page_title="Forecast Explorer", layout="wide")
st.title("📈 Forecast Explorer · Secciones 1 & 23")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings

try:
    from app import backend
except ImportError:  # pragma: no cover
    import backend  # type: ignore

_RANK_HEIGHT = 35 + 5 * 35  # ~5 filas visibles

# ─────────────────────────────────────────────────────────────────────────────
# Carga / contexto (pre-agregado UNA vez por archivo + unidad + CACHE EN DISCO)
# ─────────────────────────────────────────────────────────────────────────────
# El contexto completo se arma en el PRIMER render y se persiste a disco por
# (origen, tamaño, mtime, unidad); los arranques siguientes lo leen de cache
# en vez de re-parsear el parquet y re-escandear el panel (arranque rápido).
# En sesión se mantiene en st.session_state con clave escalar para que cada
# interacción sea O(subconjunto) sobre el contexto ya armado.
_CACHE_DIR = Path(settings.ROOT) / "data" / ".dashcache"


def _cache_key(source: tuple) -> str:
    raw = "|".join(str(x) for x in source)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _save_context(key: str, ctx: backend.DashboardContext) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ctx.unit_df.write_parquet(_CACHE_DIR / f"{key}.unit.parquet", compression="zstd")
    ctx.per_id.write_parquet(_CACHE_DIR / f"{key}.per.parquet", compression="zstd")
    pl.DataFrame(
        {
            "unique_id": list(ctx.label_map.keys()),
            "label": list(ctx.label_map.values()),
            "desc": [ctx.desc_map.get(u, "") for u in ctx.label_map],
        }
    ).write_parquet(_CACHE_DIR / f"{key}.labels.parquet", compression="zstd")
    (_CACHE_DIR / f"{key}.meta.json").write_text(
        json.dumps(
            {
                "all_ids": ctx.all_ids,
                "has_rolling": ctx.has_rolling,
                "has_value_cols": ctx.has_value_cols,
                "has_effects": ctx.has_effects,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load_cached(key: str) -> backend.DashboardContext | None:
    try:
        files = (
            _CACHE_DIR / f"{key}.unit.parquet",
            _CACHE_DIR / f"{key}.per.parquet",
            _CACHE_DIR / f"{key}.labels.parquet",
            _CACHE_DIR / f"{key}.meta.json",
        )
        if not all(p.exists() for p in files):
            return None
        unit_df = pl.read_parquet(files[0])
        per_id = pl.read_parquet(files[1])
        lbl = pl.read_parquet(files[2])
        label_map = dict(zip(lbl["unique_id"].to_list(), lbl["label"].to_list()))
        desc_map = dict(zip(lbl["unique_id"].to_list(), lbl["desc"].to_list()))
        meta = json.loads(files[3].read_text(encoding="utf-8"))
        return backend.DashboardContext(
            unit_df=unit_df,
            label_map=label_map,
            desc_map=desc_map,
            per_id=per_id,
            all_ids=meta["all_ids"],
            has_rolling=meta["has_rolling"],
            has_value_cols=meta["has_value_cols"],
            has_effects=meta.get("has_effects", False),
        )
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("No se pudo cargar el contexto cacheado: %s", exc)
        return None


def _prune_cache(keep: int = 10) -> None:
    """Mantiene acotado el dir de cache (descarta los keys más viejos)."""
    try:
        metas = sorted(_CACHE_DIR.glob("*.meta.json"), key=lambda p: p.stat().st_mtime)
        for m in metas[:-keep]:
            key = m.name[: -len(".meta.json")]
            for suffix in (
                ".meta.json",
                ".unit.parquet",
                ".per.parquet",
                ".labels.parquet",
            ):
                (_CACHE_DIR / f"{key}{suffix}").unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("No se pudo podar el cache: %s", exc)


def _load_context(uploaded_file, default_path, unidad):
    """Devuelve (ctx, source_key). Arma el contexto solo si no está cacheado
    (en disco o en sesión) o si cambió el archivo / la unidad."""
    ss = st.session_state
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        source = ("upload", uploaded_file.name, len(data), unidad)
    else:
        source = ("file", str(default_path), default_path.stat().st_mtime, unidad)

    if ss.get("ctx_key") != source:
        key = _cache_key(source)
        ctx = _load_cached(key)
        if ctx is None:
            with st.spinner("Preparando datos (primera vez)…"):
                if source[0] == "upload":
                    res_df = backend.load_forecast_bytes(
                        uploaded_file.getvalue(), uploaded_file.name
                    )
                else:
                    res_df = backend.load_forecast_parquet(default_path)
                ctx = backend.build_dashboard_context(res_df, unidad)
                _save_context(key, ctx)
                _prune_cache()
        ss["ctx"] = ctx
        ss["ctx_key"] = source
    return ss["ctx"], source


uploaded = st.sidebar.file_uploader(
    "Cargar forecasts (CSV o Parquet)", type=["csv", "parquet"]
)
default_path = Path(settings.FORECAST_PATH)
if uploaded is not None:
    st.sidebar.success(f"Cargado: {uploaded.name}")
elif default_path.exists():
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

# Contexto pre-agregado (depende de archivo + unidad) — se cachea una sola vez.
ctx, _source = _load_context(uploaded, default_path, unidad)
has_rolling = ctx.has_rolling

# Etiquetas según unidad (mapeo interno siempre yhat / yhat28)
_is_valor = unidad.startswith("Valor")
_opt_yhat = "valuehat" if _is_valor else "yhat"
_opt_yhat28 = "valuehat28" if _is_valor else "yhat28"

# El selector de series de forecast SOLO existe si el rolling 28d fue
# calculado (COMPUTE_ROLLING28=True en el pipeline). Si no, solo yhat.
if has_rolling:
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
    series_forecast = [_opt_yhat]
    show_yhat, show_yhat28 = True, False

all_ids = ctx.all_ids
label_map, desc_map = ctx.label_map, ctx.desc_map
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

# Corte in/out = train_end de la sección: lo resuelve backend dentro de
# prepare_dashboard_state_from_context (sin duplicar filtros del sidebar).
cutoff_date = None

# ─────────────────────────────────────────────────────────────────────────────
# ViewModel (solo subconjuntos sobre el contexto pre-agregado → rápido).
# ─────────────────────────────────────────────────────────────────────────────
ss = st.session_state
view_key = (
    "view",
    *_source,
    freq,
    selected_id,
    tuple(candidatos or []),
    nombre_nivel_actual,
)
if ss.get("view_key") != view_key:
    ss["view"] = backend.prepare_dashboard_state_from_context(
        ctx,
        unidad=unidad,
        freq=freq,
        selected_id=selected_id,
        cutoff_date=cutoff_date,
        candidatos=candidatos,
        nombre_nivel=nombre_nivel_actual,
    )
    ss["view_key"] = view_key
view = ss["view"]

st.subheader(f"NIVEL: **{view.label}**")
if view.store_id:
    store_lbl = (
        f"{view.store_id} — {view.store_name}" if view.store_name else view.store_id
    )
    st.caption(
        f"🏬 Tienda: **{store_lbl}** · los SKUs listados pertenecen a esta tienda."
    )


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
            "wMAPE (%)": st.column_config.TextColumn("wMAPE (%)"),
            "Rotación": st.column_config.TextColumn("Rotación"),
            "N puntos ≠0": st.column_config.NumberColumn("N puntos ≠0"),
            "% ≠0": st.column_config.TextColumn("% ≠0"),
        },
    )
    if event and event.selection and event.selection.rows:
        st.session_state["sel_from_table"] = display["unique_id"][
            event.selection.rows[0]
        ]
        st.rerun()


st.markdown("#### Ranking wMAPE + Rotación")
_seg = selected_id.split("||")
_crumbs = [f"Sección {_seg[0]}"]
_pref = _seg[0]
for _i in range(1, len(_seg)):
    _pref += f"||{_seg[_i]}"
    _crumbs.append(label_map.get(_pref, _seg[_i]))
st.caption("🧭 **Ruta:** " + " › ".join(_crumbs))
st.caption(
    f"Nivel actual: **{view.nombre_nivel}** · Unidad: **{view.unidad}** · "
    f"Se muestran solo los **{view.ranking_n_series} hijos directos** del nivel "
    f"seleccionado (sin series de otro nivel). ~5 filas visibles, scroll."
)
st.caption(
    "🏷️ **Total de puntos (spine):** "
    f"**{view.ranking_total_points}** · "
    "`N puntos ≠0` = períodos con venta (y≠0) · `% ≠0` = proporción sobre el total."
)
_show_ranking(view.ranking_wmape, f"tabla_wmape_{view.selected_id}")

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

if has_rolling:
    st.markdown("#### Métricas Rolling 28d (`yhat28`)")
    st.caption(
        "Walk-forward por bloques de 28 días. Priors = modelo final (opción B). "
        "WMAPE₂₈ = Σ|y−ŷ₂₈|/Σ|y| (excl. y=0)."
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
                mode="lines",
                line=dict(color="black", width=2),
            )
        )
    if ch["fcst_ds"] and show_yhat:
        fig.add_trace(
            go.Scatter(
                x=ch["fcst_ds"],
                y=ch["fcst_yhat"],
                name=f"{_opt_yhat} (solo forecast)",
                mode="lines",
                line=dict(color="black", dash="dot", width=2),
            )
        )

if show_yhat28 and ch.get("hist_yhat28"):
    fig.add_trace(
        go.Scatter(
            x=ch["hist_ds"],
            y=ch["hist_yhat28"],
            name=f"{_opt_yhat28} (rolling 28d)",
            mode="lines",
            line=dict(color="rgba(44, 160, 44, 1)", width=2),
        )
    )
if show_yhat28 and ch.get("fcst_yhat28"):
    fig.add_trace(
        go.Scatter(
            x=ch["fcst_ds"],
            y=ch["fcst_yhat28"],
            name=f"{_opt_yhat28} (forecast)",
            mode="lines",
            line=dict(color="rgba(44, 160, 44, 0.8)", dash="dot", width=2),
        )
    )

# ── Descomposición de efectos causales (solo nivel sección) ──────────────
show_effects = False
selected_effects: list[str] = []
if view.has_effects and ch.get("effects"):
    eff_opts = [c for c in backend.EFFECT_COLUMNS if c in ch["effects"]]
    if eff_opts:
        selected_effects = st.sidebar.multiselect(
            "Descomposición de efectos (sección)",
            options=eff_opts,
            default=[c for c in eff_opts if c not in ("ef_total", "ef_resid")],
            key="effects_visible",
            help=(
                "Áreas apiladas de contribuciones causales diarias en "
                "log₁ₚ(ŷ, unidades): Σ factores = ln(1+ŷ) sección. Eje derecho."
            ),
        )
        show_effects = bool(selected_effects)


def _to_rgb(hexcolor: str) -> str:
    h = hexcolor.lstrip("#")
    return ", ".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))


if show_effects and selected_effects:
    # Paleta estilo imagen de referencia (áreas apiladas):
    #   Level → gris, Trend → gris oscuro, Seasonality → amarillo,
    #   EDP → verde claro, Discount → púrpura, Feature/Display → rosa,
    #   Volume → gris muy oscuro, Price → rojo.
    _EFFECT_PALETTE = {
        "ef_level": "#7f8c8d",
        "ef_trend": "#5d6d7e",
        "ef_seasonality": "#f7dc6f",
        "ef_edp": "#82e0aa",
        "ef_discount": "#af7ac5",
        "ef_feature_display": "#f1948a",
        "ef_volume": "#2c3e50",
        "ef_price": "#e74c3c",
        "ef_total": "#7f7f7f",
        "ef_resid": "#7f7f7f",
    }
    for e in selected_effects:
        lbl = backend.EFFECT_LABELS.get(e, e)
        color = _EFFECT_PALETTE.get(e, "#7f7f7f")
        is_total = e == "ef_total"
        fillcolor = (
            f"rgba({_to_rgb(color)}, 0.35)" if not is_total else "rgba(0, 0, 0, 0)"
        )
        hist_y = ch["effects"][e]["hist"] or []
        fcst_y = ch["effects"][e]["fcst"] or []
        xs: list = []
        ys: list = []
        if hist_y:
            xs += list(ch["hist_ds"])
            ys += list(hist_y)
        if fcst_y:
            xs += list(ch["fcst_ds"])
            ys += list(fcst_y)
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=lbl,
                mode="lines",
                line=dict(color=color, width=0.6),
                fill="tonexty",
                fillcolor=fillcolor,
                stackgroup="ef_stack",
                yaxis="y2",
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
_layout = dict(
    title=f"y vs yhat — {view.label} ({view.freq.lower()}, {unidad_label})",
    xaxis_title="Fecha",
    yaxis=dict(title=f"{unidad_label} / {freq_label}"),
    legend_title_text="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    margin=dict(l=40, r=40, t=80, b=40),
)
if show_effects:
    # Anclar el eje derecho a 0 (parte inferior) usando como techo el
    # ln(1+ŷ) total de la sección. Así las áreas apiladas colorean SOLO la
    # fracción inferior (Σ efectos seleccionados) y dejan sin color la parte
    # superior, en lugar de colorear todo el fondo al hacer auto-range sobre
    # los valores log1p (~14-16).
    _y2 = dict(
        title="ln(1+ŷ) contrib. causal",
        overlaying="y",
        side="right",
        showgrid=False,
    )
    _eftot = ch["effects"].get("ef_total", {})
    _cand = [
        float(v)
        for v in (_eftot.get("hist", []) + _eftot.get("fcst", []))
        if v is not None and v == v
    ]
    if _cand:
        _y2["range"] = [0, max(_cand) * 1.03]
    _layout["yaxis2"] = _y2
fig.update_layout(**_layout)
st.caption("Mostrando forecast: **" + ", ".join(series_forecast) + "**")
if show_effects:
    st.caption(
        "⚗️ Efectos causales (eje derecho, log₁ₚ unidades): "
        "ln(1+ŷ sección) = Σ(level, trend, seasonality, edp, discount, "
        "feature/display, volume, price) + residuo."
    )
st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Detalle
# ─────────────────────────────────────────────────────────────────────────────
_DETAIL_LABELS = {
    "ds": "Fecha",
    "y": "Real",
    "yhat": "Pronóstico",
    "yhat28": "Rolling 28d",
    "value": "Valor real",
    "valuehat": "Valor pron.",
    "valuehat28": "Valor rolling",
    "period_type": "Período",
    "unique_id": "ID",
    "sku_desc": "SKU",
    "store_name": "Tienda",
    "seccion": "Sección",
    "abs_error": "Error abs.",
    "ef_level": "Efecto nivel",
    "ef_trend": "Efecto tendencia",
    "ef_seasonality": "Efecto estacionalidad",
    "ef_edp": "Efecto EDP",
    "ef_discount": "Efecto descuento",
    "ef_feature_display": "Efecto feature/display",
    "ef_volume": "Efecto volumen",
    "ef_price": "Efecto precio",
    "ef_total": "ln(1+ŷ) total",
    "ef_resid": "Residuo (corrección)",
}

with st.expander("Ver datos detallados"):
    st.caption(
        f"Agregación **{view.freq}** · Unidad **{view.unidad}** · "
        f"Solo este elemento de la jerarquía: **{view.label}**"
    )
    detail = view.detail.rename(
        {c: _DETAIL_LABELS[c] for c in view.detail.columns if c in _DETAIL_LABELS}
    )
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
        height=_RANK_HEIGHT,
    )
