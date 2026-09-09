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
        prepare_dashboard_state,
        prepare_dashboard_state_fast,
    )
    from app.dashboard_diagnostics import add_leaf_edp
    from app.discrepancy import discrepancy_points
except ImportError:  # pragma: no cover
    import backend  # type: ignore
    import dashboard_artifacts as artifacts  # type: ignore
    from dashboard_data import (  # type: ignore
        global_leaf_ranking_from_metrics,
        prepare_dashboard_state,
        prepare_dashboard_state_fast,
    )
    from dashboard_diagnostics import add_leaf_edp  # type: ignore
    from discrepancy import discrepancy_points  # type: ignore

_RANK_HEIGHT = 35 + 5 * 35  # ~5 filas visibles
_SENTINEL_TIENDA = "— Todas las tiendas —"
_SENTINEL_SKU = "— Todos los SKU —"


def _ranking_styler(df: pl.DataFrame, *, yellow: bool = False) -> pl.DataFrame:
    """Return Polars directly; Streamlit applies display formats via column_config.

    Keeping the grid in Polars preserves a single tabular engine and keeps the
    dashboard aligned with the project architecture. ``yellow`` is
    retained only for call-site compatibility; it does not alter the data.
    """
    return df

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
                if df.columns[j] in {"wMAPE (%)", "BIAS (%)", "wMAPE incl. y=0 (%)", "BIAS incl. y=0 (%)", "% ≠0"}:
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
            "wMAPE (%)": 12, "BIAS (%)": 12, "wMAPE incl. y=0 (%)": 18, "BIAS incl. y=0 (%)": 18, "Rotación": 14, "N puntos": 10,
            "Días con venta": 14, "% ≠0": 10, "Cohort": 14,
            "Volumen OOS": 16, "Pronóstico OOS": 16, "Error abs. OOS": 16, "unique_id": 30,
        }
        for j, c in enumerate(df.columns):
            ws.set_column(j, j, widths.get(c, 14))
    return buf.getvalue()





@st.cache_data(show_spinner=False, max_entries=8)
def _load_leaf_audit_source(mtime_key: float, forecast_path: str, unique_id: str, unidad: str) -> pl.DataFrame:
    """Carga bajo demanda SOLO una hoja y columnas de trazabilidad.

    No forma parte del hot path del dashboard: se invoca únicamente al pulsar
    "Preparar auditoría Excel". Predicate + projection pushdown evitan cargar el
    forecast completo en memoria y permiten mantener livianos los artefactos del dashboard.
    """
    path = Path(forecast_path)
    scan = pl.scan_parquet(path)
    names = set(scan.collect_schema().names())
    audit_cols = [
        "unique_id", "ds", "period_type", "update_block_days", "seccion", "sku_desc", "store_name",
        "y", "yhat", "value", "valuehat", "yhat_raw", "valuehat_raw",
        "initial_level_y", "initial_level_value", "warmup_positive_days", "leaf_start", "leaf_warmup_end",
        "ses_level_y", "ses_level_value", "ses_alpha_y", "ses_alpha_value",
        "driver_effect", "driver_effect_value", "driver_factor_y", "driver_factor_value",
        "parent_model_y", "parent_model_value", "parent_wmape_y", "parent_wmape_value",
        "parent_driver_mode_y", "parent_driver_mode_value",
        "leaf_level_method_y", "leaf_level_method_value", "modelo_seleccionado",
        "rls_metric_eligible", "rls_block", "rls_train_days",
    ]
    cols = [c for c in audit_cols if c in names]
    df = (
        scan.filter(pl.col("unique_id") == unique_id)
        .select(cols)
        .collect(engine="streaming")
        .sort("ds")
    )
    if df.height == 0:
        return df
    prepared = backend.prepare_unit_df(df, unidad, "value" in names and "valuehat" in names)
    return add_leaf_edp(prepared)


def _leaf_audit_excel_bytes(df: pl.DataFrame, *, unidad: str, label: str) -> bytes:
    """Excel exhaustivo de trazabilidad para una hoja SKU+Tienda.

    Incluye actual, EDP observado, nivel SES, observación desestacionalizada,
    parent RLS, efecto/factor de drivers, reconstrucción algebraica del forecast,
    errores, wMAPE/BIAS y la nueva bandera de discrepancia Actual-vs-Forecast.
    Se genera bajo demanda para una sola hoja, por lo que no penaliza el hot path.
    """
    buf = io.BytesIO()
    if df is None or df.height == 0:
        return b""

    work = df.sort("ds") if "ds" in df.columns else df
    actual_col = "y"
    forecast_col = "yhat"
    is_value = unidad.startswith("Valor")
    init_col = "initial_level_value" if is_value else "initial_level_y"
    level_col = "ses_level_value" if is_value else "ses_level_y"
    alpha_col = "ses_alpha_value" if is_value else "ses_alpha_y"
    effect_col = "driver_effect_value" if is_value else "driver_effect"
    factor_col = "driver_factor_value" if is_value else "driver_factor_y"
    parent_col = "parent_model_value" if is_value else "parent_model_y"
    parent_wmape_col = "parent_wmape_value" if is_value else "parent_wmape_y"
    driver_mode_col = "parent_driver_mode_value" if is_value else "parent_driver_mode_y"
    raw_fc_col = "valuehat_raw" if is_value else "yhat_raw"

    # Overall metrics. Official support excludes y=0; all-points is diagnostic.
    def _metrics(frame: pl.DataFrame) -> tuple[float | None, float | None, float | None, float | None]:
        if frame.height == 0:
            return None, None, None, None
        def calc(f: pl.DataFrame):
            if f.height == 0:
                return None, None
            agg = f.select(
                pl.col(actual_col).abs().sum().alias("den"),
                (pl.col(forecast_col) - pl.col(actual_col)).abs().sum().alias("ae"),
                (pl.col(forecast_col) - pl.col(actual_col)).sum().alias("se"),
            ).row(0, named=True)
            den = float(agg["den"] or 0.0)
            if den <= 0:
                return None, None
            return 100.0 * float(agg["ae"] or 0.0) / den, 100.0 * float(agg["se"] or 0.0) / den
        pos = frame.filter(pl.col(actual_col).fill_null(0.0) > 0)
        wm, bias = calc(pos)
        wm_all, bias_all = calc(frame)
        return wm, bias, wm_all, bias_all

    wm, bias, wm_all, bias_all = _metrics(work)
    pos = work.filter(pl.col(actual_col).fill_null(0.0) > 0)
    n_pos = pos.height
    mean_pos = float(pos.select(pl.col(actual_col).mean()).item()) if n_pos else None
    mean_fc = float(work.select(pl.col(forecast_col).mean()).item()) if work.height else None

    # Build the audit calculation table in business order.
    driver_factor = pl.col(factor_col).cast(pl.Float64).fill_null(1.0) if factor_col in work.columns else pl.lit(1.0)
    driver_effect = pl.col(effect_col).cast(pl.Float64).fill_null(0.0) if effect_col in work.columns else pl.lit(0.0)
    ses_level = pl.col(level_col).cast(pl.Float64).fill_null(0.0) if level_col in work.columns else pl.lit(0.0)
    actual = pl.col(actual_col).cast(pl.Float64).fill_null(0.0)
    forecast = pl.col(forecast_col).cast(pl.Float64).fill_null(0.0)
    eligible = actual > 0

    daily = work.with_columns(
        pl.col("ds").dt.strftime("%A").alias("weekday") if "ds" in work.columns else pl.lit(None).alias("weekday"),
        pl.col("ds").dt.month().alias("month") if "ds" in work.columns else pl.lit(None).alias("month"),
        pl.when(eligible)
          .then((actual + 1.0).log() - driver_factor.log())
          .otherwise(None)
          .alias("_deseason_log"),
        (forecast - actual).abs().alias("abs_error"),
        (forecast - actual).alias("signed_error"),
        eligible.alias("metric_eligible_y_ne_0"),
        pl.when(eligible).then((forecast - actual).abs()).otherwise(0.0).alias("wmape_numerator"),
        pl.when(eligible).then(actual.abs()).otherwise(0.0).alias("wmape_denominator"),
        pl.when(eligible).then(forecast - actual).otherwise(0.0).alias("bias_numerator"),
        pl.when(eligible).then(actual.abs()).otherwise(0.0).alias("bias_denominator"),
        (((ses_level + 1.0).log() + driver_effect).exp() - 1.0).clip(lower_bound=0.0).alias("forecast_rebuilt_raw"),
    ).with_columns(
        pl.when(pl.col("_deseason_log").is_not_null())
          .then(pl.col("_deseason_log").exp() - 1.0)
          .otherwise(None)
          .alias("actual_deseasonalized")
    ).drop("_deseason_log")

    # Cumulative official metrics make the formula auditable row by row.
    daily = daily.with_columns(
        pl.col("wmape_numerator").cum_sum().alias("cum_wmape_num"),
        pl.col("wmape_denominator").cum_sum().alias("cum_wmape_den"),
        pl.col("bias_numerator").cum_sum().alias("cum_bias_num"),
        pl.col("bias_denominator").cum_sum().alias("cum_bias_den"),
    ).with_columns(
        pl.when(pl.col("cum_wmape_den") > 0).then(100 * pl.col("cum_wmape_num") / pl.col("cum_wmape_den")).otherwise(None).alias("wmape_cumulative_pct"),
        pl.when(pl.col("cum_bias_den") > 0).then(100 * pl.col("cum_bias_num") / pl.col("cum_bias_den")).otherwise(None).alias("bias_cumulative_pct"),
    )

    # Same discrepancy rule as the chart.
    disc = discrepancy_points(
        daily["ds"].to_list() if "ds" in daily.columns else [],
        daily[actual_col].to_list(),
        daily[forecast_col].to_list(),
    )
    if disc:
        disc_df = pl.DataFrame(disc).select(
            "ds",
            pl.lit(True).alias("forecast_discrepancy_outlier"),
            pl.col("factor_gap").alias("discrepancy_factor_gap"),
            pl.col("scaled_gap").alias("discrepancy_scaled_gap"),
        ).with_columns(pl.col("ds").cast(pl.Date))
        daily = daily.join(disc_df, on="ds", how="left")
    daily = daily.with_columns(
        pl.col("forecast_discrepancy_outlier").fill_null(False) if "forecast_discrepancy_outlier" in daily.columns else pl.lit(False).alias("forecast_discrepancy_outlier"),
        pl.col("discrepancy_factor_gap") if "discrepancy_factor_gap" in daily.columns else pl.lit(None).alias("discrepancy_factor_gap"),
        pl.col("discrepancy_scaled_gap") if "discrepancy_scaled_gap" in daily.columns else pl.lit(None).alias("discrepancy_scaled_gap"),
    )

    # Explicit, review-friendly column order. Driver values are shown as date
    # calendar context + leaf EDP; the applied model contribution is the parent
    # RLS total non-intercept effect/factor stored by production.
    ordered_exprs: list[pl.Expr] = []
    def add(src: str, alias: str | None = None):
        if src in daily.columns:
            ordered_exprs.append(pl.col(src).alias(alias or src))
    add("ds", "Fecha")
    add("period_type", "Periodo")
    add("rls_block", "Bloque RLS")
    add("rls_train_days", "Días historia RLS")
    add("weekday", "Día semana")
    add("month", "Mes")
    add(actual_col, "Actual")
    add("units_actual", "Actual unidades")
    add("sales_value_actual", "Actual valor ($)")
    add("asp_observed", "ASP observado SKU+Tienda")
    add("edp_observed", "EDP observado SKU+Tienda")
    add("discount_observed", "Descuento observado")
    add("edp_source", "Fuente EDP")
    add(init_col, "Nivel inicial mediana positiva")
    add(level_col, "Nivel SES antes de drivers")
    add("actual_deseasonalized", "Actual desestacionalizado")
    add(alpha_col, "Alpha SES")
    add(parent_col, "Parent RLS seleccionado")
    add(parent_wmape_col, "wMAPE histórico parent")
    add(driver_mode_col, "Modo aplicación drivers")
    add(effect_col, "Efecto conjunto drivers (log)")
    add(factor_col, "Factor conjunto drivers")
    add("forecast_rebuilt_raw", "Forecast reconstruido raw")
    add(raw_fc_col, "Forecast raw almacenado")
    add(forecast_col, "Forecast final")
    add("abs_error", "Error absoluto")
    add("signed_error", "Error firmado")
    add("metric_eligible_y_ne_0", "Elegible métrica oficial")
    add("wmape_numerator", "wMAPE numerador")
    add("wmape_denominator", "wMAPE denominador")
    add("bias_numerator", "BIAS numerador")
    add("bias_denominator", "BIAS denominador")
    add("wmape_cumulative_pct", "wMAPE acumulado (%)")
    add("bias_cumulative_pct", "BIAS acumulado (%)")
    add("forecast_discrepancy_outlier", "Discrepancia outlier")
    add("discrepancy_factor_gap", "Factor discrepancia")
    add("discrepancy_scaled_gap", "Gap escalado")
    detail = daily.select(ordered_exprs)

    group_cols = [c for c in ("period_type", "rls_block") if c in daily.columns]
    if group_cols:
        blocks = (
            daily.group_by(group_cols, maintain_order=True)
            .agg(
                pl.len().alias("calendar_rows"),
                eligible.sum().alias("positive_days"),
                pl.when(eligible).then(actual).otherwise(None).mean().alias("mean_actual_positive"),
                actual.sum().alias("sum_actual_all"),
                forecast.mean().alias("mean_forecast"),
                forecast.sum().alias("sum_forecast"),
                pl.col("wmape_numerator").sum().alias("abs_error_official"),
                pl.col("wmape_denominator").sum().alias("den_official"),
                pl.col("bias_numerator").sum().alias("signed_error_official"),
                pl.col("forecast_discrepancy_outlier").sum().alias("n_discrepancy_outliers"),
            )
            .with_columns(
                pl.when(pl.col("den_official") > 0).then(100 * pl.col("abs_error_official") / pl.col("den_official")).otherwise(None).alias("wmape_official_pct"),
                pl.when(pl.col("den_official") > 0).then(100 * pl.col("signed_error_official") / pl.col("den_official")).otherwise(None).alias("bias_official_pct"),
            )
        )
    else:
        blocks = pl.DataFrame()

    with xlsxwriter.Workbook(buf, {"in_memory": True}) as wb:
        title = wb.add_format({"bold": True, "font_size": 14})
        hdr = wb.add_format({"bold": True, "border": 1, "bg_color": "#E8EEF8"})
        pct = wb.add_format({"num_format": '0.00"%"'})
        num = wb.add_format({"num_format": "#,##0.00"})
        integer = wb.add_format({"num_format": "0"})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})

        ws = wb.add_worksheet("Resumen")
        ws.write(0, 0, "Auditoría SKU+Tienda", title)
        rows = [
            ("Serie", label), ("Unidad", unidad), ("Filas calendario", work.height),
            ("Días/puntos con actual > 0", n_pos), ("Media actual condicional (y>0)", mean_pos),
            ("Media forecast", mean_fc), ("wMAPE oficial (excluye y=0)", wm),
            ("BIAS oficial (excluye y=0)", bias), ("wMAPE incl. y=0", wm_all),
            ("BIAS incl. y=0", bias_all), ("Discrepancias marcadas", len(disc)),
        ]
        for i, (k, v) in enumerate(rows, start=2):
            ws.write(i, 0, k, hdr)
            fmt = pct if "wMAPE" in k or "BIAS" in k else num if isinstance(v, float) else None
            ws.write(i, 1, v, fmt)
        ws.write(15, 0, "Identidad productiva", hdr)
        ws.write(15, 1, "forecast_raw = max(exp(log(1+nivel_SES) + efecto_drivers_RLS) - 1, 0); forecast final = redondeo del raw.", wrap)
        ws.write(17, 0, "EDP", hdr)
        ws.write(17, 1, "EDP observado SKU+Tienda se calcula ex-post con actuals de la hoja. Forecast-only usa carry-forward del último EDP observado. Es diagnóstico; el modelo productivo aplica el efecto RLS del parent Sección/Tienda.", wrap)
        ws.write(19, 0, "Outlier Actual vs Forecast", hdr)
        ws.write(19, 1, "Se marca si factor de discrepancia ≥4 y error ≥15% del nivel típico de la serie, o si el error absoluto ≥3× ese nivel. La regla es simétrica y evita marcar discrepancias pequeñas solo por tener gran ratio.", wrap)
        ws.set_column(0, 0, 38); ws.set_column(1, 1, 105)

        def write_df(sheet_name: str, frame: pl.DataFrame):
            w = wb.add_worksheet(sheet_name)
            for j, c in enumerate(frame.columns):
                w.write(0, j, c, hdr)
            for i, row in enumerate(frame.iter_rows(), start=1):
                for j, v in enumerate(row):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        v = None
                    c = frame.columns[j]
                    fmt = pct if c.endswith("(%)") or c.endswith("_pct") else num if isinstance(v, float) else integer if isinstance(v, int) and not isinstance(v, bool) else None
                    w.write(i, j, v, fmt)
            w.freeze_panes(1, 0)
            if frame.columns:
                w.autofilter(0, 0, max(frame.height, 1), len(frame.columns) - 1)
            for j, c in enumerate(frame.columns):
                width = 14
                if c in {"Fecha", "Periodo"}: width = 15
                elif "Parent" in c or "Modo" in c or "Fuente" in c: width = 23
                elif "actual" in c.lower() or "forecast" in c.lower() or "nivel" in c.lower() or "driver" in c.lower(): width = 22
                elif len(c) > 24: width = 26
                w.set_column(j, j, width)
        write_df("Resumen por bloque", blocks)
        write_df("Detalle cálculo", detail)

        wf = wb.add_worksheet("Fórmulas")
        formula_rows = [
            ("Nivel inicial", "MEDIANA(actual positivo en warm-up inicial de 28 días calendario)."),
            ("Actual desestacionalizado", "exp(log(1+actual) - log(factor_drivers)) - 1, solo actual>0."),
            ("SES", "El estado se actualiza causalmente solo con actual positivo desestacionalizado; y=0 mantiene el nivel."),
            ("Aplicación drivers", "factor_drivers = exp(efecto_drivers_RLS)."),
            ("Forecast raw", "max(exp(log(1+nivel_SES) + efecto_drivers_RLS) - 1, 0)."),
            ("wMAPE oficial", "SUM(|forecast-actual| para actual>0) / SUM(|actual| para actual>0)."),
            ("BIAS oficial", "SUM(forecast-actual para actual>0) / SUM(|actual| para actual>0)."),
            ("Outlier", "factor_gap=(max(actual,forecast)+floor)/(min(actual,forecast)+floor), floor=5% de la mediana positiva. Flag si factor_gap≥4 y abs_error/mediana≥0.15, o abs_error/mediana≥3."),
        ]
        for j, c in enumerate(("Concepto", "Definición")): wf.write(0, j, c, hdr)
        for i, row in enumerate(formula_rows, start=1):
            wf.write(i, 0, row[0]); wf.write(i, 1, row[1], wrap)
        wf.set_column(0, 0, 28); wf.set_column(1, 1, 115)
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
    first = ["Código", "Descripción", "wMAPE (%)", "BIAS (%)", "wMAPE incl. y=0 (%)", "BIAS incl. y=0 (%)"]
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
    # Escenarios precalculados: el selector solo cambia archivos/artefactos;
    # nunca entrena ni recalcula modelos dentro de Streamlit.
    # Solo exponer escenarios realmente operativos de la APP_VERSION actual.
    # Un forecast stale o con artefactos incompletos no debe aparecer en el
    # selector y provocar un st.stop al usuario.
    try:
        from app.forecasting.run_status import validation_errors as _run_status_errors
    except ImportError:  # pragma: no cover
        from forecasting.run_status import validation_errors as _run_status_errors  # type: ignore

    scenario_paths: dict[int, Path] = {}
    _scenario_problems: dict[int, list[str]] = {}
    for _days in getattr(settings, "UPDATE_BLOCK_OPTIONS", (1, 7, 14, 28)):
        _days = int(_days)
        _fp = settings.update_block_forecast_path(_days)
        _errs: list[str] = []
        if not _fp.exists():
            _errs.append("forecast ausente")
        else:
            _errs.extend(_run_status_errors(_fp))
            if not artifacts.artifacts_exist(_fp):
                _errs.append("artefactos ausentes/desactualizados")
        if _errs:
            _scenario_problems[_days] = _errs
        else:
            scenario_paths[_days] = _fp

    if _scenario_problems:
        _missing_label = ", ".join(f"{d}d" for d in sorted(_scenario_problems))
        st.sidebar.warning(
            "Escenarios todavía no listos: " + _missing_label + ". "
            "Preparar/reparar con `uv run python main.py --run dashboard-ready-all --n-jobs 8`."
        )

    selected_update_days: int | None = None
    if scenario_paths:
        st.sidebar.caption(
            "Escenarios listos: " + ", ".join(f"{d}d" for d in sorted(scenario_paths))
        )
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
        if not default_path.exists():
            st.info(
                "No hay escenarios vigentes para el dashboard. Ejecuta "
                "`uv run python main.py --run dashboard-ready-all --n-jobs 8`."
            )
            st.stop()
        st.sidebar.info(
            "No hay escenarios multi-bloque vigentes; mostrando forecast baseline si sus artefactos son válidos."
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
                    "No se cargará `forecast.parquet` completo. Para dejar operativo el dashboard "
                    "en todos los bloques disponibles ejecuta: "
                    "`uv run python -m app.dashboard_artifacts --all-update-blocks`"
                )
                st.info(
                    f"El forecast de {selected_update_days}d ya existe; solo faltan/sobran sincronizar "
                    "los artefactos rápidos. `--all-update-blocks` reconstruye 1d/7d/14d/28d "
                    "que existan, sin recalcular modelos."
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
    "**Modelo v13:** RLS en Sección/Tienda + SES en SKU+Tienda. "
    "In-sample, OOS y forecast-only usan la **misma familia de modelo**; solo cambia la información "
    "disponible en cada origen. Los **wMAPE oficiales son bottom-up** desde SKU+Tienda y excluyen y=0."
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
_show_acid_metrics = st.toggle(
    "Mostrar métricas incluyendo y=0",
    value=False,
    key="show_all_points_metrics",
    help=(
        "Añade wMAPE y BIAS calculados incluyendo también los puntos con y=0. "
        "No modifica el orden del ranking, que continúa basado en las métricas oficiales con y!=0."
    ),
)
st.markdown(f"##### Mostrando métricas: **{_ranking_period_label.upper()}**")
_ranking_scope = []
if view.store:
    _ranking_scope.append(f"SKU en tienda **{view.store}**")
if view.sku:
    _ranking_scope.append(f"tiendas con SKU **{view.sku}**")
_scope_txt = " · ".join(_ranking_scope) if _ranking_scope else "sin filtro cruzado tienda/SKU"
st.caption(
    f"Sección **{view.seccion}** · **{view.unidad}** · {_scope_txt} · "
    "wMAPE bottom-up desde hojas SKU+Tienda Active; ordenado siempre por la métrica oficial con y≠0."
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
    acid = ["wMAPE incl. y=0 (%)", "BIAS incl. y=0 (%)"]
    if _show_acid_metrics:
        preferred = preferred + acid
    hidden = set(acid) if not _show_acid_metrics else set()
    visible = [c for c in preferred if c in display.columns] + [
        c for c in display.columns if c not in preferred and c != "unique_id" and c not in hidden
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
            "wMAPE incl. y=0 (%)": st.column_config.NumberColumn("wMAPE incl. y=0 (%)", format="%.2f%%"),
            "BIAS incl. y=0 (%)": st.column_config.NumberColumn("BIAS incl. y=0 (%)", format="%.2f%%"),
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
            "Tienda", "SKU", "SKU descripción", "wMAPE (%)", "BIAS (%)",
            "wMAPE incl. y=0 (%)", "BIAS incl. y=0 (%)", "Rotación", "Rank",
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
        _acid_cols = {"wMAPE incl. y=0 (%)", "BIAS incl. y=0 (%)"}
        _global_cols = [
            c for c in _global_rank_display.columns
            if c != "unique_id" and (_show_acid_metrics or c not in _acid_cols)
        ]
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
                "Tienda": st.column_config.TextColumn("Tienda", width="small"),
                "SKU": st.column_config.TextColumn("SKU", width="small"),
                # Mantener compacta la descripción para que wMAPE y BIAS queden
                # visibles sin desplazamiento horizontal en el ranking hoja.
                "SKU descripción": st.column_config.TextColumn("SKU descripción", width="medium"),
                "wMAPE (%)": st.column_config.NumberColumn("wMAPE (%)", format="%.2f%%", width="small"),
                "BIAS (%)": st.column_config.NumberColumn("BIAS (%)", format="%.2f%%", width="small"),
                "wMAPE incl. y=0 (%)": st.column_config.NumberColumn("wMAPE incl. y=0 (%)", format="%.2f%%"),
                "BIAS incl. y=0 (%)": st.column_config.NumberColumn("BIAS incl. y=0 (%)", format="%.2f%%"),
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

if _show_acid_metrics:
    st.caption("Prueba ácida · mismas hojas/cohortes del KPI oficial, pero incluyendo también los puntos y=0 en numerador de error y BIAS.")
    # Mantener exactamente la misma alineación horizontal que los KPI oficiales:
    # In-sample en la primera columna y OOS en la segunda; BIAS debajo de wMAPE.
    _acid_in, _acid_oos, _acid_spacer_1, _acid_spacer_2 = st.columns([1.15, 1.15, 1, 1])
    _wi_all = _m_in.get("wmape_all_points")
    _bi_all = _m_in.get("bias_all_points")
    _wo_all = m.get("wmape_all_points")
    _bo_all = m.get("bias_all_points")
    with _acid_in:
        st.metric("wMAPE in-sample · incl. y=0", f"{float(_wi_all):.2%}" if _wi_all is not None else "N/A")
        _bias_badge("BIAS in-sample · incl. y=0", None if _bi_all is None else float(_bi_all))
    with _acid_oos:
        st.metric("wMAPE OOS · incl. y=0", f"{float(_wo_all):.2%}" if _wo_all is not None else "N/A")
        _bias_badge("BIAS OOS · incl. y=0", None if _bo_all is None else float(_bo_all))

if view.has_rolling28:
    st.markdown("#### Métricas Rolling 28d (`yhat28`)")
    st.caption(
        "Walk-forward por bloques de 28 días cuando el artefacto rolling está disponible. WMAPE₂₈ = Σ|y−ŷ₂₈|/Σ|y| (excl. y=0)."
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
    "Marcar discrepancias Actual vs Forecast", value=True, key=f"outliers_{_view_signature}",
    help=(
        "Marca puntos donde Actual y Forecast difieren materialmente: factor ≥4 con error ≥15% "
        "del nivel típico de la serie, o error absoluto ≥3× ese nivel. Es simétrico, robusto "
        "a valores pequeños y solo visual; no cambia métricas ni forecasts."
    ),
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
    _dx = list(ch.get("in_ds", [])) + list(ch.get("oos_ds", []))
    _da = list(ch.get("in_y", [])) + list(ch.get("oos_y", []))
    _df = list(ch.get("in_yhat", [])) + list(ch.get("oos_yhat", []))
    _disc = discrepancy_points(_dx, _da, _df)
    if _disc:
        _custom = [[d["abs_error"], d["factor_gap"], d["scaled_gap"]] for d in _disc]
        fig.add_trace(go.Scatter(
            x=[d["ds"] for d in _disc], y=[d["actual"] for d in _disc],
            name="Discrepancia · actual", mode="markers",
            marker=dict(size=10, symbol="diamond-open", color="#b91c1c", line=dict(width=2)),
            customdata=_custom,
            hovertemplate="Actual=%{y:.2f}<br>Error abs=%{customdata[0]:.2f}<br>Factor gap=%{customdata[1]:.2f}×<br>Gap escalado=%{customdata[2]:.2f}×<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[d["ds"] for d in _disc], y=[d["forecast"] for d in _disc],
            name="Discrepancia · forecast", mode="markers",
            marker=dict(size=10, symbol="x", color="#b91c1c"),
            customdata=_custom,
            hovertemplate="Forecast=%{y:.2f}<br>Error abs=%{customdata[0]:.2f}<br>Factor gap=%{customdata[1]:.2f}×<br>Gap escalado=%{customdata[2]:.2f}×<extra></extra>",
        ))
# EDP leaf diagnostic on secondary axis. It is derived from the selected SKU+Tienda
# actual price history; forecast-only carries the last observed state forward.
_has_edp = view.node_kind == "tienda_sku" and any(
    v is not None for key in ("in_edp", "oos_edp", "fcst_edp") for v in ch.get(key, [])
)
if _has_edp:
    _edp_style = dict(width=2.2, dash="dash", color="#0f766e")
    if ch.get("in_edp"):
        fig.add_trace(go.Scatter(x=ch["in_ds"], y=ch["in_edp"], name="EDP SKU+Tienda", mode="lines", line=_edp_style, yaxis="y2"))
    if ch.get("oos_edp"):
        fig.add_trace(go.Scatter(x=ch["oos_ds"], y=ch["oos_edp"], name="EDP SKU+Tienda · OOS", mode="lines", line=_edp_style, yaxis="y2", showlegend=False))
    if ch.get("fcst_edp"):
        fig.add_trace(go.Scatter(x=ch["fcst_ds"], y=ch["fcst_edp"], name="EDP SKU+Tienda · carry-forward", mode="lines", line=dict(width=2.0, dash="dot", color="#0f766e"), yaxis="y2", showlegend=False))

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
    yaxis2=(dict(title="EDP ($/unidad)", overlaying="y", side="right", showgrid=False) if _has_edp else None),
    legend_title_text="",
    hovermode="x unified",
    margin=dict(l=40, r=(85 if _has_edp else 40), t=50, b=40),
)
_series_caption = [_opt_yhat] if show_yhat else []
if view.has_rolling28 and show_yhat28:
    _series_caption.append(_opt_yhat28)
if _series_caption:
    st.caption("Mostrando forecast: **" + ", ".join(_series_caption) + "**")
st.plotly_chart(fig, width="stretch")
if _has_edp:
    st.caption(
        "EDP en segundo eje: descomposición ex-post del precio observado de la hoja SKU+Tienda; "
        "en forecast-only se prolonga el último EDP observado. Es diagnóstico y no sustituye al "
        "driver RLS del parent Sección/Tienda."
    )

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
    if view.node_kind == "tienda_sku":
        _audit_state_key = f"leaf_audit_bytes_{forecast_path_str}_{view.selected_id}_{view.unidad}"
        if st.button(
            "Preparar auditoría Excel SKU+Tienda",
            help="Carga bajo demanda únicamente esta hoja y sus columnas de trazabilidad; no penaliza la navegación normal.",
            key=f"prepare_leaf_audit_{view.selected_id}_{view.unidad}",
        ):
            with st.spinner("Preparando auditoría de la hoja seleccionada…"):
                _audit_source = _load_leaf_audit_source(
                    _mtime_key, forecast_path_str, view.selected_id, view.unidad
                )
                st.session_state[_audit_state_key] = _leaf_audit_excel_bytes(
                    _audit_source, unidad=view.unidad, label=view.label
                )
        _audit_bytes = st.session_state.get(_audit_state_key)
        if _audit_bytes:
            _safe_uid = view.selected_id.replace("||", "_").replace(":", "-")
            st.download_button(
                "Descargar auditoría Excel SKU+Tienda",
                data=_audit_bytes,
                file_name=f"auditoria_{_safe_uid}_{view.unidad.replace(' ', '_').replace('($)', 'valor')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Incluye resumen, cálculo por bloque, detalle diario, SES, RLS padre y fórmulas de métricas.",
                key=f"download_leaf_audit_{view.selected_id}_{view.unidad}",
            )
