"""
Capa de datos del dashboard: prepara un ViewModel listo para renderizar.
Sin Streamlit ni Plotly.

Modo rápido (artefactos precalculados):
  - rankings y métricas salen de metrics.parquet
  - series del gráfico salen de series.parquet filtrado por unique_id
  - cero wmape_por_id / scans del panel completo en el hot path

Modo legacy (sin artefactos): mantiene prepare_dashboard_state original
sobre el DataFrame completo (lento; solo fallback).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

try:
    from app import backend
    from app import dashboard_artifacts as artifacts
except ImportError:  # pragma: no cover
    import backend  # type: ignore
    import dashboard_artifacts as artifacts  # type: ignore

import settings


@dataclass
class DashboardView:
    """Todo lo que el dashboard necesita pintar."""

    selected_id: str
    seccion: str
    store: str | None
    sku: str | None
    node_kind: str  # "seccion" | "tienda" | "sku" | "tienda_sku"
    label: str
    unidad: str
    freq: str
    horizons: dict[str, dt.date | None]
    ranking_tiendas: pl.DataFrame
    ranking_skus: pl.DataFrame
    n_spine: int
    ranking_days: int
    metrics: dict[str, dict[str, Any]]
    metrics_28: dict[str, float | int]
    has_rolling28: bool
    store_context: str | None
    chart: dict[str, Any]
    detail: pl.DataFrame
    ds_min: dt.date | None
    ds_max: dt.date | None
    cutoff: dt.date
    consistency_warnings: list[str]
    model_compare: dict[str, Any] | None = None
    has_value_cols: bool = False


def _node_kind(store: str | None, sku: str | None) -> str:
    if store is not None and sku is not None:
        return "tienda_sku"
    if sku is not None:
        return "sku"
    if store is not None:
        return "tienda"
    return "seccion"


def _parse_date(v: str | None) -> dt.date | None:
    if not v:
        return None
    if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
        return v
    if isinstance(v, dt.datetime):
        return v.date()
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def _horizons_from_index(
    index: dict[str, Any], seccion: str, df_daily: pl.DataFrame
) -> dict[str, dt.date | None]:
    """Horizontes del dashboard SIEMPRE a nivel sección.

    No deben derivarse desde la serie seleccionada: un SKU+Tienda puede tener
    una historia más corta o días finales sin venta y desplazar falsamente el
    inicio OOS. El index se construye desde el forecast completo por sección.
    """
    raw = (index.get("horizons_by_sec") or {}).get(seccion) or {}
    hz = {k: _parse_date(v) for k, v in raw.items()}
    required = ("train_end", "test_start", "test_end", "forecast_start", "forecast_end")
    if all(hz.get(k) is not None for k in ("test_start", "test_end", "forecast_start", "forecast_end")):
        return {k: hz.get(k) for k in required}

    # Compatibilidad con artefactos antiguos sin horizons_by_sec completo.
    cols = set(df_daily.columns)
    resolved = backend.resolve_horizons(df_daily, seccion, cols) if df_daily.height else {}
    fallback = settings.section_horizons(seccion)
    return {k: hz.get(k) or resolved.get(k) or fallback.get(k) for k in required}


def _pin_selected(display: pl.DataFrame, code: str | None) -> pl.DataFrame:
    """Marca la selección sin reordenar filas.

    Reordenar la fila seleccionada en cada rerun interfería con el sort interactivo
    de Streamlit y hacía que el usuario percibiera un orden aparentemente aleatorio.
    """
    if display.height == 0 or code is None or "Código" not in display.columns:
        return display
    selected = str(code)
    if display.filter(pl.col("Código").cast(pl.Utf8) == selected).height == 0:
        return display
    return display.with_columns(
        pl.when(pl.col("Código").cast(pl.Utf8) == selected)
        .then(pl.lit("✓"))
        .otherwise(pl.lit(""))
        .alias("Sel.")
    )


def _metric_consistency_warnings(
    metrics: pl.DataFrame,
    *,
    selected_id: str,
    unidad: str,
    observed: dict[str, dict[str, float | int]],
    tolerance: float = 1e-9,
) -> list[str]:
    """Verifica que KPI OOS y metrics.parquet usan exactamente la misma métrica."""
    if metrics.height == 0:
        return []
    row = metrics.filter(
        (pl.col("unique_id") == selected_id)
        & (pl.col("unidad") == unidad)
    )
    if row.height == 0:
        return [
            f"No existe métrica OOS precalculada para {selected_id} / {unidad}. "
            "Regenera los artefactos del dashboard."
        ]
    expected_raw = row["wmape"][0]
    row_active = (
        bool(row["metric_active"][0])
        if "metric_active" in row.columns and row["metric_active"][0] is not None
        else True
    )
    observed_key = "out" if row_active else "out_all"
    actual_raw = (observed.get(observed_key) or {}).get("wmape")
    if expected_raw is None and actual_raw is None:
        return []
    if (expected_raw is None) != (actual_raw is None):
        return [
            "Inconsistencia detectada entre ranking/metrics.parquet y KPI OOS "
            f"del gráfico para {selected_id}: uno es N/A y el otro no. "
            "Regenera los artefactos del dashboard."
        ]
    expected = float(expected_raw)
    actual = float(actual_raw)
    if abs(expected - actual) > tolerance:
        return [
            "Inconsistencia detectada entre ranking/metrics.parquet y KPI OOS "
            f"del gráfico para {selected_id}: {expected:.6%} vs {actual:.6%}. "
            "Los artefactos pueden estar desactualizados o construidos con otra definición."
        ]
    return []


def _official_oos_metrics_from_artifact(
    metrics: pl.DataFrame,
    *,
    selected_id: str,
    unidad: str,
) -> dict[str, dict[str, Any]]:
    """KPI OOS oficial directamente desde ``metrics.parquet``.

    En modo rápido el parquet de métricas es la única fuente de verdad para
    wMAPE/BIAS OOS.  No se recalcula la métrica desde la serie agregada del
    gráfico porque una serie padre no conserva la semántica bottom-up Active.
    """
    row = metrics.filter(
        (pl.col("unique_id") == selected_id)
        & (pl.col("unidad") == unidad)
    )
    if row.height == 0:
        return {}

    def _get(name: str, default=None):
        if name not in row.columns:
            return default
        value = row[name][0]
        return default if value is None else value

    den = float(_get("sum_abs_y", 0.0) or 0.0)
    wm_checked, bias_checked = _metric_values_from_row(row)
    zero_forecast = float(_get("zero_forecast_sum", 0.0) or 0.0)
    zero_error = float(_get("zero_abs_error_sum", 0.0) or 0.0)
    return {
        "out": {
            "wmape": wm_checked,
            "bias": bias_checked,
            "n": int(_get("n_points", 0) or 0),
            "defined": _get("wmape") is not None,
            "sum_abs_y": den,
            "sum_abs_error": float(_get("sum_abs_error", 0.0) or 0.0),
            "sum_signed_error": float(_get("sum_signed_error", 0.0) or 0.0),
            "cohort": str(_get("metric_cohort", "active")),
        },
        "zero_demand": {
            "leaf_count": int(_get("n_leaf_zero", 0) or 0),
            "forecast_sum": zero_forecast,
            "abs_error_sum": zero_error,
            "impact_vs_active": zero_forecast / den if den > 0 else 0.0,
        },
        "cohorts": {
            "active": int(_get("n_leaf_active", 0) or 0),
            "sparse": int(_get("n_leaf_sparse", 0) or 0),
            "zero": int(_get("n_leaf_zero", 0) or 0),
        },
    }


def load_model_compare_fast(
    *,
    seccion: str,
    store: str | None,
    sku: str | None,
    unidad: str,
    forecast_path: str | Path | None,
    has_value: bool,
) -> dict[str, Any] | None:
    """Carga bajo demanda el diagnóstico pesado v11/v12.

    Se mantiene fuera del hot path del dashboard: una vista de sección puede
    implicar millones de filas hoja y no debe bloquear el arranque normal.
    """
    fpath = Path(forecast_path) if forecast_path else None
    leaves = artifacts.load_leaves_for_scope(
        seccion,
        store=store,
        sku=sku,
        forecast_path=fpath,
        unidad=unidad,
        has_value=has_value,
    )
    return _v12_model_compare(leaves)



def global_leaf_ranking_from_metrics(
    metrics: pl.DataFrame,
    *,
    label_map: dict[str, str] | None = None,
    desc_map: dict[str, str] | None = None,
    period: str = "oos",
) -> pl.DataFrame:
    """Ranking global vectorizado de hojas tienda+SKU.

    ``period`` puede ser ``oos`` o ``in_sample``. El universo Active se mantiene
    contractual (clasificación OOS); solo cambian los sumatorios/metricas mostrados.
    La implementación evita ``iter_rows`` sobre ~90k métricas para que cambiar de
    bloque precalculado vuelva a ser prácticamente instantáneo.
    """
    is_in = str(period).lower() in {"in", "in_sample", "in-sample"}
    vol_label = "Volumen in-sample" if is_in else "Volumen OOS"
    fc_label = "Pronóstico in-sample" if is_in else "Pronóstico OOS"
    err_label = "Error abs. in-sample" if is_in else "Error abs. OOS"
    total_col = "forecast_in_sample_total" if is_in else "forecast_oos_total"
    cols = [
        "Rank", "Unidad", "Sección", "Tienda", "Tienda descripción",
        "SKU", "SKU descripción", "wMAPE (%)", "BIAS (%)", "Rotación",
        "N puntos", "Días con venta", "% ≠0", "Cohort",
        vol_label, fc_label, err_label, "unique_id",
    ]
    if metrics.height == 0:
        return pl.DataFrame({c: [] for c in cols})

    base = metrics
    if "store" not in base.columns or "sku" not in base.columns:
        base = base.with_columns(
            pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("store"),
            pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("sku"),
        )
    base = base.filter(pl.col("store").is_not_null() & pl.col("sku").is_not_null())
    if base.height == 0:
        return pl.DataFrame({c: [] for c in cols})
    subset = ["unique_id", "unidad"] if "unidad" in base.columns else ["unique_id"]
    base = base.unique(subset=subset, keep="first")

    den = pl.col("sum_abs_y").cast(pl.Float64).fill_null(0.0)
    wm = pl.when(den > 0).then(pl.col("sum_abs_error").cast(pl.Float64).fill_null(0.0) / den).otherwise(pl.col("wmape"))
    if "sum_signed_error" in base.columns:
        bi = pl.when(den > 0).then(pl.col("sum_signed_error").cast(pl.Float64).fill_null(0.0) / den).otherwise(pl.col("bias"))
    else:
        bi = pl.col("bias")
    if total_col in base.columns:
        total_expr = pl.col(total_col).cast(pl.Float64).fill_null(0.0)
    elif "sum_yhat" in base.columns:
        total_expr = pl.col("sum_yhat").cast(pl.Float64).fill_null(0.0)
    else:
        total_expr = pl.lit(0.0)
    rotation_expr = (
        pl.col("sum_y").cast(pl.Float64).fill_null(0.0)
        if "sum_y" in base.columns
        else den
    )

    x = base.with_columns(
        pl.col("unique_id").str.split("||").list.get(0).alias("Sección"),
        pl.col("store").cast(pl.Utf8).alias("Tienda"),
        pl.col("sku").cast(pl.Utf8).alias("SKU"),
        (wm * 100.0).alias("wMAPE (%)"),
        (bi * 100.0).alias("BIAS (%)"),
        rotation_expr.alias("Rotación"),
        pl.col("n_points").fill_null(0).cast(pl.Int64).alias("N puntos"),
        pl.col("n_with_sales").fill_null(0).cast(pl.Int64).alias("Días con venta"),
        pl.col("metric_cohort").fill_null("").cast(pl.Utf8).alias("Cohort"),
        den.alias(vol_label),
        total_expr.alias(fc_label),
        pl.col("sum_abs_error").cast(pl.Float64).fill_null(0.0).alias(err_label),
        pl.col("unidad").cast(pl.Utf8).alias("Unidad"),
    ).with_columns(
        pl.when(pl.col("N puntos") > 0)
        .then(100.0 * pl.col("Días con venta") / pl.col("N puntos"))
        .otherwise(0.0).alias("% ≠0")
    )

    # Descripciones por joins vectorizados, sin Python por fila.
    desc_map = desc_map or {}
    label_map = label_map or {}
    if desc_map or label_map:
        keys = sorted(set(desc_map) | set(label_map))
        lookup = pl.DataFrame({
            "_uid": keys,
            "_desc": [str(desc_map.get(k) or label_map.get(k) or "") for k in keys],
        })
        store_lookup = lookup.filter(pl.col("_uid").str.contains("||T:", literal=True) & ~pl.col("_uid").str.contains("||S:", literal=True)).rename({"_uid":"_store_uid", "_desc":"Tienda descripción"})
        sku_lookup = lookup.filter(pl.col("_uid").str.contains("||S:", literal=True) & ~pl.col("_uid").str.contains("||T:", literal=True)).rename({"_uid":"_sku_uid", "_desc":"SKU descripción"})
        leaf_lookup = lookup.rename({"_uid": "unique_id", "_desc": "_leaf_desc"})
        x = x.with_columns(
            pl.concat_str([pl.col("Sección"), pl.lit("||T:"), pl.col("Tienda")]).alias("_store_uid"),
            pl.concat_str([pl.col("Sección"), pl.lit("||S:"), pl.col("SKU")]).alias("_sku_uid"),
        ).join(store_lookup, on="_store_uid", how="left").join(sku_lookup, on="_sku_uid", how="left").join(leaf_lookup, on="unique_id", how="left").with_columns(
            pl.when(pl.col("SKU descripción").fill_null("") != "")
            .then(pl.col("SKU descripción"))
            .otherwise(pl.col("_leaf_desc").fill_null(""))
            .alias("SKU descripción")
        ).drop("_store_uid", "_sku_uid", "_leaf_desc")
    else:
        x = x.with_columns(pl.lit("").alias("Tienda descripción"), pl.lit("").alias("SKU descripción"))

    x = x.with_columns(
        pl.col("Tienda descripción").fill_null(""),
        pl.col("SKU descripción").fill_null(""),
    ).sort(["Unidad", "wMAPE (%)", "Sección", "Tienda", "SKU"], nulls_last=True)
    x = x.with_columns(pl.col("Unidad").cum_count().over("Unidad").alias("Rank"))
    return x.select(cols)

def _metric_values_from_row(row: pl.DataFrame) -> tuple[float | None, float | None]:
    """Obtiene wMAPE/BIAS y verifica su identidad contra los sumatorios OOS."""
    if row.height == 0:
        return None, None
    wm = row["wmape"][0] if "wmape" in row.columns else None
    bi = row["bias"][0] if "bias" in row.columns else None
    if all(c in row.columns for c in ("sum_abs_y", "sum_abs_error", "sum_signed_error")):
        den = float(row["sum_abs_y"][0] or 0.0)
        if den > 0:
            wm_calc = float(row["sum_abs_error"][0] or 0.0) / den
            bi_calc = float(row["sum_signed_error"][0] or 0.0) / den
            # Los sumatorios son la fuente algebraica; evita mostrar valores stale/incoherentes.
            if wm is None or abs(float(wm) - wm_calc) > 1e-10:
                wm = wm_calc
            if bi is None or abs(float(bi) - bi_calc) > 1e-10:
                bi = bi_calc
    return (None if wm is None else float(wm), None if bi is None else float(bi))


def _sync_selected_leaf_metric(
    ranking: pl.DataFrame,
    metrics: pl.DataFrame,
    *,
    seccion: str,
    unidad: str,
    store: str | None,
    sku: str | None,
    axis: str,
) -> pl.DataFrame:
    """Sincroniza la fila seleccionada del ranking con el KPI OOS exacto de la hoja.

    Cuando hay tienda+SKU seleccionados, las dos tablas deben mostrar exactamente
    el mismo wMAPE/BIAS que el panel OOS de esa hoja, nunca una métrica padre o stale.
    """
    if ranking.height == 0 or store is None or sku is None:
        return ranking
    uid = settings.make_unique_id(seccion, store=str(store), sku=str(sku))
    exact = metrics.filter(
        (pl.col("unique_id") == uid) & (pl.col("unidad") == unidad)
    )
    if exact.height == 0:
        return ranking
    wm, bi = _metric_values_from_row(exact)
    code = str(store if axis == "store" else sku)
    wm_val = None if wm is None else float(wm) * 100.0
    bi_val = None if bi is None else float(bi) * 100.0
    return ranking.with_columns(
        pl.when(pl.col("Código").cast(pl.Utf8) == code).then(pl.lit(wm_val)).otherwise(pl.col("wMAPE (%)")).alias("wMAPE (%)"),
        pl.when(pl.col("Código").cast(pl.Utf8) == code).then(pl.lit(bi_val)).otherwise(pl.col("BIAS (%)")).alias("BIAS (%)"),
    )


def _ranking_from_metrics(
    metrics: pl.DataFrame,
    *,
    seccion: str,
    unidad: str,
    axis: str,
    fixed_peer: str | None,
    exclude: str | None,
    desc_map: dict[str, str],
    n_spine: int,
    selected_code: str | None = None,
) -> pl.DataFrame:
    """Ranking barato sobre metrics.parquet (misma semántica que ranking_table)."""
    label_rot = "Rotación ($)" if unidad.startswith("Valor") else "Rotación (unid.)"
    empty = pl.DataFrame(
        schema={
            "Código": pl.Utf8,
            "Descripción": pl.Utf8,
            "wMAPE (%)": pl.Float64,
            "BIAS (%)": pl.Float64,
            label_rot: pl.Float64,
            "N puntos": pl.UInt32,
            "% ≠0": pl.Float64,
            "Impacto error (%)": pl.Float64,
            "Estado": pl.Utf8,
            "unique_id": pl.Utf8,
        }
    )
    base = metrics.filter(
        (pl.col("seccion") == seccion) & (pl.col("unidad") == unidad)
    )
    if base.height == 0:
        # La unidad seleccionada es parte del contrato del dashboard.
        # Nunca reutilizar métricas de otra unidad: eso desincroniza ranking,
        # KPIs y gráfico. Si faltan artefactos de la unidad, devolver vacío y
        # obligar a regenerarlos.
        return empty

    # Defensa: metrics mal generados no deben duplicar filas del ranking
    if "unidad" in base.columns:
        base = base.unique(subset=["unique_id", "unidad"], keep="first")
    else:
        base = base.unique(subset=["unique_id"], keep="first")

    if "store" not in base.columns or "sku" not in base.columns:
        uids = base["unique_id"].to_list()
        stores, skus = [], []
        for u in uids:
            p = settings.split_unique_id(u)
            stores.append(p.get("store"))
            skus.append(p.get("sku"))
        base = base.with_columns(
            pl.Series("store", stores),
            pl.Series("sku", skus),
        )

    # Normalizar códigos a Utf8 para comparaciones estables
    if "store" in base.columns:
        base = base.with_columns(pl.col("store").cast(pl.Utf8))
    if "sku" in base.columns:
        base = base.with_columns(pl.col("sku").cast(pl.Utf8))

    if axis == "store":
        if fixed_peer is not None:
            # SKU seleccionado → comparar ESE MISMO SKU entre tiendas.
            peer = str(fixed_peer)
            tabla = base.filter(
                pl.col("store").is_not_null()
                & (pl.col("sku") == peer)
            )
        else:
            # Sin SKU fijo → métricas bottom-up derivadas por tienda.
            tabla = base.filter(
                pl.col("store").is_not_null() & pl.col("sku").is_null()
            )
        if exclude is not None:
            tabla = tabla.filter(pl.col("store") != str(exclude))
        code_col = "store"
    else:
        # Ranking SKU
        if fixed_peer is not None:
            # Tienda seleccionada → SOLO hojas sku+tienda de ESA tienda
            peer = str(fixed_peer)
            tabla = base.filter(
                (pl.col("store") == peer) & pl.col("sku").is_not_null()
            )
            if tabla.height == 0:
                # Fallback por patrón de unique_id (por si store no parseó bien)
                needle = f"||T:{peer}||"
                tabla = base.filter(
                    pl.col("unique_id").str.contains(needle, literal=True)
                    & (
                        pl.col("sku").is_not_null()
                        | pl.col("unique_id").str.contains("||S:", literal=True)
                    )
                )
                if tabla.height and (
                    "sku" not in tabla.columns
                    or tabla.filter(pl.col("sku").is_not_null()).height == 0
                ):
                    tabla = tabla.with_columns(
                        pl.col("unique_id")
                        .str.extract(r"\|\|S:([^|]+)", 1)
                        .alias("sku")
                    )
            if exclude is not None:
                tabla = tabla.filter(pl.col("sku") != str(exclude))
            code_col = "sku"
        else:
            # Sin tienda: preferir nodos SKU puro; si no, agregar hojas
            pure = base.filter(
                pl.col("sku").is_not_null() & pl.col("store").is_null()
            )
            if exclude is not None:
                pure = pure.filter(pl.col("sku") != str(exclude))
            if pure.height > 0:
                tabla = pure
                code_col = "sku"
            else:
                leaves = base.filter(
                    pl.col("store").is_not_null() & pl.col("sku").is_not_null()
                )
                if exclude is not None:
                    leaves = leaves.filter(pl.col("sku") != str(exclude))
                if leaves.height == 0:
                    return empty
                tabla = (
                    leaves.group_by("sku")
                    .agg(
                        pl.col("sum_abs_error").sum().alias("sum_abs_error"),
                        pl.col("sum_abs_y").sum().alias("sum_abs_y"),
                        pl.col("sum_signed_error").sum().alias("sum_signed_error"),
                        pl.col("sum_y").sum().alias("sum_y"),
                        pl.col("sum_yhat").sum().alias("sum_yhat"),
                        pl.col("n_with_sales").sum().alias("n_with_sales"),
                        pl.col("n_points").first().alias("n_points"),
                    )
                    .with_columns(
                        pl.when(pl.col("sum_abs_y") > 0)
                        .then(pl.col("sum_abs_error") / pl.col("sum_abs_y"))
                        .otherwise(0.0)
                        .alias("wmape"),
                        pl.when(pl.col("sum_abs_y") > 0)
                        .then(pl.col("sum_signed_error") / pl.col("sum_abs_y"))
                        .otherwise(0.0)
                        .alias("bias"),
                        pl.concat_str(
                            [pl.lit(seccion), pl.lit("||S:"), pl.col("sku")]
                        ).alias("unique_id"),
                    )
                )
                code_col = "sku"

    # Participación en el error absoluto del alcance ANTES de ocultar filas
    # por elegibilidad de ranking. Un wMAPE enorme en un SKU de poca rotación
    # puede así distinguirse de un SKU que realmente domina el error agregado.
    scope_metric_rows = (
        tabla.filter(pl.col("metric_active").fill_null(False))
        if "metric_active" in tabla.columns
        else tabla
    )
    scope_abs_error = float(
        scope_metric_rows.select(pl.col("sum_abs_error").sum()).item()
        or 0.0
    )

    # Elegibilidad del ranking. La selección actual se conserva aunque quede
    # fuera del criterio en el nuevo alcance (p.ej. SKU con 15 días a nivel
    # sección pero solo 6 días de venta en una tienda concreta).
    eligible_expr = (
        (pl.col("sum_abs_y") > 0)
        & pl.col("wmape").is_not_null()
        & (pl.col("wmape") > 0)
    )
    if "metric_active" in tabla.columns:
        eligible_expr = eligible_expr & pl.col("metric_active").fill_null(False)
    min_nonzero = int(getattr(settings, "RANKING_SKU_MIN_NONZERO_POINTS", 15))
    if axis == "sku":
        eligible_expr = eligible_expr & (pl.col("n_with_sales") >= min_nonzero)
    tabla = tabla.with_columns(eligible_expr.alias("_eligible"))

    selected_row = pl.DataFrame()
    if selected_code is not None:
        selected_row = tabla.filter(
            pl.col(code_col).cast(pl.Utf8) == str(selected_code)
        )

    ranked = tabla.filter(pl.col("_eligible"))
    if selected_row.height:
        ranked = pl.concat(
            [ranked, selected_row], how="diagonal_relaxed"
        ).unique(subset=[code_col], keep="first", maintain_order=True)
    tabla = ranked
    if tabla.height == 0:
        return empty
    if code_col in tabla.columns:
        # Un código por fila; keep first tras sort ASC = menor wMAPE
        tabla = tabla.sort("wmape", descending=False, nulls_last=True).unique(
            subset=[code_col], keep="first", maintain_order=True
        )
    else:
        tabla = tabla.sort("wmape", descending=False, nulls_last=True)

    codes = tabla[code_col].to_list()
    uids2 = tabla["unique_id"].to_list()
    n_with_sales = tabla["n_with_sales"].to_list()
    pct = [
        (float(nw) / float(n_spine) * 100) if n_spine else 0.0
        for nw in n_with_sales
    ]
    impacts = [
        (
            float(err) / scope_abs_error * 100.0
            if scope_abs_error > 0
            else 0.0
        )
        for err in tabla["sum_abs_error"].to_list()
    ]

    descriptions: list[str] = []
    for u, code in zip(uids2, codes):
        if axis == "store":
            store_uid = settings.make_unique_id(seccion, store=str(code))
            d = desc_map.get(store_uid, "")
        else:
            sku_uid = settings.make_unique_id(seccion, sku=str(code))
            d = desc_map.get(sku_uid, "") or desc_map.get(u, "")
            if not d:
                d = next(
                    (
                        desc_map[k]
                        for k in desc_map
                        if f"||S:{code}" in k and desc_map[k]
                    ),
                    "",
                )
        descriptions.append(d)

    return pl.DataFrame(
        {
            "Código": [str(c) for c in codes],
            "Descripción": descriptions,
            "wMAPE (%)": [float(w) * 100.0 if w is not None else None for w in tabla["wmape"].to_list()],
            "BIAS (%)": [float(b) * 100.0 if b is not None else None for b in tabla["bias"].to_list()],
            label_rot: [float(v or 0.0) for v in tabla["sum_y"].to_list()],
            "N puntos": [int(x) for x in n_with_sales],
            "% ≠0": [float(p) for p in pct],
            "Impacto error (%)": [float(v) for v in impacts],
            "Estado": [
                (
                    "✓ criterio"
                    if bool(ok)
                    else (
                        "sin demanda OOS"
                        if cohort == "zero"
                        else (
                            f"sparse OOS (<{int(getattr(settings, 'OOS_ACTIVE_MIN_NONZERO_DAYS', 7))} días ≠0)"
                            if cohort == "sparse"
                            else f"fuera criterio (<{min_nonzero} días ≠0)"
                        )
                    )
                )
                for ok, cohort in zip(
                    tabla["_eligible"].to_list(),
                    (
                        tabla["metric_cohort"].to_list()
                        if "metric_cohort" in tabla.columns
                        else ["active"] * tabla.height
                    ),
                )
            ],
            "unique_id": uids2,
        }
    )



def _v12_model_compare(leaves: pl.DataFrame) -> dict[str, Any]:
    """Comparación OOS final/v11/v12 + oracle por leaf (solo diagnóstico).

    El oracle usa verdad OOS y por definición NO es causal ni se usa para
    producir forecasts. Sirve para cuantificar el límite del selector actual.
    """
    needed = {"unique_id", "ds", "y", "yhat", "period_type", "_compare_v11", "_compare_v12"}
    if leaves.height == 0 or not needed.issubset(leaves.columns):
        return {}
    x = leaves.filter(pl.col("period_type") == "out_sample")
    if x.height == 0:
        return {}
    min_days = int(getattr(settings, "OOS_ACTIVE_MIN_NONZERO_DAYS", 7))
    active = (
        x.group_by("unique_id")
        .agg(pl.col("ds").filter(pl.col("y") > 0).n_unique().alias("_nz"))
        .filter(pl.col("_nz") >= min_days)
        .select("unique_id")
    )
    x = x.join(active, on="unique_id", how="semi").with_columns(
        pl.coalesce([pl.col("_compare_v12"), pl.col("_compare_v11")]).alias("_v12")
    )
    if x.height == 0:
        return {}
    by_leaf = (
        x.group_by("unique_id")
        .agg(
            pl.when(pl.col("y") > 0).then(pl.col("y").abs()).otherwise(0.0).sum().alias("den"),
            pl.when(pl.col("y") > 0).then((pl.col("y") - pl.col("_compare_v11")).abs()).otherwise(0.0).sum().alias("ae11"),
            pl.when(pl.col("y") > 0).then((pl.col("y") - pl.col("_v12")).abs()).otherwise(0.0).sum().alias("ae12"),
            pl.when(pl.col("y") > 0).then((pl.col("y") - pl.col("yhat")).abs()).otherwise(0.0).sum().alias("aef"),
            pl.when(pl.col("y") > 0).then(pl.col("_compare_v11") - pl.col("y")).otherwise(0.0).sum().alias("se11"),
            pl.when(pl.col("y") > 0).then(pl.col("_v12") - pl.col("y")).otherwise(0.0).sum().alias("se12"),
            pl.when(pl.col("y") > 0).then(pl.col("yhat") - pl.col("y")).otherwise(0.0).sum().alias("sef"),
            pl.col("_v12_selected").fill_null(False).max().alias("selected_v12")
            if "_v12_selected" in x.columns else pl.lit(False).alias("selected_v12"),
            pl.col("_cal_factor").drop_nulls().median().alias("cal_factor")
            if "_cal_factor" in x.columns else pl.lit(None).cast(pl.Float64).alias("cal_factor"),
            pl.col("_shape_applied").fill_null(False).max().alias("shape_applied")
            if "_shape_applied" in x.columns else pl.lit(False).alias("shape_applied"),
            pl.col("_cal_applied").fill_null(False).max().alias("cal_applied")
            if "_cal_applied" in x.columns else pl.lit(False).alias("cal_applied"),
            pl.col("_meta_probability").drop_nulls().median().alias("meta_probability")
            if "_meta_probability" in x.columns else pl.lit(None).cast(pl.Float64).alias("meta_probability"),
            pl.col("_meta_available").fill_null(False).max().alias("meta_available")
            if "_meta_available" in x.columns else pl.lit(False).alias("meta_available"),
            pl.col("_meta_top_driver").drop_nulls().first().alias("meta_top_driver")
            if "_meta_top_driver" in x.columns else pl.lit(None).cast(pl.Utf8).alias("meta_top_driver"),
            pl.col("_meta_threshold").drop_nulls().first().alias("meta_threshold")
            if "_meta_threshold" in x.columns else pl.lit(None).cast(pl.Float64).alias("meta_threshold"),
            pl.col("_meta_portfolio_mode").drop_nulls().first().alias("meta_portfolio_mode")
            if "_meta_portfolio_mode" in x.columns else pl.lit(None).cast(pl.Utf8).alias("meta_portfolio_mode"),
            pl.col("_meta_bias_guard_pass").fill_null(False).max().alias("meta_bias_guard_pass")
            if "_meta_bias_guard_pass" in x.columns else pl.lit(False).alias("meta_bias_guard_pass"),
            pl.col("_meta_policy_available").fill_null(False).max().alias("meta_policy_available")
            if "_meta_policy_available" in x.columns else pl.lit(False).alias("meta_policy_available"),
            pl.col("_meta_policy_utility_gain").drop_nulls().first().alias("meta_policy_utility_gain")
            if "_meta_policy_utility_gain" in x.columns else pl.lit(None).cast(pl.Float64).alias("meta_policy_utility_gain"),
            pl.col("_value_safety_dominance_pass").fill_null(False).max().alias("value_safety_dominance_pass")
            if "_value_safety_dominance_pass" in x.columns else pl.lit(False).alias("value_safety_dominance_pass"),
            pl.col("_value_safety_recent_confirmations").drop_nulls().first().alias("value_safety_recent_confirmations")
            if "_value_safety_recent_confirmations" in x.columns else pl.lit(None).cast(pl.Int8).alias("value_safety_recent_confirmations"),
            pl.col("_value_safety_recent_blocks").drop_nulls().first().alias("value_safety_recent_blocks")
            if "_value_safety_recent_blocks" in x.columns else pl.lit(None).cast(pl.Int8).alias("value_safety_recent_blocks"),
            pl.col("_value_safety_bias_coverage").drop_nulls().first().alias("value_safety_bias_coverage")
            if "_value_safety_bias_coverage" in x.columns else pl.lit(None).cast(pl.Float64).alias("value_safety_bias_coverage"),
            pl.col("_value_safety_bias_coverage_threshold").drop_nulls().first().alias("value_safety_bias_coverage_threshold")
            if "_value_safety_bias_coverage_threshold" in x.columns else pl.lit(None).cast(pl.Float64).alias("value_safety_bias_coverage_threshold"),
            pl.col("_value_safety_bias_coverage_pass").fill_null(False).max().alias("value_safety_bias_coverage_pass")
            if "_value_safety_bias_coverage_pass" in x.columns else pl.lit(False).alias("value_safety_bias_coverage_pass"),
            pl.col("_value_safety_meta_margin_pass").fill_null(False).max().alias("value_safety_meta_margin_pass")
            if "_value_safety_meta_margin_pass" in x.columns else pl.lit(False).alias("value_safety_meta_margin_pass"),
            pl.col("_value_safety_best_all_mode").drop_nulls().first().alias("value_safety_best_all_mode")
            if "_value_safety_best_all_mode" in x.columns else pl.lit(None).cast(pl.Utf8).alias("value_safety_best_all_mode"),
            pl.col("_value_safety_reason").drop_nulls().first().alias("value_safety_reason")
            if "_value_safety_reason" in x.columns else pl.lit(None).cast(pl.Utf8).alias("value_safety_reason"),
            pl.col("_v129_wf_enabled").fill_null(False).max().alias("v129_wf_enabled")
            if "_v129_wf_enabled" in x.columns else pl.lit(False).alias("v129_wf_enabled"),
            pl.col("_v129_wf_available").fill_null(False).max().alias("v129_wf_available")
            if "_v129_wf_available" in x.columns else pl.lit(False).alias("v129_wf_available"),
            pl.col("_v129_wf_folds").drop_nulls().first().alias("v129_wf_folds")
            if "_v129_wf_folds" in x.columns else pl.lit(None).cast(pl.Int8).alias("v129_wf_folds"),
            pl.col("_v129_wf_win_rate").drop_nulls().first().alias("v129_wf_win_rate")
            if "_v129_wf_win_rate" in x.columns else pl.lit(None).cast(pl.Float64).alias("v129_wf_win_rate"),
            pl.col("_v129_wf_median_gain").drop_nulls().first().alias("v129_wf_median_gain")
            if "_v129_wf_median_gain" in x.columns else pl.lit(None).cast(pl.Float64).alias("v129_wf_median_gain"),
            pl.col("_v129_wf_worst_gain").drop_nulls().first().alias("v129_wf_worst_gain")
            if "_v129_wf_worst_gain" in x.columns else pl.lit(None).cast(pl.Float64).alias("v129_wf_worst_gain"),
            pl.col("_v129_wf_weighted_gain").drop_nulls().first().alias("v129_wf_weighted_gain")
            if "_v129_wf_weighted_gain" in x.columns else pl.lit(None).cast(pl.Float64).alias("v129_wf_weighted_gain"),
            pl.col("_v129_wf_weighted_utility_gain").drop_nulls().first().alias("v129_wf_weighted_utility_gain")
            if "_v129_wf_weighted_utility_gain" in x.columns else pl.lit(None).cast(pl.Float64).alias("v129_wf_weighted_utility_gain"),
            pl.col("_v129_wf_bias_worsen_max").drop_nulls().first().alias("v129_wf_bias_worsen_max")
            if "_v129_wf_bias_worsen_max" in x.columns else pl.lit(None).cast(pl.Float64).alias("v129_wf_bias_worsen_max"),
            pl.col("_v129_wf_meta_folds").drop_nulls().first().alias("v129_wf_meta_folds")
            if "_v129_wf_meta_folds" in x.columns else pl.lit(None).cast(pl.Int8).alias("v129_wf_meta_folds"),
            pl.col("_v129_wf_reason").drop_nulls().first().alias("v129_wf_reason")
            if "_v129_wf_reason" in x.columns else pl.lit(None).cast(pl.Utf8).alias("v129_wf_reason"),
        )
        .with_columns((pl.col("ae12") < pl.col("ae11")).alias("oracle_v12"))
        .with_columns(
            pl.when(pl.col("oracle_v12")).then(pl.col("ae12")).otherwise(pl.col("ae11")).alias("ae_oracle"),
            pl.when(pl.col("oracle_v12")).then(pl.col("se12")).otherwise(pl.col("se11")).alias("se_oracle"),
        )
    )
    sums = by_leaf.select(
        pl.col("den").sum(), pl.col("aef").sum(), pl.col("ae11").sum(),
        pl.col("ae12").sum(), pl.col("ae_oracle").sum(), pl.col("sef").sum(),
        pl.col("se11").sum(), pl.col("se12").sum(), pl.col("se_oracle").sum(),
        pl.col("selected_v12").sum().alias("selected_v12"),
        pl.col("oracle_v12").sum().alias("oracle_v12"),
        pl.col("cal_factor").drop_nulls().median().alias("cal_factor"),
        pl.col("shape_applied").sum().alias("shape_applied"),
        pl.col("cal_applied").sum().alias("cal_applied"),
        pl.col("meta_probability").drop_nulls().median().alias("meta_probability"),
        pl.col("meta_probability").filter(pl.col("selected_v12")).drop_nulls().median().alias("meta_probability_selected"),
        pl.col("meta_available").sum().alias("meta_available"),
        pl.col("meta_threshold").drop_nulls().median().alias("meta_threshold"),
        pl.col("meta_bias_guard_pass").sum().alias("meta_bias_guard_pass"),
        pl.col("meta_policy_available").sum().alias("meta_policy_available"),
        pl.col("meta_policy_utility_gain").drop_nulls().median().alias("meta_policy_utility_gain"),
        pl.col("value_safety_dominance_pass").max().alias("value_safety_dominance_pass"),
        pl.col("value_safety_recent_confirmations").drop_nulls().median().alias("value_safety_recent_confirmations"),
        pl.col("value_safety_recent_blocks").drop_nulls().median().alias("value_safety_recent_blocks"),
        pl.col("value_safety_bias_coverage").drop_nulls().median().alias("value_safety_bias_coverage"),
        pl.col("value_safety_bias_coverage_threshold").drop_nulls().median().alias("value_safety_bias_coverage_threshold"),
        pl.col("value_safety_bias_coverage_pass").max().alias("value_safety_bias_coverage_pass"),
        pl.col("value_safety_meta_margin_pass").max().alias("value_safety_meta_margin_pass"),
        pl.col("value_safety_best_all_mode").drop_nulls().first().alias("value_safety_best_all_mode"),
        pl.col("value_safety_reason").drop_nulls().first().alias("value_safety_reason"),
        pl.col("v129_wf_enabled").max().alias("v129_wf_enabled"),
        pl.col("v129_wf_available").max().alias("v129_wf_available"),
        pl.col("v129_wf_folds").drop_nulls().first().alias("v129_wf_folds"),
        pl.col("v129_wf_win_rate").drop_nulls().first().alias("v129_wf_win_rate"),
        pl.col("v129_wf_median_gain").drop_nulls().first().alias("v129_wf_median_gain"),
        pl.col("v129_wf_worst_gain").drop_nulls().first().alias("v129_wf_worst_gain"),
        pl.col("v129_wf_weighted_gain").drop_nulls().first().alias("v129_wf_weighted_gain"),
        pl.col("v129_wf_weighted_utility_gain").drop_nulls().first().alias("v129_wf_weighted_utility_gain"),
        pl.col("v129_wf_bias_worsen_max").drop_nulls().first().alias("v129_wf_bias_worsen_max"),
        pl.col("v129_wf_meta_folds").drop_nulls().first().alias("v129_wf_meta_folds"),
        pl.col("v129_wf_reason").drop_nulls().first().alias("v129_wf_reason"),
    ).row(0, named=True)
    den = float(sums["den"] or 0.0)
    if den <= 0:
        return {}
    def scenario(ae: str, se: str) -> dict[str, float]:
        return {"wmape": float(sums[ae] or 0.0) / den, "bias": float(sums[se] or 0.0) / den}

    daily = (
        x.group_by("ds")
        .agg(
            pl.col("y").sum().alias("actual"),
            pl.col("yhat").sum().alias("final"),
            pl.col("_compare_v11").sum().alias("v11"),
            pl.col("_v12").sum().alias("v12"),
        )
        .sort("ds")
    )
    n = by_leaf.height
    top_driver = None
    if "meta_top_driver" in by_leaf.columns:
        td = (
            by_leaf.filter(pl.col("selected_v12") & pl.col("meta_top_driver").is_not_null())
            .group_by("meta_top_driver")
            .len()
            .sort("len", descending=True)
            .head(1)
        )
        if td.height:
            top_driver = str(td["meta_top_driver"][0])
    portfolio_mode = None
    if "meta_portfolio_mode" in by_leaf.columns:
        pm = by_leaf.group_by("meta_portfolio_mode").len().sort("len", descending=True).head(1)
        if pm.height and pm["meta_portfolio_mode"][0] is not None:
            portfolio_mode = str(pm["meta_portfolio_mode"][0])
    return {
        "scenarios": {
            "Final seleccionado": scenario("aef", "sef"),
            "v11 incumbent": scenario("ae11", "se11"),
            "v12 all": scenario("ae12", "se12"),
            "Oracle v11/v12 (no causal)": scenario("ae_oracle", "se_oracle"),
        },
        "n_active": n,
        "selected_v12": int(sums["selected_v12"] or 0),
        "oracle_v12": int(sums["oracle_v12"] or 0),
        "selector_gap": scenario("aef", "sef")["wmape"] - scenario("ae_oracle", "se_oracle")["wmape"],
        "cal_factor_median": float(sums["cal_factor"]) if sums["cal_factor"] is not None else None,
        "shape_applied_pct": float(sums["shape_applied"] or 0) / max(n, 1),
        "cal_applied_pct": float(sums["cal_applied"] or 0) / max(n, 1),
        "meta_probability_median": float(sums["meta_probability"]) if sums["meta_probability"] is not None else None,
        "meta_probability_selected_median": float(sums["meta_probability_selected"]) if sums["meta_probability_selected"] is not None else None,
        "meta_available_pct": float(sums["meta_available"] or 0) / max(n, 1),
        "meta_top_driver": top_driver,
        "meta_threshold": (
            float(sums["meta_threshold"])
            if sums["meta_threshold"] is not None
            else float(getattr(settings, "V12_META_SELECTOR_THRESHOLD", 0.55))
        ),
        "meta_portfolio_mode": portfolio_mode,
        "meta_bias_guard_pass_pct": float(sums["meta_bias_guard_pass"] or 0) / max(n, 1),
        "meta_policy_available_pct": float(sums["meta_policy_available"] or 0) / max(n, 1),
        "meta_policy_utility_gain": float(sums["meta_policy_utility_gain"]) if sums["meta_policy_utility_gain"] is not None else None,
        "value_safety_dominance_pass": bool(sums["value_safety_dominance_pass"]),
        "value_safety_recent_confirmations": int(sums["value_safety_recent_confirmations"] or 0),
        "value_safety_recent_blocks": int(sums["value_safety_recent_blocks"] or 0),
        "value_safety_bias_coverage": float(sums["value_safety_bias_coverage"]) if sums["value_safety_bias_coverage"] is not None else None,
        "value_safety_bias_coverage_threshold": float(sums["value_safety_bias_coverage_threshold"]) if sums["value_safety_bias_coverage_threshold"] is not None else None,
        "value_safety_bias_coverage_pass": bool(sums["value_safety_bias_coverage_pass"]),
        "value_safety_meta_margin_pass": bool(sums["value_safety_meta_margin_pass"]),
        "value_safety_best_all_mode": str(sums["value_safety_best_all_mode"]) if sums["value_safety_best_all_mode"] is not None else None,
        "value_safety_reason": str(sums["value_safety_reason"]) if sums["value_safety_reason"] is not None else None,
        "v129_wf_enabled": bool(sums["v129_wf_enabled"]),
        "v129_wf_available": bool(sums["v129_wf_available"]),
        "v129_wf_folds": int(sums["v129_wf_folds"] or 0),
        "v129_wf_win_rate": float(sums["v129_wf_win_rate"]) if sums["v129_wf_win_rate"] is not None else None,
        "v129_wf_median_gain": float(sums["v129_wf_median_gain"]) if sums["v129_wf_median_gain"] is not None else None,
        "v129_wf_worst_gain": float(sums["v129_wf_worst_gain"]) if sums["v129_wf_worst_gain"] is not None else None,
        "v129_wf_weighted_gain": float(sums["v129_wf_weighted_gain"]) if sums["v129_wf_weighted_gain"] is not None else None,
        "v129_wf_weighted_utility_gain": float(sums["v129_wf_weighted_utility_gain"]) if sums["v129_wf_weighted_utility_gain"] is not None else None,
        "v129_wf_bias_worsen_max": float(sums["v129_wf_bias_worsen_max"]) if sums["v129_wf_bias_worsen_max"] is not None else None,
        "v129_wf_meta_folds": int(sums["v129_wf_meta_folds"] or 0),
        "v129_wf_reason": str(sums["v129_wf_reason"]) if sums["v129_wf_reason"] is not None else None,
        "daily": {c: daily[c].to_list() for c in daily.columns},
    }

def prepare_dashboard_state_fast(
    *,
    index: dict[str, Any],
    metrics: pl.DataFrame,
    ranking_metrics: pl.DataFrame | None = None,
    unidad: str,
    freq: str,
    seccion: str,
    store: str | None,
    sku: str | None,
    cutoff_date: dt.date,
    forecast_path: str | Path | None = None,
    df_daily: pl.DataFrame | None = None,
) -> DashboardView:
    """Path rápido: index + metrics en memoria, 1 serie (pre-cargada o leída)."""
    selected_id = settings.make_unique_id(seccion, store=store, sku=sku)
    node_kind = _node_kind(store, sku)
    label_map: dict[str, str] = index.get("label_map") or {}
    desc_map: dict[str, str] = index.get("desc_map") or {}
    has_value = bool(index.get("has_value"))
    n_spine = int((index.get("n_spine_by_sec") or {}).get(seccion, 0) or 0)

    if df_daily is None:
        fpath = Path(forecast_path) if forecast_path else None
        df_daily = artifacts.load_series(
            selected_id,
            fpath,
            unidad=unidad,
            has_value=has_value,
        )

    horizons = _horizons_from_index(index, seccion, df_daily)
    test_start = horizons.get("test_start")
    test_end = horizons.get("test_end")
    ranking_days = (
        (test_end - test_start).days + 1
        if test_start is not None and test_end is not None
        else int(getattr(settings, "METRIC_HORIZON_DAYS", 28))
    )
    fcst_start = horizons.get("forecast_start")
    fcst_end = horizons.get("forecast_end")
    train_end = horizons.get("train_end")

    ds_min, ds_max = backend.ds_range(df_daily)
    # El corte contractual es train_end DE LA SECCIÓN. build_chart_series
    # dibuja OOS en fechas > cutoff, por lo que su primera fecha es test_start.
    cutoff = train_end or cutoff_date
    if ds_min and ds_max and not (ds_min <= cutoff <= ds_max):
        cutoff = ds_min

    df_view = backend.aggregate_temporal(df_daily, freq)
    has_period = "period_type" in df_view.columns

    ranking_source = metrics if ranking_metrics is None else ranking_metrics
    ranking_tiendas = _ranking_from_metrics(
        ranking_source,
        seccion=seccion,
        unidad=unidad,
        axis="store",
        fixed_peer=sku,
        exclude=None,
        desc_map=desc_map,
        n_spine=ranking_days or 1,
        selected_code=store,
    )
    ranking_skus = _ranking_from_metrics(
        ranking_source,
        seccion=seccion,
        unidad=unidad,
        axis="sku",
        fixed_peer=store,
        exclude=None,
        desc_map=desc_map,
        n_spine=ranking_days or 1,
        selected_code=sku,
    )
    ranking_tiendas = _sync_selected_leaf_metric(
        ranking_tiendas, ranking_source, seccion=seccion, unidad=unidad,
        store=store, sku=sku, axis="store",
    )
    ranking_skus = _sync_selected_leaf_metric(
        ranking_skus, ranking_source, seccion=seccion, unidad=unidad,
        store=store, sku=sku, axis="sku",
    )
    ranking_tiendas = _pin_selected(ranking_tiendas, store)
    ranking_skus = _pin_selected(ranking_skus, sku)

    # Hot path: ``metrics.parquet`` ya contiene la métrica OOS bottom-up
    # Active oficial. Recalcularla cargando todas las hojas de una sección es
    # lento y, además, comparar contra la serie padre agregada mezcla
    # definiciones. El diagnóstico v11/v12 queda bajo demanda en dashboard.py.
    metrics_io = _official_oos_metrics_from_artifact(
        metrics, selected_id=selected_id, unidad=unidad
    )
    metrics_28 = backend.metrics_rolling28(df_view)
    model_compare = None
    consistency_warnings = (
        []
        if metrics_io
        else [
            f"No existe métrica OOS precalculada para {selected_id} / {unidad}. "
            "Regenera los artefactos del dashboard."
        ]
    )

    has_rolling28 = (
        "yhat28" in df_daily.columns
        and df_daily.filter(pl.col("yhat28").is_not_null()).height > 0
    )

    store_context: str | None = None
    if store is not None and sku is not None:
        store_only_id = settings.make_unique_id(seccion, store=store)
        store_context = label_map.get(
            store_only_id, settings.display_label(store_only_id)
        )

    chart = backend.build_chart_series(
        df_view,
        test_end or cutoff,
        fcst_start or (cutoff + dt.timedelta(days=1)),
        fcst_end or (cutoff + dt.timedelta(days=28)),
        cutoff,
        has_period,
    )
    detail = backend.format_detail_display(backend.detail_view(df_view))
    label = label_map.get(selected_id, settings.display_label(selected_id))

    return DashboardView(
        selected_id=selected_id,
        seccion=seccion,
        store=store,
        sku=sku,
        node_kind=node_kind,
        label=label,
        unidad=unidad,
        freq=freq,
        horizons=horizons,
        ranking_tiendas=ranking_tiendas,
        ranking_skus=ranking_skus,
        n_spine=n_spine,
        ranking_days=ranking_days,
        metrics=metrics_io,
        metrics_28=metrics_28,
        has_rolling28=has_rolling28,
        store_context=store_context,
        chart=chart,
        detail=detail,
        ds_min=ds_min,
        ds_max=ds_max,
        cutoff=cutoff,
        consistency_warnings=consistency_warnings,
        model_compare=model_compare,
        has_value_cols=has_value,
    )


def _aggregate_pure_sku(
    unit_df: pl.DataFrame, seccion: str, sku: str
) -> pl.DataFrame:
    prefix = f"{seccion}||"
    suffix = f"||S:{sku}"
    leaves = unit_df.filter(
        pl.col("unique_id").str.starts_with(prefix)
        & pl.col("unique_id").str.ends_with(suffix)
        & pl.col("unique_id").str.contains(r"\|\|T:", literal=False)
    )
    if leaves.height == 0:
        return leaves
    aggs = [pl.col("y").sum(), pl.col("yhat").sum()]
    if "yhat28" in leaves.columns:
        aggs.append(pl.col("yhat28").sum())
    if "value" in leaves.columns:
        aggs.append(pl.col("value").sum())
    if "valuehat" in leaves.columns:
        aggs.append(pl.col("valuehat").sum())
    if "valuehat28" in leaves.columns:
        aggs.append(pl.col("valuehat28").sum())
    extra = [
        c
        for c in (
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
        )
        if c in leaves.columns
    ]
    for c in extra:
        aggs.append(pl.col(c).first())
    return (
        leaves.group_by("ds")
        .agg(aggs)
        .with_columns(
            pl.lit(settings.make_unique_id(seccion, sku=sku)).alias("unique_id")
        )
        .sort("ds")
    )


def prepare_dashboard_state(
    res_df: pl.DataFrame,
    *,
    unidad: str,
    freq: str,
    seccion: str,
    store: str | None,
    sku: str | None,
    cutoff_date: dt.date,
    all_ids: list[str] | None = None,
    label_map: dict[str, str] | None = None,
    desc_map: dict[str, str] | None = None,
    tabla_base: pl.DataFrame | None = None,
    n_spine: int | None = None,
) -> DashboardView:
    """Legacy: cálculos sobre el DataFrame completo (solo si no hay artefactos)."""
    selected_id = settings.make_unique_id(seccion, store=store, sku=sku)
    node_kind = _node_kind(store, sku)

    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    has_period = "period_type" in cols

    unit_df = backend.prepare_unit_df(res_df, unidad, has_value)
    if label_map is None or desc_map is None:
        label_map, desc_map = backend.build_label_maps(unit_df)
    if all_ids is None:
        all_ids = res_df["unique_id"].unique().to_list()

    if store is None and sku is not None:
        df_daily = _aggregate_pure_sku(unit_df, seccion, sku)
    else:
        df_daily = backend.filter_series(unit_df, selected_id)

    horizons = backend.resolve_horizons(df_daily, seccion, cols)
    train_end = horizons["train_end"]
    test_start = horizons["test_start"]
    test_end = horizons["test_end"]
    ranking_days = (
        (test_end - test_start).days + 1
        if test_start is not None and test_end is not None
        else int(getattr(settings, "METRIC_HORIZON_DAYS", 28))
    )
    fcst_start = horizons["forecast_start"]
    fcst_end = horizons["forecast_end"]

    ds_min, ds_max = backend.ds_range(df_daily)
    cutoff = train_end or cutoff_date
    if ds_min and ds_max and not (ds_min <= cutoff <= ds_max):
        cutoff = ds_min

    df_view = backend.aggregate_temporal(df_daily, freq)

    if n_spine is None or tabla_base is None:
        hz_spine = settings.section_horizons(seccion)
        n_spine_calc = (hz_spine["forecast_end"] - hz_spine["train_start"]).days + 1
        candidatos = [
            uid for uid in all_ids if uid == seccion or uid.startswith(f"{seccion}||")
        ]
        n_data = backend.spine_n_fechas(unit_df, candidatos)
        if n_data > n_spine_calc:
            n_spine_calc = n_data
        if n_spine is None:
            n_spine = n_spine_calc
        if tabla_base is None:
            # Rankings reportan wMAPE out-of-sample
            tabla_base = backend.wmape_por_id(
                candidatos,
                unit_df,
                n_fechas_spine=ranking_days,
                period_types=["out_sample"],
            )

    ranking_tiendas = backend.ranking_table(
        tabla_base,
        seccion=seccion,
        axis="store",
        fixed_peer=sku,
        exclude=None,
        desc_map=desc_map,
        unidad=unidad,
        selected_code=store,
    )
    ranking_skus = backend.ranking_table(
        tabla_base,
        seccion=seccion,
        axis="sku",
        fixed_peer=store,
        exclude=None,
        desc_map=desc_map,
        unidad=unidad,
        selected_code=sku,
    )

    # Diagnóstico v12.7 sobre las hojas del alcance, usando la unidad ya normalizada.
    leaves_scope = unit_df.filter(pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
    if store is not None:
        leaves_scope = leaves_scope.filter(pl.col("unique_id").str.contains(f"||T:{store}||", literal=True))
    if sku is not None:
        leaves_scope = leaves_scope.filter(pl.col("unique_id").str.ends_with(f"||S:{sku}"))
    model_compare = _v12_model_compare(leaves_scope)

    # Métricas oficiales SIEMPRE bottom-up desde hojas SKU+tienda.
    metrics = backend.metrics_in_out_bottom_up(
        unit_df,
        seccion=seccion,
        store=store,
        sku=sku,
        cutoff=cutoff,
        test_end=test_end or cutoff,
    )
    metrics_28 = backend.metrics_rolling28(df_view)

    has_rolling28 = (
        "yhat28" in df_daily.columns
        and df_daily.filter(pl.col("yhat28").is_not_null()).height > 0
    )

    store_context: str | None = None
    if store is not None and sku is not None:
        store_only_id = settings.make_unique_id(seccion, store=store)
        store_context = (label_map or {}).get(
            store_only_id, settings.display_label(store_only_id)
        )

    chart = backend.build_chart_series(
        df_view,
        test_end or cutoff,
        fcst_start or (cutoff + dt.timedelta(days=1)),
        fcst_end or (cutoff + dt.timedelta(days=28)),
        cutoff,
        has_period,
    )

    detail = backend.format_detail_display(backend.detail_view(df_view))
    label = label_map.get(selected_id, settings.display_label(selected_id))

    return DashboardView(
        selected_id=selected_id,
        seccion=seccion,
        store=store,
        sku=sku,
        node_kind=node_kind,
        label=label,
        unidad=unidad,
        freq=freq,
        horizons=horizons,
        ranking_tiendas=ranking_tiendas,
        ranking_skus=ranking_skus,
        n_spine=n_spine,
        ranking_days=ranking_days,
        metrics=metrics,
        metrics_28=metrics_28,
        has_rolling28=has_rolling28,
        store_context=store_context,
        chart=chart,
        detail=detail,
        ds_min=ds_min,
        ds_max=ds_max,
        cutoff=cutoff,
        consistency_warnings=[],
        model_compare=model_compare,
        has_value_cols=has_value,
    )
