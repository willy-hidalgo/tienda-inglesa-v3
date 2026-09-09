"""Reporte de aceptación estadística v12.x sin reentrenar el modelo.

Lee exclusivamente forecast.parquet + dashboard/metrics.parquet ya generados.
Uso:
    uv run python -m app.forecasting_acceptance_report
"""
from __future__ import annotations

import argparse

from pathlib import Path

import polars as pl

import settings
from app import dashboard_artifacts as artifacts
from app.forecasting.run_status import validation_errors as run_status_errors


def _parts(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
        pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("store"),
        pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("sku"),
    )


def _pct(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{100.0 * float(x):.2f}%"


def _print_section_summary(metrics: pl.DataFrame) -> None:
    sec = metrics.filter(
        pl.col("store").is_null() & pl.col("sku").is_null()
    ).select(
        "unidad", "seccion", "wmape", "bias", "sum_abs_error", "sum_abs_y",
        "n_leaf_active", "n_leaf_sparse", "n_leaf_zero",
        "zero_forecast_sum", "zero_abs_error_sum",
    ).sort(["unidad", "seccion"])
    print("\n=== 1) WMAPE / BIAS OOS POR SECCIÓN (ACTIVE, bottom-up) ===")
    for r in sec.iter_rows(named=True):
        print(
            f"{r['unidad']:10s} | sec={r['seccion']:>3s} | "
            f"wMAPE={_pct(r['wmape']):>9s} | BIAS={_pct(r['bias']):>9s} | "
            f"active={int(r['n_leaf_active'] or 0):5d} sparse={int(r['n_leaf_sparse'] or 0):5d} "
            f"zero={int(r['n_leaf_zero'] or 0):5d}"
        )


def _print_all_cohort_summary(metrics: pl.DataFrame) -> None:
    leaves = metrics.filter(pl.col("store").is_not_null() & pl.col("sku").is_not_null())
    all_sec = (
        leaves.group_by(["unidad", "seccion"])
        .agg(
            pl.col("sum_abs_error").sum().alias("sum_abs_error"),
            pl.col("sum_abs_y").sum().alias("sum_abs_y"),
            pl.col("sum_signed_error").sum().alias("sum_signed_error"),
            pl.col("zero_forecast_sum").sum().alias("zero_forecast_sum"),
            pl.col("zero_abs_error_sum").sum().alias("zero_abs_error_sum"),
        )
        .with_columns(
            pl.when(pl.col("sum_abs_y") > 0)
            .then(pl.col("sum_abs_error") / pl.col("sum_abs_y"))
            .otherwise(None).alias("wmape_all"),
            pl.when(pl.col("sum_abs_y") > 0)
            .then(pl.col("sum_signed_error") / pl.col("sum_abs_y"))
            .otherwise(None).alias("bias_all"),
        )
        .sort(["unidad", "seccion"])
    )
    print("\n=== 2) TODAS LAS HOJAS (active+sparse+zero) ===")
    for r in all_sec.iter_rows(named=True):
        print(
            f"{r['unidad']:10s} | sec={r['seccion']:>3s} | "
            f"wMAPE All={_pct(r['wmape_all']):>9s} | BIAS All={_pct(r['bias_all']):>9s} | "
            f"forecast sin demanda={float(r['zero_forecast_sum'] or 0):,.2f} | "
            f"impacto zero-demand={float(r['zero_abs_error_sum'] or 0):,.2f}"
        )


def _print_leaf_distribution(metrics: pl.DataFrame) -> None:
    active = metrics.filter(
        pl.col("store").is_not_null()
        & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
        & pl.col("wmape").is_not_null()
    )
    print("\n=== 3) DISTRIBUCIÓN wMAPE SKU+TIENDA ACTIVE ===")
    for unidad in ("Unidades", "Valor ($)"):
        u = active.filter(pl.col("unidad") == unidad)
        if u.height == 0:
            continue
        q = u.select(
            pl.col("wmape").median().alias("p50"),
            pl.col("wmape").quantile(0.75, interpolation="nearest").alias("p75"),
            pl.col("wmape").quantile(0.90, interpolation="nearest").alias("p90"),
            pl.col("wmape").quantile(0.95, interpolation="nearest").alias("p95"),
            pl.col("wmape").quantile(0.99, interpolation="nearest").alias("p99"),
            pl.col("wmape").max().alias("max"),
            (pl.col("wmape") > 1.0).sum().alias("gt100"),
            (pl.col("wmape") > 2.0).sum().alias("gt200"),
            (pl.col("wmape") > 5.0).sum().alias("gt500"),
        ).row(0, named=True)
        print(
            f"{unidad:10s} | n={u.height:,} | P50={_pct(q['p50'])} P75={_pct(q['p75'])} "
            f"P90={_pct(q['p90'])} P95={_pct(q['p95'])} P99={_pct(q['p99'])} "
            f"MAX={_pct(q['max'])} | >100%={q['gt100']:,} >200%={q['gt200']:,} >500%={q['gt500']:,}"
        )


def _print_store_summary(metrics: pl.DataFrame) -> None:
    stores = metrics.filter(
        pl.col("store").is_not_null() & pl.col("sku").is_null()
    ).sort(["unidad", "wmape"], descending=[False, True])
    print("\n=== 4) PEORES TIENDAS POR wMAPE OOS ACTIVE ===")
    for unidad in ("Unidades", "Valor ($)"):
        print(f"\n{unidad}:")
        x = stores.filter(pl.col("unidad") == unidad).head(10)
        for r in x.iter_rows(named=True):
            print(
                f"  sec={r['seccion']} store={r['store']} "
                f"wMAPE={_pct(r['wmape'])} BIAS={_pct(r['bias'])}"
            )


def _print_error_contributors(metrics: pl.DataFrame) -> None:
    leaves = metrics.filter(
        pl.col("store").is_not_null()
        & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
    )
    print("\n=== 5) MAYORES CONTRIBUYENTES AL ERROR ABSOLUTO ===")
    for unidad in ("Unidades", "Valor ($)"):
        print(f"\n{unidad}:")
        x = (
            leaves.filter(pl.col("unidad") == unidad)
            .sort("sum_abs_error", descending=True)
            .head(15)
        )
        for r in x.iter_rows(named=True):
            print(
                f"  {r['unique_id']:<32s} wMAPE={_pct(r['wmape']):>9s} "
                f"BIAS={_pct(r['bias']):>9s} abs_error={float(r['sum_abs_error'] or 0):,.2f} "
                f"vol={float(r['sum_abs_y'] or 0):,.2f}"
            )


def _print_methods(forecast: Path, metrics: pl.DataFrame) -> None:
    schema = pl.scan_parquet(forecast).collect_schema().names()
    wanted = [
        "unique_id", "period_type", "leaf_level_method_y",
        "leaf_level_method_value", "ses_guard_status_y", "ses_guard_status_value",
        "parent_driver_mode_y", "parent_driver_mode_value",
        "driver_level_factor_y", "driver_level_factor_value",
        "sku_seasonal_multiplier_y", "sku_seasonal_multiplier_value",
        "leaf_model_family_y", "leaf_model_family_value",
        "v12_selected_y", "v12_selected_value",
        "v12_validation_wmape_v11_y", "v12_validation_wmape_v12_y",
        "v12_validation_wmape_v11_value", "v12_validation_wmape_v12_value",
        "v12_validation_blocks_y", "v12_validation_blocks_value",
        "v12_validation_wins_y", "v12_validation_wins_value",
        "v12_validation_improvement_y", "v12_validation_improvement_value",
        "v12_section_portfolio_gate_y", "v12_section_portfolio_gate_value",
        "v12_section_portfolio_gate_joint", "v12_joint_target_preselect",
    ]
    cols = [c for c in wanted if c in schema]
    if not {"unique_id", "period_type"}.issubset(cols):
        return
    methods = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(cols)
        .unique(subset=["unique_id"])
        .collect()
    )
    print("\n=== 6) MÉTODO DE NIVEL USADO EN OOS ===")
    unit_defs = (
        ("Unidades", "leaf_level_method_y"),
        ("Valor ($)", "leaf_level_method_value"),
    )
    for unidad, col in unit_defs:
        if col not in methods.columns:
            continue
        active_ids = metrics.filter(
            (pl.col("unidad") == unidad)
            & pl.col("store").is_not_null()
            & pl.col("sku").is_not_null()
            & pl.col("metric_active").fill_null(False)
        ).select("unique_id")
        x = methods.join(active_ids, on="unique_id", how="semi")
        counts = x.group_by(col).len().sort("len", descending=True)
        print(f"\n{unidad} (solo hojas Active):")
        for r in counts.iter_rows(named=True):
            print(f"  {str(r[col]):22s} {int(r['len']):,}")

    for col in ("ses_guard_status_y", "ses_guard_status_value"):
        if col in methods.columns:
            print(f"\n{col}:")
            counts = methods.group_by(col).len().sort("len", descending=True)
            for r in counts.iter_rows(named=True):
                print(f"  {str(r[col]):30s} {int(r['len']):,}")



def _oos_leaf_methods(forecast: Path) -> pl.DataFrame:
    schema = pl.scan_parquet(forecast).collect_schema().names()
    wanted = [
        "unique_id", "period_type", "leaf_level_method_y",
        "leaf_level_method_value", "ses_guard_status_y", "ses_guard_status_value",
        "parent_driver_mode_y", "parent_driver_mode_value",
        "driver_level_factor_y", "driver_level_factor_value",
        "sku_seasonal_multiplier_y", "sku_seasonal_multiplier_value",
        "leaf_model_family_y", "leaf_model_family_value",
        "v12_selected_y", "v12_selected_value",
        "v12_validation_wmape_v11_y", "v12_validation_wmape_v12_y",
        "v12_validation_wmape_v11_value", "v12_validation_wmape_v12_value",
        "v12_validation_blocks_y", "v12_validation_blocks_value",
        "v12_validation_wins_y", "v12_validation_wins_value",
        "v12_validation_improvement_y", "v12_validation_improvement_value",
        "v12_validation_recent_improvement_y", "v12_validation_recent_improvement_value",
        "v12_validation_utility_improvement_y", "v12_validation_utility_improvement_value",
        "v12_validation_recent_utility_improvement_y", "v12_validation_recent_utility_improvement_value",
        "v12_validation_utility_mean_y", "v12_validation_utility_mean_value",
        "v12_validation_utility_std_y", "v12_validation_utility_std_value",
        "v12_section_portfolio_gate_y", "v12_section_portfolio_gate_value",
        "v12_section_portfolio_gate_joint", "v12_joint_target_preselect",
        "v12_section_portfolio_wins_y", "v12_section_portfolio_wins_value",
        "v12_section_portfolio_gain_y", "v12_section_portfolio_gain_value",
        "v12_section_portfolio_recent_gain_y", "v12_section_portfolio_recent_gain_value",
    ]
    cols = [c for c in wanted if c in schema]
    if not {"unique_id", "period_type"}.issubset(cols):
        return pl.DataFrame(schema={"unique_id": pl.Utf8})
    return (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(cols)
        .unique(subset=["unique_id"])
        .collect()
    )


def _print_method_quality(forecast: Path, metrics: pl.DataFrame) -> None:
    methods = _oos_leaf_methods(forecast)
    if methods.height == 0:
        return
    leaves = metrics.filter(
        pl.col("store").is_not_null()
        & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
    )
    print("\n=== 7) CALIDAD OOS POR MÉTODO DE NIVEL (ACTIVE, bottom-up) ===")
    for unidad, method_col in (
        ("Unidades", "leaf_level_method_y"),
        ("Valor ($)", "leaf_level_method_value"),
    ):
        if method_col not in methods.columns:
            continue
        x = leaves.filter(pl.col("unidad") == unidad).join(
            methods.select("unique_id", method_col), on="unique_id", how="left"
        )
        g = (
            x.group_by(method_col)
            .agg(
                pl.len().alias("n"),
                pl.col("sum_abs_error").sum().alias("ae"),
                pl.col("sum_abs_y").sum().alias("den"),
                pl.col("sum_signed_error").sum().alias("se"),
                pl.col("wmape").median().alias("median_wmape"),
            )
            .with_columns(
                (pl.col("ae") / pl.col("den").replace(0.0, None)).alias("wmape_bu"),
                (pl.col("se") / pl.col("den").replace(0.0, None)).alias("bias_bu"),
            )
            .sort("wmape_bu")
        )
        print(f"\n{unidad}:")
        for r in g.iter_rows(named=True):
            print(
                f"  {str(r[method_col]):22s} n={int(r['n']):5d} "
                f"wMAPE BU={_pct(r['wmape_bu']):>9s} BIAS BU={_pct(r['bias_bu']):>9s} "
                f"median={_pct(r['median_wmape']):>9s}"
            )


def _print_sales_frequency_quality(metrics: pl.DataFrame) -> None:
    leaves = metrics.filter(
        pl.col("store").is_not_null()
        & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
        & pl.col("n_with_sales").is_not_null()
    ).with_columns(
        pl.when(pl.col("n_with_sales") <= 9).then(pl.lit("07-09"))
        .when(pl.col("n_with_sales") <= 13).then(pl.lit("10-13"))
        .when(pl.col("n_with_sales") <= 20).then(pl.lit("14-20"))
        .otherwise(pl.lit("21-28")).alias("sales_days_bin")
    )
    print("\n=== 8) CALIDAD OOS SEGÚN DÍAS CON VENTA (ACTIVE) ===")
    for unidad in ("Unidades", "Valor ($)"):
        x = leaves.filter(pl.col("unidad") == unidad)
        g = (
            x.group_by("sales_days_bin")
            .agg(
                pl.len().alias("n"),
                pl.col("sum_abs_error").sum().alias("ae"),
                pl.col("sum_abs_y").sum().alias("den"),
                pl.col("sum_signed_error").sum().alias("se"),
                pl.col("wmape").median().alias("median_wmape"),
            )
            .with_columns(
                (pl.col("ae") / pl.col("den").replace(0.0, None)).alias("wmape_bu"),
                (pl.col("se") / pl.col("den").replace(0.0, None)).alias("bias_bu"),
            )
            .sort("sales_days_bin")
        )
        print(f"\n{unidad}:")
        for r in g.iter_rows(named=True):
            print(
                f"  días={r['sales_days_bin']} n={int(r['n']):5d} "
                f"wMAPE BU={_pct(r['wmape_bu']):>9s} BIAS BU={_pct(r['bias_bu']):>9s} "
                f"median={_pct(r['median_wmape']):>9s}"
            )

def _print_driver_mode_quality(forecast: Path, metrics: pl.DataFrame) -> None:
    methods = _oos_leaf_methods(forecast)
    if methods.height == 0:
        return
    leaves = metrics.filter(
        pl.col("store").is_not_null()
        & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
    )
    print("\n=== 9) CALIDAD OOS POR MODO DEL DRIVER PADRE (ACTIVE, bottom-up) ===")
    for unidad, mode_col, level_col in (
        ("Unidades", "parent_driver_mode_y", "driver_level_factor_y"),
        ("Valor ($)", "parent_driver_mode_value", "driver_level_factor_value"),
    ):
        if mode_col not in methods.columns:
            continue
        cols = ["unique_id", mode_col] + ([level_col] if level_col in methods.columns else [])
        x = leaves.filter(pl.col("unidad") == unidad).join(
            methods.select(cols), on="unique_id", how="left"
        )
        aggs = [
            pl.len().alias("n"),
            pl.col("sum_abs_error").sum().alias("ae"),
            pl.col("sum_abs_y").sum().alias("den"),
            pl.col("sum_signed_error").sum().alias("se"),
            pl.col("wmape").median().alias("median_wmape"),
        ]
        if level_col in x.columns:
            aggs.append(pl.col(level_col).median().alias("median_level_factor"))
        g = (
            x.group_by(["seccion", mode_col]).agg(*aggs)
            .with_columns(
                (pl.col("ae") / pl.col("den").replace(0.0, None)).alias("wmape_bu"),
                (pl.col("se") / pl.col("den").replace(0.0, None)).alias("bias_bu"),
            )
            .sort(["seccion", "wmape_bu"])
        )
        print(f"\n{unidad}:")
        for r in g.iter_rows(named=True):
            extra = (
                f" level-factor-med={float(r.get('median_level_factor') or 1.0):.3f}"
                if "median_level_factor" in r else ""
            )
            print(
                f"  sec={str(r['seccion']):>3s} {str(r[mode_col]):14s} n={int(r['n']):5d} "
                f"wMAPE BU={_pct(r['wmape_bu']):>9s} BIAS BU={_pct(r['bias_bu']):>9s} "
                f"median={_pct(r['median_wmape']):>9s}{extra}"
            )



def _print_sku_seasonal_quality(forecast: Path, metrics: pl.DataFrame) -> None:
    methods = _oos_leaf_methods(forecast)
    if methods.height == 0:
        return
    leaves = metrics.filter(
        pl.col("store").is_not_null()
        & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
    )
    print("\n=== 10) CALIDAD OOS DEL FACTOR ESTACIONAL SKU 364D (ACTIVE, bottom-up) ===")
    for unidad, method_col, mult_col in (
        ("Unidades", "leaf_level_method_y", "sku_seasonal_multiplier_y"),
        ("Valor ($)", "leaf_level_method_value", "sku_seasonal_multiplier_value"),
    ):
        if method_col not in methods.columns or mult_col not in methods.columns:
            continue
        x = leaves.filter(pl.col("unidad") == unidad).join(
            methods.select("unique_id", method_col, mult_col), on="unique_id", how="left"
        )
        g = (
            x.with_columns(
                (pl.col(method_col) == "sku_yoy_seasonal").alias("seasonal_selected")
            )
            .group_by(["seccion", "seasonal_selected"])
            .agg(
                pl.len().alias("n"),
                pl.col("sum_abs_error").sum().alias("ae"),
                pl.col("sum_abs_y").sum().alias("den"),
                pl.col("sum_signed_error").sum().alias("se"),
                pl.col(mult_col).median().alias("median_multiplier"),
            )
            .with_columns(
                (pl.col("ae") / pl.col("den").replace(0.0, None)).alias("wmape_bu"),
                (pl.col("se") / pl.col("den").replace(0.0, None)).alias("bias_bu"),
            )
            .sort(["seccion", "seasonal_selected"])
        )
        print(f"\n{unidad}:")
        for r in g.iter_rows(named=True):
            label = "sku_yoy_seasonal" if r["seasonal_selected"] else "other"
            print(
                f"  sec={str(r['seccion']):>3s} {label:18s} n={int(r['n']):5d} "
                f"wMAPE BU={_pct(r['wmape_bu']):>9s} BIAS BU={_pct(r['bias_bu']):>9s} "
                f"mult-med={float(r['median_multiplier'] or 1.0):.3f}"
            )


def _print_v12_family_quality(forecast: Path, metrics: pl.DataFrame) -> None:
    methods = _oos_leaf_methods(forecast)
    if methods.height == 0 or "leaf_model_family_y" not in methods.columns:
        return
    leaves = metrics.filter(
        pl.col("store").is_not_null()
        & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
    )
    print("\n=== 11) CALIDAD OOS POR FAMILIA v11 vs v12 (ACTIVE, bottom-up) ===")
    for unidad, family_col, selected_col in (
        ("Unidades", "leaf_model_family_y", "v12_selected_y"),
        ("Valor ($)", "leaf_model_family_value", "v12_selected_value"),
    ):
        if family_col not in methods.columns:
            continue
        cols = ["unique_id", family_col] + ([selected_col] if selected_col in methods.columns else [])
        x = leaves.filter(pl.col("unidad") == unidad).join(
            methods.select(cols), on="unique_id", how="left"
        )
        g = (
            x.group_by(["seccion", family_col])
            .agg(
                pl.len().alias("n"),
                pl.col("sum_abs_error").sum().alias("ae"),
                pl.col("sum_abs_y").sum().alias("den"),
                pl.col("sum_signed_error").sum().alias("se"),
                pl.col("wmape").median().alias("median_wmape"),
            )
            .with_columns(
                (pl.col("ae") / pl.col("den").replace(0.0, None)).alias("wmape_bu"),
                (pl.col("se") / pl.col("den").replace(0.0, None)).alias("bias_bu"),
            )
            .sort(["seccion", "wmape_bu"])
        )
        print(f"\n{unidad}:")
        for r in g.iter_rows(named=True):
            print(
                f"  sec={str(r['seccion']):>3s} {str(r[family_col]):16s} n={int(r['n']):5d} "
                f"wMAPE BU={_pct(r['wmape_bu']):>9s} BIAS BU={_pct(r['bias_bu']):>9s} "
                f"median={_pct(r['median_wmape']):>9s}"
            )


def _counterfactual_bu(
    df: pl.DataFrame,
    actual_col: str,
    pred_col: str,
) -> pl.DataFrame:
    return (
        df.group_by("seccion")
        .agg(
            pl.when(pl.col(actual_col) > 0)
            .then((pl.col(actual_col) - pl.col(pred_col)).abs())
            .otherwise(0.0).sum().alias("ae"),
            pl.when(pl.col(actual_col) > 0)
            .then(pl.col(actual_col))
            .otherwise(0.0).sum().alias("den"),
            pl.when(pl.col(actual_col) > 0)
            .then(pl.col(pred_col) - pl.col(actual_col))
            .otherwise(0.0).sum().alias("se"),
        )
        .with_columns(
            (pl.col("ae") / pl.col("den").replace(0.0, None)).alias("wmape"),
            (pl.col("se") / pl.col("den").replace(0.0, None)).alias("bias"),
        )
    )


def _print_v12_counterfactual(forecast: Path, metrics: pl.DataFrame) -> None:
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "period_type", "y", "value", "yhat_raw", "valuehat_raw",
        "v11_yhat_raw_before_v12", "v11_valuehat_raw_before_v12",
        "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
    }
    if not required.issubset(schema):
        return
    active_y = metrics.filter(
        (pl.col("unidad") == "Unidades")
        & pl.col("store").is_not_null() & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
    ).select("unique_id")
    active_v = metrics.filter(
        (pl.col("unidad") == "Valor ($)")
        & pl.col("store").is_not_null() & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
    ).select("unique_id")
    base = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(
            "unique_id", "y", "value", "yhat_raw", "valuehat_raw",
            "v11_yhat_raw_before_v12", "v11_valuehat_raw_before_v12",
            "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
        )
        .collect()
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.col("v11_yhat_raw_before_v12").round(0).alias("_cf_v11_y"),
            pl.col("v11_valuehat_raw_before_v12").round(2).alias("_cf_v11_v"),
            pl.coalesce([pl.col("v12_candidate_yhat_raw"), pl.col("v11_yhat_raw_before_v12")]).round(0).alias("_cf_v12_y"),
            pl.coalesce([pl.col("v12_candidate_valuehat_raw"), pl.col("v11_valuehat_raw_before_v12")]).round(2).alias("_cf_v12_v"),
            pl.col("yhat_raw").round(0).alias("_cf_final_y"),
            pl.col("valuehat_raw").round(2).alias("_cf_final_v"),
        )
    )
    print("\n=== 12) A/B CONTRAFACTUAL OOS v12 (ACTIVE, bottom-up) ===")
    for unidad, active_ids, actual, incumbent, challenger, final in (
        ("Unidades", active_y, "y", "_cf_v11_y", "_cf_v12_y", "_cf_final_y"),
        ("Valor ($)", active_v, "value", "_cf_v11_v", "_cf_v12_v", "_cf_final_v"),
    ):
        x = base.join(active_ids, on="unique_id", how="semi")
        parts = []
        for label, pred in (("v11_incumbent", incumbent), ("v12_all", challenger), ("final_selected", final)):
            g = _counterfactual_bu(x, actual, pred).with_columns(pl.lit(label).alias("scenario"))
            parts.append(g)
        g = pl.concat(parts, how="vertical_relaxed").sort(["seccion", "scenario"])
        print(f"\n{unidad}:")
        for r in g.iter_rows(named=True):
            print(
                f"  sec={str(r['seccion']):>3s} {str(r['scenario']):15s} "
                f"wMAPE={_pct(r['wmape']):>9s} BIAS={_pct(r['bias']):>9s}"
            )


def _print_v12_selection_stability(forecast: Path, metrics: pl.DataFrame) -> None:
    methods = _oos_leaf_methods(forecast)
    required = {
        "unique_id", "v12_selected_y", "v12_selected_value",
        "v12_validation_blocks_y", "v12_validation_blocks_value",
        "v12_validation_wins_y", "v12_validation_wins_value",
        "v12_validation_improvement_y", "v12_validation_improvement_value",
        "v12_validation_recent_improvement_y", "v12_validation_recent_improvement_value",
        "v12_section_portfolio_gate_y", "v12_section_portfolio_gate_value",
        "v12_section_portfolio_gate_joint", "v12_joint_target_preselect",
        "v12_section_portfolio_gain_y", "v12_section_portfolio_gain_value",
        "v12_section_portfolio_recent_gain_y", "v12_section_portfolio_recent_gain_value",
    }
    if methods.height == 0 or not required.issubset(methods.columns):
        return
    print("\n=== 13) HISTÓRICO DE SELECCIÓN + LEGACY GATE DIAGNÓSTICO v12.9.7 ===")
    for unidad, selected, blocks, wins, gain, recent_gain, gate, port_gain, port_recent in (
        (
            "Unidades", "v12_selected_y", "v12_validation_blocks_y",
            "v12_validation_wins_y", "v12_validation_improvement_y",
            "v12_validation_recent_improvement_y", "v12_section_portfolio_gate_y",
            "v12_section_portfolio_gain_y", "v12_section_portfolio_recent_gain_y",
        ),
        (
            "Valor ($)", "v12_selected_value", "v12_validation_blocks_value",
            "v12_validation_wins_value", "v12_validation_improvement_value",
            "v12_validation_recent_improvement_value", "v12_section_portfolio_gate_value",
            "v12_section_portfolio_gain_value", "v12_section_portfolio_recent_gain_value",
        ),
    ):
        active_ids = metrics.filter(
            (pl.col("unidad") == unidad)
            & pl.col("store").is_not_null()
            & pl.col("sku").is_not_null()
            & pl.col("metric_active").fill_null(False)
        ).select("unique_id")
        x = (
            methods.join(active_ids, on="unique_id", how="semi")
            .with_columns(pl.col("unique_id").str.split("||").list.get(0).alias("seccion"))
        )
        total = x.group_by("seccion").agg(pl.len().alias("n"))
        sel = x.filter(pl.col(selected).fill_null(False))
        if sel.height:
            selected_stats = (
                sel.group_by("seccion")
                .agg(
                    pl.len().alias("selected_n"),
                    pl.col(blocks).median().alias("blocks_med"),
                    pl.col(wins).median().alias("wins_med"),
                    pl.col(gain).median().alias("gain_med"),
                    pl.col(recent_gain).median().alias("recent_gain_med"),
                    pl.col(gate).first().alias("portfolio_gate"),
                    pl.col(port_gain).first().alias("portfolio_gain"),
                    pl.col(port_recent).first().alias("portfolio_recent_gain"),
                )
            )
            g = total.join(selected_stats, on="seccion", how="left")
        else:
            g = total.with_columns(
                pl.lit(0).alias("selected_n"),
                pl.lit(None, dtype=pl.Float64).alias("blocks_med"),
                pl.lit(None, dtype=pl.Float64).alias("wins_med"),
                pl.lit(None, dtype=pl.Float64).alias("gain_med"),
                pl.lit(None, dtype=pl.Float64).alias("recent_gain_med"),
                pl.lit(False).alias("portfolio_gate"),
                pl.lit(None, dtype=pl.Float64).alias("portfolio_gain"),
                pl.lit(None, dtype=pl.Float64).alias("portfolio_recent_gain"),
            )
        g = g.sort("seccion")
        print(f"\n{unidad}:")
        for r in g.iter_rows(named=True):
            print(
                f"  sec={str(r['seccion']):>3s} selected={int(r['selected_n'] or 0):5d}/{int(r['n']):5d} "
                f"disc-blocks-med={float(r['blocks_med'] or 0):.1f} disc-wins-med={float(r['wins_med'] or 0):.1f} "
                f"discovery-gain-med={_pct(r['gain_med']):>9s} confirm-med={_pct(r['recent_gain_med']):>9s} | "
                f"portfolio={_pct(r['portfolio_gain']):>9s} recent={_pct(r['portfolio_recent_gain']):>9s} "
                f"legacy-gate(diag)={bool(r['portfolio_gate'])}"
            )



def _print_v12_selected_counterfactual(forecast: Path, metrics: pl.DataFrame) -> None:
    """Compare v11 vs v12 on the exact same causally selected Active leaves."""
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "period_type", "y", "value",
        "v11_yhat_raw_before_v12", "v11_valuehat_raw_before_v12",
        "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
        "v12_selected_y", "v12_selected_value",
    }
    if not required.issubset(schema):
        return
    base = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(
            "unique_id", "y", "value", "v12_selected_y", "v12_selected_value",
            pl.col("v11_yhat_raw_before_v12").round(0).alias("_v11_y"),
            pl.col("v11_valuehat_raw_before_v12").round(2).alias("_v11_v"),
            pl.col("v12_candidate_yhat_raw").round(0).alias("_v12_y"),
            pl.col("v12_candidate_valuehat_raw").round(2).alias("_v12_v"),
        )
        .collect()
        .with_columns(pl.col("unique_id").str.split("||").list.get(0).alias("seccion"))
    )
    print("\n=== 14) CONTRAFACTUAL SOBRE LAS MISMAS HOJAS SELECCIONADAS v12.8 ===")
    for unidad, active_unit, selected, actual, p11, p12 in (
        ("Unidades", "Unidades", "v12_selected_y", "y", "_v11_y", "_v12_y"),
        ("Valor ($)", "Valor ($)", "v12_selected_value", "value", "_v11_v", "_v12_v"),
    ):
        active_ids = metrics.filter(
            (pl.col("unidad") == active_unit)
            & pl.col("store").is_not_null() & pl.col("sku").is_not_null()
            & pl.col("metric_active").fill_null(False)
        ).select("unique_id")
        x = base.join(active_ids, on="unique_id", how="semi").filter(pl.col(selected).fill_null(False))
        print(f"\n{unidad}:")
        if x.height == 0:
            print("  sin hojas Active seleccionadas")
            continue
        n_by_sec = x.select("unique_id", "seccion").unique().group_by("seccion").len().rename({"len": "n"})
        g11 = _counterfactual_bu(x, actual, p11).rename({"wmape": "wmape11", "bias": "bias11"})
        g12 = _counterfactual_bu(x, actual, p12).rename({"wmape": "wmape12", "bias": "bias12"})
        g = (
            g11.join(g12, on="seccion", how="inner")
            .join(n_by_sec, on="seccion", how="left")
            .with_columns((pl.col("wmape11") - pl.col("wmape12")).alias("gain"))
            .sort("seccion")
        )
        for r in g.iter_rows(named=True):
            print(
                f"  sec={str(r['seccion']):>3s} n={int(r['n'] or 0):5d} | "
                f"v11={_pct(r['wmape11']):>9s} v12={_pct(r['wmape12']):>9s} "
                f"gain={_pct(r['gain']):>9s} | BIAS11={_pct(r['bias11']):>9s} BIAS12={_pct(r['bias12']):>9s}"
            )


def _print_v12_joint_consistency(forecast: Path) -> None:
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "period_type", "v12_selected_y", "v12_selected_value",
        "v12_occurrence_gate_open_y", "v12_occurrence_gate_open_value",
        "v12_section_portfolio_gate_joint",
    }
    if not required.issubset(schema):
        return
    x = (
        pl.scan_parquet(forecast)
        .filter(
            pl.col("period_type").is_in(["out_sample", "forecast_only"])
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .collect()
    )
    sel_mismatch = x.filter(
        pl.col("v12_selected_y").fill_null(False) != pl.col("v12_selected_value").fill_null(False)
    ).height
    gate_mismatch = x.filter(
        pl.col("v12_occurrence_gate_open_y").fill_null(False)
        != pl.col("v12_occurrence_gate_open_value").fill_null(False)
    ).height
    selected_y = x.filter(pl.col("v12_selected_y").fill_null(False)).get_column("unique_id").n_unique()
    selected_v = x.filter(pl.col("v12_selected_value").fill_null(False)).get_column("unique_id").n_unique()
    joint_gate = bool(x.select(pl.col("v12_section_portfolio_gate_joint").fill_null(False).any()).item())
    print("\n=== 15) SELECCIÓN TARGET-SPECIFIC v12.8 + OCCURRENCE COMPARTIDO ===")
    print(
        f"selected Qty={selected_y:,} | selected Valor={selected_v:,} | "
        f"selection differences esperadas={sel_mismatch:,} | occurrence-gate mismatches={gate_mismatch:,} | "
        f"joint diagnostic gate any={joint_gate}"
    )



def _print_v12_sku_total_ensemble_quality(forecast: Path) -> None:
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "ds", "period_type", "y", "value",
        "v12_sku_forecast_y", "v12_sku_forecast_value",
        "v12_sku_model_y", "v12_sku_model_value",
        "v12_sku_validation_wmape_y", "v12_sku_validation_wmape_value",
    }
    if not required.issubset(schema):
        return
    sku_day = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("sku"),
        )
        .group_by(["seccion", "sku", "ds"])
        .agg(
            pl.col("y").sum().alias("sku_actual_y"),
            pl.col("value").sum().alias("sku_actual_v"),
            pl.col("v12_sku_forecast_y").drop_nulls().first().alias("sku_fc_y"),
            pl.col("v12_sku_forecast_value").drop_nulls().first().alias("sku_fc_v"),
            pl.col("v12_sku_model_y").drop_nulls().first().alias("sku_model_y"),
            pl.col("v12_sku_model_value").drop_nulls().first().alias("sku_model_v"),
            pl.col("v12_sku_validation_wmape_y").drop_nulls().first().alias("sku_val_w_y"),
            pl.col("v12_sku_validation_wmape_value").drop_nulls().first().alias("sku_val_w_v"),
        )
        .collect()
    )
    print("\n=== 16) CALIDAD OOS DEL ENSAMBLE DE TOTAL SKU v12.8 ===")
    for unidad, actual, pred, model, valw in (
        ("Unidades", "sku_actual_y", "sku_fc_y", "sku_model_y", "sku_val_w_y"),
        ("Valor ($)", "sku_actual_v", "sku_fc_v", "sku_model_v", "sku_val_w_v"),
    ):
        print(f"\n{unidad}:")
        x = sku_day.filter(pl.col(model).is_not_null() & pl.col(pred).is_not_null())
        if x.height == 0:
            print("  sin candidatos SKU-total")
            continue
        g = (
            x.group_by(["seccion", model])
            .agg(
                pl.col("sku").n_unique().alias("n_sku"),
                pl.when(pl.col(actual) > 0).then((pl.col(actual) - pl.col(pred)).abs()).otherwise(0.0).sum().alias("ae"),
                pl.when(pl.col(actual) > 0).then(pl.col(actual)).otherwise(0.0).sum().alias("den"),
                pl.when(pl.col(actual) > 0).then(pl.col(pred) - pl.col(actual)).otherwise(0.0).sum().alias("se"),
                pl.col(valw).median().alias("hist_wmape_med"),
            )
            .with_columns(
                (pl.col("ae") / pl.col("den").replace(0.0, None)).alias("wmape"),
                (pl.col("se") / pl.col("den").replace(0.0, None)).alias("bias"),
            )
            .sort(["seccion", "wmape"])
        )
        for r in g.iter_rows(named=True):
            print(
                f"  sec={str(r['seccion']):>3s} {str(r[model]):18s} nSKU={int(r['n_sku']):5d} "
                f"wMAPE={_pct(r['wmape']):>9s} BIAS={_pct(r['bias']):>9s} "
                f"hist-med={_pct(r['hist_wmape_med']):>9s}"
            )


def _print_v12_share_strategy(forecast: Path) -> None:
    """Audit the v12.5 28d/84d store-share production contract."""
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "ds", "period_type",
        "v12_store_share_y", "v12_store_share_value",
        "v12_occurrence_gate_open_joint",
        "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
    }
    if not required.issubset(schema):
        return
    x = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
            & (pl.col("v12_candidate_yhat_raw").is_not_null() | pl.col("v12_candidate_valuehat_raw").is_not_null())
        )
        .select(*sorted(required))
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("sku"),
        )
        .collect()
    )
    if x.height == 0:
        return
    sums = (
        x.group_by(["seccion", "sku", "ds"])
        .agg(
            pl.col("v12_store_share_y").sum().alias("sy"),
            pl.col("v12_store_share_value").sum().alias("sv"),
        )
        .with_columns(
            (pl.col("sy") - 1.0).abs().alias("dy"),
            (pl.col("sv") - 1.0).abs().alias("dv"),
        )
    )
    bad_sum = sums.filter((pl.col("dy") > 1e-6) | (pl.col("dv") > 1e-6)).height
    hurdle = x.with_columns(
        pl.col("v12_occurrence_gate_open_joint").fill_null(False).any().over(["seccion", "sku", "ds"]).alias("_any_open")
    )
    bad_closed = hurdle.filter(
        pl.col("_any_open")
        & ~pl.col("v12_occurrence_gate_open_joint").fill_null(False)
        & ((pl.col("v12_store_share_y").abs() > 1e-9) | (pl.col("v12_store_share_value").abs() > 1e-9))
    ).height
    print("\n=== 17) SHARE v12.8 — RECENCIA 28D×84D ===")
    print(
        f"share={100.0 * float(getattr(settings, 'V12_SHARE_RECENT_WEIGHT', 0.70)):.0f}%/"
        f"{100.0 * (1.0 - float(getattr(settings, 'V12_SHARE_RECENT_WEIGHT', 0.70))):.0f}% "
        f"windows={int(getattr(settings, 'V12_SHARE_RECENT_DAYS', 28))}d/"
        f"{int(getattr(settings, 'V12_SHARE_STABLE_DAYS', 84))}d | "
        f"SKU/día share!=1={bad_sum:,} | closed-gate con share>0={bad_closed:,}"
    )



def _print_v12_lgbm_shape_quality(forecast: Path) -> None:
    """Audit v12.6 pooled LightGBM shape and exact SKU-total preservation."""
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "ds", "period_type", "y", "value",
        "v12_sku_forecast_base_y", "v12_sku_forecast_base_value",
        "v12_sku_forecast_y", "v12_sku_forecast_value",
        "v12_sku_shape_model_y", "v12_sku_shape_model_value",
        "v12_sku_shape_training_rows_y", "v12_sku_shape_training_rows_value",
    }
    if not required.issubset(schema):
        return
    sku_day = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("sku"),
        )
        .group_by(["seccion", "sku", "ds"])
        .agg(
            pl.col("y").sum().alias("actual_y"),
            pl.col("value").sum().alias("actual_v"),
            pl.col("v12_sku_forecast_base_y").drop_nulls().first().alias("base_y"),
            pl.col("v12_sku_forecast_y").drop_nulls().first().alias("corr_y"),
            pl.col("v12_sku_forecast_base_value").drop_nulls().first().alias("base_v"),
            pl.col("v12_sku_forecast_value").drop_nulls().first().alias("corr_v"),
            pl.col("v12_sku_shape_model_y").drop_nulls().first().alias("model_y"),
            pl.col("v12_sku_shape_model_value").drop_nulls().first().alias("model_v"),
            pl.col("v12_sku_shape_training_rows_y").drop_nulls().first().alias("train_y"),
            pl.col("v12_sku_shape_training_rows_value").drop_nulls().first().alias("train_v"),
        )
        .collect()
    )
    if sku_day.height == 0:
        return
    print("\n=== 18) v12.8 POOLED LIGHTGBM — SHAPE SKU-TOTAL ===")
    tol = float(getattr(settings, "V12_SKU_SHAPE_LGBM_TOTAL_TOLERANCE", 1e-9))
    for label, actual, base, corr, model, train in (
        ("Unidades", "actual_y", "base_y", "corr_y", "model_y", "train_y"),
        ("Valor ($)", "actual_v", "base_v", "corr_v", "model_v", "train_v"),
    ):
        print(f"\n{label}:")
        x = sku_day.filter(pl.col(base).is_not_null() & pl.col(corr).is_not_null())
        stats = (
            x.group_by("seccion")
            .agg(
                pl.when(pl.col(actual) > 0).then(pl.col(actual)).otherwise(0.0).sum().alias("den"),
                pl.when(pl.col(actual) > 0).then((pl.col(actual) - pl.col(base)).abs()).otherwise(0.0).sum().alias("ae_base"),
                pl.when(pl.col(actual) > 0).then((pl.col(actual) - pl.col(corr)).abs()).otherwise(0.0).sum().alias("ae_corr"),
                pl.when(pl.col(actual) > 0).then(pl.col(base) - pl.col(actual)).otherwise(0.0).sum().alias("se_base"),
                pl.when(pl.col(actual) > 0).then(pl.col(corr) - pl.col(actual)).otherwise(0.0).sum().alias("se_corr"),
                pl.col("sku").n_unique().alias("n_sku"),
                pl.col(model).drop_nulls().first().alias("model"),
                pl.col(train).median().alias("train_med"),
            )
            .with_columns(
                (pl.col("ae_base") / pl.col("den").replace(0.0, None)).alias("w_base"),
                (pl.col("ae_corr") / pl.col("den").replace(0.0, None)).alias("w_corr"),
                (pl.col("se_base") / pl.col("den").replace(0.0, None)).alias("b_base"),
                (pl.col("se_corr") / pl.col("den").replace(0.0, None)).alias("b_corr"),
            )
            .sort("seccion")
        )
        totals = (
            x.group_by(["seccion", "sku"])
            .agg(
                pl.col(base).sum().alias("base_total"),
                pl.col(corr).sum().alias("corr_total"),
            )
            .with_columns(
                (pl.col("base_total") - pl.col("corr_total")).abs().alias("diff"),
                (
                    tol * pl.max_horizontal(pl.lit(1.0), pl.col("base_total").abs())
                ).alias("tol_abs"),
            )
            .group_by("seccion")
            .agg(
                (pl.col("diff") > pl.col("tol_abs")).sum().alias("bad_total"),
                pl.col("diff").max().alias("max_diff"),
            )
        )
        stats = stats.join(totals, on="seccion", how="left")
        for r in stats.iter_rows(named=True):
            gain = None if r["w_base"] is None or r["w_corr"] is None else float(r["w_base"] - r["w_corr"])
            print(
                f"  sec={str(r['seccion']):>3s} nSKU={int(r['n_sku'] or 0):5d} "
                f"base={_pct(r['w_base']):>9s} -> LGBM={_pct(r['w_corr']):>9s} "
                f"gain={_pct(gain):>9s} | BIAS {_pct(r['b_base']):>9s}->{_pct(r['b_corr']):>9s} | "
                f"model={r['model']} train-med={int(r['train_med'] or 0):,} | "
                f"total28 bad={int(r['bad_total'] or 0):,} maxdiff={float(r['max_diff'] or 0.0):.3g}"
            )


def _print_v12_level_calibration_quality(forecast: Path) -> None:
    """Audit v12.8 calibration challenger against restored v12.6 SKU baseline."""
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "ds", "period_type", "y", "value",
        "v12_sku_forecast_uncalibrated_y", "v12_sku_forecast_uncalibrated_value",
        "v12_sku_forecast_calibrated_challenger_y", "v12_sku_forecast_calibrated_challenger_value",
        "v12_sku_level_calibration_applied_y", "v12_sku_level_calibration_applied_value",
        "v12_sku_level_calibration_factor_y", "v12_sku_level_calibration_factor_value",
        "v12_sku_level_calibration_blocks_y", "v12_sku_level_calibration_blocks_value",
    }
    if not required.issubset(schema):
        return
    sku_day = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("sku"),
        )
        .group_by(["seccion", "sku", "ds"])
        .agg(
            pl.col("y").sum().alias("actual_y"),
            pl.col("value").sum().alias("actual_v"),
            pl.col("v12_sku_forecast_uncalibrated_y").drop_nulls().first().alias("uncal_y"),
            pl.col("v12_sku_forecast_uncalibrated_value").drop_nulls().first().alias("uncal_v"),
            pl.col("v12_sku_forecast_calibrated_challenger_y").drop_nulls().first().alias("cal_y"),
            pl.col("v12_sku_forecast_calibrated_challenger_value").drop_nulls().first().alias("cal_v"),
            pl.col("v12_sku_level_calibration_applied_y").fill_null(False).any().alias("applied_y"),
            pl.col("v12_sku_level_calibration_applied_value").fill_null(False).any().alias("applied_v"),
            pl.col("v12_sku_level_calibration_factor_y").drop_nulls().first().alias("factor_y"),
            pl.col("v12_sku_level_calibration_factor_value").drop_nulls().first().alias("factor_v"),
            pl.col("v12_sku_level_calibration_blocks_y").drop_nulls().first().alias("blocks_y"),
            pl.col("v12_sku_level_calibration_blocks_value").drop_nulls().first().alias("blocks_v"),
        )
        .collect()
    )
    if sku_day.height == 0:
        return
    print("\n=== 19) v12.8 CALIBRACIÓN SKU-TOTAL — CHALLENGER NO APLICADO POR DEFECTO ===")
    for label, actual, uncal, cal, factor, blocks, applied in (
        ("Unidades", "actual_y", "uncal_y", "cal_y", "factor_y", "blocks_y", "applied_y"),
        ("Valor ($)", "actual_v", "uncal_v", "cal_v", "factor_v", "blocks_v", "applied_v"),
    ):
        x = sku_day.filter(pl.col(uncal).is_not_null() & pl.col(cal).is_not_null())
        g = (
            x.group_by("seccion")
            .agg(
                pl.when(pl.col(actual) > 0).then(pl.col(actual)).otherwise(0.0).sum().alias("den"),
                pl.when(pl.col(actual) > 0).then((pl.col(actual) - pl.col(uncal)).abs()).otherwise(0.0).sum().alias("ae_uncal"),
                pl.when(pl.col(actual) > 0).then((pl.col(actual) - pl.col(cal)).abs()).otherwise(0.0).sum().alias("ae_cal"),
                pl.when(pl.col(actual) > 0).then(pl.col(uncal) - pl.col(actual)).otherwise(0.0).sum().alias("se_uncal"),
                pl.when(pl.col(actual) > 0).then(pl.col(cal) - pl.col(actual)).otherwise(0.0).sum().alias("se_cal"),
                pl.col("sku").n_unique().alias("n_sku"),
                pl.col(factor).median().alias("factor_med"),
                pl.col(factor).quantile(0.10).alias("factor_p10"),
                pl.col(factor).quantile(0.90).alias("factor_p90"),
                pl.col(blocks).median().alias("blocks_med"),
                pl.col(applied).cast(pl.Float64).mean().alias("applied_pct"),
            )
            .with_columns(
                (pl.col("ae_uncal") / pl.col("den").replace(0.0, None)).alias("w_uncal"),
                (pl.col("ae_cal") / pl.col("den").replace(0.0, None)).alias("w_cal"),
                (pl.col("se_uncal") / pl.col("den").replace(0.0, None)).alias("b_uncal"),
                (pl.col("se_cal") / pl.col("den").replace(0.0, None)).alias("b_cal"),
            )
            .sort("seccion")
        )
        print(f"\n{label}:")
        for r in g.iter_rows(named=True):
            gain = None if r["w_uncal"] is None or r["w_cal"] is None else float(r["w_uncal"] - r["w_cal"])
            print(
                f"  sec={str(r['seccion']):>3s} nSKU={int(r['n_sku'] or 0):5d} "
                f"baseline={_pct(r['w_uncal']):>9s} -> cal-challenger={_pct(r['w_cal']):>9s} "
                f"gain={_pct(gain):>9s} | BIAS {_pct(r['b_uncal']):>9s}->{_pct(r['b_cal']):>9s} | "
                f"factor p10/med/p90={float(r['factor_p10'] or 1.0):.3f}/"
                f"{float(r['factor_med'] or 1.0):.3f}/{float(r['factor_p90'] or 1.0):.3f} "
                f"blocks-med={float(r['blocks_med'] or 0):.1f} applied={100.0*float(r['applied_pct'] or 0.0):.1f}%"
            )


def _print_v12_oracle_selector_bound(forecast: Path, metrics: pl.DataFrame) -> None:
    """Diagnostic-only oracle: choose v11 or v12 once per leaf using OOS truth.

    This is intentionally NON-causal and is never used to produce forecasts. It
    quantifies how much error remains attributable to family selection versus
    forecast generation when the two already-computed candidates are fixed.
    """
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "period_type", "y", "value", "yhat_raw", "valuehat_raw",
        "v11_yhat_raw_before_v12", "v11_valuehat_raw_before_v12",
        "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
    }
    if not required.issubset(schema):
        return
    base = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .collect()
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.col("v11_yhat_raw_before_v12").round(0).alias("_v11_y"),
            pl.col("v11_valuehat_raw_before_v12").round(2).alias("_v11_v"),
            pl.coalesce([pl.col("v12_candidate_yhat_raw"), pl.col("v11_yhat_raw_before_v12")]).round(0).alias("_v12_y"),
            pl.coalesce([pl.col("v12_candidate_valuehat_raw"), pl.col("v11_valuehat_raw_before_v12")]).round(2).alias("_v12_v"),
            pl.col("yhat_raw").round(0).alias("_final_y"),
            pl.col("valuehat_raw").round(2).alias("_final_v"),
        )
    )
    print("\n=== 20) ORACLE DIAGNÓSTICO v11/v12 POR LEAF (NO CAUSAL, SOLO COTA) ===")
    for unidad, actual, p11, p12, pfinal in (
        ("Unidades", "y", "_v11_y", "_v12_y", "_final_y"),
        ("Valor ($)", "value", "_v11_v", "_v12_v", "_final_v"),
    ):
        active_ids = metrics.filter(
            (pl.col("unidad") == unidad)
            & pl.col("store").is_not_null()
            & pl.col("sku").is_not_null()
            & pl.col("metric_active").fill_null(False)
        ).select("unique_id")
        x = base.join(active_ids, on="unique_id", how="semi")
        if x.height == 0:
            continue
        by_leaf = (
            x.group_by(["seccion", "unique_id"])
            .agg(
                pl.when(pl.col(actual) > 0).then(pl.col(actual)).otherwise(0.0).sum().alias("den"),
                pl.when(pl.col(actual) > 0).then((pl.col(actual) - pl.col(p11)).abs()).otherwise(0.0).sum().alias("ae11"),
                pl.when(pl.col(actual) > 0).then((pl.col(actual) - pl.col(p12)).abs()).otherwise(0.0).sum().alias("ae12"),
                pl.when(pl.col(actual) > 0).then((pl.col(actual) - pl.col(pfinal)).abs()).otherwise(0.0).sum().alias("aef"),
                pl.when(pl.col(actual) > 0).then(pl.col(p11) - pl.col(actual)).otherwise(0.0).sum().alias("se11"),
                pl.when(pl.col(actual) > 0).then(pl.col(p12) - pl.col(actual)).otherwise(0.0).sum().alias("se12"),
                pl.when(pl.col(actual) > 0).then(pl.col(pfinal) - pl.col(actual)).otherwise(0.0).sum().alias("sef"),
            )
            .with_columns((pl.col("ae12") < pl.col("ae11")).alias("oracle_v12"))
            .with_columns(
                pl.when(pl.col("oracle_v12")).then(pl.col("ae12")).otherwise(pl.col("ae11")).alias("ae_oracle"),
                pl.when(pl.col("oracle_v12")).then(pl.col("se12")).otherwise(pl.col("se11")).alias("se_oracle"),
            )
        )
        g = (
            by_leaf.group_by("seccion")
            .agg(
                pl.col("den").sum().alias("den"),
                pl.col("aef").sum().alias("aef"),
                pl.col("ae11").sum().alias("ae11"),
                pl.col("ae12").sum().alias("ae12"),
                pl.col("ae_oracle").sum().alias("aeo"),
                pl.col("sef").sum().alias("sef"),
                pl.col("se_oracle").sum().alias("seo"),
                pl.len().alias("n"),
                pl.col("oracle_v12").sum().alias("oracle_v12_n"),
            )
            .with_columns(
                (pl.col("aef") / pl.col("den").replace(0.0, None)).alias("wf"),
                (pl.col("ae11") / pl.col("den").replace(0.0, None)).alias("w11"),
                (pl.col("ae12") / pl.col("den").replace(0.0, None)).alias("w12"),
                (pl.col("aeo") / pl.col("den").replace(0.0, None)).alias("wo"),
                (pl.col("sef") / pl.col("den").replace(0.0, None)).alias("bf"),
                (pl.col("seo") / pl.col("den").replace(0.0, None)).alias("bo"),
            )
            .sort("seccion")
        )
        print(f"\n{unidad}:")
        for r in g.iter_rows(named=True):
            selector_gap = None if r["wf"] is None or r["wo"] is None else float(r["wf"] - r["wo"])
            print(
                f"  sec={str(r['seccion']):>3s} final={_pct(r['wf']):>9s} "
                f"v11_all={_pct(r['w11']):>9s} v12_all={_pct(r['w12']):>9s} "
                f"oracle={_pct(r['wo']):>9s} selector-gap={_pct(selector_gap):>9s} | "
                f"BIAS final={_pct(r['bf']):>9s} oracle={_pct(r['bo']):>9s} | "
                f"oracle-v12={int(r['oracle_v12_n'] or 0):,}/{int(r['n'] or 0):,} leaves"
            )



def _print_v12_meta_selector(forecast: Path, metrics: pl.DataFrame) -> None:
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "period_type", "v12_selected_y", "v12_selected_value",
        "v12_meta_probability_y", "v12_meta_probability_value",
        "v12_meta_model_available_y", "v12_meta_model_available_value",
        "v12_meta_training_rows_y", "v12_meta_training_rows_value",
        "v12_meta_recent_gain_y", "v12_meta_recent_gain_value",
        "v12_meta_weighted_gain_y", "v12_meta_weighted_gain_value",
        "v12_meta_top_driver_y", "v12_meta_top_driver_value",
        "v12_meta_threshold_y", "v12_meta_threshold_value",
        "v12_meta_portfolio_mode_y", "v12_meta_portfolio_mode_value",
        "v12_meta_policy_available_y", "v12_meta_policy_available_value",
        "v12_meta_policy_utility_gain_y", "v12_meta_policy_utility_gain_value",
        "v12_meta_bias_guard_pass_y", "v12_meta_bias_guard_pass_value",
    }
    if not required.issubset(schema):
        return
    x = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .unique(subset=["unique_id"])
        .with_columns(pl.col("unique_id").str.split("||").list.get(0).alias("seccion"))
        .collect()
    )
    print("\n=== 21) v12.9.7 META-SELECTOR + PORTFOLIO MODE ===")
    for unidad, selected, prob, avail, train_rows, recent, weighted, driver, threshold, mode, policy_avail, policy_gain, bias_guard in (
        ("Unidades", "v12_selected_y", "v12_meta_probability_y", "v12_meta_model_available_y", "v12_meta_training_rows_y", "v12_meta_recent_gain_y", "v12_meta_weighted_gain_y", "v12_meta_top_driver_y", "v12_meta_threshold_y", "v12_meta_portfolio_mode_y", "v12_meta_policy_available_y", "v12_meta_policy_utility_gain_y", "v12_meta_bias_guard_pass_y"),
        ("Valor ($)", "v12_selected_value", "v12_meta_probability_value", "v12_meta_model_available_value", "v12_meta_training_rows_value", "v12_meta_recent_gain_value", "v12_meta_weighted_gain_value", "v12_meta_top_driver_value", "v12_meta_threshold_value", "v12_meta_portfolio_mode_value", "v12_meta_policy_available_value", "v12_meta_policy_utility_gain_value", "v12_meta_bias_guard_pass_value"),
    ):
        active_ids = metrics.filter(
            (pl.col("unidad") == unidad)
            & pl.col("store").is_not_null()
            & pl.col("sku").is_not_null()
            & pl.col("metric_active").fill_null(False)
        ).select("unique_id")
        z = x.join(active_ids, on="unique_id", how="semi")
        if z.height == 0:
            continue
        stats = (
            z.group_by("seccion")
            .agg(
                pl.len().alias("n"),
                pl.col(selected).fill_null(False).sum().alias("selected"),
                pl.col(avail).fill_null(False).sum().alias("available"),
                pl.col(prob).drop_nulls().median().alias("p50"),
                pl.col(prob).filter(pl.col(selected).fill_null(False)).drop_nulls().median().alias("p50_selected"),
                pl.col(train_rows).median().alias("train_med"),
                pl.col(recent).drop_nulls().median().alias("recent_med"),
                pl.col(weighted).drop_nulls().median().alias("weighted_med"),
                pl.col(threshold).drop_nulls().median().alias("threshold"),
                pl.col(mode).drop_nulls().first().alias("mode"),
                pl.col(policy_avail).fill_null(False).any().alias("policy_available"),
                pl.col(policy_gain).drop_nulls().median().alias("policy_gain"),
                pl.col(bias_guard).fill_null(False).mean().alias("bias_guard_rate"),
            )
            .sort("seccion")
        )
        print(f"\n{unidad}:")
        for r in stats.iter_rows(named=True):
            sec = str(r["seccion"])
            drivers = (
                z.filter((pl.col("seccion") == sec) & pl.col(selected).fill_null(False))
                .group_by(driver).len().sort("len", descending=True).head(1)
            )
            top = "NA" if drivers.height == 0 or drivers[driver][0] is None else str(drivers[driver][0])
            print(
                f"  sec={sec:>3s} selected={int(r['selected'] or 0):5d}/{int(r['n'] or 0):5d} "
                f"meta-available={int(r['available'] or 0):5d} p50={float(r['p50'] or 0.0):.3f} "
                f"p50-selected={float(r['p50_selected'] or 0.0):.3f} train-med={int(r['train_med'] or 0):,} | "
                f"mode={str(r['mode'] or 'NA'):>9s} threshold={float(r['threshold'] or 0.0):.2f} "
                f"policy={'yes' if bool(r['policy_available']) else 'fallback'} gain={_pct(r['policy_gain']):>9s} | "
                f"bias-guard={float(r['bias_guard_rate'] or 0.0):.1%} recent-gain-med={_pct(r['recent_med']):>9s} "
                f"weighted-gain-med={_pct(r['weighted_med']):>9s} | top-driver={top}"
            )


def _print_v12_value_portfolio_safety(forecast: Path, metrics: pl.DataFrame) -> None:
    schema = pl.scan_parquet(forecast).collect_schema().names()
    required = {
        "unique_id", "period_type", "v12_selected_value", "v12_meta_portfolio_mode_value",
        "v12_meta_threshold_value", "v12_meta_policy_utility_gain_value",
        "v12_value_safety_enabled", "v12_value_safety_dominance_pass",
        "v12_value_safety_recent_confirmations", "v12_value_safety_recent_blocks",
        "v12_value_safety_bias_coverage", "v12_value_safety_bias_coverage_threshold",
        "v12_value_safety_bias_coverage_pass", "v12_value_safety_meta_margin_pass",
        "v12_value_safety_best_all_mode", "v12_value_safety_reason",
    }
    if not required.issubset(schema):
        return
    active_ids = metrics.filter(
        (pl.col("unidad") == "Valor ($)")
        & pl.col("store").is_not_null()
        & pl.col("sku").is_not_null()
        & pl.col("metric_active").fill_null(False)
    ).select("unique_id")
    x = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .unique(subset=["unique_id"])
        .with_columns(pl.col("unique_id").str.split("||").list.get(0).alias("seccion"))
        .collect()
        .join(active_ids, on="unique_id", how="semi")
    )
    if x.height == 0:
        return
    stats = (
        x.group_by("seccion")
        .agg(
            pl.len().alias("n"),
            pl.col("v12_selected_value").fill_null(False).sum().alias("selected"),
            pl.col("v12_meta_portfolio_mode_value").drop_nulls().first().alias("mode"),
            pl.col("v12_meta_threshold_value").drop_nulls().first().alias("threshold"),
            pl.col("v12_meta_policy_utility_gain_value").drop_nulls().first().alias("policy_gain"),
            pl.col("v12_value_safety_enabled").fill_null(False).any().alias("enabled"),
            pl.col("v12_value_safety_dominance_pass").fill_null(False).any().alias("dominance"),
            pl.col("v12_value_safety_recent_confirmations").drop_nulls().first().alias("confirmations"),
            pl.col("v12_value_safety_recent_blocks").drop_nulls().first().alias("recent_blocks"),
            pl.col("v12_value_safety_bias_coverage").drop_nulls().first().alias("coverage"),
            pl.col("v12_value_safety_bias_coverage_threshold").drop_nulls().first().alias("coverage_threshold"),
            pl.col("v12_value_safety_bias_coverage_pass").fill_null(False).any().alias("coverage_pass"),
            pl.col("v12_value_safety_meta_margin_pass").fill_null(False).any().alias("meta_margin_pass"),
            pl.col("v12_value_safety_best_all_mode").drop_nulls().first().alias("best_all"),
            pl.col("v12_value_safety_reason").drop_nulls().first().alias("reason"),
        )
        .sort("seccion")
    )
    print("\n=== 22) v12.9.7 VALUE PORTFOLIO SAFETY (LEGACY DIAGNÓSTICO) ===")
    for r in stats.iter_rows(named=True):
        cov = "N/A" if r["coverage"] is None else f"{float(r['coverage']):.1%}"
        cov_th = "N/A" if r["coverage_threshold"] is None else f"{float(r['coverage_threshold']):.0%}"
        print(
            f"  sec={str(r['seccion']):>3s} selected={int(r['selected'] or 0):5d}/{int(r['n'] or 0):5d} "
            f"mode={str(r['mode'] or 'NA'):>9s} threshold={float(r['threshold'] or 0.0):.2f} "
            f"gain={_pct(r['policy_gain']):>9s} | dominance={'yes' if bool(r['dominance']) else 'no'} "
            f"recent={int(r['confirmations'] or 0)}/{int(r['recent_blocks'] or 0)} "
            f"bias-coverage={cov} (min={cov_th}, pass={'yes' if bool(r['coverage_pass']) else 'no'}) | "
            f"best-all={str(r['best_all'] or 'NA'):>9s} meta-margin={'yes' if bool(r['meta_margin_pass']) else 'no'} "
            f"reason={str(r['reason'] or 'NA')}"
        )


def _print_v129_value_walkforward(forecast: Path, metrics: pl.DataFrame) -> None:
    required = {
        "unique_id", "period_type", "v12_selected_value", "v12_meta_portfolio_mode_value",
        "v12_meta_threshold_value", "v129_value_wf_enabled", "v129_value_wf_available",
        "v129_value_wf_folds", "v129_value_wf_win_rate", "v129_value_wf_median_gain",
        "v129_value_wf_worst_gain", "v129_value_wf_weighted_gain",
        "v129_value_wf_weighted_utility_gain", "v129_value_wf_bias_worsen_max",
        "v129_value_wf_meta_folds", "v129_value_wf_fold1_gain", "v129_value_wf_fold2_gain",
        "v129_value_wf_fold3_gain", "v129_value_wf_reason",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    active_ids = metrics.filter(
        (pl.col("unidad") == "Valor ($)")
        & (pl.col("node_kind") == "tienda_sku")
        & pl.col("metric_active").fill_null(False)
    ).select("unique_id")
    x = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .unique(subset=["unique_id"])
        .with_columns(pl.col("unique_id").str.split("||").list.get(0).alias("seccion"))
        .collect()
        .join(active_ids, on="unique_id", how="semi")
    )
    if x.height == 0:
        return
    stats = (
        x.group_by("seccion")
        .agg(
            pl.len().alias("n"),
            pl.col("v12_selected_value").fill_null(False).sum().alias("selected"),
            pl.col("v12_meta_portfolio_mode_value").drop_nulls().first().alias("mode"),
            pl.col("v12_meta_threshold_value").drop_nulls().first().alias("threshold"),
            pl.col("v129_value_wf_enabled").fill_null(False).any().alias("enabled"),
            pl.col("v129_value_wf_available").fill_null(False).any().alias("available"),
            pl.col("v129_value_wf_folds").drop_nulls().first().alias("folds"),
            pl.col("v129_value_wf_win_rate").drop_nulls().first().alias("win_rate"),
            pl.col("v129_value_wf_median_gain").drop_nulls().first().alias("median_gain"),
            pl.col("v129_value_wf_worst_gain").drop_nulls().first().alias("worst_gain"),
            pl.col("v129_value_wf_weighted_gain").drop_nulls().first().alias("weighted_gain"),
            pl.col("v129_value_wf_weighted_utility_gain").drop_nulls().first().alias("utility_gain"),
            pl.col("v129_value_wf_bias_worsen_max").drop_nulls().first().alias("bias_worsen"),
            pl.col("v129_value_wf_meta_folds").drop_nulls().first().alias("meta_folds"),
            pl.col("v129_value_wf_fold1_gain").drop_nulls().first().alias("fold1_gain"),
            pl.col("v129_value_wf_fold2_gain").drop_nulls().first().alias("fold2_gain"),
            pl.col("v129_value_wf_fold3_gain").drop_nulls().first().alias("fold3_gain"),
            pl.col("v129_value_wf_reason").drop_nulls().first().alias("reason"),
        )
        .sort("seccion")
    )
    print("\n=== 23) v12.9.2 VALUE WALK-FORWARD PORTFOLIO SELECTOR ===")
    for r in stats.iter_rows(named=True):
        wr = "N/A" if r["win_rate"] is None else f"{float(r['win_rate']):.0%}"
        print(
            f"  sec={str(r['seccion']):>3s} selected={int(r['selected'] or 0):5d}/{int(r['n'] or 0):5d} "
            f"mode={str(r['mode'] or 'NA'):>9s} threshold={float(r['threshold'] or 0.0):.2f} "
            f"folds={int(r['folds'] or 0)} win-rate={wr} meta-folds={int(r['meta_folds'] or 0)} | "
            f"median-gain={_pct(r['median_gain']):>9s} worst-gain={_pct(r['worst_gain']):>9s} "
            f"weighted-gain={_pct(r['weighted_gain']):>9s} utility-gain={_pct(r['utility_gain']):>9s} | "
            f"max-bias-worsen={_pct(r['bias_worsen']):>9s} "
            f"fold-gains=[{_pct(r['fold1_gain'])},{_pct(r['fold2_gain'])},{_pct(r['fold3_gain'])}] "
            f"reason={str(r['reason'] or 'NA')}"
        )


def _print_v1291_expected_gain_selector(forecast: Path, metrics: pl.DataFrame) -> None:
    required = {
        "unique_id", "period_type", "value", "valuehat_raw",
        "v11_valuehat_raw_before_v12", "v12_candidate_valuehat_raw", "v12_selected_value",
        "v1291_expected_gain_enabled_value", "v1291_expected_gain_available_value",
        "v1291_expected_gain_value", "v1291_expected_gain_confidence_value",
        "v1291_expected_gain_training_rows_value", "v1291_expected_gain_resid_rmse_value",
        "v1291_expected_gain_top_driver_value", "v1291_expected_gain_portfolio_pass_value",
        "v1291_expected_gain_portfolio_gain_value", "v1291_expected_gain_portfolio_share_value",
        "v1292_raw_expected_gain_value", "v1292_calibrated_gain_value",
        "v1292_calibration_bucket_value", "v1292_bucket_loss_rate_value",
        "v1292_bucket_p10_gain_value", "v1292_leaf_loss_rate_value",
        "v1292_leaf_p10_gain_value", "v1292_impact_percentile_value",
        "v1292_budget_selected_value", "v1292_portfolio_budget_value",
        "v1292_portfolio_selected_share_value", "v1292_portfolio_expected_gain_value",
        "v1292_portfolio_pass_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    active_ids = metrics.filter(
        (pl.col("unidad") == "Valor ($)")
        & (pl.col("node_kind") == "tienda_sku")
        & pl.col("metric_active").fill_null(False)
    ).select("unique_id")
    x = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .group_by("unique_id")
        .agg(
            pl.when(pl.col("value") > 0).then(pl.col("value")).otherwise(0.0).sum().alias("den"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("v11_valuehat_raw_before_v12").round(2)).abs()).otherwise(0.0).sum().alias("ae11"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("v12_candidate_valuehat_raw").round(2)).abs()).otherwise(0.0).sum().alias("ae12"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("valuehat_raw").round(2)).abs()).otherwise(0.0).sum().alias("aes"),
            pl.col("v12_selected_value").first().alias("selected"),
            pl.col("v1291_expected_gain_enabled_value").first().alias("enabled"),
            pl.col("v1291_expected_gain_available_value").first().alias("available"),
            pl.col("v1291_expected_gain_value").first().alias("expected_gain"),
            pl.col("v1291_expected_gain_confidence_value").first().alias("confidence"),
            pl.col("v1291_expected_gain_training_rows_value").first().alias("train_rows"),
            pl.col("v1291_expected_gain_resid_rmse_value").first().alias("rmse"),
            pl.col("v1291_expected_gain_top_driver_value").first().alias("top_driver"),
            pl.col("v1291_expected_gain_portfolio_pass_value").first().alias("portfolio_pass"),
            pl.col("v1291_expected_gain_portfolio_gain_value").first().alias("portfolio_gain"),
            pl.col("v1291_expected_gain_portfolio_share_value").first().alias("portfolio_share"),
            pl.col("v1292_raw_expected_gain_value").first().alias("raw_gain"),
            pl.col("v1292_calibrated_gain_value").first().alias("calibrated_gain"),
            pl.col("v1292_calibration_bucket_value").first().alias("cal_bucket"),
            pl.col("v1292_bucket_loss_rate_value").first().alias("bucket_loss"),
            pl.col("v1292_bucket_p10_gain_value").first().alias("bucket_p10"),
            pl.col("v1292_leaf_loss_rate_value").first().alias("leaf_loss"),
            pl.col("v1292_leaf_p10_gain_value").first().alias("leaf_p10"),
            pl.col("v1292_impact_percentile_value").first().alias("impact_pct"),
            pl.col("v1292_budget_selected_value").first().alias("budget_selected"),
            pl.col("v1292_portfolio_budget_value").first().alias("portfolio_budget"),
            pl.col("v1292_portfolio_selected_share_value").first().alias("portfolio_selected_share"),
            pl.col("v1292_portfolio_expected_gain_value").first().alias("portfolio_expected_gain"),
            pl.col("v1292_portfolio_pass_value").first().alias("portfolio_pass_v1292"),
        )
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.when(pl.col("den") > 0).then((pl.col("ae11") - pl.col("ae12")) / pl.col("den")).otherwise(None).alias("realized_gain_v12"),
            pl.when(pl.col("den") > 0).then((pl.col("ae11") - pl.col("aes")) / pl.col("den")).otherwise(None).alias("realized_gain_selected"),
        )
        .collect()
        .join(active_ids, on="unique_id", how="semi")
    )
    if x.height == 0:
        return
    print("\n=== 24) v12.9.2 VALUE CALIBRATED EXPECTED-IMPACT SELECTOR (DIAGNÓSTICO) ===")
    by_sec = x.group_by("seccion").agg(
        pl.len().alias("n"),
        pl.col("selected").fill_null(False).sum().alias("selected_n"),
        pl.col("den").sum().alias("den"),
        pl.when(pl.col("selected")).then(pl.col("den")).otherwise(0.0).sum().alias("selected_den"),
        (pl.col("expected_gain") * pl.col("den")).sum().alias("eg_num"),
        ((pl.col("ae11") - pl.col("aes")).sum()).alias("realized_num"),
        pl.col("train_rows").max().alias("train_rows"),
        pl.col("rmse").drop_nulls().first().alias("rmse"),
        pl.col("portfolio_pass").fill_null(False).any().alias("portfolio_pass"),
        pl.col("portfolio_gain").drop_nulls().first().alias("portfolio_gain"),
        pl.col("portfolio_share").drop_nulls().first().alias("portfolio_share"),
    ).sort("seccion")
    for r in by_sec.iter_rows(named=True):
        den = float(r["den"] or 0.0)
        expected = float(r["eg_num"] or 0.0) / den if den > 0 else 0.0
        realized = float(r["realized_num"] or 0.0) / den if den > 0 else 0.0
        vol_share = float(r["selected_den"] or 0.0) / den if den > 0 else 0.0
        print(
            f"sec={str(r['seccion']):>3s} selected={int(r['selected_n'] or 0):5d}/{int(r['n'] or 0):5d} "
            f"volume-share={vol_share:6.1%} expected-gain(BU)={_pct(expected):>9s} "
            f"realized-gain(BU)={_pct(realized):>9s} train={int(r['train_rows'] or 0):,} "
            f"rmse={_pct(r['rmse']):>9s} portfolio-pass={'yes' if r['portfolio_pass'] else 'no'} "
            f"portfolio-expected={_pct(r['portfolio_gain']):>9s}"
        )

    # OOF calibration buckets used by production.  Report raw ridge score,
    # calibrated gain and current realized v12-v11 gain side by side.
    available = x.filter(pl.col("available").fill_null(False) & pl.col("cal_bucket").is_not_null())
    if available.height:
        cal = available.group_by(["seccion", "cal_bucket"]).agg(
            pl.len().alias("n"),
            pl.col("den").sum().alias("den"),
            (pl.col("raw_gain") * pl.col("den")).sum().alias("raw_num"),
            (pl.col("calibrated_gain") * pl.col("den")).sum().alias("cal_num"),
            (pl.col("ae11") - pl.col("ae12")).sum().alias("realized_num"),
            pl.col("bucket_loss").drop_nulls().first().alias("hist_loss"),
            pl.col("bucket_p10").drop_nulls().first().alias("hist_p10"),
        ).sort(["seccion", "cal_bucket"])
        print("  OOF calibration buckets (raw -> calibrated -> realized):")
        for r in cal.iter_rows(named=True):
            den = float(r["den"] or 0.0)
            raw = float(r["raw_num"] or 0.0) / den if den > 0 else 0.0
            cg = float(r["cal_num"] or 0.0) / den if den > 0 else 0.0
            rg = float(r["realized_num"] or 0.0) / den if den > 0 else 0.0
            print(
                f"    sec={r['seccion']} B{int(r['cal_bucket'] or 0)} n={int(r['n'])} "
                f"raw={_pct(raw)} calibrated={_pct(cg)} realized={_pct(rg)} "
                f"hist-loss={float(r['hist_loss'] or 0.0):.0%} hist-p10={_pct(r['hist_p10'])}"
            )

    # Top error contributors, with both alternatives and the oracle decision.
    top = x.with_columns(
        pl.min_horizontal("ae11", "ae12").alias("ae_oracle"),
        pl.when(pl.col("ae12") < pl.col("ae11")).then(pl.lit("v12")).otherwise(pl.lit("v11")).alias("oracle"),
    ).sort("aes", descending=True).head(50)
    print("\n=== 25) TOP 50 VALUE ERROR CONTRIBUTORS — v12.9.2 RISK AUDIT (DIAGNÓSTICO) ===")
    for r in top.iter_rows(named=True):
        den = float(r["den"] or 0.0)
        def w(a): return (float(a or 0.0) / den) if den > 0 else float("nan")
        print(
            f"{str(r['unique_id']):32s} vol={den:,.2f} selected={'v12' if r['selected'] else 'v11':>3s} "
            f"oracle={r['oracle']:>3s} | v11={w(r['ae11']):7.2%} v12={w(r['ae12']):7.2%} "
            f"final={w(r['aes']):7.2%} oracle={w(r['ae_oracle']):7.2%} "
            f"raw={_pct(r['raw_gain'])} cal={_pct(r['calibrated_gain'])} "
            f"conf={float(r['confidence'] or 0.0):.2f} budget={'yes' if r['budget_selected'] else 'no'} "
            f"impact-pct(high=100%)={float(r['impact_pct'] or 1.0):.1%} "
            f"loss(bucket/leaf)={float(r['bucket_loss'] or 0.0):.0%}/{float(r['leaf_loss'] or 0.0):.0%}"
        )


def _print_v1293_hierarchical_segment_selector(forecast: Path, metrics: pl.DataFrame) -> None:
    required = {
        "unique_id", "period_type", "value", "valuehat_raw", "v11_valuehat_raw_before_v12",
        "v12_candidate_valuehat_raw", "v12_selected_value",
        "v1293_segment_selector_enabled_value", "v1293_segment_prebudget_value",
        "v1293_segment_budget_selected_value", "v1293_segment_level_value",
        "v1293_segment_reason_value", "v1293_segment_key_value",
        "v1293_segment_folds_value", "v1293_segment_win_rate_value",
        "v1293_segment_median_gain_value", "v1293_segment_worst_gain_value",
        "v1293_segment_weighted_gain_value", "v1293_segment_utility_gain_value",
        "v1293_segment_bias_worsen_value", "v1293_segment_min_leaves_value",
        "v1293_segment_selected_volume_share_value", "v1293_segment_section_strong_value",
        "v1293_segment_budget_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    active_ids = metrics.filter(
        (pl.col("unidad") == "Valor ($)")
        & (pl.col("node_kind") == "tienda_sku")
        & pl.col("metric_active").fill_null(False)
    ).select("unique_id")
    x = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .group_by("unique_id")
        .agg(
            pl.when(pl.col("value") > 0).then(pl.col("value")).otherwise(0.0).sum().alias("den"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("v11_valuehat_raw_before_v12").round(2)).abs()).otherwise(0.0).sum().alias("ae11"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("v12_candidate_valuehat_raw").round(2)).abs()).otherwise(0.0).sum().alias("ae12"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("valuehat_raw").round(2)).abs()).otherwise(0.0).sum().alias("aes"),
            pl.col("v12_selected_value").first().alias("selected"),
            pl.col("v1293_segment_prebudget_value").first().alias("prebudget"),
            pl.col("v1293_segment_budget_selected_value").first().alias("budget_selected"),
            pl.col("v1293_segment_level_value").first().alias("level"),
            pl.col("v1293_segment_reason_value").first().alias("reason"),
            pl.col("v1293_segment_key_value").first().alias("segment_key"),
            pl.col("v1293_segment_folds_value").first().alias("folds"),
            pl.col("v1293_segment_win_rate_value").first().alias("win_rate"),
            pl.col("v1293_segment_median_gain_value").first().alias("median_gain"),
            pl.col("v1293_segment_worst_gain_value").first().alias("worst_gain"),
            pl.col("v1293_segment_weighted_gain_value").first().alias("weighted_gain"),
            pl.col("v1293_segment_utility_gain_value").first().alias("utility_gain"),
            pl.col("v1293_segment_bias_worsen_value").first().alias("bias_worsen"),
            pl.col("v1293_segment_min_leaves_value").first().alias("min_leaves"),
            pl.col("v1293_segment_selected_volume_share_value").first().alias("portfolio_share"),
            pl.col("v1293_segment_section_strong_value").first().alias("section_strong"),
            pl.col("v1293_segment_budget_value").first().alias("budget"),
        )
        .with_columns(pl.col("unique_id").str.split("||").list.get(0).alias("seccion"))
        .collect()
        .join(active_ids, on="unique_id", how="semi")
    )
    if x.height == 0:
        return
    print("\n=== 26) v12.9.7 HIERARCHICAL SEGMENT WALK-FORWARD SELECTOR ===")
    by_sec = x.group_by("seccion").agg(
        pl.len().alias("n"),
        pl.col("selected").fill_null(False).sum().alias("selected_n"),
        pl.col("prebudget").fill_null(False).sum().alias("pre_n"),
        pl.col("den").sum().alias("den"),
        pl.when(pl.col("selected")).then(pl.col("den")).otherwise(0.0).sum().alias("selected_den"),
        (pl.col("ae11") - pl.col("aes")).sum().alias("realized_num"),
        pl.col("portfolio_share").drop_nulls().first().alias("policy_share"),
        pl.col("section_strong").fill_null(False).any().alias("section_strong"),
        pl.col("budget").drop_nulls().first().alias("budget"),
    ).sort("seccion")
    for r in by_sec.iter_rows(named=True):
        den = float(r["den"] or 0.0)
        realized = float(r["realized_num"] or 0.0) / den if den > 0 else 0.0
        vol_share = float(r["selected_den"] or 0.0) / den if den > 0 else 0.0
        print(
            f"sec={str(r['seccion']):>3s} selected={int(r['selected_n'] or 0):5d}/{int(r['n'] or 0):5d} "
            f"prebudget={int(r['pre_n'] or 0):5d} volume-share={vol_share:6.1%} "
            f"realized-gain(BU)={_pct(realized):>9s} section-strong={'yes' if r['section_strong'] else 'no'} "
            f"budget={float(r['budget'] or 0.0):.0%}"
        )
        levels = (
            x.filter(pl.col("seccion") == r["seccion"])
            .group_by(["level", "selected"])
            .agg(pl.len().alias("n"), pl.col("den").sum().alias("den"))
            .sort(["selected", "den"], descending=[True, True])
        )
        lev_txt = ", ".join(
            f"{z['level']}:{'v12' if z['selected'] else 'v11'} n={int(z['n'])} vol={float(z['den'] or 0.0)/den:.0%}"
            for z in levels.iter_rows(named=True)
        )
        print(f"  hierarchy: {lev_txt}")
        reasons = (
            x.filter(pl.col("seccion") == r["seccion"])
            .group_by("reason").agg(pl.len().alias("n"), pl.col("den").sum().alias("den"))
            .sort("den", descending=True).head(6)
        )
        print("  reasons: " + "; ".join(
            f"{z['reason']}={int(z['n'])} ({float(z['den'] or 0.0)/den:.0%} vol)"
            for z in reasons.iter_rows(named=True)
        ))

    top_seg = (
        x.filter(pl.col("segment_key").is_not_null())
        .group_by(["seccion", "level", "segment_key", "reason"])
        .agg(
            pl.len().alias("n"), pl.col("den").sum().alias("den"),
            pl.col("selected").fill_null(False).any().alias("selected"),
            pl.col("folds").first().alias("folds"), pl.col("win_rate").first().alias("win_rate"),
            pl.col("median_gain").first().alias("median_gain"), pl.col("worst_gain").first().alias("worst_gain"),
            pl.col("weighted_gain").first().alias("weighted_gain"), pl.col("utility_gain").first().alias("utility_gain"),
            pl.col("bias_worsen").first().alias("bias_worsen"), pl.col("min_leaves").first().alias("min_leaves"),
            (pl.col("ae11") - pl.col("aes")).sum().alias("realized_num"),
        )
        .sort("den", descending=True).head(15)
    )
    print("  Top segmentos por impacto OOS (solo auditoría, no usados para selección):")
    for r in top_seg.iter_rows(named=True):
        den = float(r["den"] or 0.0)
        realized = float(r["realized_num"] or 0.0) / den if den > 0 else 0.0
        print(
            f"    sec={r['seccion']} {r['level']} selected={'v12' if r['selected'] else 'v11'} n={int(r['n'])} "
            f"hist gain={_pct(r['weighted_gain'])} med={_pct(r['median_gain'])} worst={_pct(r['worst_gain'])} "
            f"win={float(r['win_rate'] or 0.0):.0%} bias-worsen={_pct(r['bias_worsen'])} "
            f"realized={_pct(realized)} key={r['segment_key']}"
        )


def _print_v1294_volume_risk_control(forecast: Path, metrics: pl.DataFrame) -> None:
    required = {
        "unique_id", "period_type", "value", "valuehat_raw", "v11_valuehat_raw_before_v12",
        "v12_selected_value", "v1293_segment_prebudget_value", "v1293_segment_budget_selected_value",
        "v1293_segment_key_value", "v1293_segment_level_value",
        "v1293_segment_selected_volume_share_value", "v1293_segment_budget_value",
        "v1294_segment_risk_score_value", "v1294_segment_exposure_budget_value",
        "v1294_impact_percentile_value", "v1294_high_impact_guard_pass_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    active_ids = metrics.filter(
        (pl.col("unidad") == "Valor ($)") & (pl.col("node_kind") == "tienda_sku")
        & pl.col("metric_active").fill_null(False)
    ).select("unique_id")
    x = (
        pl.scan_parquet(forecast)
        .filter((pl.col("period_type") == "out_sample") & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2))
        .select(*sorted(required))
        .group_by("unique_id").agg(
            pl.when(pl.col("value") > 0).then(pl.col("value")).otherwise(0.0).sum().alias("den"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("v11_valuehat_raw_before_v12").round(2)).abs()).otherwise(0.0).sum().alias("ae11"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("valuehat_raw").round(2)).abs()).otherwise(0.0).sum().alias("aes"),
            pl.col("v12_selected_value").first().alias("selected"),
            pl.col("v1293_segment_prebudget_value").first().alias("prebudget"),
            pl.col("v1293_segment_budget_selected_value").first().alias("budget_selected"),
            pl.col("v1293_segment_key_value").first().alias("segment_key"),
            pl.col("v1293_segment_level_value").first().alias("level"),
            pl.col("v1293_segment_selected_volume_share_value").first().alias("portfolio_share"),
            pl.col("v1293_segment_budget_value").first().alias("budget"),
            pl.col("v1294_segment_risk_score_value").first().alias("risk_score"),
            pl.col("v1294_segment_exposure_budget_value").first().alias("segment_budget"),
            pl.col("v1294_impact_percentile_value").first().alias("impact_pct"),
            pl.col("v1294_high_impact_guard_pass_value").first().alias("high_guard"),
        )
        .with_columns(pl.col("unique_id").str.split("||").list.get(0).alias("seccion"))
        .collect().join(active_ids, on="unique_id", how="semi")
    )
    if x.height == 0:
        return
    print("\n=== 27) v12.9.7 SEC23 VOLUME-RISK CONTROL ===")
    for sec in ["1", "23"]:
        q = x.filter(pl.col("seccion") == sec)
        if q.height == 0:
            continue
        den = float(q.select(pl.col("den").sum()).item() or 0.0)
        selected_den = float(q.select(pl.when(pl.col("selected")).then(pl.col("den")).otherwise(0.0).sum()).item() or 0.0)
        gain_num = float(q.select((pl.col("ae11") - pl.col("aes")).sum()).item() or 0.0)
        selected = int(q.select(pl.col("selected").fill_null(False).sum()).item() or 0)
        pre = int(q.select(pl.col("prebudget").fill_null(False).sum()).item() or 0)
        budgets = q.get_column("budget").drop_nulls()
        budget = float(budgets[0]) if len(budgets) else 0.0
        rejected_hi = q.filter(pl.col("prebudget").fill_null(False) & ~pl.col("high_guard").fill_null(False)).height
        print(
            f"sec={sec:>3s} selected={selected:5d}/{q.height:5d} prebudget={pre:5d} "
            f"volume-share={(selected_den / den if den else 0):6.1%} hard-budget={budget:.0%} "
            f"realized-gain={_pct(gain_num / den if den else 0.0)} high-impact-rejected={rejected_hi}"
        )
    seg = (
        x.filter((pl.col("seccion") == "23") & pl.col("prebudget").fill_null(False) & pl.col("segment_key").is_not_null())
        .group_by(["level", "segment_key"]).agg(
            pl.col("den").sum().alias("den"),
            pl.col("selected").fill_null(False).sum().alias("selected_n"),
            pl.col("segment_budget").max().alias("segment_budget"),
            pl.col("risk_score").max().alias("risk_score"),
            pl.col("impact_pct").max().alias("max_impact_pct"),
            (pl.col("ae11") - pl.col("aes")).sum().alias("gain_num"),
        ).sort("den", descending=True).head(12)
    )
    if seg.height:
        print("  Sec23 top segmentos por exposición/riesgo:")
        for z in seg.iter_rows(named=True):
            d = float(z["den"] or 0.0)
            print(
                f"    {z['level']} selected={int(z['selected_n'] or 0):4d} "
                f"seg-budget={float(z['segment_budget'] or 0.0):5.1%} "
                f"risk={float(z['risk_score'] or 0.0):.4f} max-impact={float(z['max_impact_pct'] or 0.0):.1%} "
                f"realized={_pct(float(z['gain_num'] or 0.0) / d if d else 0.0)} key={z['segment_key']}"
            )


def _print_v1295_long_horizon_stress(forecast: Path, metrics: pl.DataFrame) -> None:
    required = {
        "unique_id", "period_type", "value", "v11_valuehat_raw_before_v12",
        "v12_candidate_valuehat_raw", "v1295_stress_enabled_value",
        "v1295_stress_requested_blocks_value", "v1295_stress_materialized_blocks_value", "v1295_stress_level_value",
        "v1295_stress_key_value", "v1295_stress_folds_value",
        "v1295_stress_win_rate_value", "v1295_stress_median_gain_value",
        "v1295_stress_p25_gain_value", "v1295_stress_p10_gain_value",
        "v1295_stress_worst_gain_value", "v1295_stress_std_gain_value",
        "v1295_stress_weighted_gain_value", "v1295_stress_bias_p90_worsen_value",
        "v1295_stress_max_consecutive_losses_value", "v1295_stress_min_leaves_value",
        "v1295_stress_pass_value", "v1295_stress_reason_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    active_ids = metrics.filter(
        (pl.col("unidad") == "Valor ($)") & (pl.col("node_kind") == "tienda_sku")
        & pl.col("metric_active").fill_null(False)
    ).select("unique_id")
    x = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .group_by("unique_id").agg(
            pl.when(pl.col("value") > 0).then(pl.col("value")).otherwise(0.0).sum().alias("den"),
            pl.when(pl.col("value") > 0)
            .then((pl.col("value") - pl.col("v11_valuehat_raw_before_v12").round(2)).abs())
            .otherwise(0.0).sum().alias("ae11"),
            pl.when(pl.col("value") > 0)
            .then((pl.col("value") - pl.col("v12_candidate_valuehat_raw").round(2)).abs())
            .otherwise(0.0).sum().alias("ae12"),
            pl.col("v1295_stress_enabled_value").first().alias("enabled"),
            pl.col("v1295_stress_requested_blocks_value").first().alias("requested"),
            pl.col("v1295_stress_materialized_blocks_value").first().alias("materialized"),
            pl.col("v1295_stress_level_value").first().alias("level"),
            pl.col("v1295_stress_key_value").first().alias("key"),
            pl.col("v1295_stress_folds_value").first().alias("folds"),
            pl.col("v1295_stress_win_rate_value").first().alias("win_rate"),
            pl.col("v1295_stress_median_gain_value").first().alias("median_gain"),
            pl.col("v1295_stress_p25_gain_value").first().alias("p25_gain"),
            pl.col("v1295_stress_p10_gain_value").first().alias("p10_gain"),
            pl.col("v1295_stress_worst_gain_value").first().alias("worst_gain"),
            pl.col("v1295_stress_std_gain_value").first().alias("std_gain"),
            pl.col("v1295_stress_weighted_gain_value").first().alias("weighted_gain"),
            pl.col("v1295_stress_bias_p90_worsen_value").first().alias("bias_p90"),
            pl.col("v1295_stress_max_consecutive_losses_value").first().alias("max_loss_run"),
            pl.col("v1295_stress_min_leaves_value").first().alias("min_leaves"),
            pl.col("v1295_stress_pass_value").first().alias("stress_pass"),
            pl.col("v1295_stress_reason_value").first().alias("reason"),
        )
        .with_columns(pl.col("unique_id").str.split("||").list.first().alias("seccion"))
        .collect()
        .join(active_ids, on="unique_id", how="semi")
    )
    if x.height == 0:
        return
    print("\n=== 28) v12.9.7 LONG-HORIZON SEGMENT STRESS TEST (DIAGNÓSTICO, NO PRODUCTIVO) ===")
    print("El OOS actual NO participa en estos folds; solo se usa abajo para auditar generalización.")
    for sec in ["1", "23"]:
        q = x.filter(pl.col("seccion") == sec)
        if q.height == 0:
            continue
        den = float(q.select(pl.col("den").sum()).item() or 0.0)
        pass_den = float(q.select(pl.when(pl.col("stress_pass")).then(pl.col("den")).otherwise(0.0).sum()).item() or 0.0)
        pass_n = int(q.select(pl.col("stress_pass").fill_null(False).sum()).item() or 0)
        folds_med = q.select(pl.col("folds").median()).item()
        folds_min = q.select(pl.col("folds").min()).item()
        folds_max = q.select(pl.col("folds").max()).item()
        requested = q.select(pl.col("requested").max()).item()
        materialized = q.select(pl.col("materialized").max()).item()
        cf_num = float(q.select((pl.col("ae11") - pl.col("ae12")).sum()).item() or 0.0)
        pass_cf_num = float(q.select(
            pl.when(pl.col("stress_pass")).then(pl.col("ae11") - pl.col("ae12")).otherwise(0.0).sum()
        ).item() or 0.0)
        print(
            f"sec={sec:>3s} requested/materialized={int(requested or 0):2d}/{int(materialized or 0):2d} folds(min/med/max)="
            f"{int(folds_min or 0)}/{float(folds_med or 0):.1f}/{int(folds_max or 0)} "
            f"stress-pass={pass_n:5d}/{q.height:5d} volume={pass_den/den if den else 0.0:6.1%} "
            f"all-v12 OOS gain={_pct(cf_num/den if den else 0.0)} "
            f"stress-pass-only OOS contribution={_pct(pass_cf_num/den if den else 0.0)}"
        )
        levels = q.group_by(["level", "stress_pass"]).agg(
            pl.len().alias("n"), pl.col("den").sum().alias("den")
        ).sort("den", descending=True)
        print("  hierarchy: " + ", ".join(
            f"{r['level']}:{'pass' if r['stress_pass'] else 'risk'} n={int(r['n'])} vol={float(r['den'] or 0.0)/den:.0%}"
            for r in levels.iter_rows(named=True)
        ))

    seg = (
        x.filter(pl.col("key").is_not_null())
        .group_by(["seccion", "level", "key"]).agg(
            pl.len().alias("n"), pl.col("den").sum().alias("den"),
            pl.col("folds").first().alias("folds"), pl.col("win_rate").first().alias("win_rate"),
            pl.col("median_gain").first().alias("median_gain"), pl.col("p25_gain").first().alias("p25_gain"),
            pl.col("p10_gain").first().alias("p10_gain"), pl.col("worst_gain").first().alias("worst_gain"),
            pl.col("std_gain").first().alias("std_gain"), pl.col("weighted_gain").first().alias("weighted_gain"),
            pl.col("bias_p90").first().alias("bias_p90"), pl.col("max_loss_run").first().alias("max_loss_run"),
            pl.col("stress_pass").first().alias("stress_pass"),
            (pl.col("ae11") - pl.col("ae12")).sum().alias("oos_gain_num"),
        )
        .sort("den", descending=True)
        .head(20)
    )
    if seg.height:
        print("  Top segmentos por volumen (historia larga vs OOS final):")
        for r in seg.iter_rows(named=True):
            d = float(r["den"] or 0.0)
            print(
                f"    sec={r['seccion']} {r['level']} {'PASS' if r['stress_pass'] else 'RISK'} n={int(r['n'])} "
                f"folds={int(r['folds'] or 0)} win={float(r['win_rate'] or 0.0):.0%} "
                f"med={_pct(r['median_gain'])} p25={_pct(r['p25_gain'])} p10={_pct(r['p10_gain'])} "
                f"worst={_pct(r['worst_gain'])} std={_pct(r['std_gain'])} loss-run={int(r['max_loss_run'] or 0)} "
                f"bias-p90={_pct(r['bias_p90'])} hist-BU={_pct(r['weighted_gain'])} "
                f"OOS={_pct(float(r['oos_gain_num'] or 0.0)/d if d else 0.0)} key={r['key']}"
            )


def _print_v1296_sec23_stress_gate(forecast: Path, metrics: pl.DataFrame) -> None:
    required = {
        "unique_id", "period_type", "value", "valuehat_raw",
        "v11_valuehat_raw_before_v12", "v12_candidate_valuehat_raw",
        "v12_selected_value", "v1295_stress_level_value", "v1295_stress_key_value",
        "v1295_stress_folds_value", "v1295_stress_win_rate_value",
        "v1295_stress_median_gain_value", "v1295_stress_p25_gain_value",
        "v1295_stress_p10_gain_value", "v1295_stress_worst_gain_value",
        "v1295_stress_bias_p90_worsen_value", "v1295_stress_max_consecutive_losses_value",
        "v1296_stress_gate_enabled_value", "v1296_stress_eligible_value",
        "v1296_stress_gate_pass_value", "v1296_budget_selected_value",
        "v1296_high_impact_guard_pass_value", "v1296_reason_value",
        "v1296_budget_value", "v1296_selected_volume_share_value",
        "v1296_segment_score_value", "v1294_impact_percentile_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    active_ids = metrics.filter(
        (pl.col("unidad") == "Valor ($)")
        & (pl.col("node_kind") == "tienda_sku")
        & (pl.col("metric_cohort") == "active")
    ).select("unique_id").unique()
    x = (
        pl.scan_parquet(forecast)
        .filter((pl.col("period_type") == "out_sample") & pl.col("unique_id").str.starts_with("23||"))
        .group_by("unique_id")
        .agg(
            pl.when(pl.col("value") > 0).then(pl.col("value")).otherwise(0.0).sum().alias("den"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("v11_valuehat_raw_before_v12").round(2)).abs()).otherwise(0.0).sum().alias("ae11"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("v12_candidate_valuehat_raw").round(2)).abs()).otherwise(0.0).sum().alias("ae12"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("valuehat_raw").round(2)).abs()).otherwise(0.0).sum().alias("aef"),
            pl.col("v12_selected_value").first().alias("selected"),
            pl.col("v1295_stress_level_value").first().alias("level"),
            pl.col("v1295_stress_key_value").first().alias("key"),
            pl.col("v1295_stress_folds_value").first().alias("folds"),
            pl.col("v1295_stress_win_rate_value").first().alias("win"),
            pl.col("v1295_stress_median_gain_value").first().alias("median"),
            pl.col("v1295_stress_p25_gain_value").first().alias("p25"),
            pl.col("v1295_stress_p10_gain_value").first().alias("p10"),
            pl.col("v1295_stress_worst_gain_value").first().alias("worst"),
            pl.col("v1295_stress_bias_p90_worsen_value").first().alias("bias_p90"),
            pl.col("v1295_stress_max_consecutive_losses_value").first().alias("loss_run"),
            pl.col("v1296_stress_eligible_value").first().alias("eligible"),
            pl.col("v1296_stress_gate_pass_value").first().alias("gate_pass"),
            pl.col("v1296_budget_selected_value").first().alias("budget_selected"),
            pl.col("v1296_high_impact_guard_pass_value").first().alias("high_guard"),
            pl.col("v1296_reason_value").first().alias("reason"),
            pl.col("v1296_budget_value").first().alias("budget"),
            pl.col("v1296_selected_volume_share_value").first().alias("selected_share"),
            pl.col("v1296_segment_score_value").first().alias("score"),
            pl.col("v1294_impact_percentile_value").first().alias("impact_pct"),
        )
        .collect().join(active_ids, on="unique_id", how="semi")
    )
    if x.height == 0:
        return
    print("\n=== 29) v12.9.7 SEC23 L4 LONG-HORIZON STRESS-GATED SELECTOR ===")
    min_folds = int(getattr(settings, "V1296_SEC23_MIN_FOLDS", 8))
    l4 = x.filter(pl.col("level") == "L4")
    l4_ins = l4.filter(pl.col("folds").fill_null(0) < min_folds).height
    l4_pass = l4.filter(pl.col("eligible").fill_null(False)).height
    l4_fail = max(0, l4.height - l4_pass - l4_ins)
    selected_n = int(x.select(pl.col("selected").fill_null(False).sum()).item() or 0)
    den = float(x.select(pl.col("den").sum()).item() or 0.0)
    selected_den = float(x.select(pl.when(pl.col("selected")).then(pl.col("den")).otherwise(0.0).sum()).item() or 0.0)
    gain_num = float(x.select((pl.col("ae11") - pl.col("aef")).sum()).item() or 0.0)
    bias11_num = float(x.select(pl.col("ae11").sum()).item() or 0.0)
    budgets = x.get_column("budget").drop_nulls()
    budget = float(budgets[0]) if len(budgets) else 0.0
    print(
        f"sec=23 L4 eligible/pass={l4_pass}/{l4.height} fail={l4_fail} insufficient={l4_ins} | "
        f"selected={selected_n}/{x.height} volume-share={(selected_den/den if den else 0):.1%} "
        f"budget={budget:.0%} realized-gain-vs-v11={_pct(gain_num/den if den else 0.0)}"
    )
    reasons = x.group_by("reason").agg(pl.len().alias("n"), pl.col("den").sum().alias("den")).sort("den", descending=True)
    print("  razones:")
    for r in reasons.iter_rows(named=True):
        print(f"    {r['reason']}: n={int(r['n'])} vol={(float(r['den'] or 0.0)/den if den else 0):.1%}")
    seg = (
        x.filter(pl.col("key").is_not_null())
        .group_by(["level", "key"])
        .agg(
            pl.len().alias("n"), pl.col("den").sum().alias("den"),
            pl.col("selected").fill_null(False).sum().alias("selected_n"),
            pl.col("eligible").first().alias("eligible"),
            pl.col("win").first().alias("win"), pl.col("median").first().alias("median"),
            pl.col("p25").first().alias("p25"), pl.col("p10").first().alias("p10"),
            pl.col("worst").first().alias("worst"), pl.col("bias_p90").first().alias("bias_p90"),
            pl.col("loss_run").first().alias("loss_run"), pl.col("score").first().alias("score"),
            (pl.col("ae11") - pl.col("ae12")).sum().alias("cand_gain_num"),
        )
        .sort("den", descending=True)
    )
    chosen = seg.filter(pl.col("selected_n") > 0).head(10)
    if chosen.height:
        print("  top segmentos seleccionados:")
        for r in chosen.iter_rows(named=True):
            d=float(r['den'] or 0.0)
            print(
                f"    {r['level']} n={int(r['n'])} selected={int(r['selected_n'])} win={float(r['win'] or 0):.0%} "
                f"med={_pct(r['median'])} p25={_pct(r['p25'])} p10={_pct(r['p10'])} worst={_pct(r['worst'])} "
                f"loss-run={int(r['loss_run'] or 0)} OOS-cand-gain={_pct(float(r['cand_gain_num'] or 0.0)/d if d else 0.0)} key={r['key']}"
            )
    rejected = seg.filter((pl.col("level") == "L4") & ~pl.col("eligible").fill_null(False)).head(10)
    if rejected.height:
        print("  top L4 rechazados por stress:")
        for r in rejected.iter_rows(named=True):
            d=float(r['den'] or 0.0)
            print(
                f"    n={int(r['n'])} win={float(r['win'] or 0):.0%} med={_pct(r['median'])} p25={_pct(r['p25'])} "
                f"p10={_pct(r['p10'])} worst={_pct(r['worst'])} loss-run={int(r['loss_run'] or 0)} "
                f"OOS-cand-gain={_pct(float(r['cand_gain_num'] or 0.0)/d if d else 0.0)} key={r['key']}"
            )


def _print_v1297_temporal_replay(forecast: Path, metrics: pl.DataFrame) -> None:
    required = {
        "unique_id", "period_type", "v1297_temporal_replay_enabled_value",
        "v1297_temporal_replay_cutoffs_value", "v1297_temporal_replay_win_rate_value",
        "v1297_temporal_replay_median_gain_value", "v1297_temporal_replay_worst_gain_value",
        "v1297_temporal_replay_weighted_gain_value", "v1297_temporal_replay_bias_worst_value",
        "v1297_temporal_replay_selected_share_median_value",
        "v1297_temporal_replay_fold_gains_value", "v1297_temporal_replay_fold_shares_value",
        "v1297_temporal_replay_fold_bias_value", "v1297_temporal_replay_pass_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    row = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & pl.col("unique_id").str.starts_with("23||")
            & pl.col("unique_id").str.contains("||T:", literal=True)
            & pl.col("v1297_temporal_replay_enabled_value").fill_null(False)
        )
        .select(
            pl.col("v1297_temporal_replay_enabled_value").first().alias("enabled"),
            pl.col("v1297_temporal_replay_cutoffs_value").first().alias("cutoffs"),
            pl.col("v1297_temporal_replay_win_rate_value").first().alias("win"),
            pl.col("v1297_temporal_replay_median_gain_value").first().alias("median"),
            pl.col("v1297_temporal_replay_worst_gain_value").first().alias("worst"),
            pl.col("v1297_temporal_replay_weighted_gain_value").first().alias("weighted"),
            pl.col("v1297_temporal_replay_bias_worst_value").first().alias("bias_worst"),
            pl.col("v1297_temporal_replay_selected_share_median_value").first().alias("share_med"),
            pl.col("v1297_temporal_replay_fold_gains_value").first().alias("gains"),
            pl.col("v1297_temporal_replay_fold_shares_value").first().alias("shares"),
            pl.col("v1297_temporal_replay_fold_bias_value").first().alias("biases"),
            pl.col("v1297_temporal_replay_pass_value").first().alias("passed"),
        )
        .collect()
    )
    if row.height == 0:
        return
    r = row.row(0, named=True)
    print("\n=== 30) v12.9.7 MULTI-CUTOFF TEMPORAL ROBUSTNESS REPLAY (DIAGNÓSTICO) ===")
    print("Cada pseudo-OOS se decide usando únicamente bloques 28d más antiguos; el OOS actual queda fuera.")
    print(
        f"sec=23 cutoffs={int(r['cutoffs'] or 0)} win-rate={float(r['win'] or 0.0):.0%} "
        f"median-gain={_pct(r['median'])} worst-gain={_pct(r['worst'])} "
        f"weighted-gain={_pct(r['weighted'])} worst-bias-worsen={_pct(r['bias_worst'])} "
        f"median-selected-volume={float(r['share_med'] or 0.0):.1%} replay-pass={'yes' if r['passed'] else 'no'}"
    )
    print(f"  fold gains : [{r['gains'] or 'N/A'}]")
    print(f"  fold shares: [{r['shares'] or 'N/A'}]")
    print(f"  fold bias  : [{r['biases'] or 'N/A'}]")


def _print_v1298_sparse_bias_diagnostic(forecast: Path) -> None:
    required = {
        "unique_id", "period_type", "y", "value", "yhat", "valuehat",
        "v1298_sparse_bias_enabled", "v1298_sparse_bucket_y", "v1298_sparse_bucket_value",
        "v1298_sparse_folds_y", "v1298_sparse_folds_value",
        "v1298_sparse_median_ratio_y", "v1298_sparse_median_ratio_value",
        "v1298_sparse_p25_ratio_y", "v1298_sparse_p25_ratio_value",
        "v1298_sparse_worst_ratio_y", "v1298_sparse_worst_ratio_value",
        "v1298_sparse_factor_y", "v1298_sparse_factor_value",
        "v1298_sparse_candidate_y", "v1298_sparse_candidate_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return

    select_cols = set(required)
    if "v12910_valuehat_raw_before_sparse" in schema:
        select_cols.add("v12910_valuehat_raw_before_sparse")
    leaf = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(select_cols))
        .with_columns(pl.col("unique_id").str.split("||").list.first().alias("seccion"))
        .collect()
    )
    if leaf.height == 0:
        return

    print("\n=== 31) v12.9.8 SPARSE-DEMAND BIAS RESCUE (DIAGNÓSTICO CAUSAL, NO PRODUCTIVO) ===")
    print("Factor aprendido solo con bloques cerrados; el OOS actual se usa únicamente para auditar el challenger.")
    for sec in sorted(leaf.get_column("seccion").drop_nulls().unique().to_list()):
        q = leaf.filter(pl.col("seccion") == sec)
        for label, actual, pred, bucket, folds, med, p25, worst, factor, cand in (
            ("Unidades", "y", "yhat", "v1298_sparse_bucket_y", "v1298_sparse_folds_y", "v1298_sparse_median_ratio_y", "v1298_sparse_p25_ratio_y", "v1298_sparse_worst_ratio_y", "v1298_sparse_factor_y", "v1298_sparse_candidate_y"),
            ("Valor ($)", "value", "valuehat", "v1298_sparse_bucket_value", "v1298_sparse_folds_value", "v1298_sparse_median_ratio_value", "v1298_sparse_p25_ratio_value", "v1298_sparse_worst_ratio_value", "v1298_sparse_factor_value", "v1298_sparse_candidate_value"),
        ):
            base_pred = (
                pl.col("v12910_valuehat_raw_before_sparse").round(2)
                if label == "Valor ($)" and "v12910_valuehat_raw_before_sparse" in q.columns
                else pl.col(pred)
            )
            scored = q.filter(pl.col(actual) > 0).with_columns(
                base_pred.alias("_diag_base")
            ).with_columns(
                pl.when(pl.col(cand).fill_null(False))
                .then(pl.col("_diag_base") * pl.col(factor).fill_null(1.0))
                .otherwise(pl.col("_diag_base"))
                .alias("_challenger")
            )
            if scored.height == 0:
                continue
            agg = scored.select(
                pl.col(actual).sum().alias("den"),
                (pl.col("_diag_base") - pl.col(actual)).abs().sum().alias("ae0"),
                (pl.col("_diag_base") - pl.col(actual)).sum().alias("se0"),
                (pl.col("_challenger") - pl.col(actual)).abs().sum().alias("ae1"),
                (pl.col("_challenger") - pl.col(actual)).sum().alias("se1"),
                pl.col("unique_id").filter(pl.col(cand).fill_null(False)).n_unique().alias("n_candidate"),
            ).row(0, named=True)
            den = float(agg["den"] or 0.0)
            if den <= 0:
                continue
            wm0 = float(agg["ae0"] or 0.0) / den
            wm1 = float(agg["ae1"] or 0.0) / den
            b0 = float(agg["se0"] or 0.0) / den
            b1 = float(agg["se1"] or 0.0) / den
            print(
                f"{label:10s} sec={sec:>3s} candidates={int(agg['n_candidate'] or 0):4d} | "
                f"baseline={_pct(wm0)} -> challenger={_pct(wm1)} gain={_pct(wm0-wm1)} | "
                f"BIAS {_pct(b0)}->{_pct(b1)}"
            )
            meta = (
                q.filter(pl.col(cand).fill_null(False))
                .select(bucket, folds, med, p25, worst, factor)
                .unique()
                .sort(bucket)
            )
            for r in meta.iter_rows(named=True):
                print(
                    f"  bucket={r[bucket] or 'NA'} folds={int(r[folds] or 0)} "
                    f"ratio med/p25/worst={float(r[med] or 0.0):.3f}/{float(r[p25] or 0.0):.3f}/{float(r[worst] or 0.0):.3f} "
                    f"factor={float(r[factor] or 1.0):.3f}"
                )


def _print_v1299_value_sparse_segmented(forecast: Path) -> None:
    required = {
        "unique_id", "period_type", "value", "valuehat",
        "v1299_value_sparse_enabled", "v1299_value_sparse_bucket",
        "v1299_value_sparse_level_method", "v1299_value_sparse_seasonal",
        "v1299_value_sparse_segment", "v1299_value_sparse_factor",
        "v1299_value_sparse_folds", "v1299_value_sparse_win_rate",
        "v1299_value_sparse_median_gain", "v1299_value_sparse_worst_gain",
        "v1299_value_sparse_weighted_gain", "v1299_value_sparse_bias_worst",
        "v1299_value_sparse_candidate", "v1299_value_sparse_replay_cutoffs",
        "v1299_value_sparse_replay_win_rate", "v1299_value_sparse_replay_median_gain",
        "v1299_value_sparse_replay_worst_gain", "v1299_value_sparse_replay_weighted_gain",
        "v1299_value_sparse_replay_bias_worst", "v1299_value_sparse_replay_selected_share_median",
        "v1299_value_sparse_replay_fold_gains", "v1299_value_sparse_replay_fold_shares",
        "v1299_value_sparse_replay_fold_bias", "v1299_value_sparse_replay_pass",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    select_cols = set(required)
    if "v12910_valuehat_raw_before_sparse" in schema:
        select_cols.add("v12910_valuehat_raw_before_sparse")
    leaf = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(select_cols))
        .with_columns(pl.col("unique_id").str.split("||").list.first().alias("seccion"))
        .collect()
    )
    if leaf.height == 0:
        return
    print("\n=== 32) v12.9.9 VALUE SPARSE BIAS RESCUE SEGMENTADO + MULTI-CUTOFF (DIAGNÓSTICO) ===")
    print("Diagnóstico v12.9.9 sobre baseline pre-v12.9.10; segmento causal=bucket días venta × método de nivel × flag estacional.")
    for sec in sorted(leaf.get_column("seccion").drop_nulls().unique().to_list()):
        q = leaf.filter(pl.col("seccion") == sec)
        base_value = (
            pl.col("v12910_valuehat_raw_before_sparse").round(2)
            if "v12910_valuehat_raw_before_sparse" in q.columns
            else pl.col("valuehat")
        )
        scored = q.filter(pl.col("value") > 0).with_columns(
            base_value.alias("_diag_base")
        ).with_columns(
            pl.when(pl.col("v1299_value_sparse_candidate").fill_null(False))
            .then(pl.col("_diag_base") * pl.col("v1299_value_sparse_factor").fill_null(1.0))
            .otherwise(pl.col("_diag_base"))
            .alias("_challenger")
        )
        if scored.height == 0:
            continue
        r = scored.select(
            pl.col("value").sum().alias("den"),
            (pl.col("_diag_base") - pl.col("value")).abs().sum().alias("ae0"),
            (pl.col("_diag_base") - pl.col("value")).sum().alias("se0"),
            (pl.col("_challenger") - pl.col("value")).abs().sum().alias("ae1"),
            (pl.col("_challenger") - pl.col("value")).sum().alias("se1"),
            pl.col("unique_id").filter(pl.col("v1299_value_sparse_candidate").fill_null(False)).n_unique().alias("n"),
        ).row(0, named=True)
        den = float(r["den"] or 0.0)
        if den <= 0:
            continue
        wm0, wm1 = float(r["ae0"] or 0.0)/den, float(r["ae1"] or 0.0)/den
        b0, b1 = float(r["se0"] or 0.0)/den, float(r["se1"] or 0.0)/den
        replay = q.select(
            pl.col("v1299_value_sparse_replay_cutoffs").max().alias("cutoffs"),
            pl.col("v1299_value_sparse_replay_win_rate").max().alias("win"),
            pl.col("v1299_value_sparse_replay_median_gain").drop_nulls().first().alias("med"),
            pl.col("v1299_value_sparse_replay_worst_gain").drop_nulls().first().alias("worst"),
            pl.col("v1299_value_sparse_replay_weighted_gain").drop_nulls().first().alias("weighted"),
            pl.col("v1299_value_sparse_replay_bias_worst").drop_nulls().first().alias("bias"),
            pl.col("v1299_value_sparse_replay_selected_share_median").drop_nulls().first().alias("share"),
            pl.col("v1299_value_sparse_replay_fold_gains").drop_nulls().first().alias("gains"),
            pl.col("v1299_value_sparse_replay_fold_shares").drop_nulls().first().alias("shares"),
            pl.col("v1299_value_sparse_replay_fold_bias").drop_nulls().first().alias("biases"),
            pl.col("v1299_value_sparse_replay_pass").max().alias("passed"),
        ).row(0, named=True)
        print(
            f"Valor ($) sec={sec:>3s} candidates={int(r['n'] or 0):4d} | "
            f"baseline={_pct(wm0)} -> challenger={_pct(wm1)} gain={_pct(wm0-wm1)} | "
            f"BIAS {_pct(b0)}->{_pct(b1)}"
        )
        print(
            f"  replay cutoffs={int(replay['cutoffs'] or 0)} win-rate={float(replay['win'] or 0.0):.0%} "
            f"median={_pct(replay['med'])} worst={_pct(replay['worst'])} weighted={_pct(replay['weighted'])} "
            f"worst-bias-worsen={_pct(replay['bias'])} selected-volume-med={float(replay['share'] or 0.0):.1%} "
            f"pass={'yes' if replay['passed'] else 'no'}"
        )
        print(f"  fold gains : [{replay['gains'] or 'N/A'}]")
        print(f"  fold shares: [{replay['shares'] or 'N/A'}]")
        print(f"  fold bias  : [{replay['biases'] or 'N/A'}]")
        segs = (
            q.filter(pl.col("v1299_value_sparse_candidate").fill_null(False))
            .select(
                "v1299_value_sparse_segment", "v1299_value_sparse_factor",
                "v1299_value_sparse_folds", "v1299_value_sparse_win_rate",
                "v1299_value_sparse_median_gain", "v1299_value_sparse_worst_gain",
                "v1299_value_sparse_weighted_gain", "v1299_value_sparse_bias_worst",
            )
            .unique()
            .sort("v1299_value_sparse_weighted_gain", descending=True)
        )
        for x in segs.head(12).iter_rows(named=True):
            print(
                f"  segment={x['v1299_value_sparse_segment']} factor={float(x['v1299_value_sparse_factor'] or 1.0):.3f} "
                f"folds={int(x['v1299_value_sparse_folds'] or 0)} win={float(x['v1299_value_sparse_win_rate'] or 0.0):.0%} "
                f"med={_pct(x['v1299_value_sparse_median_gain'])} worst={_pct(x['v1299_value_sparse_worst_gain'])} "
                f"weighted={_pct(x['v1299_value_sparse_weighted_gain'])} bias-worst={_pct(x['v1299_value_sparse_bias_worst'])}"
            )


def _print_v12910_value_sparse_promotion(forecast: Path) -> None:
    required = {
        "unique_id", "period_type", "value", "valuehat",
        "v12910_valuehat_raw_before_sparse", "v12910_value_sparse_applied",
        "v12910_value_sparse_promoted", "v12910_value_sparse_promotion_reason",
        "v12910_value_sparse_active_replay_cutoffs",
        "v12910_value_sparse_active_replay_win_rate",
        "v12910_value_sparse_active_replay_median_gain",
        "v12910_value_sparse_active_replay_worst_gain",
        "v12910_value_sparse_active_replay_weighted_gain",
        "v12910_value_sparse_active_replay_bias_worst",
        "v12910_value_sparse_active_replay_selected_share_median",
        "v12910_value_sparse_active_replay_fold_gains",
        "v12910_value_sparse_active_replay_fold_shares",
        "v12910_value_sparse_active_replay_fold_bias",
        "v1299_value_sparse_candidate", "v1299_value_sparse_factor",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    active_min = int(getattr(settings, "OOS_ACTIVE_MIN_NONZERO_DAYS", 7))
    leaf = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .with_columns(pl.col("unique_id").str.split("||").list.first().alias("seccion"))
        .collect()
    )
    if leaf.height == 0:
        return
    print("\n=== 33) v12.9.10 VALUE SPARSE RESCUE — ACTIVE PROMOTION GATE (PRODUCTIVO) ===")
    print("La promoción usa SOLO pseudo-OOS históricos ACTIVE; el OOS actual se muestra abajo únicamente como auditoría ex-post.")
    for sec in sorted(leaf.get_column("seccion").drop_nulls().unique().to_list()):
        q = leaf.filter(pl.col("seccion") == sec)
        meta = q.select(
            pl.col("v12910_value_sparse_promoted").max().alias("promoted"),
            pl.col("v12910_value_sparse_promotion_reason").drop_nulls().first().alias("reason"),
            pl.col("v12910_value_sparse_active_replay_cutoffs").max().alias("cutoffs"),
            pl.col("v12910_value_sparse_active_replay_win_rate").max().alias("win"),
            pl.col("v12910_value_sparse_active_replay_median_gain").drop_nulls().first().alias("med"),
            pl.col("v12910_value_sparse_active_replay_worst_gain").drop_nulls().first().alias("worst"),
            pl.col("v12910_value_sparse_active_replay_weighted_gain").drop_nulls().first().alias("weighted"),
            pl.col("v12910_value_sparse_active_replay_bias_worst").drop_nulls().first().alias("bias"),
            pl.col("v12910_value_sparse_active_replay_selected_share_median").drop_nulls().first().alias("share"),
            pl.col("v12910_value_sparse_active_replay_fold_gains").drop_nulls().first().alias("gains"),
            pl.col("v12910_value_sparse_active_replay_fold_shares").drop_nulls().first().alias("shares"),
            pl.col("v12910_value_sparse_active_replay_fold_bias").drop_nulls().first().alias("biases"),
            pl.col("unique_id").filter(pl.col("v12910_value_sparse_applied").fill_null(False)).n_unique().alias("applied"),
        ).row(0, named=True)
        print(
            f"Valor ($) sec={sec:>3s} promoted={'yes' if meta['promoted'] else 'no'} applied-leaves={int(meta['applied'] or 0)} "
            f"reason={meta['reason'] or 'NA'}"
        )
        print(
            f"  ACTIVE replay cutoffs={int(meta['cutoffs'] or 0)} win-rate={float(meta['win'] or 0.0):.0%} "
            f"median={_pct(meta['med'])} worst={_pct(meta['worst'])} weighted={_pct(meta['weighted'])} "
            f"worst-bias-worsen={_pct(meta['bias'])} selected-volume-med={float(meta['share'] or 0.0):.1%}"
        )
        print(f"  fold gains : [{meta['gains'] or 'N/A'}]")
        print(f"  fold shares: [{meta['shares'] or 'N/A'}]")
        print(f"  fold bias  : [{meta['biases'] or 'N/A'}]")

        # Current OOS ACTIVE is an audit only. Cohort membership is derived from
        # current actuals exactly like backend.wmape_bottom_up: >= active_min
        # non-zero days at leaf level.
        active_ids = (
            q.group_by("unique_id")
            .agg((pl.col("value") != 0).sum().alias("n_sales"))
            .filter(pl.col("n_sales") >= active_min)
            .select("unique_id")
        )
        qa = q.join(active_ids, on="unique_id", how="inner").filter(pl.col("value") > 0)
        if qa.height:
            r = qa.select(
                pl.col("value").sum().alias("den"),
                (pl.col("v12910_valuehat_raw_before_sparse").round(2) - pl.col("value")).abs().sum().alias("ae0"),
                (pl.col("v12910_valuehat_raw_before_sparse").round(2) - pl.col("value")).sum().alias("se0"),
                (pl.col("valuehat") - pl.col("value")).abs().sum().alias("ae1"),
                (pl.col("valuehat") - pl.col("value")).sum().alias("se1"),
            ).row(0, named=True)
            den = float(r["den"] or 0.0)
            if den > 0:
                wm0, wm1 = float(r["ae0"] or 0.0)/den, float(r["ae1"] or 0.0)/den
                b0, b1 = float(r["se0"] or 0.0)/den, float(r["se1"] or 0.0)/den
                print(
                    f"  OOS ACTIVE audit: baseline={_pct(wm0)} -> final={_pct(wm1)} gain={_pct(wm0-wm1)} | "
                    f"BIAS {_pct(b0)}->{_pct(b1)}"
                )


def _print_v12911_residual_selector_gap(forecast: Path, metrics: pl.DataFrame) -> None:
    """Audit the v12.9.11 residual Sec23 Value challenger on current OOS.

    The policy is diagnostic only.  Historical replay metadata was fixed before
    current OOS was observed.  Here we only measure the hypothetical incremental
    switch on the official ACTIVE support while keeping the productive v12.9.10
    sparse factor, when applicable, on top of the alternative family.
    """
    required = {
        "unique_id", "period_type", "value", "valuehat",
        "v12_candidate_valuehat_raw", "v12_selected_value",
        "v1299_value_sparse_factor", "v12910_value_sparse_applied",
        "v12911_residual_candidate_value", "v12911_residual_folds_value",
        "v12911_residual_win_rate_value", "v12911_residual_median_gain_value",
        "v12911_residual_p25_gain_value", "v12911_residual_worst_gain_value",
        "v12911_residual_weighted_gain_value", "v12911_residual_recent_gain_value",
        "v12911_residual_bias_worst_value", "v12911_residual_score_value",
        "v12911_residual_selected_share_value", "v12911_residual_replay_cutoffs_value",
        "v12911_residual_replay_win_rate_value", "v12911_residual_replay_median_gain_value",
        "v12911_residual_replay_worst_gain_value", "v12911_residual_replay_weighted_gain_value",
        "v12911_residual_replay_bias_worst_value", "v12911_residual_replay_selected_share_median_value",
        "v12911_residual_replay_fold_gains_value", "v12911_residual_replay_fold_shares_value",
        "v12911_residual_replay_fold_bias_value", "v12911_residual_replay_pass_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    active_ids = metrics.filter(
        (pl.col("unidad") == "Valor ($)")
        & (pl.col("node_kind") == "tienda_sku")
        & pl.col("metric_active").fill_null(False)
        & (pl.col("seccion").cast(pl.Utf8) == "23")
    ).select("unique_id")
    q = (
        pl.scan_parquet(forecast)
        .filter(
            (pl.col("period_type") == "out_sample")
            & pl.col("unique_id").str.starts_with("23||")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .select(*sorted(required))
        .collect()
        .join(active_ids, on="unique_id", how="semi")
    )
    if q.height == 0:
        return
    meta = q.select(
        pl.col("v12911_residual_candidate_value").fill_null(False).any().alias("any_candidate"),
        pl.col("unique_id").filter(pl.col("v12911_residual_candidate_value").fill_null(False)).n_unique().alias("selected_n"),
        pl.col("v12911_residual_selected_share_value").drop_nulls().first().alias("target_share"),
        pl.col("v12911_residual_replay_cutoffs_value").max().alias("cutoffs"),
        pl.col("v12911_residual_replay_win_rate_value").drop_nulls().first().alias("win"),
        pl.col("v12911_residual_replay_median_gain_value").drop_nulls().first().alias("med"),
        pl.col("v12911_residual_replay_worst_gain_value").drop_nulls().first().alias("worst"),
        pl.col("v12911_residual_replay_weighted_gain_value").drop_nulls().first().alias("weighted"),
        pl.col("v12911_residual_replay_bias_worst_value").drop_nulls().first().alias("bias"),
        pl.col("v12911_residual_replay_selected_share_median_value").drop_nulls().first().alias("share_med"),
        pl.col("v12911_residual_replay_fold_gains_value").drop_nulls().first().alias("gains"),
        pl.col("v12911_residual_replay_fold_shares_value").drop_nulls().first().alias("shares"),
        pl.col("v12911_residual_replay_fold_bias_value").drop_nulls().first().alias("biases"),
        pl.col("v12911_residual_replay_pass_value").fill_null(False).any().alias("pass"),
    ).row(0, named=True)
    print("\n=== 34) v12.9.11 SEC23 VALUE RESIDUAL SELECTOR-GAP CHALLENGER (DIAGNÓSTICO) ===")
    print("v12.9.10 permanece productivo. El challenger solo reconsidera hojas Sec23 que siguen en v11; OOS actual es auditoría ex-post.")
    print(
        f"selected-leaves={int(meta['selected_n'] or 0)} target-volume-share={float(meta['target_share'] or 0.0):.1%} | "
        f"replay cutoffs={int(meta['cutoffs'] or 0)} win-rate={float(meta['win'] or 0.0):.0%} "
        f"median={_pct(meta['med'])} worst={_pct(meta['worst'])} weighted={_pct(meta['weighted'])} "
        f"worst-bias-worsen={_pct(meta['bias'])} replay-share-med={float(meta['share_med'] or 0.0):.1%} "
        f"pass={'yes' if meta['pass'] else 'no'}"
    )
    print(f"  fold gains : [{meta['gains'] or 'N/A'}]")
    print(f"  fold shares: [{meta['shares'] or 'N/A'}]")
    print(f"  fold bias  : [{meta['biases'] or 'N/A'}]")

    positive = q.filter(pl.col("value") > 0).with_columns(
        pl.when(pl.col("v12910_value_sparse_applied").fill_null(False))
        .then(pl.col("v12_candidate_valuehat_raw") * pl.col("v1299_value_sparse_factor").fill_null(1.0))
        .otherwise(pl.col("v12_candidate_valuehat_raw"))
        .round(2).alias("_v12911_alt"),
    ).with_columns(
        pl.when(pl.col("v12911_residual_candidate_value").fill_null(False))
        .then(pl.col("_v12911_alt"))
        .otherwise(pl.col("valuehat"))
        .alias("_v12911_challenger")
    )
    if positive.height:
        r = positive.select(
            pl.col("value").sum().alias("den"),
            (pl.col("valuehat") - pl.col("value")).abs().sum().alias("ae0"),
            (pl.col("valuehat") - pl.col("value")).sum().alias("se0"),
            (pl.col("_v12911_challenger") - pl.col("value")).abs().sum().alias("ae1"),
            (pl.col("_v12911_challenger") - pl.col("value")).sum().alias("se1"),
        ).row(0, named=True)
        den = float(r["den"] or 0.0)
        if den > 0:
            w0, w1 = float(r["ae0"] or 0.0)/den, float(r["ae1"] or 0.0)/den
            b0, b1 = float(r["se0"] or 0.0)/den, float(r["se1"] or 0.0)/den
            print(
                f"  OOS ACTIVE hypothetical: baseline v12.9.10={_pct(w0)} -> challenger={_pct(w1)} "
                f"gain={_pct(w0-w1)} | BIAS {_pct(b0)}->{_pct(b1)}"
            )

    top = (
        q.filter(pl.col("v12911_residual_candidate_value").fill_null(False))
        .select(
            "unique_id", "v12911_residual_folds_value", "v12911_residual_win_rate_value",
            "v12911_residual_median_gain_value", "v12911_residual_p25_gain_value",
            "v12911_residual_worst_gain_value", "v12911_residual_weighted_gain_value",
            "v12911_residual_recent_gain_value", "v12911_residual_bias_worst_value",
            "v12911_residual_score_value",
        )
        .unique("unique_id")
        .sort("v12911_residual_score_value", descending=True)
        .head(12)
    )
    if top.height:
        print("  top residual leaves por score causal:")
        for x in top.iter_rows(named=True):
            print(
                f"    {str(x['unique_id']):32s} folds={int(x['v12911_residual_folds_value'] or 0):2d} "
                f"win={float(x['v12911_residual_win_rate_value'] or 0.0):.0%} "
                f"med={_pct(x['v12911_residual_median_gain_value'])} p25={_pct(x['v12911_residual_p25_gain_value'])} "
                f"worst={_pct(x['v12911_residual_worst_gain_value'])} weighted={_pct(x['v12911_residual_weighted_gain_value'])} "
                f"recent={_pct(x['v12911_residual_recent_gain_value'])} bias-worst={_pct(x['v12911_residual_bias_worst_value'])}"
            )


def _print_v12912_ultra_stable_residual(forecast: Path) -> None:
    """Report the strict v12.9.12 residual diagnostic; never promotes forecasts."""
    required = {
        "period_type", "unique_id", "v12912_ultra_enabled_value",
        "v12912_ultra_candidate_value", "v12912_ultra_replay_pass_value",
        "v12912_ultra_reason_value",
    }
    schema = set(pl.scan_parquet(forecast).collect_schema().names())
    if not required.issubset(schema):
        return
    q = (
        pl.scan_parquet(forecast)
        .filter((pl.col("period_type") == "out_sample") & pl.col("unique_id").str.starts_with("23||"))
        .select(*sorted(required))
        .collect()
        .unique("unique_id")
    )
    if q.height == 0:
        return
    n = q.filter(pl.col("v12912_ultra_candidate_value").fill_null(False)).height
    replay = bool(q.select(pl.col("v12912_ultra_replay_pass_value").fill_null(False).any()).item())
    actionable = q.filter(
        pl.col("v12912_ultra_candidate_value").fill_null(False)
        & pl.col("v12912_ultra_replay_pass_value").fill_null(False)
    ).height
    print("\n=== 35) v12.9.12 ULTRA-STABLE RESIDUAL GATE (DIAGNÓSTICO) ===")
    print("Productivo congelado: esta capa no modifica valuehat/yhat.")
    print(
        f"strict-leaf-candidates={n} | replay-pass={'yes' if replay else 'no'} | "
        f"actionable={actionable} | budget<=3%"
    )
    if actionable:
        ids = q.filter(
            pl.col("v12912_ultra_candidate_value").fill_null(False)
            & pl.col("v12912_ultra_replay_pass_value").fill_null(False)
        ).get_column("unique_id").head(12).to_list()
        print("  candidates:", ", ".join(map(str, ids)))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reporte de aceptación estadística")
    parser.add_argument("--forecast", default=None)
    parser.add_argument("--update-block-days", type=int, choices=list(getattr(settings, "UPDATE_BLOCK_OPTIONS", (1,7,14,28))), default=None)
    args = parser.parse_args(argv)
    forecast = (
        settings.update_block_forecast_path(args.update_block_days)
        if args.update_block_days is not None
        else Path(args.forecast or settings.FORECAST_PATH)
    )
    adir = artifacts.artifacts_dir(forecast)
    mpath = adir / "metrics.parquet"
    if not forecast.exists():
        raise FileNotFoundError(f"No existe forecast: {forecast}")
    status_errors = run_status_errors(forecast)
    if status_errors:
        raise RuntimeError(
            "No se puede ejecutar aceptación sobre un forecast stale/no válido: "
            + "; ".join(status_errors)
        )
    if not mpath.exists():
        raise FileNotFoundError(f"No existe metrics dashboard: {mpath}")
    index = artifacts.load_index(forecast)
    if not artifacts.artifacts_match_source(forecast, index):
        raise RuntimeError(
            "No se puede ejecutar aceptación: metrics/index del dashboard no "
            "corresponden al forecast actual. Reconstruya artefactos después de una corrida exitosa."
        )

    metrics = _parts(pl.read_parquet(mpath))
    print(f"ACEPTACIÓN ESTADÍSTICA | APP_VERSION={getattr(settings, 'APP_VERSION', '?')}")
    if index.get("update_block_days"):
        print(f"Bloque de actualización: {index.get('update_block_days')} días | OOS comparable: {index.get('metric_horizon_days', 28)} días")
    print(f"Forecast: {forecast}")
    print(f"Metrics : {mpath}")
    _print_section_summary(metrics)
    _print_all_cohort_summary(metrics)
    _print_leaf_distribution(metrics)
    _print_store_summary(metrics)
    _print_error_contributors(metrics)
    _print_methods(forecast, metrics)
    _print_method_quality(forecast, metrics)
    _print_sales_frequency_quality(metrics)
    _print_driver_mode_quality(forecast, metrics)
    _print_sku_seasonal_quality(forecast, metrics)
    _print_v12_family_quality(forecast, metrics)
    _print_v12_counterfactual(forecast, metrics)
    _print_v12_selection_stability(forecast, metrics)
    _print_v12_selected_counterfactual(forecast, metrics)
    _print_v12_joint_consistency(forecast)
    _print_v12_sku_total_ensemble_quality(forecast)
    _print_v12_share_strategy(forecast)
    _print_v12_lgbm_shape_quality(forecast)
    _print_v12_level_calibration_quality(forecast)
    _print_v12_oracle_selector_bound(forecast, metrics)
    _print_v12_meta_selector(forecast, metrics)
    _print_v12_value_portfolio_safety(forecast, metrics)
    _print_v129_value_walkforward(forecast, metrics)
    _print_v1291_expected_gain_selector(forecast, metrics)
    _print_v1293_hierarchical_segment_selector(forecast, metrics)
    _print_v1294_volume_risk_control(forecast, metrics)
    _print_v1295_long_horizon_stress(forecast, metrics)
    _print_v1296_sec23_stress_gate(forecast, metrics)
    _print_v1297_temporal_replay(forecast, metrics)
    _print_v1298_sparse_bias_diagnostic(forecast)
    _print_v1299_value_sparse_segmented(forecast)
    _print_v12910_value_sparse_promotion(forecast)
    _print_v12911_residual_selector_gap(forecast, metrics)
    _print_v12912_ultra_stable_residual(forecast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
