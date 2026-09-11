"""Auditoría de continuidad en la frontera in-sample -> OOS.

Diagnóstico puro: no modifica forecasts, métricas ni artefactos del dashboard.
Sirve para localizar si un salto agregado al iniciar OOS proviene del nivel SES,
del factor RLS/driver seleccionado o del cap leaf.

Ejemplos:
    uv run python -m app.forecasting.oos_boundary_audit --update-block-days 1
    uv run python -m app.forecasting.oos_boundary_audit --all-update-blocks
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import settings

_LEAF_RE = r"\|\|T:[^|]+\|\|S:[^|]+"
_TARGETS = {
    "Unidades": {
        "actual": "y",
        "forecast": "yhat_raw",
        "level": "ses_level_y",
        "factor": "driver_factor_y",
        "cap": "leaf_forecast_cap_y",
        "alpha": "ses_alpha_y",
        "parent": "parent_model_y",
        "gap": "leaf_gap_observable_days_y",
        "gap_factor": "leaf_gap_decay_factor_y",
        "effect": "driver_effect",
        "effect_raw": "driver_effect_raw",
        "effect_ref": "driver_effect_reference",
        "effect_center": "driver_effect_center",
        "parent_rls": "driver_parent_rls_forecast_y",
        "reference": "driver_reference_level_y",
    },
    "Valor ($)": {
        "actual": "value",
        "forecast": "valuehat_raw",
        "level": "ses_level_value",
        "factor": "driver_factor_value",
        "cap": "leaf_forecast_cap_value",
        "alpha": "ses_alpha_value",
        "parent": "parent_model_value",
        "gap": "leaf_gap_observable_days_value",
        "gap_factor": "leaf_gap_decay_factor_value",
        "effect": "driver_effect_value",
        "effect_raw": "driver_effect_value_raw",
        "effect_ref": "driver_effect_value_reference",
        "effect_center": "driver_effect_value_center",
        "parent_rls": "driver_parent_rls_forecast_value",
        "reference": "driver_reference_level_value",
    },
}


def _scan(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("unique_id").cast(pl.Utf8).str.contains(_LEAF_RE))
        .with_columns(
            pl.col("unique_id").cast(pl.Utf8),
            pl.col("unique_id").cast(pl.Utf8).str.extract(r"^([^|]+)", 1).alias("section"),
        )
    )


def _required_columns(schema: dict[str, pl.DataType]) -> None:
    required = {
        "unique_id", "ds", "period_type", "y", "value", "yhat_raw", "valuehat_raw",
        "ses_level_y", "ses_level_value", "driver_factor_y", "driver_factor_value",
        "leaf_forecast_cap_y", "leaf_forecast_cap_value",
        "parent_model_y", "parent_model_value",
    }
    missing = sorted(required - set(schema))
    if missing:
        raise RuntimeError("forecast.parquet no contiene trazas requeridas: " + ", ".join(missing))


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.abs() > 1e-12).then(num / den).otherwise(None)


def _uncapped_expr(level: str, factor: str) -> pl.Expr:
    # Identidad productiva exacta del leaf SES+RLS.
    return (
        ((pl.col(level).clip(lower_bound=0.0) + 1.0)
         * pl.col(factor).fill_null(1.0).clip(lower_bound=1e-12)
         - 1.0)
        .clip(lower_bound=0.0)
    )


def _audit_target(lf: pl.LazyFrame, days: int, target: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    c = _TARGETS[target]
    forecast = c["forecast"]
    level = c["level"]
    factor = c["factor"]
    cap = c["cap"]

    # Frontera global OOS por corrida/sección. El horizonte es común a hojas.
    bounds = (
        lf.group_by("section")
        .agg(
            pl.col("ds").filter(pl.col("period_type") == "in_sample").max().alias("last_in_sample_ds"),
            pl.col("ds").filter(pl.col("period_type") == "out_sample").min().alias("first_oos_ds"),
        )
        .collect()
    )
    if bounds.height == 0:
        return pl.DataFrame(), pl.DataFrame()

    base = lf.join(bounds.lazy(), on="section", how="left")
    recent = base.filter(
        (pl.col("period_type") == "in_sample")
        & (pl.col("ds") > pl.col("last_in_sample_ds") - pl.duration(days=28))
        & (pl.col("ds") <= pl.col("last_in_sample_ds"))
    )
    first = base.filter(
        (pl.col("period_type") == "out_sample") & (pl.col("ds") == pl.col("first_oos_ds"))
    )

    # Baseline leaf: mediana diaria de forecast in-sample de los 28 días previos.
    leaf_recent = (
        recent.group_by(["section", "unique_id"])
        .agg(
            pl.col(forecast).median().alias("recent_insample_forecast_median"),
            pl.col(c["actual"]).filter(pl.col("y") > 0).median().alias("recent_positive_actual_median"),
        )
    )

    optional = [
        c["alpha"], c["parent"], c["gap"], c["gap_factor"], c["effect"], c["effect_raw"],
        c["effect_ref"], c["effect_center"], c["parent_rls"], c["reference"],
    ]
    first_cols = ["section", "unique_id", "ds", forecast, level, factor, cap] + [
        x for x in optional if x in lf.collect_schema().names()
    ]
    leaf = (
        first.select(first_cols)
        .join(leaf_recent, on=["section", "unique_id"], how="left")
        .with_columns(
            _uncapped_expr(level, factor).alias("first_oos_uncapped_reconstructed"),
            pl.when(pl.col(cap) > 0)
            .then(pl.min_horizontal(_uncapped_expr(level, factor), pl.col(cap)))
            .otherwise(_uncapped_expr(level, factor))
            .alias("first_oos_capped_reconstructed"),
            (pl.col(forecast) - pl.col("recent_insample_forecast_median").fill_null(0.0)).alias("boundary_delta"),
            _safe_div(pl.col(forecast), pl.col("recent_insample_forecast_median")).alias("boundary_ratio"),
            _safe_div(_uncapped_expr(level, factor), pl.col(level).clip(lower_bound=1e-12)).alias("factor_amplification_vs_level"),
            pl.lit(target).alias("target"),
            pl.lit(int(days)).alias("update_block_days"),
        )
        .collect()
        .sort("boundary_delta", descending=True)
    )

    # Agregado por sección con descomposición exacta del primer día OOS.
    section = (
        leaf.lazy()
        .group_by("section")
        .agg(
            pl.col("recent_insample_forecast_median").sum().alias("recent_insample_leaf_median_sum"),
            pl.col(level).sum().alias("first_oos_ses_level_sum"),
            pl.col("first_oos_uncapped_reconstructed").sum().alias("first_oos_uncapped_sum"),
            pl.col(forecast).sum().alias("first_oos_final_sum"),
            pl.col("boundary_delta").sum().alias("aggregate_boundary_delta"),
            (pl.col(factor) > 1.05).sum().alias("leaves_factor_gt_1_05"),
            (pl.col(factor) > 1.25).sum().alias("leaves_factor_gt_1_25"),
            (pl.col(factor) > 2.0).sum().alias("leaves_factor_gt_2"),
            (pl.col(cap) > 0).sum().alias("leaves_with_cap"),
            pl.len().alias("n_leaves"),
        )
        .with_columns(
            _safe_div(pl.col("first_oos_ses_level_sum"), pl.col("recent_insample_leaf_median_sum")).alias("ratio_level_vs_recent"),
            _safe_div(pl.col("first_oos_uncapped_sum"), pl.col("first_oos_ses_level_sum")).alias("ratio_after_factor_vs_level"),
            _safe_div(pl.col("first_oos_final_sum"), pl.col("recent_insample_leaf_median_sum")).alias("ratio_final_vs_recent"),
            pl.lit(target).alias("target"),
            pl.lit(int(days)).alias("update_block_days"),
        )
        .collect()
        .sort(["section", "target"])
    )
    return section, leaf


def audit_path(path: Path, days: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    lf = _scan(path)
    _required_columns(dict(lf.collect_schema()))
    sections: list[pl.DataFrame] = []
    leaves: list[pl.DataFrame] = []
    for target in _TARGETS:
        sec, leaf = _audit_target(lf, days, target)
        if sec.height:
            sections.append(sec)
        if leaf.height:
            leaves.append(leaf)
    return (
        pl.concat(sections, how="diagonal_relaxed") if sections else pl.DataFrame(),
        pl.concat(leaves, how="diagonal_relaxed") if leaves else pl.DataFrame(),
    )


def _print_summary(summary: pl.DataFrame) -> None:
    if summary.height == 0:
        print("Sin filas para auditar.")
        return
    cols = [
        "update_block_days", "section", "target",
        "recent_insample_leaf_median_sum", "first_oos_ses_level_sum",
        "first_oos_uncapped_sum", "first_oos_final_sum",
        "ratio_level_vs_recent", "ratio_after_factor_vs_level", "ratio_final_vs_recent",
        "leaves_factor_gt_1_25", "leaves_factor_gt_2", "n_leaves",
    ]
    print(summary.select([c for c in cols if c in summary.columns]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Auditoría de continuidad in-sample -> OOS.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--update-block-days", type=int, choices=tuple(int(x) for x in settings.UPDATE_BLOCK_OPTIONS))
    group.add_argument("--all-update-blocks", action="store_true")
    parser.add_argument("--top", type=int, default=50, help="Top hojas por contribución al salto por target/sección.")
    parser.add_argument("--section", type=str, default=None, help="Filtra impresión de detalle a una sección, p.ej. 1 o 23.")
    parser.add_argument("--target", choices=tuple(_TARGETS), default=None, help="Filtra impresión de detalle a un target.")
    args = parser.parse_args(argv)

    days_list = tuple(int(x) for x in settings.UPDATE_BLOCK_OPTIONS) if args.all_update_blocks else (int(args.update_block_days),)
    sec_parts: list[pl.DataFrame] = []
    leaf_parts: list[pl.DataFrame] = []
    for days in days_list:
        path = settings.update_block_forecast_path(days)
        if not path.exists():
            raise SystemExit(f"Falta {path}")
        sec, leaf = audit_path(path, days)
        if sec.height:
            sec_parts.append(sec)
        if leaf.height:
            ranked = (
                leaf.with_columns(
                    pl.col("boundary_delta").rank("dense", descending=True).over(["section", "target"]).alias("boundary_rank")
                )
                .filter(pl.col("boundary_rank") <= int(args.top))
            )
            leaf_parts.append(ranked)

    summary = pl.concat(sec_parts, how="diagonal_relaxed") if sec_parts else pl.DataFrame()
    leaves = pl.concat(leaf_parts, how="diagonal_relaxed") if leaf_parts else pl.DataFrame()

    out_summary = settings.OUT_DIR / "oos_boundary_section_summary.parquet"
    out_leaves = settings.OUT_DIR / "oos_boundary_leaf_contributors.parquet"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_summary)
    leaves.write_parquet(out_leaves)

    print(f"AUDITORÍA FRONTERA OOS v{settings.APP_VERSION}")
    _print_summary(summary)
    detail = leaves
    if args.section is not None and detail.height:
        detail = detail.filter(pl.col("section") == str(args.section))
    if args.target is not None and detail.height:
        detail = detail.filter(pl.col("target") == str(args.target))
    if detail.height:
        detail_cols = [
            "update_block_days", "section", "target", "boundary_rank", "unique_id",
            "recent_insample_forecast_median", "ses_level_y", "ses_level_value",
            "driver_factor_y", "driver_factor_value", "first_oos_uncapped_reconstructed",
            "yhat_raw", "valuehat_raw", "boundary_delta", "boundary_ratio",
            "parent_model_y", "parent_model_value", "leaf_gap_observable_days_y",
            "leaf_gap_observable_days_value", "leaf_forecast_cap_y", "leaf_forecast_cap_value",
        ]
        print("\nTOP CONTRIBUYENTES AL SALTO")
        print(detail.select([c for c in detail_cols if c in detail.columns]).sort(
            ["update_block_days", "section", "target", "boundary_rank"]
        ).head(max(int(args.top), 1)))

    print(f"\nGuardado: {out_summary}")
    print(f"Guardado: {out_leaves}")


if __name__ == "__main__":
    main()
