"""Diagnóstico de nivel/driver para una hoja SKU+tienda.

Uso:
    python -m app.forecasting.diagnose_leaf --uid "1||T:00001||S:105322"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import settings


FIELDS = [
    "ds",
    "period_type",
    "y",
    "value",
    "yhat",
    "valuehat",
    "yhat_raw",
    "valuehat_raw",
    "ses_level_y",
    "ses_level_value",
    "recent28_mean_y",
    "recent28_mean_value",
    "ses_vs_recent28_ratio_y",
    "ses_vs_recent28_ratio_value",
    "driver_factor_y",
    "driver_factor_value",
    "driver_strength_y",
    "driver_strength_value",
    "ses_recent28_y",
    "ses_recent28_value",
    "ses_recent14_y",
    "ses_recent14_value",
    "ses_recent28_coverage_y",
    "ses_recent28_coverage_value",
    "ses_regime_anchor_y",
    "ses_regime_anchor_value",
    "ses_regime_class_y",
    "ses_regime_class_value",
    "ses_model_reference_block",
    "ses_structural_level_y",
    "ses_structural_level_value",
    "ses_structural_history_blocks_y",
    "ses_structural_history_blocks_value",
    "ses_block_b0_y",
    "ses_block_b0_value",
    "ses_block_b1_y",
    "ses_block_b1_value",
    "ses_block_b2_y",
    "ses_block_b2_value",
    "ses_block_b3_y",
    "ses_block_b3_value",
    "parent_model_y",
    "parent_model_value",
    "ses_alpha_y",
    "ses_alpha_value",
    "ses_alpha_unconstrained_y",
    "ses_alpha_unconstrained_value",
    "pure_ses_wmape_y",
    "pure_ses_bias_y",
    "pure_ses_wmape_value",
    "pure_ses_bias_value",
    "pure_ses_score_y",
    "pure_ses_score_value",
    "ses_score_history_blocks_y",
    "ses_score_history_blocks_value",
    "ses_score_window_y",
    "ses_score_window_value",
    "ses_guard_status_y",
    "ses_guard_status_value",
    "ses_stability_reference_y",
    "ses_stability_reference_value",
    "ses_stability_guard_y",
    "ses_stability_guard_value",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audita SES level → driver factor → forecast de una hoja."
    )
    parser.add_argument("--uid", required=True, help="unique_id SKU+tienda")
    parser.add_argument(
        "--forecast",
        default=str(settings.FORECAST_PATH),
        help="Ruta a forecast.parquet",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Imprime las 28 filas del período; por defecto muestra un resumen.",
    )
    args = parser.parse_args()

    path = Path(args.forecast)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pl.read_parquet(path)
    if "unique_id" not in df.columns:
        raise ValueError("forecast.parquet no contiene unique_id")

    leaf = df.filter(pl.col("unique_id") == args.uid).sort("ds")
    if leaf.height == 0:
        print(f"No se encontró {args.uid}")
        return 1

    cols = [c for c in FIELDS if c in leaf.columns]
    leaf = leaf.select(cols)

    print(f"=== Diagnóstico hoja {args.uid} ===")
    for period in ("in_sample", "out_sample", "forecast_only"):
        g = leaf.filter(pl.col("period_type") == period)
        if g.height == 0:
            continue
        print(f"\n[{period}] {g['ds'].min()} → {g['ds'].max()} | {g.height} filas")
        expressions = []
        for c in (
            "y", "value", "yhat", "valuehat", "yhat_raw", "valuehat_raw",
            "ses_level_y", "ses_level_value",
            "recent28_mean_y", "recent28_mean_value",
            "driver_factor_y", "driver_factor_value",
        ):
            if c in g.columns:
                expressions.extend([
                    pl.col(c).min().alias(f"{c}_min"),
                    pl.col(c).mean().alias(f"{c}_mean"),
                    pl.col(c).max().alias(f"{c}_max"),
                ])
        if expressions:
            print(g.select(expressions).to_dicts()[0])

        if period in ("out_sample", "forecast_only"):
            diag_cols = [
                c for c in (
                    "ses_model_reference_block",
                    "ses_regime_class_value",
                    "ses_structural_level_value",
                    "ses_structural_history_blocks_value",
                    "ses_block_b0_value",
                    "ses_block_b1_value",
                    "ses_block_b2_value",
                    "ses_block_b3_value",
                    "ses_alpha_value",
                    "ses_guard_status_value",
                    "pure_ses_wmape_value",
                    "pure_ses_bias_value",
                    "pure_ses_score_value",
                    "ses_level_value",
                    "recent28_mean_value",
                    "parent_model_value",
                    "driver_strength_value",
                )
                if c in g.columns
            ]
            if diag_cols:
                print("modelo:", g.select(diag_cols).head(1).to_dicts()[0])

            if "driver_factor_value" in g.columns:
                factor = g.select(
                    pl.col("driver_factor_value").min().alias("min"),
                    pl.col("driver_factor_value").mean().alias("mean"),
                    pl.col("driver_factor_value").max().alias("max"),
                    pl.col("driver_factor_value").std().fill_null(0.0).alias("std"),
                ).to_dicts()[0]
                print("driver_factor_value:", factor)

            if args.all_rows:
                print(
                    g.select(
                        [c for c in (
                            "ds", "value", "valuehat", "valuehat_raw",
                            "ses_level_value", "driver_factor_value",
                            "parent_model_value",
                        ) if c in g.columns]
                    ).to_dicts()
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
