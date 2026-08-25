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
    "driver_strength_selected_y",
    "driver_strength_selected_value",
    "driver_strength_y",
    "driver_strength_value",
    "driver_direction_guard_y",
    "driver_direction_guard_value",
    "ses_recent28_y",
    "ses_recent28_value",
    "ses_recent14_y",
    "ses_recent14_value",
    "ses_recent28_coverage_y",
    "ses_recent28_coverage_value",
    "ses_regime_anchor_y",
    "ses_regime_anchor_value",
    "ses_sparse_robust_y",
    "ses_sparse_robust_value",
    "ses_sparse_shock_y",
    "ses_sparse_shock_value",
    "ses_robust_block_median_y",
    "ses_robust_block_median_value",
    "parent_model_y",
    "parent_model_value",
    "ses_alpha_y",
    "ses_alpha_value",
    "pure_ses_wmape_y",
    "pure_ses_wmape_value",
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
            print(
                g.select(
                    [c for c in (
                        "ds", "value", "valuehat", "valuehat_raw",
                        "ses_level_value", "recent28_mean_value",
                        "ses_stability_reference_value",
                        "ses_sparse_robust_value",
                        "ses_sparse_shock_value",
                        "ses_robust_block_median_value",
                        "ses_stability_guard_value",
                        "pure_ses_wmape_value",
                        "driver_factor_value", "driver_strength_value", "parent_model_value",
                        "ses_alpha_value",
                    ) if c in g.columns]
                ).to_dicts()
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
