"""Forecast accuracy metrics."""
from __future__ import annotations
import polars as pl

import settings

def compute_wmape(
    res_df: pl.DataFrame,
    period_types: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame:
        """
        WMAPE bottom-up, separado por in_sample / out_sample.

        1. Métrica oficial del cliente: solo días con venta real y != 0.
             Los falsos positivos en días y=0 se auditan por separado como
             zero-demand y NO inflan el wMAPE/BIAS oficial.
        2. WMAPE hoja = Σe / Σ|y| por unique_id y period_type
        3. Tienda / sección = suma de numeradores y denominadores de sus hojas
           (no usa el yhat del RLS de esos niveles).
        """
        empty = pl.DataFrame(
            schema={
                "unique_id": pl.Utf8,
                "period_type": pl.Utf8,
                "nivel": pl.Utf8,
                "wmape": pl.Float64,
                "bias": pl.Float64,
                "n_points": pl.UInt32,
                "sum_y": pl.Float64,
                "sum_yhat": pl.Float64,
                "sum_abs_y": pl.Float64,
                "sum_abs_error": pl.Float64,
                "sum_signed_error": pl.Float64,
            }
        )
        if res_df.height == 0 or "yhat" not in res_df.columns:
            return empty

        # Solo hojas: sec||T:xx||S:yy
        leaves = res_df.filter(
            pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2
        )
        leaves = leaves.filter(
            pl.col("y").is_not_null()
            & pl.col("yhat").is_not_null()
            & pl.col("y").is_finite()
            & pl.col("yhat").is_finite()
        )
        if (
            "rls_metric_eligible" in leaves.columns
            and str(getattr(settings, "METRICS_MODE", "rolling_28")).lower()
            == "rolling_28"
        ):
            leaves = leaves.filter(pl.col("rls_metric_eligible") == True)

        if "period_type" in leaves.columns:
            allowed = list(period_types) if period_types is not None else ["in_sample", "out_sample"]
            leaves = leaves.filter(pl.col("period_type").is_in(allowed))
        else:
            leaves = leaves.with_columns(pl.lit("in_sample").alias("period_type"))
            if period_types is not None and "in_sample" not in period_types:
                return empty

        if leaves.height == 0:
            return empty

        # Parse store_uid / seccion desde unique_id
        leaves = leaves.with_columns(
            pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("_store_uid"),
            pl.col("unique_id").str.split("||").list.first().alias("_seccion"),
            pl.when(pl.col("y") != 0)
            .then((pl.col("y") - pl.col("yhat")).abs())
            .otherwise(0.0)
            .alias("_abs_error"),
            pl.when(pl.col("y") != 0)
            .then(pl.col("y").abs())
            .otherwise(0.0)
            .alias("_abs_y"),
            pl.when(pl.col("y") != 0)
            .then(pl.col("yhat") - pl.col("y"))
            .otherwise(0.0)
            .alias("_signed_error"),
            pl.when(pl.col("y") != 0)
            .then(pl.col("yhat"))
            .otherwise(0.0)
            .alias("_metric_yhat"),
        )

        def _finalize(df: pl.DataFrame, nivel: str) -> pl.DataFrame:
            return df.with_columns(
                pl.when(pl.col("sum_abs_y") > 0)
                .then(pl.col("sum_abs_error") / pl.col("sum_abs_y"))
                .otherwise(None)
                .alias("wmape"),
                pl.when(pl.col("sum_abs_y") > 0)
                .then(
                    pl.col("sum_signed_error")
                    / pl.col("sum_abs_y").replace(0, None)
                )
                .otherwise(None)
                .alias("bias"),
                pl.lit(nivel).alias("nivel"),
            ).select(
                [
                    "unique_id",
                    "period_type",
                    "nivel",
                    "wmape",
                    "bias",
                    "n_points",
                    "sum_y",
                    "sum_yhat",
                    "sum_abs_y",
                    "sum_abs_error",
                    "sum_signed_error",
                ]
            )

        # ── Hojas SKU+tienda ───────────────────────────────────────────────
        leaf_agg = leaves.group_by(["unique_id", "period_type"]).agg(
            pl.col("_abs_error").sum().alias("sum_abs_error"),
            pl.col("_abs_y").sum().alias("sum_abs_y"),
            pl.col("_signed_error").sum().alias("sum_signed_error"),
            pl.col("y").sum().alias("sum_y"),
            pl.col("_metric_yhat").sum().alias("sum_yhat"),
            pl.len().alias("n_points"),
        )
        out_leaf = _finalize(leaf_agg, "sku_tienda")

        # ── Tienda (bottom-up desde hojas) ─────────────────────────────────
        store_agg = (
            leaves.group_by(["_store_uid", "period_type"])
            .agg(
                pl.col("_abs_error").sum().alias("sum_abs_error"),
                pl.col("_abs_y").sum().alias("sum_abs_y"),
                pl.col("_signed_error").sum().alias("sum_signed_error"),
                pl.col("y").sum().alias("sum_y"),
                pl.col("_metric_yhat").sum().alias("sum_yhat"),
                pl.len().alias("n_points"),
            )
            .rename({"_store_uid": "unique_id"})
        )
        out_store = _finalize(store_agg, "tienda")

        # ── Sección (bottom-up desde hojas) ────────────────────────────────
        sec_agg = (
            leaves.group_by(["_seccion", "period_type"])
            .agg(
                pl.col("_abs_error").sum().alias("sum_abs_error"),
                pl.col("_abs_y").sum().alias("sum_abs_y"),
                pl.col("_signed_error").sum().alias("sum_signed_error"),
                pl.col("y").sum().alias("sum_y"),
                pl.col("_metric_yhat").sum().alias("sum_yhat"),
                pl.len().alias("n_points"),
            )
            .rename({"_seccion": "unique_id"})
        )
        out_sec = _finalize(sec_agg, "seccion")

        return pl.concat([out_leaf, out_store, out_sec], how="vertical")
