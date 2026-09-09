"""Auditoría offline de coherencia de artefactos del dashboard.

Uso:
    python -m app.dashboard_consistency

No modifica datos. Falla con exit code 1 si encuentra inconsistencias estructurales.
"""
from __future__ import annotations

import argparse

import sys
from pathlib import Path

import polars as pl

import settings
from app.forecasting.run_status import validation_errors as run_status_errors

try:
    from app import dashboard_artifacts as artifacts
except ImportError:  # pragma: no cover
    import dashboard_artifacts as artifacts  # type: ignore


def _parts(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("unique_id").str.split("||").list.get(0).alias("_sec"),
        pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("_store"),
        pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("_sku"),
    )


def audit_metrics(metrics: pl.DataFrame) -> list[str]:
    errors: list[str] = []
    if metrics.height == 0:
        return ["metrics.parquet está vacío"]

    dup = (
        metrics.group_by(["unique_id", "unidad"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if dup.height:
        errors.append(f"metrics.parquet tiene {dup.height} claves duplicadas")

    # Ranking OOS: n_points debe representar 28 días, no el spine histórico.
    expected_days = int(getattr(settings, "METRIC_HORIZON_DAYS", 28))
    bad_days = metrics.filter(
        pl.col("n_points").is_not_null()
        & (pl.col("n_points") != expected_days)
    )
    if bad_days.height:
        errors.append(
            f"{bad_days.height} métricas tienen n_points != {expected_days} días OOS"
        )

    m = _parts(metrics)
    for unidad in m["unidad"].unique().to_list():
        u = m.filter(pl.col("unidad") == unidad)
        leaves = u.filter(
            pl.col("_store").is_not_null() & pl.col("_sku").is_not_null()
        )
        if leaves.height == 0:
            errors.append(f"{unidad}: no hay métricas de hojas SKU+tienda")
            continue

        required = {
            "sum_abs_error", "sum_abs_y", "sum_signed_error", "wmape", "bias",
            "metric_active", "metric_cohort",
        }
        missing = sorted(required - set(u.columns))
        if missing:
            errors.append(
                f"{unidad}: faltan componentes bottom-up en metrics.parquet: {missing}"
            )
            continue

        tol = float(
            getattr(settings, "DASHBOARD_METRIC_IDENTITY_TOLERANCE", 1e-9)
        )

        def _bad_identity(
            rebuilt: pl.DataFrame,
            nodes: pl.DataFrame,
            *,
            key: str,
        ) -> pl.DataFrame:
            return (
                rebuilt.join(nodes, on=key, how="left", suffix="_node")
                .with_columns(
                    pl.when(pl.col("sum_abs_y") > 0)
                    .then(pl.col("sum_abs_error") / pl.col("sum_abs_y"))
                    .otherwise(0.0)
                    .alias("_wmape_leaf"),
                    pl.when(pl.col("sum_abs_y") > 0)
                    .then(pl.col("sum_signed_error") / pl.col("sum_abs_y"))
                    .otherwise(0.0)
                    .alias("_bias_leaf"),
                )
                .with_columns(
                    (
                        (pl.col("_wmape_leaf") - pl.col("wmape")).abs()
                        / pl.max_horizontal(
                            pl.col("_wmape_leaf").abs(), pl.lit(1.0)
                        )
                    ).alias("_dw"),
                    (
                        (pl.col("_bias_leaf") - pl.col("bias")).abs()
                        / pl.max_horizontal(
                            pl.col("_bias_leaf").abs(), pl.lit(1.0)
                        )
                    ).alias("_db"),
                    (
                        (pl.col("sum_abs_error") - pl.col("sum_abs_error_node")).abs()
                        / pl.max_horizontal(
                            pl.col("sum_abs_error").abs(), pl.lit(1.0)
                        )
                    ).alias("_de"),
                    (
                        (pl.col("sum_abs_y") - pl.col("sum_abs_y_node")).abs()
                        / pl.max_horizontal(
                            pl.col("sum_abs_y").abs(), pl.lit(1.0)
                        )
                    ).alias("_dy"),
                    (
                        (
                            pl.col("sum_signed_error")
                            - pl.col("sum_signed_error_node")
                        ).abs()
                        / pl.max_horizontal(
                            pl.col("sum_signed_error").abs(), pl.lit(1.0)
                        )
                    ).alias("_ds"),
                )
                .filter(
                    pl.col("wmape").is_null()
                    | (pl.col("_dw") > tol)
                    | (pl.col("_db") > tol)
                    | (pl.col("_de") > tol)
                    | (pl.col("_dy") > tol)
                    | (pl.col("_ds") > tol)
                )
            )

        # Sección = suma exacta de componentes de todas sus hojas.
        active_leaves = leaves.filter(
            pl.col("metric_active").fill_null(False)
        )

        leaf_by_sec = active_leaves.group_by("_sec").agg(
            pl.col("sum_abs_error").sum(),
            pl.col("sum_abs_y").sum(),
            pl.col("sum_signed_error").sum(),
        )
        sec = u.filter(
            pl.col("_store").is_null() & pl.col("_sku").is_null()
        ).select(
            "_sec",
            "wmape",
            "bias",
            "sum_abs_error",
            "sum_abs_y",
            "sum_signed_error",
        )
        cmp = _bad_identity(leaf_by_sec, sec, key="_sec")
        if cmp.height:
            errors.append(
                f"{unidad}: {cmp.height} secciones no cumplen identidad bottom-up"
            )

        # Tienda = suma exacta de componentes de sus hojas.
        leaf_by_store = (
            active_leaves.with_columns(
                pl.concat_str(
                    [pl.col("_sec"), pl.lit("||T:"), pl.col("_store")]
                ).alias("_scope_uid")
            )
            .group_by("_scope_uid")
            .agg(
                pl.col("sum_abs_error").sum(),
                pl.col("sum_abs_y").sum(),
                pl.col("sum_signed_error").sum(),
            )
        )
        store_nodes = u.filter(
            pl.col("_store").is_not_null() & pl.col("_sku").is_null()
        ).select(
            pl.col("unique_id").alias("_scope_uid"),
            "wmape",
            "bias",
            "sum_abs_error",
            "sum_abs_y",
            "sum_signed_error",
        )
        cmp_store = _bad_identity(
            leaf_by_store, store_nodes, key="_scope_uid"
        )
        if cmp_store.height:
            errors.append(
                f"{unidad}: {cmp_store.height} tiendas no cumplen identidad bottom-up"
            )

        # SKU puro = suma exacta del mismo SKU a través de tiendas.
        leaf_by_sku = (
            active_leaves.with_columns(
                pl.concat_str(
                    [pl.col("_sec"), pl.lit("||S:"), pl.col("_sku")]
                ).alias("_scope_uid")
            )
            .group_by("_scope_uid")
            .agg(
                pl.col("sum_abs_error").sum(),
                pl.col("sum_abs_y").sum(),
                pl.col("sum_signed_error").sum(),
            )
        )
        sku_nodes = u.filter(
            pl.col("_store").is_null() & pl.col("_sku").is_not_null()
        ).select(
            pl.col("unique_id").alias("_scope_uid"),
            "wmape",
            "bias",
            "sum_abs_error",
            "sum_abs_y",
            "sum_signed_error",
        )
        cmp_sku = _bad_identity(
            leaf_by_sku, sku_nodes, key="_scope_uid"
        )
        if cmp_sku.height:
            errors.append(
                f"{unidad}: {cmp_sku.height} SKU puros no cumplen identidad bottom-up"
            )

    return errors


def audit_metric_sample(
    forecast: Path,
    metrics: pl.DataFrame,
    *,
    sample_per_unit: int = 12,
    tolerance: float = 1e-9,
) -> list[str]:
    """Recalcula una muestra de hojas directamente desde forecast.parquet.

    La recomputación es independiente de ``metrics.parquet`` pero respeta la
    definición oficial: los días con actual == 0 no participan en wMAPE/BIAS.
    El sobreforecast en demanda cero se audita por los componentes específicos
    de zero-demand, no mezclándolo en la métrica oficial.
    """
    errors: list[str] = []
    if metrics.height == 0:
        return errors

    m = _parts(metrics)
    leaves = m.filter(
        pl.col("_store").is_not_null() & pl.col("_sku").is_not_null()
    )
    for unidad, actual_col, pred_col in (
        ("Unidades", "y", "yhat"),
        ("Valor ($)", "value", "valuehat"),
    ):
        target = (
            leaves.filter(pl.col("unidad") == unidad)
            .select(
                "unique_id", "wmape", "bias", "sum_abs_y",
                "sum_abs_error", "sum_signed_error", "metric_cohort",
            )
            .head(sample_per_unit)
        )
        if target.height == 0:
            continue
        ids = target["unique_id"].to_list()

        lf = pl.scan_parquet(forecast)
        schema_names = set(lf.collect_schema().names())
        source_cols = {"unique_id", actual_col, pred_col, "period_type"}
        if "rls_metric_eligible" in schema_names:
            source_cols.add("rls_metric_eligible")
        source = (
            lf.filter(
                pl.col("unique_id").is_in(ids)
                & (pl.col("period_type") == "out_sample")
            )
            .select(sorted(source_cols))
            .collect()
        )
        if (
            "rls_metric_eligible" in source.columns
            and str(getattr(settings, "METRICS_MODE", "rolling_28")).lower()
            == "rolling_28"
        ):
            source = source.filter(pl.col("rls_metric_eligible") == True)
        source = source.filter(
            pl.col(actual_col).is_not_null()
            & pl.col(pred_col).is_not_null()
            & pl.col(actual_col).is_finite()
            & pl.col(pred_col).is_finite()
        )
        if source.height == 0:
            errors.append(f"{unidad}: muestra OOS ausente en forecast.parquet")
            continue

        scored = pl.col(actual_col) != 0
        abs_err = (pl.col(actual_col) - pl.col(pred_col)).abs()
        signed_err = pl.col(pred_col) - pl.col(actual_col)
        calc = (
            source.group_by("unique_id")
            .agg(
                pl.when(scored).then(abs_err).otherwise(0.0)
                .sum().alias("_sum_abs_error_calc"),
                pl.when(scored).then(pl.col(actual_col).abs()).otherwise(0.0)
                .sum().alias("_sum_abs_y_calc"),
                pl.when(scored).then(signed_err).otherwise(0.0)
                .sum().alias("_sum_signed_error_calc"),
            )
            .with_columns(
                pl.when(pl.col("_sum_abs_y_calc") > 0)
                .then(pl.col("_sum_abs_error_calc") / pl.col("_sum_abs_y_calc"))
                .otherwise(None)
                .alias("_wmape_calc"),
                pl.when(pl.col("_sum_abs_y_calc") > 0)
                .then(pl.col("_sum_signed_error_calc") / pl.col("_sum_abs_y_calc"))
                .otherwise(None)
                .alias("_bias_calc"),
            )
        )

        cmp = (
            target.rename({
                "wmape": "_wmape_metric",
                "bias": "_bias_metric",
                "sum_abs_y": "_sum_abs_y_metric",
                "sum_abs_error": "_sum_abs_error_metric",
                "sum_signed_error": "_sum_signed_error_metric",
            })
            .join(calc, on="unique_id", how="left")
            .filter(
                (pl.col("_wmape_calc").is_null() != pl.col("_wmape_metric").is_null())
                | (pl.col("_bias_calc").is_null() != pl.col("_bias_metric").is_null())
                | (
                    pl.col("_wmape_calc").is_not_null()
                    & pl.col("_wmape_metric").is_not_null()
                    & ((pl.col("_wmape_calc") - pl.col("_wmape_metric")).abs() > tolerance)
                )
                | (
                    pl.col("_bias_calc").is_not_null()
                    & pl.col("_bias_metric").is_not_null()
                    & ((pl.col("_bias_calc") - pl.col("_bias_metric")).abs() > tolerance)
                )
                | ((pl.col("_sum_abs_error_calc") - pl.col("_sum_abs_error_metric")).abs() > tolerance)
                | ((pl.col("_sum_abs_y_calc") - pl.col("_sum_abs_y_metric")).abs() > tolerance)
                | ((pl.col("_sum_signed_error_calc") - pl.col("_sum_signed_error_metric")).abs() > tolerance)
            )
        )
        if cmp.height:
            errors.append(
                f"{unidad}: {cmp.height} hojas de la muestra no coinciden "
                "entre forecast.parquet y metrics.parquet"
            )
    return errors


def _audit_one(forecast: Path) -> list[str]:
    adir = artifacts.artifacts_dir(forecast)
    index = artifacts.load_index(forecast)
    metrics = artifacts.load_metrics(forecast)
    errors: list[str] = []
    errors.extend([f"artefacto stale/no válido: {e}" for e in run_status_errors(forecast)])
    if int(index.get("version") or 0) != artifacts.ARTIFACT_VERSION:
        errors.append(f"index version={index.get('version')} != {artifacts.ARTIFACT_VERSION}")
    if not artifacts.artifacts_match_source(forecast, index):
        errors.append("fingerprint de forecast.parquet no coincide con index.json")
    expected_rows = int((index.get("forecast_fingerprint") or {}).get("n_rows") or -1)
    if expected_rows >= 0:
        actual_rows = int(pl.scan_parquet(forecast).select(pl.len()).collect().item())
        if actual_rows != expected_rows:
            errors.append(f"forecast row count={actual_rows} != index n_rows={expected_rows}")
    errors.extend(audit_metrics(metrics))
    errors.extend(audit_metric_sample(forecast, metrics))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditar artefactos dashboard")
    parser.add_argument("--forecast", default=None)
    parser.add_argument("--update-block-days", type=int, choices=list(getattr(settings, "UPDATE_BLOCK_OPTIONS", (1,7,14,28))), default=None)
    parser.add_argument("--all-update-blocks", action="store_true")
    args = parser.parse_args(argv)
    if args.all_update_blocks:
        targets = [(d, settings.update_block_forecast_path(d)) for d in settings.UPDATE_BLOCK_OPTIONS if settings.update_block_forecast_path(d).exists()]
    elif args.update_block_days is not None:
        targets = [(args.update_block_days, settings.update_block_forecast_path(args.update_block_days))]
    else:
        targets = [(None, Path(args.forecast or settings.FORECAST_PATH))]
    all_errors: list[str] = []
    for days, forecast in targets:
        errors = _audit_one(forecast)
        if errors:
            prefix = f"[{days}d] " if days else ""
            all_errors.extend(prefix + e for e in errors)
        else:
            index = artifacts.load_index(forecast)
            metrics = artifacts.load_metrics(forecast)
            label = f" ({days}d)" if days else ""
            print(f"AUDITORÍA DASHBOARD{label}: OK")
            print(f" - artifact version: {artifacts.ARTIFACT_VERSION}")
            print(f" - update block: {index.get('update_block_days', 28)} días")
            print(f" - metric rows: {metrics.height:,}")
            print(" - claves unique_id/unidad únicas")
            print(" - horizonte ranking OOS consistente (28d)")
            print(" - wMAPE de sección consistente con bottom-up de hojas")
    if all_errors:
        print("AUDITORÍA DASHBOARD: ERROR")
        for e in all_errors:
            print(" -", e)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
