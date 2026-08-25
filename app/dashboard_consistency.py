"""Auditoría offline de coherencia de artefactos del dashboard.

Uso:
    python -m app.dashboard_consistency

No modifica datos. Falla con exit code 1 si encuentra inconsistencias estructurales.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

import settings

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

        # Identidad sección = suma bottom-up de errores absolutos de sus hojas.
        leaf_by_sec = (
            leaves.group_by("_sec")
            .agg(
                (pl.col("wmape") * pl.col("sum_y").abs())
                .sum()
                .alias("_ae"),
                pl.col("sum_y").sum().alias("_sum_y"),
            )
            .with_columns(
                pl.when(pl.col("_sum_y") != 0)
                .then(pl.col("_ae") / pl.col("_sum_y").abs())
                .otherwise(0.0)
                .alias("_wmape_leaf")
            )
        )
        sec = u.filter(
            pl.col("_store").is_null() & pl.col("_sku").is_null()
        ).select(
            pl.col("_sec"),
            pl.col("wmape").alias("_wmape_node"),
        )
        cmp = leaf_by_sec.join(sec, on="_sec", how="left").filter(
            pl.col("_wmape_node").is_null()
            | ((pl.col("_wmape_leaf") - pl.col("_wmape_node")).abs() > 1e-9)
        )
        if cmp.height:
            errors.append(
                f"{unidad}: {cmp.height} secciones no cumplen identidad bottom-up"
            )

        # Identidad tienda = suma bottom-up de hojas de la tienda.
        leaf_by_store = (
            leaves.with_columns(
                pl.concat_str(
                    [pl.col("_sec"), pl.lit("||T:"), pl.col("_store")]
                ).alias("_scope_uid")
            )
            .group_by("_scope_uid")
            .agg(
                (pl.col("wmape") * pl.col("sum_y").abs()).sum().alias("_ae"),
                pl.col("sum_y").sum().alias("_sum_y"),
            )
            .with_columns(
                pl.when(pl.col("_sum_y") != 0)
                .then(pl.col("_ae") / pl.col("_sum_y").abs())
                .otherwise(0.0)
                .alias("_wmape_leaf")
            )
        )
        store_nodes = u.filter(
            pl.col("_store").is_not_null() & pl.col("_sku").is_null()
        ).select(
            pl.col("unique_id").alias("_scope_uid"),
            pl.col("wmape").alias("_wmape_node"),
        )
        cmp_store = leaf_by_store.join(
            store_nodes, on="_scope_uid", how="left"
        ).filter(
            pl.col("_wmape_node").is_null()
            | ((pl.col("_wmape_leaf") - pl.col("_wmape_node")).abs() > 1e-9)
        )
        if cmp_store.height:
            errors.append(
                f"{unidad}: {cmp_store.height} tiendas no cumplen identidad bottom-up"
            )

        # Identidad SKU puro = suma de ese SKU entre tiendas.
        leaf_by_sku = (
            leaves.with_columns(
                pl.concat_str(
                    [pl.col("_sec"), pl.lit("||S:"), pl.col("_sku")]
                ).alias("_scope_uid")
            )
            .group_by("_scope_uid")
            .agg(
                (pl.col("wmape") * pl.col("sum_y").abs()).sum().alias("_ae"),
                pl.col("sum_y").sum().alias("_sum_y"),
            )
            .with_columns(
                pl.when(pl.col("_sum_y") != 0)
                .then(pl.col("_ae") / pl.col("_sum_y").abs())
                .otherwise(0.0)
                .alias("_wmape_leaf")
            )
        )
        sku_nodes = u.filter(
            pl.col("_store").is_null() & pl.col("_sku").is_not_null()
        ).select(
            pl.col("unique_id").alias("_scope_uid"),
            pl.col("wmape").alias("_wmape_node"),
        )
        cmp_sku = leaf_by_sku.join(
            sku_nodes, on="_scope_uid", how="left"
        ).filter(
            pl.col("_wmape_node").is_null()
            | ((pl.col("_wmape_leaf") - pl.col("_wmape_node")).abs() > 1e-9)
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
    """Recalcula una muestra de hojas directamente desde forecast.parquet."""
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
            .select("unique_id", "wmape")
            .head(sample_per_unit)
        )
        if target.height == 0:
            continue
        ids = target["unique_id"].to_list()
        source = (
            pl.scan_parquet(forecast)
            .filter(
                pl.col("unique_id").is_in(ids)
                & (pl.col("period_type") == "out_sample")
            )
            .select("unique_id", actual_col, pred_col)
            .collect()
        )
        if source.height == 0:
            errors.append(f"{unidad}: muestra OOS ausente en forecast.parquet")
            continue
        calc = (
            source.group_by("unique_id")
            .agg(
                (pl.col(actual_col) - pl.col(pred_col))
                .abs()
                .sum()
                .alias("_ae"),
                pl.col(actual_col).abs().sum().alias("_den"),
            )
            .with_columns(
                pl.when(pl.col("_den") > 0)
                .then(pl.col("_ae") / pl.col("_den"))
                .otherwise(0.0)
                .alias("_wmape_calc")
            )
        )
        cmp = (
            target.rename({"wmape": "_wmape_metric"})
            .join(calc, on="unique_id", how="left")
            .filter(
                pl.col("_wmape_calc").is_null()
                | (
                    (pl.col("_wmape_calc") - pl.col("_wmape_metric")).abs()
                    > tolerance
                )
            )
        )
        if cmp.height:
            errors.append(
                f"{unidad}: {cmp.height} hojas de la muestra no coinciden "
                "entre forecast.parquet y metrics.parquet"
            )
    return errors


def main() -> int:
    forecast = Path(settings.FORECAST_PATH)
    adir = artifacts.artifacts_dir(forecast)
    index = artifacts.load_index(forecast)
    metrics = artifacts.load_metrics(forecast)

    errors: list[str] = []
    if int(index.get("version") or 0) != artifacts.ARTIFACT_VERSION:
        errors.append(
            f"index version={index.get('version')} != {artifacts.ARTIFACT_VERSION}"
        )
    if not artifacts.artifacts_match_source(forecast, index):
        errors.append(
            "fingerprint de forecast.parquet no coincide con index.json"
        )
    expected_rows = int(
        (index.get("forecast_fingerprint") or {}).get("n_rows") or -1
    )
    if expected_rows >= 0:
        actual_rows = int(
            pl.scan_parquet(forecast).select(pl.len()).collect().item()
        )
        if actual_rows != expected_rows:
            errors.append(
                f"forecast row count={actual_rows} != index n_rows={expected_rows}"
            )
    errors.extend(audit_metrics(metrics))
    errors.extend(audit_metric_sample(forecast, metrics))

    if errors:
        print("AUDITORÍA DASHBOARD: ERROR")
        for e in errors:
            print(f" - {e}")
        print(f"Artefactos: {adir}")
        return 1

    print("AUDITORÍA DASHBOARD: OK")
    print(f" - artifact version: {artifacts.ARTIFACT_VERSION}")
    print(f" - metric rows: {metrics.height:,}")
    print(" - claves unique_id/unidad únicas")
    print(" - horizonte ranking OOS consistente")
    print(" - wMAPE de sección consistente con bottom-up de hojas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
