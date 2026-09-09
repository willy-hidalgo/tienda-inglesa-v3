"""Structural validator for the coherent v13 SES+RLS forecasting contract.

This module validates model semantics, temporal horizons, output identities and
schema hygiene.  It does *not* optimize or judge forecast accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import settings
from app.forecasting.run_status import validation_errors as run_status_errors


_LEAF_RE = r"\|\|T:[^|]+\|\|S:[^|]+"
_OBSOLETE_COLUMN_TOKENS = (
    "v11_",
    "v12_",
    "v129",
    "occ_share",
    "occurrence",
    "lgbm",
    "lightgbm",
    "sparse_rescue",
    "meta_selector",
    "residual_selector",
)


def _leaf_expr() -> pl.Expr:
    return pl.col("unique_id").cast(pl.Utf8).str.contains(_LEAF_RE)


def _count(lf: pl.LazyFrame) -> int:
    return int(lf.select(pl.len().alias("n")).collect().item())


def _max_abs(lf: pl.LazyFrame, expr: pl.Expr) -> float:
    val = lf.select(expr.abs().max().alias("m")).collect().item()
    return float(val or 0.0)


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    path = Path(path)
    if not path.exists():
        return [f"No existe forecast: {path}"]

    status_errors = run_status_errors(path)
    errors.extend(f"artefacto stale/no válido: {e}" for e in status_errors)

    scan = pl.scan_parquet(path)
    cols = set(scan.collect_schema().names())
    required = {
        "unique_id", "ds", "period_type", "y", "value", "yhat", "valuehat",
        "yhat_raw", "valuehat_raw", "modelo_seleccionado", "rls_metric_eligible",
        "initial_level_y", "initial_level_value", "leaf_warmup_end",
        "ses_level_y", "ses_level_value", "ses_alpha_y", "ses_alpha_value",
        "parent_model_y", "parent_model_value", "parent_wmape_y", "parent_wmape_value",
        "driver_effect", "driver_effect_value", "driver_factor_y", "driver_factor_value",
        "rls_block", "rls_train_days",
    }
    missing = sorted(required - cols)
    if missing:
        errors.append(f"faltan columnas v13 requeridas: {missing}")
        return errors

    obsolete = sorted(
        c for c in cols if any(token in c.lower() for token in _OBSOLETE_COLUMN_TOKENS)
    )
    if obsolete:
        errors.append(f"forecast conserva columnas legacy incompatibles con v13: {obsolete[:30]}")

    leaf = scan.filter(_leaf_expr()).select(sorted(required))
    if _count(leaf) == 0:
        errors.append("no hay hojas SKU+tienda en forecast.parquet")
        return errors

    allowed_periods = {"in_sample", "out_sample", "forecast_only"}
    periods = set(leaf.select("period_type").unique().collect()["period_type"].to_list())
    extra_periods = periods - allowed_periods
    missing_periods = allowed_periods - periods
    if extra_periods:
        errors.append(f"period_type inesperados: {sorted(extra_periods)}")
    if missing_periods:
        errors.append(f"faltan period_type: {sorted(missing_periods)}")

    # Exactly 28 distinct days for OOS and forecast-only for every leaf.
    horizon = int(settings.METRIC_HORIZON_DAYS)
    for period in ("out_sample", "forecast_only"):
        bad = (
            leaf.filter(pl.col("period_type") == period)
            .group_by("unique_id")
            .agg(pl.col("ds").n_unique().alias("n_days"))
            .filter(pl.col("n_days") != horizon)
        )
        n_bad = _count(bad)
        if n_bad:
            errors.append(f"{period}: {n_bad} hojas no tienen exactamente {horizon} días")

    # Forecast-only must begin the day immediately after the OOS horizon.
    bounds = (
        leaf.filter(pl.col("period_type").is_in(["out_sample", "forecast_only"]))
        .group_by(["unique_id", "period_type"])
        .agg(pl.col("ds").min().alias("d0"), pl.col("ds").max().alias("d1"))
        .collect()
    )
    if bounds.height:
        oos = bounds.filter(pl.col("period_type") == "out_sample").select(
            "unique_id", pl.col("d1").alias("oos_end")
        )
        fo = bounds.filter(pl.col("period_type") == "forecast_only").select(
            "unique_id", pl.col("d0").alias("fo_start")
        )
        bad_gap = oos.join(fo, on="unique_id", how="inner").filter(
            pl.col("fo_start") != (pl.col("oos_end") + pl.duration(days=1))
        )
        if bad_gap.height:
            errors.append(f"{bad_gap.height} hojas tienen gap OOS → forecast-only")

    # One model identity in every scored period.  Warm-up is initialization of
    # that same SES+RLS family and is intentionally metric-ineligible.
    bad_model = leaf.filter(~pl.col("modelo_seleccionado").cast(pl.Utf8).str.starts_with("SES+RLS("))
    if _count(bad_model):
        errors.append("hay hojas cuyo modelo no pertenece a la familia única SES+RLS")

    scored = leaf.filter(pl.col("ds") > pl.col("leaf_warmup_end"))
    bad_parent = scored.filter(
        ~pl.col("parent_model_y").is_in(["store", "section"])
        | ~pl.col("parent_model_value").is_in(["store", "section"])
    )
    if _count(bad_parent):
        errors.append("hay filas post-warmup sin parent RLS store/section válido")

    warmup = leaf.filter(pl.col("ds") <= pl.col("leaf_warmup_end"))
    if _count(warmup):
        if _count(warmup.filter((pl.col("driver_effect").abs() > 1e-12) | (pl.col("driver_effect_value").abs() > 1e-12))):
            errors.append("warm-up tiene driver_effect distinto de 0")
        if _count(warmup.filter((pl.col("driver_factor_y") - 1.0).abs() > 1e-12)):
            errors.append("warm-up tiene driver_factor_y distinto de 1")
        if _count(warmup.filter((pl.col("driver_factor_value") - 1.0).abs() > 1e-12)):
            errors.append("warm-up tiene driver_factor_value distinto de 1")
        if _count(warmup.filter(pl.col("rls_metric_eligible").fill_null(False))):
            errors.append("warm-up no debe ser elegible para métricas/selección")

    # Algebraic identity of the single production model.
    # El kernel productivo impone soporte no negativo antes del redondeo:
    # yhat_raw = max(expm1(log1p(SES) + efecto_RLS), 0).  La auditoría
    # debe reconstruir exactamente esa misma identidad; sin el clip, cualquier
    # efecto RLS suficientemente negativo genera un valor algebraico < 0 aunque
    # producción haya almacenado 0.0, produciendo un falso ERROR de identidad.
    ident_y = (
        (pl.col("ses_level_y").clip(lower_bound=0.0).log1p() + pl.col("driver_effect"))
        .exp()
        .sub(1.0)
        .clip(lower_bound=0.0)
    )
    ident_v = (
        (
            pl.col("ses_level_value").clip(lower_bound=0.0).log1p()
            + pl.col("driver_effect_value")
        )
        .exp()
        .sub(1.0)
        .clip(lower_bound=0.0)
    )
    tol = 1e-7
    diff_y = pl.col("yhat_raw") - ident_y
    diff_v = pl.col("valuehat_raw") - ident_v
    max_diff_y = _max_abs(leaf, diff_y)
    max_diff_v = _max_abs(leaf, diff_v)
    if max_diff_y > tol:
        bad_y = _count(leaf.filter(diff_y.abs() > tol))
        errors.append(
            "identidad yhat_raw != SES level + RLS driver effect "
            f"({bad_y} filas; max_diff={max_diff_y:.6g})"
        )
    if max_diff_v > tol:
        bad_v = _count(leaf.filter(diff_v.abs() > tol))
        errors.append(
            "identidad valuehat_raw != SES level + RLS driver effect "
            f"({bad_v} filas; max_diff={max_diff_v:.6g})"
        )
    if _max_abs(leaf, pl.col("yhat") - pl.col("yhat_raw").round(0)) > 1e-9:
        errors.append("yhat no coincide con round(yhat_raw, 0)")
    if _max_abs(leaf, pl.col("valuehat") - pl.col("valuehat_raw").round(2)) > 1e-9:
        errors.append("valuehat no coincide con round(valuehat_raw, 2)")

    # Factor is simply exp(effect); it is traceability, not a second model.
    if _max_abs(leaf, pl.col("driver_factor_y") - pl.col("driver_effect").exp()) > tol:
        errors.append("driver_factor_y != exp(driver_effect)")
    if _max_abs(leaf, pl.col("driver_factor_value") - pl.col("driver_effect_value").exp()) > tol:
        errors.append("driver_factor_value != exp(driver_effect_value)")

    # Forecast-only never enters metric selection.
    if _count(leaf.filter((pl.col("period_type") == "forecast_only") & pl.col("rls_metric_eligible").fill_null(False))):
        errors.append("forecast-only no puede ser elegible para métricas")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditoría estructural v13 SES+RLS")
    parser.add_argument("--forecast", default=None, help="Forecast explícito; por defecto data/output/forecast.parquet")
    parser.add_argument(
        "--update-block-days",
        type=int,
        choices=list(settings.UPDATE_BLOCK_OPTIONS),
        default=None,
        help="Audita un escenario multi-bloque concreto.",
    )
    parser.add_argument(
        "--all-update-blocks",
        action="store_true",
        help="Audita todos los escenarios 1/7/14/28d existentes.",
    )
    args = parser.parse_args(argv)
    if args.all_update_blocks and (args.update_block_days is not None or args.forecast is not None):
        parser.error("No combine --all-update-blocks con --update-block-days/--forecast")
    if args.update_block_days is not None and args.forecast is not None:
        parser.error("No combine --update-block-days con --forecast")

    if args.all_update_blocks:
        targets = [
            (int(days), settings.update_block_forecast_path(int(days)))
            for days in settings.UPDATE_BLOCK_OPTIONS
            if settings.update_block_forecast_path(int(days)).exists()
        ]
        if not targets:
            print("AUDITORÍA MODELO: ERROR")
            print(" - no existen escenarios multi-bloque para auditar")
            return 1
    elif args.update_block_days is not None:
        targets = [(int(args.update_block_days), settings.update_block_forecast_path(int(args.update_block_days)))]
    else:
        targets = [(None, Path(args.forecast or settings.FORECAST_PATH))]

    all_errors: list[str] = []
    for days, path in targets:
        errors = validate(path)
        label = f" ({days}d)" if days is not None else ""
        if errors:
            print(f"AUDITORÍA MODELO v{settings.APP_VERSION}{label}: ERROR")
            for err in errors:
                print(f" - {err}")
                all_errors.append(f"[{days}d] {err}" if days is not None else err)
            continue
        print(f"AUDITORÍA MODELO v{settings.APP_VERSION}{label}: OK")
        print(f"Forecast: {path}")
        print("Contrato: RLS sección/tienda + SES leaf; misma familia in-sample/OOS/forecast-only")
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
