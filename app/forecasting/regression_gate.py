"""End-to-end release gate for the v13 dashboard scenarios.

This module does not tune or alter forecasts. It blocks a release if the four
update cadences contain pathologies that the structural/algebraic validators
cannot detect: absurd leaf spikes, sentinel-series regressions, or orders-of-
magnitude inconsistency between 1d/7d/14d/28d for the same active leaf.

Run:
    uv run python -m app.forecasting.regression_gate --all-update-blocks
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import settings
from app.forecasting.run_status import validation_errors as run_status_errors

_LEAF_RE = r"\|\|T:[^|]+\|\|S:[^|]+"


def _leaf_scan(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("unique_id").cast(pl.Utf8).str.contains(_LEAF_RE))
        .with_columns(pl.col("unique_id").cast(pl.Utf8))
    )


def _safe_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den).otherwise(None)


def _spike_summary(path: Path, days: int) -> pl.DataFrame:
    lf = _leaf_scan(path)
    observed = lf.filter(pl.col("period_type").is_in(["in_sample", "out_sample"]))

    unit = (
        observed.group_by("unique_id")
        .agg(
            pl.col("y").filter(pl.col("y") > 0).median().alias("positive_median"),
            pl.col("y").filter(pl.col("y") > 0).max().alias("observed_max"),
            pl.col("yhat_raw").max().alias("forecast_max"),
            (pl.col("y") > 0).sum().alias("n_positive"),
        )
        .with_columns(
            pl.lit("Unidades").alias("target"),
            pl.lit(int(days)).alias("update_block_days"),
        )
    )
    value = (
        observed.group_by("unique_id")
        .agg(
            pl.col("value")
            .filter((pl.col("y") > 0) & pl.col("value").is_finite() & (pl.col("value") > 0))
            .median()
            .alias("positive_median"),
            pl.col("value")
            .filter((pl.col("y") > 0) & pl.col("value").is_finite() & (pl.col("value") > 0))
            .max()
            .alias("observed_max"),
            pl.col("valuehat_raw").max().alias("forecast_max"),
            ((pl.col("y") > 0) & pl.col("value").is_finite()).sum().alias("n_positive"),
        )
        .with_columns(
            pl.lit("Valor ($)").alias("target"),
            pl.lit(int(days)).alias("update_block_days"),
        )
    )
    out = pl.concat([unit.collect(), value.collect()], how="vertical_relaxed")
    return out.with_columns(
        _safe_ratio(pl.col("forecast_max"), pl.col("positive_median")).alias("ratio_to_median"),
        _safe_ratio(pl.col("forecast_max"), pl.col("observed_max")).alias("ratio_to_observed_max"),
    )


def _oos_cadence_summary(path: Path, days: int) -> pl.DataFrame:
    lf = _leaf_scan(path).filter(pl.col("period_type") == "out_sample")
    unit = (
        lf.group_by("unique_id")
        .agg(
            (pl.col("y") > 0).sum().alias("n_positive"),
            pl.col("y").filter(pl.col("y") > 0).median().alias("actual_median"),
            pl.col("yhat_raw").filter(pl.col("y") > 0).median().alias("forecast_median"),
        )
        .with_columns(
            pl.lit("Unidades").alias("target"),
            pl.lit(int(days)).alias("update_block_days"),
        )
    )
    value = (
        lf.group_by("unique_id")
        .agg(
            ((pl.col("y") > 0) & pl.col("value").is_finite()).sum().alias("n_positive"),
            pl.col("value")
            .filter((pl.col("y") > 0) & pl.col("value").is_finite())
            .median()
            .alias("actual_median"),
            pl.col("valuehat_raw")
            .filter((pl.col("y") > 0) & pl.col("value").is_finite())
            .median()
            .alias("forecast_median"),
        )
        .with_columns(
            pl.lit("Valor ($)").alias("target"),
            pl.lit(int(days)).alias("update_block_days"),
        )
    )
    return pl.concat([unit.collect(), value.collect()], how="vertical_relaxed")




def _train_window_summary(path: Path, days: int) -> pl.DataFrame:
    return (
        _leaf_scan(path)
        .filter(pl.col("period_type") == "in_sample")
        .with_columns(
            pl.col("unique_id").str.extract(r"^([^|]+)", 1).alias("section")
        )
        .group_by("section")
        .agg(
            pl.col("ds").min().alias("train_start"),
            pl.col("ds").max().alias("train_end"),
        )
        .with_columns(pl.lit(int(days)).alias("update_block_days"))
        .collect()
    )

def _long_gap_reactivation_summary(path: Path, days: int) -> pl.DataFrame:
    """Audit OOS positive reactivations after long observable zero-sale gaps."""
    lf = (
        _leaf_scan(path)
        .filter(pl.col("period_type").is_in(["in_sample", "out_sample"]))
        .filter(pl.col("y").is_finite() & (pl.col("y") > 0))
        .select(
            "unique_id", "ds", "period_type", "y", "value",
            "yhat_raw", "valuehat_raw", "leaf_observable_day_index",
        )
        .sort(["unique_id", "ds"])
        .with_columns(
            pl.col("leaf_observable_day_index")
            .shift(1)
            .over("unique_id")
            .alias("_prev_observable_seq"),
            pl.col("ds").shift(1).over("unique_id").alias("_prev_positive_ds"),
        )
        .with_columns(
            (
                pl.col("leaf_observable_day_index")
                - pl.col("_prev_observable_seq")
                - 1
            )
            .clip(lower_bound=0)
            .cast(pl.Int32)
            .alias("observable_zero_days"),
            (
                (pl.col("ds") - pl.col("_prev_positive_ds")).dt.total_days() - 1
            )
            .clip(lower_bound=0)
            .cast(pl.Int32)
            .alias("calendar_gap_days"),
        )
        .filter(
            (pl.col("period_type") == "out_sample")
            & pl.col("_prev_observable_seq").is_not_null()
        )
    )

    def _factor(actual: str, forecast: str) -> pl.Expr:
        return (
            pl.when((pl.col(actual) > 0) & (pl.col(forecast) > 0))
            .then(
                pl.max_horizontal(
                    pl.col(forecast) / pl.col(actual),
                    pl.col(actual) / pl.col(forecast),
                )
            )
            .when((pl.col(actual) > 0) & (pl.col(forecast) <= 0))
            .then(pl.lit(1e308))
            .otherwise(None)
        )

    unit = lf.select(
        "unique_id", "ds", "_prev_positive_ds", "observable_zero_days",
        "calendar_gap_days",
        pl.col("y").alias("actual"),
        pl.col("yhat_raw").alias("forecast"),
        _factor("y", "yhat_raw").alias("factor_error"),
    ).with_columns(
        pl.lit("Unidades").alias("target"),
        pl.lit(int(days)).alias("update_block_days"),
    )

    value = (
        lf.filter(pl.col("value").is_finite() & (pl.col("value") > 0))
        .select(
            "unique_id", "ds", "_prev_positive_ds", "observable_zero_days",
            "calendar_gap_days",
            pl.col("value").alias("actual"),
            pl.col("valuehat_raw").alias("forecast"),
            _factor("value", "valuehat_raw").alias("factor_error"),
        )
        .with_columns(
            pl.lit("Valor ($)").alias("target"),
            pl.lit(int(days)).alias("update_block_days"),
        )
    )
    return pl.concat([unit.collect(), value.collect()], how="vertical_relaxed")

def _sentinel_metrics(path: Path, days: int) -> pl.DataFrame:
    rows: list[dict] = []
    lf = _leaf_scan(path)
    for section, store, sku, target in getattr(settings, "RELEASE_GATE_SENTINELS", ()):
        uid = settings.make_unique_id(str(section), store=str(store), sku=str(sku))
        g = lf.filter(pl.col("unique_id") == uid)
        if target == "Unidades":
            actual, forecast = "y", "yhat_raw"
        else:
            actual, forecast = "value", "valuehat_raw"

        for period in ("in_sample", "out_sample"):
            support = (
                g.filter(
                    (pl.col("period_type") == period)
                    & pl.col("y").is_finite()
                    & (pl.col("y") > 0)
                    & pl.col(actual).is_finite()
                    & pl.col(forecast).is_finite()
                )
                .select(
                    (pl.col(forecast) - pl.col(actual)).abs().sum().alias("ae"),
                    pl.col(actual).abs().sum().alias("den"),
                    (pl.col(forecast) - pl.col(actual)).sum().alias("se"),
                    pl.len().alias("n"),
                )
                .collect()
            )
            if support.height == 0:
                continue
            r = support.row(0, named=True)
            den = float(r["den"] or 0.0)
            if den <= 0.0:
                continue
            rows.append(
                {
                    "unique_id": uid,
                    "target": str(target),
                    "update_block_days": int(days),
                    "period_type": period,
                    "wmape": float(r["ae"] or 0.0) / den,
                    "bias": float(r["se"] or 0.0) / den,
                    "n_points": int(r["n"] or 0),
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def validate_all() -> tuple[list[str], pl.DataFrame]:
    errors: list[str] = []
    spike_parts: list[pl.DataFrame] = []
    cadence_parts: list[pl.DataFrame] = []
    sentinel_parts: list[pl.DataFrame] = []
    long_gap_parts: list[pl.DataFrame] = []
    train_window_parts: list[pl.DataFrame] = []

    for days in tuple(int(x) for x in settings.UPDATE_BLOCK_OPTIONS):
        path = settings.update_block_forecast_path(days)
        if not path.exists():
            errors.append(f"{days}d: falta forecast.parquet")
            continue
        status = run_status_errors(path)
        if status:
            errors.extend(f"{days}d: {msg}" for msg in status)
            continue
        spike_parts.append(_spike_summary(path, days))
        cadence_parts.append(_oos_cadence_summary(path, days))
        train_window_parts.append(_train_window_summary(path, days))
        long_gap_parts.append(_long_gap_reactivation_summary(path, days))
        sent = _sentinel_metrics(path, days)
        if sent.height:
            sentinel_parts.append(sent)

    if errors:
        return errors, pl.DataFrame()

    spikes = pl.concat(spike_parts, how="vertical_relaxed") if spike_parts else pl.DataFrame()
    cadence = pl.concat(cadence_parts, how="vertical_relaxed") if cadence_parts else pl.DataFrame()
    sentinels = pl.concat(sentinel_parts, how="vertical_relaxed") if sentinel_parts else pl.DataFrame()
    long_gaps = pl.concat(long_gap_parts, how="vertical_relaxed") if long_gap_parts else pl.DataFrame()
    train_windows = pl.concat(train_window_parts, how="vertical_relaxed") if train_window_parts else pl.DataFrame()

    if train_windows.height:
        bad_windows = (
            train_windows.group_by("section")
            .agg(
                pl.col("train_start").n_unique().alias("n_train_start"),
                pl.col("train_end").n_unique().alias("n_train_end"),
                pl.col("update_block_days").n_unique().alias("n_cadences"),
                pl.col("train_start").min().alias("min_train_start"),
                pl.col("train_start").max().alias("max_train_start"),
            )
            .filter(
                (pl.col("n_cadences") == len(settings.UPDATE_BLOCK_OPTIONS))
                & ((pl.col("n_train_start") != 1) | (pl.col("n_train_end") != 1))
            )
        )
        if bad_windows.height:
            errors.append(
                "historia al origen OOS difiere entre cadencias: "
                + str(bad_windows.to_dicts())
            )
    else:
        bad_windows = pl.DataFrame()

    median_limit = float(getattr(settings, "RELEASE_GATE_MAX_FORECAST_TO_POSITIVE_MEDIAN", 8.0))
    max_limit = float(getattr(settings, "RELEASE_GATE_MAX_FORECAST_TO_OBSERVED_MAX", 2.0))
    bad_spikes = spikes.filter(
        (pl.col("n_positive") >= 7)
        & (pl.col("positive_median") > 0)
        & (pl.col("observed_max") > 0)
        & (pl.col("ratio_to_median") > median_limit)
        & (pl.col("ratio_to_observed_max") > max_limit)
    )
    if bad_spikes.height:
        examples = bad_spikes.sort("ratio_to_median", descending=True).head(10).to_dicts()
        errors.append(
            f"spikes leaf fuera de escala: {bad_spikes.height} series; ejemplos={examples}"
        )

    # Cross-cadence gate only on OOS-active leaves (>=7 positive actual days in
    # every available cadence). It detects orders-of-magnitude disagreement,
    # not ordinary differences caused by a faster/slower update frequency.
    if cadence.height:
        cross = (
            cadence.filter(
                (pl.col("n_positive") >= 7)
                & pl.col("forecast_median").is_finite()
                & (pl.col("forecast_median") > 0)
            )
            .group_by(["unique_id", "target"])
            .agg(
                pl.col("forecast_median").min().alias("min_forecast_median"),
                pl.col("forecast_median").max().alias("max_forecast_median"),
                pl.col("update_block_days").n_unique().alias("n_cadences"),
            )
            .with_columns(
                _safe_ratio(
                    pl.col("max_forecast_median"), pl.col("min_forecast_median")
                ).alias("cross_cadence_ratio")
            )
        )
        cross_limit = float(
            getattr(settings, "RELEASE_GATE_MAX_CROSS_CADENCE_MEDIAN_RATIO", 4.0)
        )
        bad_cross = cross.filter(
            (pl.col("n_cadences") == len(settings.UPDATE_BLOCK_OPTIONS))
            & (pl.col("cross_cadence_ratio") > cross_limit)
        )
        if bad_cross.height:
            examples = bad_cross.sort("cross_cadence_ratio", descending=True).head(10).to_dicts()
            errors.append(
                f"incoherencia OOS entre cadencias: {bad_cross.height} hojas; ejemplos={examples}"
            )
    else:
        cross = pl.DataFrame()

    if long_gaps.height:
        gap_min = int(
            getattr(settings, "RELEASE_GATE_LONG_GAP_MIN_OBSERVABLE_ZERO_DAYS", 28)
        )
        gap_factor_limit = float(
            getattr(settings, "RELEASE_GATE_LONG_GAP_MAX_FACTOR_ERROR", 4.0)
        )
        bad_long_gap = long_gaps.filter(
            (pl.col("observable_zero_days") >= gap_min)
            & pl.col("factor_error").is_finite()
            & (pl.col("factor_error") > gap_factor_limit)
        )
        if bad_long_gap.height:
            examples = (
                bad_long_gap.sort(
                    ["observable_zero_days", "factor_error"],
                    descending=[True, True],
                )
                .head(10)
                .to_dicts()
            )
            errors.append(
                "reactivaciones tras gap largo fuera de escala: "
                f"{bad_long_gap.height} casos; ejemplos={examples}"
            )
    else:
        bad_long_gap = pl.DataFrame()

    if sentinels.height:
        max_ins = float(getattr(settings, "RELEASE_GATE_SENTINEL_MAX_IN_SAMPLE_WMAPE", 3.0))
        max_bias = float(getattr(settings, "RELEASE_GATE_SENTINEL_MAX_OOS_ABS_BIAS", 1.0))
        bad_ins = sentinels.filter(
            (pl.col("period_type") == "in_sample") & (pl.col("wmape") > max_ins)
        )
        bad_oos = sentinels.filter(
            (pl.col("period_type") == "out_sample") & (pl.col("bias").abs() > max_bias)
        )
        if bad_ins.height:
            errors.append(
                "casos centinela con wMAPE in-sample patológico: "
                + str(bad_ins.to_dicts())
            )
        if bad_oos.height:
            errors.append(
                "casos centinela con |BIAS| OOS patológico: "
                + str(bad_oos.to_dicts())
            )

    report_parts: list[pl.DataFrame] = []
    if 'bad_windows' in locals() and bad_windows.height:
        report_parts.append(
            bad_windows.with_columns(pl.lit("train_window_mismatch").alias("gate_type"))
        )
    if bad_spikes.height:
        report_parts.append(
            bad_spikes.with_columns(pl.lit("spike").alias("gate_type"))
        )
    if 'bad_cross' in locals() and bad_cross.height:
        report_parts.append(
            bad_cross.with_columns(pl.lit("cross_cadence").alias("gate_type"))
        )
    if 'bad_long_gap' in locals() and bad_long_gap.height:
        report_parts.append(
            bad_long_gap.with_columns(pl.lit("long_gap_reactivation").alias("gate_type"))
        )
    if sentinels.height:
        report_parts.append(
            sentinels.with_columns(pl.lit("sentinel_metric").alias("gate_type"))
        )
    report = pl.concat(report_parts, how="diagonal_relaxed") if report_parts else pl.DataFrame()
    return errors, report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate end-to-end de no-regresión v13.")
    parser.add_argument(
        "--all-update-blocks",
        action="store_true",
        help="Valida 1d/7d/14d/28d (requerido para release).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.all_update_blocks:
        raise SystemExit("Use --all-update-blocks: el gate de release compara las cuatro cadencias.")
    errors, report = validate_all()
    report_path = settings.OUT_DIR / "release_regression_gate.parquet"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report.height:
        report.write_parquet(report_path)
    elif report_path.exists():
        report_path.unlink()

    if errors:
        print(f"GATE REGRESIÓN v{settings.APP_VERSION}: ERROR")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print(f"GATE REGRESIÓN v{settings.APP_VERSION}: OK")
    print("- misma historia al origen OOS en 1d/7d/14d/28d")
    print("- sin spikes leaf fuera de escala")
    print("- coherencia OOS 1d/7d/14d/28d dentro del contrato")
    print("- reactivaciones tras gaps largos dentro del contrato")
    print("- casos centinela sin métricas patológicas")


if __name__ == "__main__":
    main()
