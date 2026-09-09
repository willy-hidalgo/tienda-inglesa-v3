"""Post-run structural validator for Tienda Inglesa v12.x (v11-compatible).

Usage:
    python -m app.forecasting.validate_v11
    python -m app.forecasting.validate_v11 --forecast data/output/forecast.parquet

This validates model contracts, not forecast accuracy. Statistical accuracy must
still be evaluated with OOS wMAPE/BIAS and the dashboard audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import settings
from app.forecasting.run_status import validation_errors as run_status_errors


def _leaf_expr() -> pl.Expr:
    uid = pl.col("unique_id").cast(pl.Utf8)
    return (
        uid.str.contains(r"\|\|T:[^|]+")
        & uid.str.contains(r"\|\|S:[^|]+")
    )


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"No existe forecast: {path}"]

    status_errors = run_status_errors(path)
    if status_errors:
        return [f"artefacto stale/no válido: {e}" for e in status_errors]

    # Memory-safe validation: do not materialize the full ~7.5M-row / hundreds-of-columns
    # forecast.  Structural checks only need leaf OOS/forecast-only rows plus a
    # narrow projection.  Schema validation remains global and legacy-period checks
    # are executed lazily against the whole parquet.
    scan = pl.scan_parquet(path)
    schema_cols = set(scan.collect_schema().names())
    required = {
        "unique_id",
        "ds",
        "period_type",
        "yhat",
        "valuehat",
        "parent_model_y",
        "parent_model_value",
        "driver_strength_y",
        "driver_strength_value",
        "driver_factor_y",
        "driver_factor_value",
        "driver_level_factor_y",
        "driver_level_factor_value",
        "parent_driver_mode_y",
        "parent_driver_mode_value",
        "ses_level_y",
        "ses_level_value",
        "ses_guard_status_y",
        "ses_guard_status_value",
        "sku_seasonal_multiplier_y",
        "sku_seasonal_multiplier_value",
        "leaf_model_family_y",
        "leaf_model_family_value",
        "v12_selected_y",
        "v12_selected_value",
        "v12_candidate_yhat_raw",
        "v12_candidate_valuehat_raw",
        "v12_sku_forecast_y",
        "v12_sku_forecast_value",
        "v12_sku_forecast_base_y",
        "v12_sku_forecast_base_value",
        "v12_sku_forecast_uncalibrated_y",
        "v12_sku_forecast_uncalibrated_value",
        "v12_sku_forecast_calibrated_challenger_y",
        "v12_sku_forecast_calibrated_challenger_value",
        "v12_sku_level_calibration_applied_y",
        "v12_sku_level_calibration_applied_value",
        "v12_sku_level_calibration_raw_ratio_y",
        "v12_sku_level_calibration_raw_ratio_value",
        "v12_sku_level_calibration_factor_y",
        "v12_sku_level_calibration_factor_value",
        "v12_sku_level_calibration_blocks_y",
        "v12_sku_level_calibration_blocks_value",
        "v12_sku_shape_model_y",
        "v12_sku_shape_model_value",
        "v12_sku_shape_training_rows_y",
        "v12_sku_shape_training_rows_value",
        "v12_sku_shape_applied_y",
        "v12_sku_shape_applied_value",
        "v12_sku_model_y",
        "v12_sku_model_value",
        "v12_sku_validation_wmape_y",
        "v12_sku_validation_wmape_value",
        "v12_store_share_y",
        "v12_store_share_value",
        "v12_occurrence_prob_y",
        "v12_occurrence_prob_value",
        "v12_occurrence_prob_joint",
        "v12_occurrence_gate_open_y",
        "v12_occurrence_gate_open_value",
        "v12_occurrence_gate_open_joint",
        "v12_validation_blocks_y",
        "v12_validation_blocks_value",
        "v12_validation_wins_y",
        "v12_validation_wins_value",
        "v12_validation_improvement_y",
        "v12_validation_improvement_value",
        "v12_validation_recent_improvement_y",
        "v12_validation_recent_improvement_value",
        "v12_validation_utility_improvement_y",
        "v12_validation_utility_improvement_value",
        "v12_validation_recent_utility_improvement_y",
        "v12_validation_recent_utility_improvement_value",
        "v12_validation_utility_mean_y",
        "v12_validation_utility_mean_value",
        "v12_validation_utility_std_y",
        "v12_validation_utility_std_value",
        "v12_section_portfolio_gate_y",
        "v12_section_portfolio_gate_value",
        "v12_section_portfolio_gate_joint",
        "v12_joint_target_preselect",
        "v12_meta_probability_y",
        "v12_meta_probability_value",
        "v12_meta_model_available_y",
        "v12_meta_model_available_value",
        "v12_meta_training_rows_y",
        "v12_meta_training_rows_value",
        "v12_meta_recent_gain_y",
        "v12_meta_recent_gain_value",
        "v12_meta_weighted_gain_y",
        "v12_meta_weighted_gain_value",
        "v12_meta_win_rate_y",
        "v12_meta_win_rate_value",
        "v12_meta_top_driver_y",
        "v12_meta_top_driver_value",
        "v12_meta_bias11_y",
        "v12_meta_bias11_value",
        "v12_meta_bias12_y",
        "v12_meta_bias12_value",
        "v12_meta_bias_guard_pass_y",
        "v12_meta_bias_guard_pass_value",
        "v12_meta_threshold_y",
        "v12_meta_threshold_value",
        "v12_meta_portfolio_mode_y",
        "v12_meta_portfolio_mode_value",
        "v12_meta_policy_available_y",
        "v12_meta_policy_available_value",
        "v12_meta_policy_utility_gain_y",
        "v12_meta_policy_utility_gain_value",
        "v12_value_safety_enabled",
        "v12_value_safety_dominance_pass",
        "v12_value_safety_recent_confirmations",
        "v12_value_safety_recent_blocks",
        "v12_value_safety_bias_coverage",
        "v12_value_safety_bias_coverage_threshold",
        "v12_value_safety_bias_coverage_pass",
        "v12_value_safety_meta_margin_pass",
        "v12_value_safety_best_all_mode",
        "v12_value_safety_reason",
        "v129_value_wf_enabled",
        "v129_value_wf_available",
        "v129_value_wf_folds",
        "v129_value_wf_win_rate",
        "v129_value_wf_median_gain",
        "v129_value_wf_worst_gain",
        "v129_value_wf_weighted_gain",
        "v129_value_wf_weighted_utility_gain",
        "v129_value_wf_bias_worsen_max",
        "v129_value_wf_meta_folds",
        "v129_value_wf_fold1_gain",
        "v129_value_wf_fold2_gain",
        "v129_value_wf_fold3_gain",
        "v129_value_wf_reason",
        "v1291_expected_gain_enabled_value",
        "v1291_expected_gain_available_value",
        "v1291_expected_gain_value",
        "v1291_expected_gain_confidence_value",
        "v1291_expected_gain_recent_value",
        "v1291_expected_gain_range_value",
        "v1291_expected_gain_training_rows_value",
        "v1291_expected_gain_portfolio_pass_value",
        "v1291_expected_gain_portfolio_gain_value",
        "v1291_expected_gain_portfolio_share_value",
        "v1291_expected_gain_min_gain_value",
        "v1291_expected_gain_min_confidence_value",
        "v1292_raw_expected_gain_value",
        "v1292_calibrated_gain_value",
        "v1292_bucket_loss_rate_value",
        "v1292_leaf_loss_rate_value",
        "v1292_impact_percentile_value",
        "v1292_budget_selected_value",
        "v1292_portfolio_budget_value",
        "v1292_portfolio_expected_gain_value",
        "v1292_portfolio_selected_share_value",
        "v1292_portfolio_pass_value",
        "v1293_segment_selector_enabled_value",
        "v1293_segment_prebudget_value",
        "v1293_segment_budget_selected_value",
        "v1293_segment_level_value",
        "v1293_segment_reason_value",
        "v1293_segment_folds_value",
        "v1293_segment_win_rate_value",
        "v1293_segment_weighted_gain_value",
        "v1293_segment_selected_volume_share_value",
        "v1293_segment_budget_value",
        "v1294_segment_risk_score_value",
        "v1294_segment_exposure_budget_value",
        "v1294_impact_percentile_value",
        "v1294_high_impact_guard_pass_value",
        "v1295_stress_enabled_value",
        "v1295_stress_requested_blocks_value",
        "v1295_stress_materialized_blocks_value",
        "v1295_stress_level_value",
        "v1295_stress_key_value",
        "v1295_stress_folds_value",
        "v1295_stress_win_rate_value",
        "v1295_stress_median_gain_value",
        "v1295_stress_p25_gain_value",
        "v1295_stress_p10_gain_value",
        "v1295_stress_worst_gain_value",
        "v1295_stress_std_gain_value",
        "v1295_stress_weighted_gain_value",
        "v1295_stress_bias_p90_worsen_value",
        "v1295_stress_max_consecutive_losses_value",
        "v1295_stress_min_leaves_value",
        "v1295_stress_pass_value",
        "v1295_stress_reason_value",
        "v1296_stress_gate_enabled_value",
        "v1296_stress_eligible_value",
        "v1296_stress_gate_pass_value",
        "v1296_budget_selected_value",
        "v1296_high_impact_guard_pass_value",
        "v1296_reason_value",
        "v1296_budget_value",
        "v1296_selected_volume_share_value",
        "v1296_segment_score_value",
        "v1297_temporal_replay_enabled_value",
        "v1297_temporal_replay_cutoffs_value",
        "v1297_temporal_replay_win_rate_value",
        "v1297_temporal_replay_median_gain_value",
        "v1297_temporal_replay_worst_gain_value",
        "v1297_temporal_replay_weighted_gain_value",
        "v1297_temporal_replay_bias_worst_value",
        "v1297_temporal_replay_selected_share_median_value",
        "v1297_temporal_replay_pass_value",
        "v12_section_portfolio_recent_gain_y",
        "v12_section_portfolio_recent_gain_value",
    }
    missing = sorted(required - schema_cols)
    if missing:
        return [f"Faltan columnas v11: {missing}"]

    # Columns actually referenced by row-level validation.  Keep this projection
    # deliberately narrow: required-only metadata that is checked for presence does
    # not need to be loaded into RAM.
    validation_cols = {
        "unique_id", "ds", "period_type", "y", "value",
        "test_start", "test_end", "forecast_start", "forecast_end",
        "yhat_raw", "valuehat_raw", "ses_regime_class_value",
    }
    # Add every column used by the validator expressions.  This explicit set avoids
    # loading dashboard/report-only metadata and is intentionally safe to extend.
    validation_cols.update({
        "parent_model_y", "parent_model_value",
        "driver_strength_y", "driver_strength_value",
        "driver_factor_y", "driver_factor_value",
        "driver_level_factor_y", "driver_level_factor_value",
        "parent_driver_mode_y", "parent_driver_mode_value",
        "ses_level_y", "ses_level_value",
        "sku_seasonal_multiplier_y", "sku_seasonal_multiplier_value",
        "leaf_model_family_y", "leaf_model_family_value",
        "v12_selected_y", "v12_selected_value",
        "v12_candidate_available_value",
        "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
        "v12_sku_forecast_y", "v12_sku_forecast_value",
        "v12_sku_forecast_base_y", "v12_sku_forecast_base_value",
        "v12_sku_forecast_uncalibrated_y", "v12_sku_forecast_uncalibrated_value",
        "v12_sku_forecast_calibrated_challenger_y", "v12_sku_forecast_calibrated_challenger_value",
        "v12_sku_level_calibration_applied_y", "v12_sku_level_calibration_applied_value",
        "v12_sku_level_calibration_factor_y", "v12_sku_level_calibration_factor_value",
        "v12_sku_level_calibration_blocks_y", "v12_sku_level_calibration_blocks_value",
        "v12_sku_shape_model_y", "v12_sku_shape_model_value",
        "v12_sku_shape_training_rows_y", "v12_sku_shape_training_rows_value",
        "v12_sku_model_y", "v12_sku_model_value",
        "v12_store_share_y", "v12_store_share_value",
        "v12_occurrence_gate_open_y", "v12_occurrence_gate_open_value", "v12_occurrence_gate_open_joint",
        "v12_validation_blocks_y", "v12_validation_blocks_value",
        "v12_validation_wins_y", "v12_validation_wins_value",
        "v12_validation_improvement_y", "v12_validation_improvement_value",
        "v12_validation_recent_improvement_y", "v12_validation_recent_improvement_value",
        "v12_validation_utility_improvement_y", "v12_validation_utility_improvement_value",
        "v12_validation_recent_utility_improvement_y", "v12_validation_recent_utility_improvement_value",
        "v12_validation_utility_mean_y", "v12_validation_utility_mean_value",
        "v12_validation_utility_std_y", "v12_validation_utility_std_value",
        "v12_meta_probability_y", "v12_meta_probability_value",
        "v12_meta_model_available_y", "v12_meta_model_available_value",
        "v12_meta_training_rows_y", "v12_meta_training_rows_value",
        "v12_meta_recent_gain_y", "v12_meta_recent_gain_value",
        "v12_meta_bias_guard_pass_y", "v12_meta_bias_guard_pass_value",
        "v12_meta_threshold_y", "v12_meta_threshold_value",
        "v12_meta_portfolio_mode_y", "v12_meta_portfolio_mode_value",
        "v129_value_wf_enabled", "v129_value_wf_available", "v129_value_wf_folds",
        "v129_value_wf_win_rate", "v129_value_wf_meta_folds", "v129_value_wf_reason",
        "v1291_expected_gain_confidence_value",
        "v1292_bucket_loss_rate_value", "v1292_leaf_loss_rate_value", "v1292_impact_percentile_value",
        "v1293_segment_selector_enabled_value", "v1293_segment_prebudget_value",
        "v1293_segment_budget_selected_value", "v1293_segment_reason_value",
        "v1293_segment_folds_value", "v1293_segment_win_rate_value",
        "v1293_segment_selected_volume_share_value", "v1293_segment_budget_value",
        "v1294_segment_exposure_budget_value", "v1294_impact_percentile_value",
        "v1295_stress_enabled_value", "v1295_stress_requested_blocks_value",
        "v1295_stress_materialized_blocks_value", "v1295_stress_level_value", "v1295_stress_key_value",
        "v1295_stress_folds_value", "v1295_stress_win_rate_value",
        "v1295_stress_p25_gain_value", "v1295_stress_p10_gain_value",
        "v1295_stress_max_consecutive_losses_value", "v1295_stress_pass_value", "v1295_stress_reason_value",
        "v1296_stress_gate_enabled_value", "v1296_stress_eligible_value", "v1296_stress_gate_pass_value",
        "v1296_budget_selected_value", "v1296_high_impact_guard_pass_value",
        "v1296_budget_value", "v1296_selected_volume_share_value",
        "v1297_temporal_replay_enabled_value", "v1297_temporal_replay_cutoffs_value",
        "v1297_temporal_replay_win_rate_value", "v1297_temporal_replay_selected_share_median_value",
        "v1298_sparse_bucket_y", "v1298_sparse_bucket_value",
        "v1298_sparse_folds_y", "v1298_sparse_folds_value",
        "v1298_sparse_factor_y", "v1298_sparse_factor_value",
        "v1298_sparse_candidate_y", "v1298_sparse_candidate_value",
        "v1299_value_sparse_bucket", "v1299_value_sparse_folds", "v1299_value_sparse_win_rate",
        "v1299_value_sparse_factor", "v1299_value_sparse_candidate",
        "v1299_value_sparse_replay_cutoffs", "v1299_value_sparse_replay_win_rate",
        "v1299_value_sparse_replay_selected_share_median",
        "v12910_valuehat_raw_before_sparse", "v12910_value_sparse_applied",
        "v12910_value_sparse_promoted", "v12910_value_sparse_active_replay_cutoffs",
        "v12910_value_sparse_active_replay_win_rate",
        "v12910_value_sparse_active_replay_selected_share_median",
        "v12911_residual_candidate_value", "v12911_residual_folds_value",
        "v12911_residual_win_rate_value", "v12911_residual_selected_share_value",
        "v12911_residual_replay_cutoffs_value", "v12911_residual_replay_win_rate_value",
        "v12911_residual_replay_selected_share_median_value", "v12911_residual_replay_pass_value",
    })
    load_cols = sorted(validation_cols & schema_cols)
    target = (
        scan.select(load_cols)
        .filter(_leaf_expr())
        .filter(pl.col("period_type").is_in(["out_sample", "forecast_only"]))
        .with_columns(pl.col("ds").cast(pl.Date))
        .collect(engine="streaming")
    )

    if target.height == 0:
        errors.append("No existen filas SKU+tienda OOS/forecast_only")
        return errors

    bad_parent = target.filter(
        (~pl.col("parent_model_y").is_in(["store", "section"]))
        | (~pl.col("parent_model_value").is_in(["store", "section"]))
    )
    if bad_parent.height:
        errors.append(
            f"{bad_parent.height} filas leaf usan parent distinto de store/section"
        )

    tol_strength = 1e-12
    bad_strength = target.filter(
        ((pl.col("driver_strength_y") - 1.0).abs() > tol_strength)
        | ((pl.col("driver_strength_value") - 1.0).abs() > tol_strength)
    )
    if bad_strength.height:
        errors.append(
            f"{bad_strength.height} filas leaf no usan driver strength 100%"
        )

    factor_tol = float(
        getattr(settings, "LEAF_DRIVER_MEAN_TOLERANCE", 1e-6)
    )
    means = (
        target.group_by(["unique_id", "period_type"])
        .agg(
            pl.col("driver_factor_y").mean().alias("_fy"),
            pl.col("driver_factor_value").mean().alias("_fv"),
            pl.col("driver_level_factor_y").first().alias("_expected_y"),
            pl.col("driver_level_factor_value").first().alias("_expected_v"),
            pl.col("parent_driver_mode_y").first().alias("_mode_y"),
            pl.col("parent_driver_mode_value").first().alias("_mode_v"),
            pl.col("ds").n_unique().alias("_n_dates"),
        )
    )
    bad_mode = target.filter(
        (~pl.col("parent_driver_mode_y").is_in(["shape_only", "level_shape"]))
        | (~pl.col("parent_driver_mode_value").is_in(["shape_only", "level_shape"]))
    )
    if bad_mode.height:
        errors.append(f"{bad_mode.height} filas leaf usan driver mode inválido")

    bad_factor = means.filter(
        ((pl.col("_fy") - pl.col("_expected_y")).abs() > factor_tol)
        | ((pl.col("_fv") - pl.col("_expected_v")).abs() > factor_tol)
    )
    if bad_factor.height:
        errors.append(
            f"{bad_factor.height} bloques leaf no respetan la media del factor causal seleccionado"
        )

    bad_days = means.filter(pl.col("_n_dates") != 28)
    if bad_days.height:
        errors.append(
            f"{bad_days.height} bloques leaf OOS/forecast-only no tienen 28 días"
        )

    # v11.2 temporal contract: there are no actual_extension / observed_tail
    # periods after OOS. OOS is the final 28-day actual block.
    legacy_count = (
        scan.select(["unique_id", "period_type"])
        .filter(_leaf_expr())
        .filter(pl.col("period_type").is_in(["actual_extension", "observed_tail"]))
        .select(pl.len().alias("n"))
        .collect(engine="streaming")
        .item()
    )
    if legacy_count:
        errors.append(
            f"{legacy_count} filas usan periodos legacy actual_extension/observed_tail"
        )

    if {"test_start", "test_end", "forecast_start", "forecast_end"}.issubset(schema_cols):
        hz = (
            target.group_by("unique_id")
            .agg(
                pl.col("test_start").first().alias("_test_start"),
                pl.col("test_end").first().alias("_test_end"),
                pl.col("forecast_start").first().alias("_forecast_start"),
                pl.col("forecast_end").first().alias("_forecast_end"),
                pl.col("ds").filter(pl.col("period_type") == "out_sample").min().alias("_oos_min"),
                pl.col("ds").filter(pl.col("period_type") == "out_sample").max().alias("_oos_max"),
            )
            .with_columns(
                pl.col("_test_start").cast(pl.Date),
                pl.col("_test_end").cast(pl.Date),
                pl.col("_forecast_start").cast(pl.Date),
                pl.col("_forecast_end").cast(pl.Date),
            )
        )
        bad_hz = hz.filter(
            (pl.col("_oos_min") != pl.col("_test_start"))
            | (pl.col("_oos_max") != pl.col("_test_end"))
            | (pl.col("_forecast_start") != pl.col("_test_end") + pl.duration(days=1))
            | ((pl.col("_forecast_end") - pl.col("_forecast_start")).dt.total_days() != 27)
        )
        if bad_hz.height:
            errors.append(
                f"{bad_hz.height} hojas violan OOS final-28 / forecast-only contiguo"
            )

    # Forecast-only must not contain actual demand. These rows are synthetic
    # strictly after the last known actual according to section_horizons().
    fc = target.filter(pl.col("period_type") == "forecast_only")
    if fc.height:
        actual_cols = [c for c in ("y", "value") if c in fc.columns]
        for col in actual_cols:
            nonzero = fc.filter(pl.col(col).fill_null(0.0) != 0)
            if nonzero.height:
                errors.append(
                    f"forecast_only contiene {nonzero.height} actuals no cero en {col}"
                )

    # v11 incumbent identity and v12 challenger identity are validated
    # separately because v12 is not an SES rescaling.
    if {"yhat_raw", "valuehat_raw"}.issubset(schema_cols):
        scale_tol = 1e-7
        # v12.9.10 can apply the productive sparse Value rescue after the
        # v11/v12 family forecast has been selected.  Family invariants must
        # therefore be checked against the preserved pre-rescue baseline,
        # exactly as runner.fast_leaf_forecasts() does.  Using final
        # valuehat_raw here would report every legitimately rescued row as a
        # false v11/v12 identity violation.
        if "v12910_valuehat_raw_before_sparse" in target.columns:
            value_family_raw = pl.coalesce(
                [pl.col("v12910_valuehat_raw_before_sparse"), pl.col("valuehat_raw")]
            )
        else:
            value_family_raw = pl.col("valuehat_raw")

        incumbent = target.filter(
            (pl.col("leaf_model_family_y") == "v11_ses_rls")
            | (pl.col("leaf_model_family_value") == "v11_ses_rls")
        ).with_columns(
            value_family_raw.alias("_valuehat_family_raw"),
            (
                pl.col("ses_level_y")
                * pl.col("driver_factor_y")
                * pl.col("sku_seasonal_multiplier_y").fill_null(1.0)
                - pl.col("yhat_raw")
            ).abs().alias("_dy"),
            (
                pl.col("ses_level_value")
                * pl.col("driver_factor_value")
                * pl.col("sku_seasonal_multiplier_value").fill_null(1.0)
                - value_family_raw
            ).abs().alias("_dv"),
        )
        bad_incumbent = incumbent.filter(
            (
                (pl.col("leaf_model_family_y") == "v11_ses_rls")
                & (pl.col("_dy") > scale_tol * pl.max_horizontal(pl.col("yhat_raw").abs(), pl.lit(1.0)))
            )
            | (
                (pl.col("leaf_model_family_value") == "v11_ses_rls")
                & (pl.col("_dv") > scale_tol * pl.max_horizontal(pl.col("_valuehat_family_raw").abs(), pl.lit(1.0)))
            )
        )
        if bad_incumbent.height:
            errors.append(
                f"{bad_incumbent.height} filas v11 violan forecast = SES × driver × seasonal"
            )

        v12 = target.filter(
            pl.col("v12_candidate_yhat_raw").is_not_null()
            | pl.col("v12_candidate_valuehat_raw").is_not_null()
        ).with_columns(
            value_family_raw.alias("_valuehat_family_raw"),
            (
                pl.col("v12_candidate_yhat_raw")
                - pl.col("v12_sku_forecast_y") * pl.col("v12_store_share_y")
            ).abs().alias("_v12_dy"),
            (
                pl.col("v12_candidate_valuehat_raw")
                - pl.col("v12_sku_forecast_value") * pl.col("v12_store_share_value")
            ).abs().alias("_v12_dv"),
        )
        bad_v12 = v12.filter(
            (pl.col("_v12_dy") > scale_tol * pl.max_horizontal(pl.col("v12_candidate_yhat_raw").abs(), pl.lit(1.0)))
            | (pl.col("_v12_dv") > scale_tol * pl.max_horizontal(pl.col("v12_candidate_valuehat_raw").abs(), pl.lit(1.0)))
            | (
                pl.col("v12_selected_y")
                & ((pl.col("yhat_raw") - pl.col("v12_candidate_yhat_raw")).abs()
                   > scale_tol * pl.max_horizontal(pl.col("yhat_raw").abs(), pl.lit(1.0)))
            )
            | (
                pl.col("v12_selected_value")
                & ((pl.col("_valuehat_family_raw") - pl.col("v12_candidate_valuehat_raw")).abs()
                   > scale_tol * pl.max_horizontal(pl.col("_valuehat_family_raw").abs(), pl.lit(1.0)))
            )
        )
        if bad_v12.height:
            errors.append(
                f"{bad_v12.height} filas violan identidad v12 SKU-total × store-share"
            )

        allowed_sku_models = [
            "v11_sum", "recent_weekday", "same_weekday_4", "lag28", "annual_scaled", "annual_blend"
        ]
        bad_sku_model = v12.filter(
            (~pl.col("v12_sku_model_y").is_in(allowed_sku_models))
            | (~pl.col("v12_sku_model_value").is_in(allowed_sku_models))
            | (pl.col("v12_sku_forecast_y") < 0)
            | (pl.col("v12_sku_forecast_value") < 0)
        )
        if bad_sku_model.height:
            errors.append(
                f"{bad_sku_model.height} filas v12 tienen modelo/forecast SKU-total inválido"
            )

        # v12.8: calibration is a diagnostic challenger by default.  Its
        # multiplicative identity is always audited, while the production base
        # must equal either the uncalibrated path or the challenger according to
        # the explicit apply flag.
        cal_lo, cal_hi = tuple(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_CLIP", (0.60, 2.00)))
        cal_tol = 1e-8
        bad_cal = v12.filter(
            (pl.col("v12_sku_level_calibration_factor_y") < cal_lo - cal_tol)
            | (pl.col("v12_sku_level_calibration_factor_y") > cal_hi + cal_tol)
            | (pl.col("v12_sku_level_calibration_factor_value") < cal_lo - cal_tol)
            | (pl.col("v12_sku_level_calibration_factor_value") > cal_hi + cal_tol)
            | (pl.col("v12_sku_level_calibration_blocks_y").fill_null(0) < 0)
            | (pl.col("v12_sku_level_calibration_blocks_value").fill_null(0) < 0)
            | ((pl.col("v12_sku_forecast_calibrated_challenger_y")
                - pl.col("v12_sku_forecast_uncalibrated_y") * pl.col("v12_sku_level_calibration_factor_y")).abs()
               > cal_tol * pl.max_horizontal(pl.lit(1.0), pl.col("v12_sku_forecast_calibrated_challenger_y").abs()))
            | ((pl.col("v12_sku_forecast_calibrated_challenger_value")
                - pl.col("v12_sku_forecast_uncalibrated_value") * pl.col("v12_sku_level_calibration_factor_value")).abs()
               > cal_tol * pl.max_horizontal(pl.lit(1.0), pl.col("v12_sku_forecast_calibrated_challenger_value").abs()))
            | (
                pl.col("v12_sku_level_calibration_applied_y").fill_null(False)
                & ((pl.col("v12_sku_forecast_base_y") - pl.col("v12_sku_forecast_calibrated_challenger_y")).abs()
                   > cal_tol * pl.max_horizontal(pl.lit(1.0), pl.col("v12_sku_forecast_base_y").abs()))
            )
            | (
                ~pl.col("v12_sku_level_calibration_applied_y").fill_null(False)
                & ((pl.col("v12_sku_forecast_base_y") - pl.col("v12_sku_forecast_uncalibrated_y")).abs()
                   > cal_tol * pl.max_horizontal(pl.lit(1.0), pl.col("v12_sku_forecast_base_y").abs()))
            )
            | (
                pl.col("v12_sku_level_calibration_applied_value").fill_null(False)
                & ((pl.col("v12_sku_forecast_base_value") - pl.col("v12_sku_forecast_calibrated_challenger_value")).abs()
                   > cal_tol * pl.max_horizontal(pl.lit(1.0), pl.col("v12_sku_forecast_base_value").abs()))
            )
            | (
                ~pl.col("v12_sku_level_calibration_applied_value").fill_null(False)
                & ((pl.col("v12_sku_forecast_base_value") - pl.col("v12_sku_forecast_uncalibrated_value")).abs()
                   > cal_tol * pl.max_horizontal(pl.lit(1.0), pl.col("v12_sku_forecast_base_value").abs()))
            )
        )
        if bad_cal.height:
            errors.append(
                f"{bad_cal.height} filas v12.8 violan contrato baseline/calibration-challenger SKU-total"
            )

        allowed_shape_models = ["base", "lgbm_shape_g1p0"]
        bad_shape_model = v12.filter(
            (~pl.col("v12_sku_shape_model_y").is_in(allowed_shape_models))
            | (~pl.col("v12_sku_shape_model_value").is_in(allowed_shape_models))
            | (pl.col("v12_sku_shape_training_rows_y").fill_null(0) < 0)
            | (pl.col("v12_sku_shape_training_rows_value").fill_null(0) < 0)
        )
        if bad_shape_model.height:
            errors.append(
                f"{bad_shape_model.height} filas v12.8 tienen metadata LightGBM shape inválida"
            )

        if v12.height:
            # v12.6 hard invariant: the pooled LightGBM layer can redistribute
            # daily SKU shape but must preserve the pre-LGBM 28-day SKU total.
            shape_tol = float(getattr(settings, "V12_SKU_SHAPE_LGBM_TOTAL_TOLERANCE", 1e-9))
            sku_day_shape = (
                v12.with_columns(
                    pl.col("unique_id").str.extract(r"\|\|S:(.+)$", 1).alias("_sku")
                )
                .group_by(["_sku", "ds", "period_type"])
                .agg(
                    pl.col("v12_sku_forecast_base_y").drop_nulls().first().alias("_by"),
                    pl.col("v12_sku_forecast_y").drop_nulls().first().alias("_cy"),
                    pl.col("v12_sku_forecast_base_value").drop_nulls().first().alias("_bv"),
                    pl.col("v12_sku_forecast_value").drop_nulls().first().alias("_cv"),
                )
            )
            bad_shape_total = (
                sku_day_shape.group_by(["_sku", "period_type"])
                .agg(
                    pl.col("_by").sum().alias("_by"),
                    pl.col("_cy").sum().alias("_cy"),
                    pl.col("_bv").sum().alias("_bv"),
                    pl.col("_cv").sum().alias("_cv"),
                )
                .filter(
                    ((pl.col("_by") - pl.col("_cy")).abs()
                     > shape_tol * pl.max_horizontal(pl.lit(1.0), pl.col("_by").abs()))
                    | ((pl.col("_bv") - pl.col("_cv")).abs()
                       > shape_tol * pl.max_horizontal(pl.lit(1.0), pl.col("_bv").abs()))
                )
            )
            if bad_shape_total.height:
                errors.append(
                    f"{bad_shape_total.height} SKU/horizonte v12.8 violan total28 base == LGBM-shape"
                )

        if v12.height:
            share_sums = (
                v12.with_columns(
                    pl.col("unique_id").str.extract(r"\|\|S:(.+)$", 1).alias("_sku")
                )
                .group_by(["_sku", "ds", "period_type"])
                .agg(
                    pl.col("v12_store_share_y").sum().alias("_sy"),
                    pl.col("v12_store_share_value").sum().alias("_sv"),
                )
            )
            bad_share = share_sums.filter(
                ((pl.col("_sy") - 1.0).abs() > 1e-6)
                | ((pl.col("_sv") - 1.0).abs() > 1e-6)
            )
            if bad_share.height:
                errors.append(
                    f"{bad_share.height} SKU/día v12 no suman store-share=1"
                )

            # v12.3 shared true-hurdle invariant. If at least one store opens for a
            # SKU/day, stores whose occurrence gate is closed must receive zero
            # share. If every gate is closed, the documented base-share fallback
            # is allowed so the SKU total is still allocatable.
            hurdle = (
                v12.with_columns(
                    pl.col("unique_id").str.extract(r"\|\|S:(.+)$", 1).alias("_sku")
                )
                .with_columns(
                    pl.col("v12_occurrence_gate_open_y").fill_null(False)
                    .any().over(["_sku", "ds", "period_type"])
                    .alias("_any_open_y"),
                    pl.col("v12_occurrence_gate_open_value").fill_null(False)
                    .any().over(["_sku", "ds", "period_type"])
                    .alias("_any_open_v"),
                )
            )
            bad_hurdle = hurdle.filter(
                (pl.col("_any_open_y") & ~pl.col("v12_occurrence_gate_open_y").fill_null(False) & (pl.col("v12_store_share_y").abs() > 1e-9))
                | (pl.col("_any_open_v") & ~pl.col("v12_occurrence_gate_open_value").fill_null(False) & (pl.col("v12_store_share_value").abs() > 1e-9))
            )
            if bad_hurdle.height:
                errors.append(
                    f"{bad_hurdle.height} filas v12 violan true-hurdle: gate cerrado con share positivo"
                )

            # v12.3 shared-occurrence invariant: quantity and value represent
            # the same retail sale event, so their hurdle state must be equal
            # to the explicit joint gate on every candidate row.
            bad_shared_gate = hurdle.filter(
                (pl.col("v12_occurrence_gate_open_y").fill_null(False)
                 != pl.col("v12_occurrence_gate_open_value").fill_null(False))
                | (pl.col("v12_occurrence_gate_open_y").fill_null(False)
                   != pl.col("v12_occurrence_gate_open_joint").fill_null(False))
            )
            if bad_shared_gate.height:
                errors.append(
                    f"{bad_shared_gate.height} filas v12 violan shared hurdle quantity/value"
                )

        meta_enabled = bool(getattr(settings, "V12_META_SELECTOR_ENABLED", True))
        meta_max_recent_degradation = float(getattr(settings, "V12_META_SELECTOR_MAX_RECENT_DEGRADATION", 0.08))
        min_blocks = int(getattr(settings, "V12_SELECTION_MIN_BLOCKS", 2))
        min_wins = int(getattr(settings, "V12_SELECTION_MIN_WINS", 2))
        min_improvement = float(getattr(settings, "V12_MIN_IMPROVEMENT", 0.02))
        require_recent = bool(getattr(settings, "V12_REQUIRE_RECENT_WIN", True))
        recent_min = float(getattr(settings, "V12_RECENT_MIN_IMPROVEMENT", min_improvement))
        utility_min = float(getattr(settings, "V12_SELECTOR_MIN_UTILITY_IMPROVEMENT", 0.0025))
        utility_stability = float(getattr(settings, "V12_SELECTOR_STABILITY_WEIGHT", 0.10))

        # v12.8.3: selection is authorized by the adaptive causal policy; Value
        # adds a portfolio-safety layer while Qty remains v12.8.2. The legacy gate
        # is diagnostic only. v12_all is a section/target policy mode;
        # meta_leaf uses the learned row-specific threshold and BIAS guard;
        # when a meta fit is unavailable, the conservative v12.7 rule remains
        # the fallback.
        fallback_ok_y = (
            (~pl.col("v12_meta_model_available_y").fill_null(False) | ~pl.lit(meta_enabled))
            & (pl.col("v12_validation_blocks_y").fill_null(0) >= min_blocks)
            & (pl.col("v12_validation_wins_y").fill_null(0) >= min_wins)
            & (pl.col("v12_validation_improvement_y").fill_null(-999.0) >= min_improvement)
            & (pl.col("v12_validation_utility_improvement_y").fill_null(-999.0) >= utility_min)
            & ((~pl.lit(require_recent)) | (pl.col("v12_validation_recent_improvement_y").fill_null(-999.0) >= recent_min))
            & ((~pl.lit(require_recent)) | (pl.col("v12_validation_recent_utility_improvement_y").fill_null(-999.0) >= utility_min))
            & (pl.col("v12_validation_utility_mean_y").fill_null(-999.0)
               >= utility_stability * pl.col("v12_validation_utility_std_y").fill_null(0.0))
        )
        fallback_ok_v = (
            (~pl.col("v12_meta_model_available_value").fill_null(False) | ~pl.lit(meta_enabled))
            & (pl.col("v12_validation_blocks_value").fill_null(0) >= min_blocks)
            & (pl.col("v12_validation_wins_value").fill_null(0) >= min_wins)
            & (pl.col("v12_validation_improvement_value").fill_null(-999.0) >= min_improvement)
            & (pl.col("v12_validation_utility_improvement_value").fill_null(-999.0) >= utility_min)
            & ((~pl.lit(require_recent)) | (pl.col("v12_validation_recent_improvement_value").fill_null(-999.0) >= recent_min))
            & ((~pl.lit(require_recent)) | (pl.col("v12_validation_recent_utility_improvement_value").fill_null(-999.0) >= utility_min))
            & (pl.col("v12_validation_utility_mean_value").fill_null(-999.0)
               >= utility_stability * pl.col("v12_validation_utility_std_value").fill_null(0.0))
        )
        meta_ok_y = (
            pl.lit(meta_enabled)
            & pl.col("v12_meta_model_available_y").fill_null(False)
            & (pl.col("v12_meta_probability_y").fill_null(-1.0) >= pl.col("v12_meta_threshold_y"))
            & (pl.col("v12_meta_recent_gain_y").fill_null(-999.0) >= -meta_max_recent_degradation)
            & pl.col("v12_meta_bias_guard_pass_y").fill_null(False)
        )
        meta_ok_v = (
            pl.lit(meta_enabled)
            & pl.col("v12_meta_model_available_value").fill_null(False)
            & (pl.col("v12_meta_probability_value").fill_null(-1.0) >= pl.col("v12_meta_threshold_value"))
            & (pl.col("v12_meta_recent_gain_value").fill_null(-999.0) >= -meta_max_recent_degradation)
            & pl.col("v12_meta_bias_guard_pass_value").fill_null(False)
        )
        # Compatibilidad de contrato: la política causal meta/portfolio sigue
        # validándose para Qty y para cadencias donde expected-gain no gobierna Valor.
        selected_ok_y = (
            (pl.col("v12_meta_portfolio_mode_y") == "v12_all")
            | ((pl.col("v12_meta_portfolio_mode_y") == "meta_leaf") & (meta_ok_y | fallback_ok_y))
        )
        seg_enabled = bool(getattr(settings, "V1293_VALUE_SEGMENT_SELECTOR_ENABLED", True)) and int(
            getattr(settings, "RLS_BLOCK_DAYS", 28)
        ) == 28
        segment_ok_v = (
            pl.col("v1293_segment_selector_enabled_value").fill_null(False)
            & pl.col("v1293_segment_prebudget_value").fill_null(False)
            & pl.col("v1293_segment_budget_selected_value").fill_null(False)
            & (pl.col("v1293_segment_folds_value").fill_null(0) >= 3)
            & (pl.col("v1293_segment_win_rate_value").fill_null(0.0) >= (2.0 / 3.0))
            & pl.col("v1293_segment_reason_value").fill_null("").str.contains("walkforward_stable", literal=True)
        )
        v1296_enabled = bool(getattr(settings, "V1296_SEC23_STRESS_GATE_ENABLED", True)) and int(
            getattr(settings, "RLS_BLOCK_DAYS", 28)
        ) == 28
        sec23_expr = pl.col("unique_id").str.starts_with("23||")
        stress_gate_ok_v = (
            pl.col("v1296_stress_gate_enabled_value").fill_null(False)
            & pl.col("v1296_stress_eligible_value").fill_null(False)
            & pl.col("v1296_stress_gate_pass_value").fill_null(False)
            & pl.col("v1296_budget_selected_value").fill_null(False)
            & pl.col("v1296_high_impact_guard_pass_value").fill_null(False)
            & pl.col("v1295_stress_level_value").is_in(["L4", "L3"])
            & (pl.col("v1295_stress_folds_value").fill_null(0) >= int(getattr(settings, "V1296_SEC23_MIN_FOLDS", 8)))
        )
        if seg_enabled:
            selected_ok_v = (
                (sec23_expr & stress_gate_ok_v) | ((~sec23_expr) & segment_ok_v)
                if v1296_enabled else segment_ok_v
            )
        else:
            selected_ok_v = (
                (pl.col("v12_meta_portfolio_mode_value") == "v12_all")
                | ((pl.col("v12_meta_portfolio_mode_value") == "meta_leaf") & (meta_ok_v | fallback_ok_v))
            )
        bad_selection = target.filter(
            (pl.col("v12_selected_y") & ~selected_ok_y)
            | (pl.col("v12_selected_value") & ~selected_ok_v)
            | ((pl.col("v12_meta_portfolio_mode_y") == "v11_all") & pl.col("v12_selected_y"))
            | ((~pl.lit(seg_enabled)) & (pl.col("v12_meta_portfolio_mode_value") == "v11_all") & pl.col("v12_selected_value"))
        )
        if bad_selection.height:
            errors.append(
                f"{bad_selection.height} filas v12.9.7 violan política causal del selector jerárquico"
            )

        bad_policy_mode = target.filter(
            ~pl.col("v12_meta_portfolio_mode_y").is_in(["v11_all", "v12_all", "meta_leaf"])
            | ~pl.col("v12_meta_portfolio_mode_value").is_in(["v11_all", "v12_all", "meta_leaf"])
            | (pl.col("v12_meta_threshold_y") <= 0.0) | (pl.col("v12_meta_threshold_y") >= 1.0)
            | (pl.col("v12_meta_threshold_value") <= 0.0) | (pl.col("v12_meta_threshold_value") >= 1.0)
        )
        if bad_policy_mode.height:
            errors.append(f"{bad_policy_mode.height} filas v12.9.2 tienen policy mode/threshold inválido")

        # v12.9.2 Value walk-forward contract. Legacy Value Safety metadata
        # remains diagnostic, but the authoritative production gate is the
        # walk-forward policy whenever enabled.
        wf_enabled = bool(getattr(settings, "V129_VALUE_WALK_FORWARD_ENABLED", True))
        min_wf_folds = int(getattr(settings, "V129_VALUE_WF_MIN_FOLDS", 3))
        bad_wf_range = target.filter(
            (pl.col("v129_value_wf_folds") < 0)
            | (pl.col("v129_value_wf_meta_folds") < 0)
            | (pl.col("v129_value_wf_win_rate").is_not_null()
               & ((pl.col("v129_value_wf_win_rate") < 0.0) | (pl.col("v129_value_wf_win_rate") > 1.0)))
        )
        if bad_wf_range.height:
            errors.append(f"{bad_wf_range.height} filas v12.9.2 tienen metadata walk-forward inválida")

        if wf_enabled:
            bad_wf_mode = target.filter(
                pl.col("v129_value_wf_enabled").fill_null(False)
                & pl.col("v129_value_wf_available").fill_null(False)
                & pl.col("v12_meta_portfolio_mode_value").is_in(["v12_all", "meta_leaf"])
                & (pl.col("v129_value_wf_folds").fill_null(0) < min_wf_folds)
            )
            if bad_wf_mode.height:
                errors.append(
                    f"{bad_wf_mode.height} filas v12.9.2 promueven Value sin folds walk-forward suficientes"
                )
            bad_wf_reason = target.filter(
                pl.col("v129_value_wf_enabled").fill_null(False)
                & pl.col("v129_value_wf_reason").is_null()
            )
            if bad_wf_reason.height:
                errors.append(f"{bad_wf_reason.height} filas v12.9.2 carecen de razón walk-forward")

        # v12.9.1/2 expected-impact fields are diagnostic only in v12.9.7.
        # Validate bounded metadata without requiring them to authorize Value.
        bad_eg = target.filter(
            (pl.col("v1291_expected_gain_confidence_value").is_not_null()
             & ((pl.col("v1291_expected_gain_confidence_value") < 0.0)
                | (pl.col("v1291_expected_gain_confidence_value") > 1.0)))
            | (pl.col("v1292_bucket_loss_rate_value").is_not_null()
               & ((pl.col("v1292_bucket_loss_rate_value") < 0.0)
                  | (pl.col("v1292_bucket_loss_rate_value") > 1.0)))
            | (pl.col("v1292_leaf_loss_rate_value").is_not_null()
               & ((pl.col("v1292_leaf_loss_rate_value") < 0.0)
                  | (pl.col("v1292_leaf_loss_rate_value") > 1.0)))
            | (pl.col("v1292_impact_percentile_value").is_not_null()
               & ((pl.col("v1292_impact_percentile_value") <= 0.0)
                  | (pl.col("v1292_impact_percentile_value") > 1.0)))
        )
        if bad_eg.height:
            errors.append(f"{bad_eg.height} filas v12.9.7 tienen metadata diagnóstica expected-impact inválida")

        if seg_enabled:
            # v12.9.7: sparse/zero or otherwise non-applicable leaves may carry
            # null/disabled segment metadata. Validate ranges only where the
            # segment selector is actually materialized; selected v12 leaves
            # must always have complete causal segment evidence.
            seg_rows = target.filter(pl.col("v1293_segment_selector_enabled_value").fill_null(False))
            bad_seg = seg_rows.filter(
                (pl.col("v1293_segment_folds_value").fill_null(0) < 0)
                | (pl.col("v1293_segment_win_rate_value").is_not_null()
                   & ((pl.col("v1293_segment_win_rate_value") < 0.0)
                      | (pl.col("v1293_segment_win_rate_value") > 1.0)))
                | (pl.col("v1293_segment_selected_volume_share_value").fill_null(-1.0) < 0.0)
                | (pl.col("v1293_segment_selected_volume_share_value").fill_null(2.0) > 1.0)
                | (pl.col("v1293_segment_budget_value").fill_null(-1.0) < 0.0)
                | (pl.col("v1293_segment_budget_value").fill_null(2.0) > 1.0)
                | (pl.col("v1294_segment_exposure_budget_value").fill_null(0.0) < 0.0)
                | (pl.col("v1294_segment_exposure_budget_value").fill_null(0.0) > 1.0)
                | (pl.col("v1294_impact_percentile_value").is_not_null()
                   & ((pl.col("v1294_impact_percentile_value") <= 0.0)
                      | (pl.col("v1294_impact_percentile_value") > 1.0)))
            )
            if bad_seg.height:
                errors.append(f"{bad_seg.height} filas v12.9.7 tienen metadata segment-selector inválida")
            bad_selected_seg = target.filter(
                pl.col("v12_selected_value").fill_null(False)
                & (~pl.col("unique_id").str.starts_with("23||"))
                & (
                    ~pl.col("v1293_segment_selector_enabled_value").fill_null(False)
                    | ~pl.col("v1293_segment_prebudget_value").fill_null(False)
                    | ~pl.col("v1293_segment_budget_selected_value").fill_null(False)
                    | (pl.col("v1293_segment_folds_value").fill_null(0) < 3)
                    | pl.col("v1293_segment_reason_value").is_null()
                )
            )
            if bad_selected_seg.height:
                errors.append(f"{bad_selected_seg.height} filas v12.9.7 seleccionan Value v12 sin metadata segment-selector completa")
            sec23 = target.filter(
                pl.col("unique_id").str.starts_with("23||")
                & pl.col("v1293_segment_selector_enabled_value").fill_null(False)
            )
            if sec23.height:
                hard_budget = float(getattr(settings, "V1294_VALUE_SEC23_MAX_VOLUME_SHARE", 0.25))
                bad_budget = sec23.filter(
                    pl.col("v1293_segment_selected_volume_share_value").fill_null(0.0) > hard_budget + 1e-9
                )
                if bad_budget.height:
                    errors.append(f"{bad_budget.height} filas v12.9.7 exceden budget duro de volumen Sec23")

        stress_rows = target.filter(pl.col("v1295_stress_enabled_value").fill_null(False))
        if stress_rows.height:
            min_stress_folds = int(getattr(settings, "V1295_STRESS_TEST_MIN_FOLDS", 8))
            bad_stress = stress_rows.filter(
                (pl.col("v1295_stress_requested_blocks_value").fill_null(0) < min_stress_folds)
                | (pl.col("v1295_stress_materialized_blocks_value").fill_null(0) < 0)
                | (pl.col("v1295_stress_materialized_blocks_value").fill_null(0) > pl.col("v1295_stress_requested_blocks_value").fill_null(0))
                | (pl.col("v1295_stress_folds_value").fill_null(0) < 0)
                | (pl.col("v1295_stress_folds_value").fill_null(0) > pl.col("v1295_stress_requested_blocks_value").fill_null(0))
                | (pl.col("v1295_stress_win_rate_value").is_not_null()
                   & ((pl.col("v1295_stress_win_rate_value") < 0.0) | (pl.col("v1295_stress_win_rate_value") > 1.0)))
                | (pl.col("v1295_stress_max_consecutive_losses_value").fill_null(0) < 0)
                | (pl.col("v1295_stress_max_consecutive_losses_value").fill_null(0) > pl.col("v1295_stress_folds_value").fill_null(0))
            )
            if bad_stress.height:
                errors.append(f"{bad_stress.height} filas v12.9.7 tienen metadata long-horizon stress inválida")
            bad_stress_pass = stress_rows.filter(
                pl.col("v1295_stress_pass_value").fill_null(False)
                & (
                    (pl.col("v1295_stress_folds_value").fill_null(0) < min_stress_folds)
                    | (pl.col("v1295_stress_key_value").is_null())
                    | (pl.col("v1295_stress_p25_gain_value").is_null())
                    | (pl.col("v1295_stress_p10_gain_value").is_null())
                    | (pl.col("v1295_stress_reason_value") != "long_horizon_stable")
                )
            )
            if bad_stress_pass.height:
                errors.append(f"{bad_stress_pass.height} filas v12.9.7 marcan stress-pass sin evidencia histórica completa")

        if bool(getattr(settings, "V1296_SEC23_STRESS_GATE_ENABLED", True)):
            # v12.9.7 corrected: validate the Sec23 stress/replay contract only
            # on leaf rows where that metadata is actually applicable.  Sparse/
            # fallback rows may legitimately carry null diagnostic metadata.
            sec23_v1296 = target.filter(
                pl.col("unique_id").str.starts_with("23||")
                & pl.col("v1296_stress_gate_enabled_value").fill_null(False)
            )
            bad_v1296_meta = sec23_v1296.filter(
                (pl.col("v1296_budget_value").fill_null(-1.0) < 0.0)
                | (pl.col("v1296_budget_value").fill_null(2.0) > 1.0)
                | (pl.col("v1296_selected_volume_share_value").fill_null(-1.0) < 0.0)
                | (pl.col("v1296_selected_volume_share_value").fill_null(2.0) > pl.col("v1296_budget_value").fill_null(0.0) + 1e-9)
                | (pl.col("v1296_budget_selected_value").fill_null(False) & ~pl.col("v1296_stress_eligible_value").fill_null(False))
                | (pl.col("v1296_budget_selected_value").fill_null(False) & ~pl.col("v1296_high_impact_guard_pass_value").fill_null(False))
                | (pl.col("v1296_budget_selected_value").fill_null(False) & ~pl.col("v1295_stress_level_value").is_in(["L4", "L3"]))
                | ((pl.col("v1295_stress_level_value") == "L1") & pl.col("v12_selected_value").fill_null(False))
                | ((pl.col("v1295_stress_level_value") == "L2") & pl.col("v12_selected_value").fill_null(False))
            )
            if bad_v1296_meta.height:
                errors.append(f"{bad_v1296_meta.height} filas v12.9.7 violan stress-gate productivo Sec23")
            # Final family can only apply a budget-selected challenger when the
            # candidate itself exists.  Compare against that final post-gate,
            # post-budget, candidate-available decision (not the pre-application
            # budget flag alone).
            expected_v1296_final = (
                pl.col("v1296_budget_selected_value").fill_null(False)
                & pl.col("v12_candidate_available_value").fill_null(False)
            )
            mismatch_v1296 = sec23_v1296.filter(
                pl.col("v12_selected_value").fill_null(False) != expected_v1296_final
            )
            if mismatch_v1296.height:
                errors.append(f"{mismatch_v1296.height} filas v12.9.7 no coinciden entre familia final y budget stress-gate Sec23")
            replay_applicable = pl.col("v1297_temporal_replay_enabled_value").fill_null(False)
            bad_v1297 = sec23_v1296.filter(
                replay_applicable
                & (
                    (pl.col("v1297_temporal_replay_cutoffs_value").fill_null(0) <= 0)
                    | (pl.col("v1297_temporal_replay_win_rate_value").fill_null(-1.0) < 0.0)
                    | (pl.col("v1297_temporal_replay_win_rate_value").fill_null(2.0) > 1.0)
                    | (pl.col("v1297_temporal_replay_selected_share_median_value").fill_null(-1.0) < 0.0)
                    | (pl.col("v1297_temporal_replay_selected_share_median_value").fill_null(2.0) > 1.0)
                )
            )
            if bad_v1297.height:
                errors.append(f"{bad_v1297.height} filas v12.9.7 tienen metadata temporal-replay inválida")

        # v12.9.8 sparse-bias rescue is diagnostic only. Validate metadata on
        # leaf OOS/forecast-only rows when present; it must never authorize
        # negative/over-cap factors or non-sparse candidate buckets.
        sparse_cols = {
            "v1298_sparse_bias_enabled", "v1298_sparse_bucket_y", "v1298_sparse_bucket_value",
            "v1298_sparse_folds_y", "v1298_sparse_folds_value",
            "v1298_sparse_median_ratio_y", "v1298_sparse_median_ratio_value",
            "v1298_sparse_p25_ratio_y", "v1298_sparse_p25_ratio_value",
            "v1298_sparse_worst_ratio_y", "v1298_sparse_worst_ratio_value",
            "v1298_sparse_factor_y", "v1298_sparse_factor_value",
            "v1298_sparse_candidate_y", "v1298_sparse_candidate_value",
        }
        if sparse_cols.issubset(set(target.columns)):
            clip_hi = float(getattr(settings, "V1298_SPARSE_BIAS_FACTOR_CLIP", (1.0, 1.15))[1])
            sparse_buckets = list(getattr(settings, "V1298_SPARSE_BIAS_BUCKETS", ("07-09", "10-13")))
            bad_sparse = target.filter(
                (pl.col("v1298_sparse_factor_y").fill_null(1.0) < 1.0)
                | (pl.col("v1298_sparse_factor_y").fill_null(1.0) > clip_hi + 1e-9)
                | (pl.col("v1298_sparse_factor_value").fill_null(1.0) < 1.0)
                | (pl.col("v1298_sparse_factor_value").fill_null(1.0) > clip_hi + 1e-9)
                | (pl.col("v1298_sparse_folds_y").fill_null(0) < 0)
                | (pl.col("v1298_sparse_folds_value").fill_null(0) < 0)
                | (pl.col("v1298_sparse_candidate_y").fill_null(False) & ~pl.col("v1298_sparse_bucket_y").is_in(sparse_buckets))
                | (pl.col("v1298_sparse_candidate_value").fill_null(False) & ~pl.col("v1298_sparse_bucket_value").is_in(sparse_buckets))
            )
            if bad_sparse.height:
                errors.append(f"{bad_sparse.height} filas v12.9.8 tienen metadata sparse-bias inválida")

        # v12.9.9 segmented Value rescue remains diagnostic/non-productive.
        sparse99_cols = {
            "v1299_value_sparse_enabled", "v1299_value_sparse_bucket",
            "v1299_value_sparse_factor", "v1299_value_sparse_folds",
            "v1299_value_sparse_win_rate", "v1299_value_sparse_candidate",
            "v1299_value_sparse_replay_cutoffs", "v1299_value_sparse_replay_win_rate",
            "v1299_value_sparse_replay_selected_share_median",
        }
        if sparse99_cols.issubset(set(target.columns)):
            grid99 = tuple(float(x) for x in getattr(settings, "V1299_VALUE_SPARSE_FACTOR_GRID", (1.0, 1.15)))
            lo99, hi99 = min(grid99), max(grid99)
            buckets99 = list(getattr(settings, "V1299_VALUE_SPARSE_BUCKETS", ("07-09", "10-13")))
            bad99 = target.filter(
                (pl.col("v1299_value_sparse_factor").fill_null(1.0) < lo99 - 1e-9)
                | (pl.col("v1299_value_sparse_factor").fill_null(1.0) > hi99 + 1e-9)
                | (pl.col("v1299_value_sparse_folds").fill_null(0) < 0)
                | (pl.col("v1299_value_sparse_win_rate").fill_null(0.0) < 0.0)
                | (pl.col("v1299_value_sparse_win_rate").fill_null(1.0) > 1.0)
                | (pl.col("v1299_value_sparse_candidate").fill_null(False) & ~pl.col("v1299_value_sparse_bucket").is_in(buckets99))
                | (pl.col("v1299_value_sparse_replay_cutoffs").fill_null(0) < 0)
                | (pl.col("v1299_value_sparse_replay_win_rate").fill_null(0.0) < 0.0)
                | (pl.col("v1299_value_sparse_replay_win_rate").fill_null(1.0) > 1.0)
                | (pl.col("v1299_value_sparse_replay_selected_share_median").is_not_null()
                   & ((pl.col("v1299_value_sparse_replay_selected_share_median") < 0.0)
                      | (pl.col("v1299_value_sparse_replay_selected_share_median") > 1.0)))
            )
            if bad99.height:
                errors.append(f"{bad99.height} filas v12.9.9 tienen metadata sparse-segmentada inválida")

        # v12.9.10 productive Value rescue must be gated solely by ACTIVE replay
        # metadata and its write-back must exactly match the learned factor.
        sparse10_cols = {
            "v12910_value_sparse_promotion_enabled",
            "v12910_value_sparse_active_replay_cutoffs",
            "v12910_value_sparse_active_replay_win_rate",
            "v12910_value_sparse_active_replay_selected_share_median",
            "v12910_value_sparse_promoted", "v12910_value_sparse_promotion_reason",
            "v12910_value_sparse_applied", "v12910_valuehat_raw_before_sparse",
            "v1299_value_sparse_candidate", "v1299_value_sparse_factor", "valuehat_raw",
        }
        if sparse10_cols.issubset(set(target.columns)):
            max_share10 = float(getattr(settings, "V12910_VALUE_SPARSE_MAX_SELECTED_VOLUME_SHARE", 0.30))
            min_cut10 = int(getattr(settings, "V12910_VALUE_SPARSE_MIN_REPLAY_CUTOFFS", 4))
            expected_apply = (
                pl.col("v12910_value_sparse_promoted").fill_null(False)
                & pl.col("v1299_value_sparse_candidate").fill_null(False)
                & (pl.col("v1299_value_sparse_factor").fill_null(1.0) > 1.0)
            )
            bad10 = target.filter(
                (pl.col("v12910_value_sparse_active_replay_cutoffs").fill_null(0) < 0)
                | (pl.col("v12910_value_sparse_active_replay_win_rate").fill_null(0.0) < 0.0)
                | (pl.col("v12910_value_sparse_active_replay_win_rate").fill_null(1.0) > 1.0)
                | (pl.col("v12910_value_sparse_active_replay_selected_share_median").is_not_null()
                   & ((pl.col("v12910_value_sparse_active_replay_selected_share_median") < 0.0)
                      | (pl.col("v12910_value_sparse_active_replay_selected_share_median") > 1.0)))
                | (pl.col("v12910_value_sparse_promoted").fill_null(False)
                   & (pl.col("v12910_value_sparse_active_replay_cutoffs").fill_null(0) < min_cut10))
                | (pl.col("v12910_value_sparse_promoted").fill_null(False)
                   & (pl.col("v12910_value_sparse_active_replay_selected_share_median").fill_null(1.0) > max_share10 + 1e-9))
                | (pl.col("v12910_value_sparse_applied").fill_null(False) != expected_apply)
            )
            if bad10.height:
                errors.append(f"{bad10.height} filas v12.9.10 violan promotion-gate sparse ACTIVE")
            expected_raw = pl.when(expected_apply).then(
                pl.col("v12910_valuehat_raw_before_sparse") * pl.col("v1299_value_sparse_factor").fill_null(1.0)
            ).otherwise(pl.col("v12910_valuehat_raw_before_sparse"))
            bad10_raw = target.filter(
                pl.col("v12910_valuehat_raw_before_sparse").is_not_null()
                & ((pl.col("valuehat_raw") - expected_raw).abs() > 1e-6)
            )
            if bad10_raw.height:
                errors.append(f"{bad10_raw.height} filas v12.9.10 no coinciden entre baseline y write-back sparse")

        # v12.9.11 is diagnostic-only: metadata must be bounded and it may
        # nominate only Sec23 leaves that the productive family selector kept on v11.
        residual11_cols = {
            "v12911_residual_candidate_value", "v12911_residual_folds_value",
            "v12911_residual_win_rate_value", "v12911_residual_selected_share_value",
            "v12911_residual_replay_cutoffs_value", "v12911_residual_replay_win_rate_value",
            "v12911_residual_replay_selected_share_median_value", "v12911_residual_replay_pass_value",
            "v12_selected_value", "unique_id",
        }
        if residual11_cols.issubset(set(target.columns)):
            max_share11 = float(getattr(settings, "V12911_SEC23_VALUE_MAX_VOLUME_SHARE", 0.10))
            bad11 = target.filter(
                (pl.col("v12911_residual_folds_value").fill_null(0) < 0)
                | (pl.col("v12911_residual_win_rate_value").fill_null(0.0) < 0.0)
                | (pl.col("v12911_residual_win_rate_value").fill_null(1.0) > 1.0)
                | (pl.col("v12911_residual_selected_share_value").fill_null(0.0) < 0.0)
                | (pl.col("v12911_residual_selected_share_value").fill_null(0.0) > max_share11 + 1e-9)
                | (pl.col("v12911_residual_replay_cutoffs_value").fill_null(0) < 0)
                | (pl.col("v12911_residual_replay_win_rate_value").fill_null(0.0) < 0.0)
                | (pl.col("v12911_residual_replay_win_rate_value").fill_null(1.0) > 1.0)
                | (pl.col("v12911_residual_replay_selected_share_median_value").fill_null(0.0) < 0.0)
                | (pl.col("v12911_residual_replay_selected_share_median_value").fill_null(0.0) > 1.0)
                | (pl.col("v12911_residual_candidate_value").fill_null(False)
                   & (~pl.col("unique_id").str.starts_with("23||") | pl.col("v12_selected_value").fill_null(False)))
            )
            if bad11.height:
                errors.append(f"{bad11.height} filas v12.9.11 tienen metadata residual-selector inválida")

        bad_meta_prob = target.filter(
            (pl.col("v12_meta_probability_y").is_not_null()
             & ((pl.col("v12_meta_probability_y") < 0.0) | (pl.col("v12_meta_probability_y") > 1.0)))
            | (pl.col("v12_meta_probability_value").is_not_null()
               & ((pl.col("v12_meta_probability_value") < 0.0) | (pl.col("v12_meta_probability_value") > 1.0)))
            | (pl.col("v12_meta_training_rows_y").fill_null(0) < 0)
            | (pl.col("v12_meta_training_rows_value").fill_null(0) < 0)
        )
        if bad_meta_prob.height:
            errors.append(f"{bad_meta_prob.height} filas v12.8 tienen metadata meta-selector inválida")

        if bool(getattr(settings, "V12_REQUIRE_JOINT_TARGET_WIN", True)):
            bad_joint = target.filter(
                pl.col("v12_selected_y").fill_null(False)
                != pl.col("v12_selected_value").fill_null(False)
            )
            if bad_joint.height:
                errors.append(
                    f"{bad_joint.height} filas v12 violan selección conjunta quantity/value"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast", default=None, help="Ruta a forecast.parquet",
    )
    parser.add_argument(
        "--update-block-days", type=int, choices=list(getattr(settings, "UPDATE_BLOCK_OPTIONS", (1,7,14,28))), default=None,
        help="Validar un escenario precalculado de la demo.",
    )
    args = parser.parse_args()
    path = (
        settings.update_block_forecast_path(args.update_block_days)
        if args.update_block_days is not None
        else Path(args.forecast or settings.FORECAST_PATH)
    )

    errors = validate(path)
    if errors:
        print(f"AUDITORÍA MODELO v{settings.APP_VERSION}: ERROR")
        for error in errors:
            print(" -", error)
        return 1

    summary_scan = pl.scan_parquet(path)
    summary_schema = set(summary_scan.collect_schema().names())
    summary_cols = [
        c for c in ("unique_id", "period_type", "ses_guard_status_value", "ses_regime_class_value")
        if c in summary_schema
    ]
    summary_target = (
        summary_scan.select(summary_cols)
        .filter(_leaf_expr())
        .filter(pl.col("period_type").is_in(["out_sample", "forecast_only"]))
    )
    target_count = summary_target.select(pl.len().alias("n")).collect(engine="streaming").item()
    statuses = (
        summary_target.group_by(["ses_guard_status_value", "ses_regime_class_value"])
        .len()
        .sort("len", descending=True)
        .collect(engine="streaming")
        if "ses_regime_class_value" in summary_schema
        else pl.DataFrame()
    )

    print(f"AUDITORÍA MODELO v{settings.APP_VERSION}: OK")
    print(f"Forecast: {path}")
    print(f"Filas leaf OOS/forecast-only: {target_count:,}")
    if statuses.height:
        print("Intervenciones/regímenes:")
        print(statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
