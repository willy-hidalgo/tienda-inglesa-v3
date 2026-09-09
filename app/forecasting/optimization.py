"""Statistical tuning diagnostics for the coherent v13 SES+RLS architecture.

This module never changes the productive forecast.  It only summarizes
candidate errors that were already evaluated by the SES and RLS engines and a
fast contribution screen of the drivers already present in the RLS design.

No alternative model family is introduced here.  In particular, the driver
screen does not invent variables and does not promote/remove drivers
implicitly. Phase 4 also calibrates only the strength of the already-issued
RLS effect and audits calendar recurrence. Phase 5 compares the existing
Section-vs-Store parent effect in the context of each SKU+Tienda leaf, reusing
the exact same SES level and alpha. All phases remain diagnostic evidence for
the next explicit decision.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

import settings
from app.discrepancy import DiscrepancyRule

OPTIMIZATION_DIRNAME = "statistical_optimization"


def _empty() -> pl.DataFrame:
    return pl.DataFrame()


def _safe_div(num: str, den: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(den) > 0)
        .then(pl.col(num) / pl.col(den))
        .otherwise(None)
        .alias(alias)
    )


def summarize_ses(candidates: pl.DataFrame) -> pl.DataFrame:
    if candidates.height == 0:
        return _empty()
    select_cols = [
        "section", "target", "period_type", "block", "alpha",
        "sum_abs_error", "sum_abs_y", "sum_signed_error", "wmape", "bias",
    ]
    if "productive_candidate" in candidates.columns:
        select_cols.append("productive_candidate")
    fold = (
        candidates.select(select_cols)
        .with_columns(
            pl.col("wmape")
            .min()
            .over(["section", "target", "period_type", "block"])
            .alias("_best_fold_wmape")
        )
        .with_columns(
            (pl.col("wmape") <= pl.col("_best_fold_wmape") + 1e-15)
            .alias("_fold_win")
        )
    )
    group_keys = ["section", "target", "period_type", "alpha"]
    if "productive_candidate" in fold.columns:
        group_keys.append("productive_candidate")
    return (
        fold.group_by(group_keys)
        .agg(
            pl.col("sum_abs_error").sum(),
            pl.col("sum_abs_y").sum(),
            pl.col("sum_signed_error").sum(),
            pl.col("block").n_unique().alias("n_folds"),
            pl.col("wmape").median().alias("median_fold_wmape"),
            pl.col("wmape").max().alias("worst_fold_wmape"),
            pl.col("_fold_win").sum().alias("fold_wins"),
        )
        .with_columns(
            _safe_div("sum_abs_error", "sum_abs_y", "wmape"),
            _safe_div("sum_signed_error", "sum_abs_y", "bias"),
        )
        .sort(["section", "target", "period_type", "wmape", "alpha"])
    )


def summarize_rls(candidates: pl.DataFrame) -> pl.DataFrame:
    if candidates.height == 0:
        return _empty()
    group_keys = ["section", "node_level", "target", "period_type", "dynamics", "lambda"]
    if "productive_candidate" in candidates.columns:
        group_keys.append("productive_candidate")
    return (
        candidates.group_by(group_keys)
        .agg(
            pl.col("sum_abs_error").sum(),
            pl.col("sum_abs_y").sum(),
            pl.col("sum_signed_error").sum(),
            pl.col("n_points").sum(),
            pl.concat_str([pl.col("unique_id"), pl.col("block").cast(pl.Utf8)], separator="|").n_unique().alias("n_folds"),
            pl.col("wmape").median().alias("median_fold_wmape"),
            pl.col("wmape").max().alias("worst_fold_wmape"),
            pl.col("selected_by_prior").sum().alias("selected_blocks"),
        )
        .with_columns(
            _safe_div("sum_abs_error", "sum_abs_y", "wmape"),
            _safe_div("sum_signed_error", "sum_abs_y", "bias"),
        )
        .sort(
            ["section", "node_level", "target", "period_type", "wmape", "dynamics", "lambda"]
        )
    )


def summarize_rls_by_node(candidates: pl.DataFrame) -> pl.DataFrame:
    if candidates.height == 0:
        return _empty()
    group_keys = [
        "section", "node_level", "unique_id", "target",
        "period_type", "dynamics", "lambda",
    ]
    if "productive_candidate" in candidates.columns:
        group_keys.append("productive_candidate")
    return (
        candidates.group_by(group_keys)
        .agg(
            pl.col("sum_abs_error").sum(),
            pl.col("sum_abs_y").sum(),
            pl.col("sum_signed_error").sum(),
            pl.col("n_points").sum(),
            pl.col("block").n_unique().alias("n_folds"),
            pl.col("wmape").median().alias("median_fold_wmape"),
            pl.col("wmape").max().alias("worst_fold_wmape"),
            pl.col("selected_by_prior").sum().alias("selected_blocks"),
        )
        .with_columns(
            _safe_div("sum_abs_error", "sum_abs_y", "wmape"),
            _safe_div("sum_signed_error", "sum_abs_y", "bias"),
        )
        .sort(
            [
                "section", "node_level", "unique_id", "target",
                "period_type", "wmape", "dynamics", "lambda",
            ]
        )
    )


def recommend_ses(summary: pl.DataFrame) -> pl.DataFrame:
    if summary.height == 0:
        return _empty()
    keys = ["section", "target"]
    hist_cols = [*keys, "alpha"]
    if "productive_candidate" in summary.columns:
        hist_cols.append("productive_candidate")
    hist = (
        summary.filter(pl.col("period_type") == "in_sample")
        .sort(keys + ["wmape", "worst_fold_wmape", "alpha"])
        .unique(subset=keys, keep="first", maintain_order=True)
        .select(
            *hist_cols,
            pl.col("wmape").alias("history_wmape"),
            pl.col("bias").alias("history_bias"),
            pl.col("median_fold_wmape").alias("history_median_fold_wmape"),
            pl.col("worst_fold_wmape").alias("history_worst_fold_wmape"),
            "n_folds",
        )
    )
    if hist.height == 0:
        return hist
    hold = summary.filter(pl.col("period_type") == "out_sample").select(
        *keys,
        "alpha",
        pl.col("wmape").alias("holdout_wmape"),
        pl.col("bias").alias("holdout_bias"),
    )
    return hist.join(hold, on=keys + ["alpha"], how="left")


def recommend_rls(node_summary: pl.DataFrame) -> pl.DataFrame:
    if node_summary.height == 0:
        return _empty()
    keys = ["section", "node_level", "unique_id", "target"]
    hist_cols = [*keys, "dynamics", "lambda"]
    if "productive_candidate" in node_summary.columns:
        hist_cols.append("productive_candidate")
    hist = (
        node_summary.filter(pl.col("period_type") == "in_sample")
        .sort(keys + ["wmape", "worst_fold_wmape", "dynamics", "lambda"])
        .unique(subset=keys, keep="first", maintain_order=True)
        .select(
            *hist_cols,
            pl.col("wmape").alias("history_wmape"),
            pl.col("bias").alias("history_bias"),
            pl.col("median_fold_wmape").alias("history_median_fold_wmape"),
            pl.col("worst_fold_wmape").alias("history_worst_fold_wmape"),
            "n_folds",
        )
    )
    if hist.height == 0:
        return hist
    hold = node_summary.filter(pl.col("period_type") == "out_sample").select(
        *keys,
        "dynamics",
        "lambda",
        pl.col("wmape").alias("holdout_wmape"),
        pl.col("bias").alias("holdout_bias"),
    )
    return hist.join(hold, on=keys + ["dynamics", "lambda"], how="left")


def summarize_driver_screen(screen: pl.DataFrame) -> pl.DataFrame:
    if screen.height == 0:
        return _empty()
    # Positive delta means that zeroing/removing that existing group worsened
    # wMAPE, therefore the group is helping the fitted RLS path.  This is a
    # contribution screen, not a refitted ablation test.
    out = (
        screen.group_by(
            ["section", "node_level", "target", "period_type", "driver_group"]
        )
        .agg(
            pl.col("sum_abs_y").sum().alias("sum_abs_y"),
            pl.col("n_points").sum().alias("n_points"),
            (
                (pl.col("wmape_delta_if_removed") * pl.col("sum_abs_y")).sum()
                / pl.col("sum_abs_y").sum()
            ).alias("weighted_wmape_delta_if_removed"),
            pl.col("wmape_delta_if_removed").median().alias("median_fold_delta"),
            pl.col("wmape_delta_if_removed").min().alias("worst_fold_delta"),
            (pl.col("wmape_delta_if_removed") > 0).mean().alias("positive_fold_share"),
            pl.col("mean_abs_log_contribution").mean().alias("mean_abs_log_contribution"),
            pl.concat_str([pl.col("unique_id"), pl.col("block").cast(pl.Utf8)], separator="|").n_unique().alias("n_folds"),
        )
        .with_columns(
            pl.when(
                (pl.col("weighted_wmape_delta_if_removed") >= 0.005)
                & (pl.col("positive_fold_share") >= 0.60)
            )
            .then(pl.lit("useful_signal"))
            .when(
                (pl.col("weighted_wmape_delta_if_removed") <= -0.005)
                & (pl.col("positive_fold_share") <= 0.40)
            )
            .then(pl.lit("candidate_for_review"))
            .otherwise(pl.lit("weak_or_unstable"))
            .alias("screen_class")
        )
        .sort(
            [
                "section",
                "node_level",
                "target",
                "period_type",
                "weighted_wmape_delta_if_removed",
            ],
            descending=[False, False, False, False, True],
        )
    )
    return out


def summarize_driver_refit(refit: pl.DataFrame) -> pl.DataFrame:
    """Aggregate exact same-family refit tests.

    Positive ``weighted_wmape_improvement`` means the challenger action
    (remove or add) improves the CURRENT RLS specification. No action is
    promoted automatically.
    """
    if refit.height == 0:
        return _empty()
    return (
        refit.group_by(
            [
                "section", "node_level", "unique_id", "target", "period_type",
                "driver_group", "test_action", "dynamics", "lambda",
            ]
        )
        .agg(
            pl.col("sum_abs_y").sum().alias("sum_abs_y"),
            pl.col("n_points").sum().alias("n_points"),
            (
                (pl.col("wmape_improvement") * pl.col("sum_abs_y")).sum()
                / pl.col("sum_abs_y").sum()
            ).alias("weighted_wmape_improvement"),
            pl.col("wmape_improvement").median().alias("median_fold_improvement"),
            pl.col("wmape_improvement").min().alias("worst_fold_improvement"),
            (pl.col("wmape_improvement") > 0).mean().alias("positive_fold_share"),
            (
                (pl.col("abs_bias_change") * pl.col("sum_abs_y")).sum()
                / pl.col("sum_abs_y").sum()
            ).alias("weighted_abs_bias_change"),
            pl.concat_str(
                [pl.col("unique_id"), pl.col("block").cast(pl.Utf8)],
                separator="|",
            ).n_unique().alias("n_folds"),
        )
        .with_columns(
            pl.when(
                (pl.col("weighted_wmape_improvement") >= 0.005)
                & (pl.col("positive_fold_share") >= 0.60)
                & (pl.col("weighted_abs_bias_change") <= 0.005)
            )
            .then(
                pl.when(pl.col("test_action") == "remove")
                .then(pl.lit("candidate_remove"))
                .otherwise(pl.lit("candidate_add"))
            )
            .when(
                (pl.col("weighted_wmape_improvement") <= -0.005)
                & (pl.col("positive_fold_share") <= 0.40)
            )
            .then(pl.lit("keep_current"))
            .otherwise(pl.lit("review"))
            .alias("decision_signal")
        )
        .sort(
            [
                "section", "node_level", "unique_id", "target", "period_type",
                "weighted_wmape_improvement",
            ],
            descending=[False, False, False, False, False, True],
        )
    )



def validate_driver_refit_candidates(summary: pl.DataFrame) -> pl.DataFrame:
    """Validate historical driver candidates on the untouched OOS holdout.

    Holdout never participates in candidate creation.  It can only validate or
    veto a change that was already selected from closed historical folds.
    """
    if summary.height == 0:
        return _empty()

    keys = [
        "section", "node_level", "unique_id", "target",
        "driver_group", "test_action", "dynamics", "lambda",
    ]
    hist = (
        summary.filter(
            (pl.col("period_type") == "in_sample")
            & pl.col("decision_signal").is_in(["candidate_add", "candidate_remove"])
        )
        .select(
            *keys,
            pl.col("decision_signal").alias("historical_signal"),
            pl.col("weighted_wmape_improvement").alias("history_wmape_improvement"),
            pl.col("positive_fold_share").alias("history_positive_fold_share"),
            pl.col("weighted_abs_bias_change").alias("history_abs_bias_change"),
            pl.col("n_folds").alias("history_n_folds"),
        )
    )
    if hist.height == 0:
        return hist

    hold = (
        summary.filter(pl.col("period_type") == "out_sample")
        .select(
            *keys,
            pl.col("weighted_wmape_improvement").alias("holdout_wmape_improvement"),
            pl.col("weighted_abs_bias_change").alias("holdout_abs_bias_change"),
            pl.col("positive_fold_share").alias("holdout_positive_fold_share"),
            pl.col("n_folds").alias("holdout_n_folds"),
        )
    )

    return (
        hist.join(hold, on=keys, how="left")
        .with_columns(
            pl.when(pl.col("holdout_wmape_improvement").is_null())
            .then(pl.lit("no_holdout"))
            .when(
                (pl.col("holdout_wmape_improvement") >= 0.0)
                & (pl.col("holdout_abs_bias_change") <= 0.005)
            )
            .then(pl.lit("validated_candidate"))
            .when(
                (pl.col("holdout_wmape_improvement") < 0.0)
                | (pl.col("holdout_abs_bias_change") > 0.010)
            )
            .then(pl.lit("holdout_veto"))
            .otherwise(pl.lit("mixed_holdout"))
            .alias("holdout_validation")
        )
        .sort(
            ["section", "node_level", "unique_id", "target", "history_wmape_improvement"],
            descending=[False, False, False, False, True],
        )
    )




def audit_promoted_exclusions(refit_summary: pl.DataFrame) -> pl.DataFrame:
    """Re-audit active node/target exclusions after parameter reselection.

    A promoted removal is tested by the existing Phase-2 ``add`` refit.  This
    table is governance only: it makes second-order interactions visible when
    an exclusion changes the subsequently selected lambda/dynamics.
    """
    if refit_summary.height == 0:
        return _empty()
    cfg = getattr(settings, "RLS_DRIVER_GROUP_EXCLUSIONS", {}) or {}
    promoted: list[dict] = []
    if isinstance(cfg, dict):
        for uid, targets in cfg.items():
            if not isinstance(targets, dict):
                continue
            for target, groups in targets.items():
                for group in groups:
                    promoted.append({
                        "unique_id": str(uid),
                        "target": str(target),
                        "driver_group": str(group),
                        "test_action": "add",
                    })
    if not promoted:
        return _empty()
    keys = ["unique_id", "target", "driver_group", "test_action"]
    wanted = pl.DataFrame(promoted).unique(subset=keys)
    base = refit_summary.join(wanted, on=keys, how="inner")
    if base.height == 0:
        return _empty()

    meta = ["section", "node_level", "unique_id", "target", "driver_group", "test_action", "dynamics", "lambda"]
    hist = base.filter(pl.col("period_type") == "in_sample").select(
        *meta,
        pl.col("weighted_wmape_improvement").alias("history_addback_improvement"),
        pl.col("positive_fold_share").alias("history_addback_positive_share"),
        pl.col("weighted_abs_bias_change").alias("history_addback_abs_bias_change"),
        pl.col("decision_signal").alias("history_decision_signal"),
        pl.col("n_folds").alias("history_n_folds"),
    )
    hold = base.filter(pl.col("period_type") == "out_sample").select(
        *keys,
        pl.col("weighted_wmape_improvement").alias("holdout_addback_improvement"),
        pl.col("weighted_abs_bias_change").alias("holdout_addback_abs_bias_change"),
        pl.col("n_folds").alias("holdout_n_folds"),
    )
    if hist.height == 0:
        return hist
    return (
        hist.join(hold, on=keys, how="left")
        .with_columns(
            pl.when(pl.col("holdout_addback_improvement").is_null())
            .then(pl.lit("no_holdout"))
            .when(
                (pl.col("history_decision_signal") == "candidate_add")
                & (pl.col("holdout_addback_improvement") >= 0.0)
                & (pl.col("holdout_addback_abs_bias_change") <= 0.005)
            )
            .then(pl.lit("reversal_validated"))
            .when(
                (pl.col("history_decision_signal") == "candidate_add")
                & (
                    (pl.col("holdout_addback_improvement") < 0.0)
                    | (pl.col("holdout_addback_abs_bias_change") > 0.010)
                )
            )
            .then(pl.lit("reversal_holdout_veto"))
            .when(
                (pl.col("history_addback_improvement") <= 0.0)
                & (pl.col("holdout_addback_improvement") <= 0.0)
            )
            .then(pl.lit("promotion_stable"))
            .when(
                (pl.col("history_addback_improvement") <= -0.005)
                & (pl.col("holdout_addback_improvement") > 0.005)
            )
            .then(pl.lit("holdout_regression_watch"))
            .otherwise(pl.lit("mixed_reaudit"))
            .alias("reaudit_status")
        )
        .sort(["section", "unique_id", "target", "driver_group"])
    )


def build_leaf_residual_diagnostics(
    forecast: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Diagnose extreme leaf residuals without changing the SES+RLS model.

    Memory contract: only OOS plus a recent closed-history slice are analysed,
    and Unidades/Valor are processed sequentially.  The model itself still uses
    the full available history; this bounded window is diagnostic only.

    The outlier thresholds mirror the dashboard rule, while materiality uses a
    causal model scale ``max(SES level at origin, initial robust level, 1)`` so
    no OOS actual is used to define its own threshold.
    """
    empty = {
        "leaf_residual_extremes": _empty(),
        "leaf_residual_summary": _empty(),
        "residual_signal_summary": _empty(),
    }
    if forecast.height == 0:
        return empty

    required = {
        "unique_id", "ds", "period_type", "rls_metric_eligible",
        "y", "yhat", "yhat_raw", "value", "valuehat", "valuehat_raw",
        "ses_level_y", "ses_level_value", "initial_level_y", "initial_level_value",
        "ses_alpha_y", "ses_alpha_value", "parent_model_y", "parent_model_value",
        "parent_wmape_y", "parent_wmape_value", "driver_factor_y", "driver_factor_value",
        "driver_effect", "driver_effect_value",
    }
    if not required.issubset(forecast.columns):
        return empty

    optional = [c for c in ("sku_desc", "store_name") if c in forecast.columns]
    leaf = (
        forecast.filter(
            (pl.col("unique_id").str.count_matches(r"\|\|") == 2)
            & pl.col("period_type").is_in(["in_sample", "out_sample"])
            & pl.col("rls_metric_eligible").fill_null(False)
        )
        .select([*sorted(required), *optional])
        .with_columns(
            pl.col("ds").cast(pl.Date),
            pl.col("unique_id").str.extract(r"^([^|]+)", 1).alias("section"),
            pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("store_id"),
            pl.col("unique_id").str.extract(r"\|\|S:(.+)$", 1).alias("sku_id"),
        )
    )
    if leaf.height == 0:
        return empty

    # Residual pattern diagnosis does not need the full multi-year in-sample
    # panel.  Keep OOS intact and only a recent CLOSED history window.  This is
    # diagnostic-only and therefore cannot change the long-history SES/RLS state.
    history_days = max(int(getattr(settings, "STAT_OPT_RESIDUAL_HISTORY_DAYS", 168)), 28)
    hist_end = (
        leaf.filter(pl.col("period_type") == "in_sample")
        .select(pl.col("ds").max())
        .item()
    )
    if hist_end is not None:
        cutoff = hist_end - dt.timedelta(days=history_days - 1)
        leaf = leaf.filter(
            (pl.col("period_type") == "out_sample")
            | ((pl.col("period_type") == "in_sample") & (pl.col("ds") >= pl.lit(cutoff)))
        )

    rule = DiscrepancyRule()
    max_rows = max(int(getattr(settings, "STAT_OPT_MAX_RESIDUAL_EXTREMES", 50_000)), 1)
    extremes_parts: list[pl.DataFrame] = []
    summary_parts: list[pl.DataFrame] = []
    signal_parts: list[pl.DataFrame] = []

    def _target_frame(target: str) -> pl.DataFrame:
        if target == "Unidades":
            actual, fcst, raw = "y", "yhat", "yhat_raw"
            level, initial, alpha = "ses_level_y", "initial_level_y", "ses_alpha_y"
            parent, pwmape = "parent_model_y", "parent_wmape_y"
            factor, effect = "driver_factor_y", "driver_effect"
        else:
            actual, fcst, raw = "value", "valuehat", "valuehat_raw"
            level, initial, alpha = "ses_level_value", "initial_level_value", "ses_alpha_value"
            parent, pwmape = "parent_model_value", "parent_wmape_value"
            factor, effect = "driver_factor_value", "driver_effect_value"

        base_cols = [
            "section", "store_id", "sku_id", "unique_id", "ds", "period_type", "y",
            *[c for c in optional if c in leaf.columns],
        ]
        return leaf.select(
            *base_cols,
            pl.lit(target).alias("target"),
            pl.col(actual).cast(pl.Float64).alias("actual"),
            pl.col(fcst).cast(pl.Float64).alias("forecast"),
            pl.col(raw).cast(pl.Float64).alias("forecast_raw"),
            pl.col(level).cast(pl.Float64).fill_null(0.0).alias("ses_level"),
            pl.col(initial).cast(pl.Float64).fill_null(0.0).alias("initial_level"),
            pl.col(alpha).cast(pl.Float64).alias("ses_alpha"),
            pl.col(parent).cast(pl.Utf8).alias("parent_model"),
            pl.col(pwmape).cast(pl.Float64).alias("parent_wmape"),
            pl.col(factor).cast(pl.Float64).fill_null(1.0).alias("driver_factor"),
            pl.col(effect).cast(pl.Float64).fill_null(0.0).alias("driver_effect"),
        ).filter(
            pl.col("actual").is_finite() & pl.col("forecast").is_finite()
        )

    def _score_target(target: str) -> None:
        scored = _target_frame(target)
        if scored.height == 0:
            return
        scored = scored.with_columns(
            pl.col("actual").clip(lower_bound=0.0).alias("actual_nonnegative"),
            pl.col("forecast").clip(lower_bound=0.0).alias("forecast_nonnegative"),
            pl.max_horizontal(
                pl.col("ses_level").abs(),
                pl.col("initial_level").abs(),
                pl.lit(1.0),
            ).alias("causal_scale"),
        ).with_columns(
            (pl.col("causal_scale") * float(rule.floor_share)).clip(lower_bound=1e-9).alias("ratio_floor"),
            (pl.col("forecast") - pl.col("actual")).alias("signed_error"),
            (pl.col("forecast") - pl.col("actual")).abs().alias("abs_error"),
            (pl.col("actual") - pl.col("ses_level")).abs().alias("ses_only_abs_error"),
            (pl.col("y").is_finite() & (pl.col("y") > 0.0)).alias("metric_eligible"),
            (pl.col("y").is_null() | (pl.col("y") <= 0.0)).alias("zero_sale_day"),
            pl.col("ds").dt.weekday().cast(pl.Int8).alias("weekday"),
            pl.col("ds").dt.month().cast(pl.Int8).alias("month"),
        ).with_columns(
            (
                (pl.max_horizontal("actual_nonnegative", "forecast_nonnegative") + pl.col("ratio_floor"))
                / (pl.min_horizontal("actual_nonnegative", "forecast_nonnegative") + pl.col("ratio_floor"))
            ).alias("factor_gap"),
            (pl.col("abs_error") / pl.col("causal_scale")).alias("scaled_gap"),
            (
                (pl.col("actual_nonnegative") + 1.0).log()
                - (pl.col("ses_level").clip(lower_bound=0.0) + 1.0).log()
            ).alias("required_driver_effect"),
            pl.when(pl.col("forecast") < pl.col("actual"))
            .then(pl.lit("underforecast"))
            .when(pl.col("forecast") > pl.col("actual"))
            .then(pl.lit("overforecast"))
            .otherwise(pl.lit("match"))
            .alias("error_direction"),
        ).with_columns(
            (pl.col("required_driver_effect") - pl.col("driver_effect")).alias("unexplained_driver_effect"),
            pl.col("required_driver_effect").clip(-20.0, 20.0).exp().alias("required_driver_factor"),
        ).with_columns(
            pl.col("unexplained_driver_effect").clip(-20.0, 20.0).exp().alias("additional_driver_factor"),
            (pl.col("ses_only_abs_error") - pl.col("abs_error")).alias("driver_error_improvement"),
            (
                ((pl.col("factor_gap") >= float(rule.factor_threshold))
                 & (pl.col("scaled_gap") >= float(rule.material_scaled_gap)))
                | (pl.col("scaled_gap") >= float(rule.severe_scaled_gap))
            ).alias("is_extreme"),
            pl.when(pl.col("metric_eligible")).then(pl.col("abs_error")).otherwise(0.0).alias("official_abs_error"),
            pl.when(pl.col("metric_eligible")).then(pl.col("actual").abs()).otherwise(0.0).alias("official_abs_y"),
        )

        summary = (
            scored.group_by(["section", "store_id", "target", "period_type"])
            .agg(
                pl.len().alias("n_points"),
                pl.col("metric_eligible").sum().alias("n_metric_points"),
                pl.col("zero_sale_day").sum().alias("n_zero_sale_points"),
                pl.col("is_extreme").sum().alias("n_extreme"),
                (pl.col("is_extreme") & pl.col("zero_sale_day")).sum().alias("n_zero_sale_extreme"),
                (pl.col("is_extreme") & pl.col("metric_eligible")).sum().alias("n_metric_extreme"),
                pl.col("abs_error").sum().alias("sum_abs_error_all"),
                pl.col("actual").abs().sum().alias("sum_abs_actual_all"),
                pl.when(pl.col("zero_sale_day")).then(pl.col("abs_error")).otherwise(0.0).sum().alias("zero_sale_abs_error"),
                pl.col("official_abs_error").sum().alias("sum_abs_error_official"),
                pl.col("official_abs_y").sum().alias("sum_abs_y_official"),
                pl.when(pl.col("is_extreme")).then(pl.col("abs_error")).otherwise(0.0).sum().alias("extreme_abs_error_all"),
                pl.when(pl.col("is_extreme") & pl.col("metric_eligible"))
                .then(pl.col("abs_error")).otherwise(0.0).sum().alias("extreme_abs_error_official"),
                pl.col("scaled_gap").filter(pl.col("is_extreme")).median().alias("median_scaled_gap_extreme"),
                pl.col("factor_gap").filter(pl.col("is_extreme")).median().alias("median_factor_gap_extreme"),
                pl.col("unexplained_driver_effect").abs().filter(pl.col("is_extreme")).median().alias("median_abs_unexplained_effect"),
                pl.col("unexplained_driver_effect").abs().filter(pl.col("is_extreme") & pl.col("metric_eligible")).median().alias("median_abs_unexplained_effect_metric"),
                pl.col("additional_driver_factor").filter(pl.col("is_extreme")).median().alias("median_additional_driver_factor"),
                pl.col("driver_error_improvement").filter(pl.col("is_extreme")).median().alias("median_driver_error_improvement_extreme"),
                (pl.col("driver_error_improvement") > 0.0).filter(pl.col("is_extreme")).mean().alias("driver_help_extreme_share"),
                (pl.col("driver_error_improvement") > 0.0).filter(pl.col("is_extreme") & pl.col("metric_eligible")).mean().alias("driver_help_metric_extreme_share"),
                (pl.col("error_direction") == "underforecast").filter(pl.col("is_extreme")).mean().alias("underforecast_extreme_share"),
                (pl.col("error_direction") == "underforecast").filter(pl.col("is_extreme") & pl.col("metric_eligible")).mean().alias("underforecast_metric_extreme_share"),
            )
            .with_columns(
                pl.when(pl.col("sum_abs_actual_all") > 0)
                .then(pl.col("sum_abs_error_all") / pl.col("sum_abs_actual_all"))
                .otherwise(None)
                .alias("wmape_all_points"),
                pl.when(pl.col("sum_abs_y_official") > 0)
                .then(pl.col("sum_abs_error_official") / pl.col("sum_abs_y_official"))
                .otherwise(None)
                .alias("wmape_official"),
                pl.when(pl.col("sum_abs_error_all") > 0)
                .then(pl.col("zero_sale_abs_error") / pl.col("sum_abs_error_all"))
                .otherwise(0.0)
                .alias("zero_sale_error_share"),
                pl.when(pl.col("n_extreme") > 0)
                .then(pl.col("n_zero_sale_extreme") / pl.col("n_extreme"))
                .otherwise(0.0)
                .alias("zero_sale_extreme_share"),
                pl.when(pl.col("sum_abs_error_official") > 0)
                .then(pl.col("extreme_abs_error_official") / pl.col("sum_abs_error_official"))
                .otherwise(0.0)
                .alias("extreme_error_share"),
                pl.when(pl.col("n_points") > 0)
                .then(pl.col("n_extreme") / pl.col("n_points"))
                .otherwise(0.0)
                .alias("extreme_point_share"),
            )
            .with_columns(
                pl.when(
                    (pl.col("n_metric_extreme") >= 3)
                    & (pl.col("extreme_error_share") >= 0.50)
                    & (pl.col("median_abs_unexplained_effect_metric") >= 0.6931471805599453)
                )
                .then(pl.lit("strong_driver_review"))
                .when(
                    (pl.col("n_metric_extreme") >= 2)
                    & (pl.col("extreme_error_share") >= 0.25)
                    & (pl.col("median_abs_unexplained_effect_metric") >= 0.4054651081081644)
                )
                .then(pl.lit("moderate_driver_review"))
                .otherwise(pl.lit("low"))
                .alias("driver_review_signal"),
                pl.when(
                    (pl.col("n_zero_sale_extreme") >= 3)
                    & (pl.col("zero_sale_error_share") >= 0.50)
                )
                .then(pl.lit("strong_zero_sale_review"))
                .when(
                    (pl.col("n_zero_sale_extreme") >= 2)
                    & (pl.col("zero_sale_error_share") >= 0.25)
                )
                .then(pl.lit("moderate_zero_sale_review"))
                .otherwise(pl.lit("low"))
                .alias("zero_sale_review_signal")
            )
        )
        summary_parts.append(summary)

        extremes_all = scored.filter(pl.col("is_extreme"))
        if extremes_all.height == 0:
            return

        def _signal(col_name: str, dimension: str) -> pl.DataFrame:
            return (
                extremes_all.group_by(["section", "target", "period_type", col_name])
                .agg(
                    pl.len().alias("n_extreme"),
                    pl.col("unique_id").n_unique().alias("n_leaves"),
                    pl.col("official_abs_error").sum().alias("extreme_abs_error_official"),
                    pl.col("signed_error").sum().alias("signed_error"),
                    pl.col("scaled_gap").median().alias("median_scaled_gap"),
                    pl.col("unexplained_driver_effect").abs().median().alias("median_abs_unexplained_effect"),
                    pl.col("zero_sale_day").mean().alias("zero_sale_extreme_share"),
                    (pl.col("driver_error_improvement") > 0.0).mean().alias("driver_help_extreme_share"),
                    (pl.col("error_direction") == "underforecast").mean().alias("underforecast_share"),
                    (pl.col("error_direction") == "underforecast").filter(pl.col("metric_eligible")).mean().alias("underforecast_metric_share"),
                )
                .with_columns(
                    pl.lit(dimension).alias("dimension"),
                    pl.col(col_name).cast(pl.Utf8).alias("dimension_value"),
                )
                .drop(col_name)
                .with_columns(
                    pl.when(
                        pl.col("extreme_abs_error_official").sum().over(["section", "target", "period_type"]) > 0
                    )
                    .then(
                        pl.col("extreme_abs_error_official")
                        / pl.col("extreme_abs_error_official").sum().over(["section", "target", "period_type"])
                    )
                    .otherwise(0.0)
                    .alias("extreme_error_share_within_target")
                )
            )

        signal_parts.extend([
            _signal("store_id", "store"),
            _signal("ds", "date"),
            _signal("weekday", "weekday"),
            _signal("month", "month"),
            _signal("parent_model", "parent_model"),
        ])
        # Keep OOS detail first; summaries above are exact over all rows.
        detail = (
            extremes_all.with_columns(
                pl.when(pl.col("period_type") == "out_sample").then(0).otherwise(1).alias("_period_rank")
            )
            .sort(["_period_rank", "official_abs_error", "scaled_gap"], descending=[False, True, True])
            .head(max_rows)
            .drop("_period_rank")
        )
        extremes_parts.append(detail)

    # Process targets sequentially to avoid a 2× duplication of the leaf panel.
    _score_target("Unidades")
    _score_target("Valor ($)")

    summary = pl.concat(summary_parts, how="diagonal_relaxed") if summary_parts else _empty()
    if summary.height:
        summary = summary.sort(
            ["period_type", "extreme_abs_error_official", "wmape_official"],
            descending=[True, True, True],
        )

    signals = pl.concat(signal_parts, how="diagonal_relaxed") if signal_parts else _empty()
    if signals.height:
        signals = signals.sort(
            ["period_type", "extreme_abs_error_official"],
            descending=[True, True],
        )

    extremes = pl.concat(extremes_parts, how="diagonal_relaxed") if extremes_parts else _empty()
    if extremes.height:
        extremes = (
            extremes.with_columns(
                pl.when(pl.col("period_type") == "out_sample").then(0).otherwise(1).alias("_period_rank")
            )
            .sort(["_period_rank", "official_abs_error", "scaled_gap"], descending=[False, True, True])
            .head(max_rows)
            .drop("_period_rank")
        )

    return {
        "leaf_residual_extremes": extremes,
        "leaf_residual_summary": summary,
        "residual_signal_summary": signals,
    }



def build_leaf_driver_strength_diagnostics(
    forecast: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Tune only the strength of the EXISTING parent RLS effect at leaf level.

    ``gamma=1`` is the productive v13 identity.  Other gamma values do not
    create drivers or models: they only rescale the already-issued causal RLS
    non-intercept effect in

        log1p(leaf forecast) = log1p(SES level) + gamma * RLS effect.

    Selection evidence comes from closed in-sample blocks.  OOS is joined only
    afterwards as a holdout validation and can never create a candidate.
    """
    empty = {
        "driver_strength_candidates": _empty(),
        "driver_strength_summary": _empty(),
        "driver_strength_recommendation": _empty(),
    }
    if forecast.height == 0:
        return empty

    required = {
        "unique_id", "period_type", "rls_metric_eligible", "rls_block",
        "y", "value", "ses_level_y", "ses_level_value",
        "parent_model_y", "parent_model_value", "driver_effect", "driver_effect_value",
    }
    if not required.issubset(forecast.columns):
        return empty

    gammas = tuple(sorted({
        float(x) for x in getattr(
            settings,
            "STAT_OPT_DRIVER_STRENGTH_CANDIDATES",
            (0.0, 0.25, 0.50, 0.75, 0.90, 1.0, 1.10, 1.25, 1.50),
        )
        if 0.0 <= float(x) <= 2.0
    } | {1.0}))

    base = (
        forecast.filter(
            (pl.col("unique_id").str.count_matches(r"\|\|") == 2)
            & pl.col("period_type").is_in(["in_sample", "out_sample"])
            & pl.col("rls_metric_eligible").fill_null(False)
        )
        .select(
            "unique_id", "period_type", "rls_metric_eligible", "rls_block",
            "y", "value", "ses_level_y", "ses_level_value",
            "parent_model_y", "parent_model_value", "driver_effect", "driver_effect_value",
        )
        .with_columns(
            pl.col("unique_id").str.extract(r"^([^|]+)", 1).alias("section"),
            pl.col("rls_block").cast(pl.Int32),
        )
    )
    if base.height == 0:
        return empty

    fold_parts: list[pl.DataFrame] = []
    for target in ("Unidades", "Valor ($)"):
        if target == "Unidades":
            actual, level, effect, parent = "y", "ses_level_y", "driver_effect", "parent_model_y"
            decimals = 0
        else:
            actual, level, effect, parent = "value", "ses_level_value", "driver_effect_value", "parent_model_value"
            decimals = 2

        rows = (
            base.select(
                "section", "period_type", "rls_block", "y",
                pl.col(actual).cast(pl.Float64).alias("actual"),
                pl.col(level).cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0).alias("ses_level"),
                pl.col(effect).cast(pl.Float64).fill_null(0.0).alias("driver_effect"),
                pl.col(parent).cast(pl.Utf8).fill_null("unknown").alias("parent_model"),
            )
            .filter(
                pl.col("y").is_finite()
                & (pl.col("y") > 0.0)
                & pl.col("actual").is_finite()
                & pl.col("ses_level").is_finite()
                & pl.col("driver_effect").is_finite()
            )
        )
        if rows.height == 0:
            continue

        for gamma in gammas:
            pred_raw = (
                (
                    (pl.col("ses_level") + 1.0).log()
                    + pl.lit(float(gamma)) * pl.col("driver_effect")
                ).clip(-50.0, 50.0).exp()
                - 1.0
            ).clip(lower_bound=0.0)
            scored = rows.with_columns(
                pl.lit(target).alias("target"),
                pl.lit(float(gamma)).alias("gamma"),
                pred_raw.round(decimals).alias("candidate_forecast"),
            ).with_columns(
                (pl.col("candidate_forecast") - pl.col("actual")).abs().alias("abs_error"),
                (pl.col("candidate_forecast") - pl.col("actual")).alias("signed_error"),
                pl.col("actual").abs().alias("abs_actual"),
            )
            fold_parts.append(
                scored.group_by(
                    ["section", "target", "parent_model", "period_type", "rls_block", "gamma"]
                ).agg(
                    pl.col("abs_error").sum().alias("sum_abs_error"),
                    pl.col("abs_actual").sum().alias("sum_abs_y"),
                    pl.col("signed_error").sum().alias("sum_signed_error"),
                    pl.len().alias("n_points"),
                ).with_columns(
                    _safe_div("sum_abs_error", "sum_abs_y", "wmape"),
                    _safe_div("sum_signed_error", "sum_abs_y", "bias"),
                )
            )

    folds = pl.concat(fold_parts, how="diagonal_relaxed") if fold_parts else _empty()
    if folds.height == 0:
        return empty

    baseline_fold = folds.filter(pl.col("gamma") == 1.0).select(
        "section", "target", "parent_model", "period_type", "rls_block",
        pl.col("wmape").alias("baseline_fold_wmape"),
    )
    folds = folds.join(
        baseline_fold,
        on=["section", "target", "parent_model", "period_type", "rls_block"],
        how="left",
    ).with_columns(
        (pl.col("baseline_fold_wmape") - pl.col("wmape")).alias("fold_wmape_improvement"),
        (pl.col("wmape") <= pl.col("baseline_fold_wmape") + 1e-15).alias("beats_gamma_1"),
    ).sort(["section", "target", "parent_model", "period_type", "rls_block", "gamma"])

    summary = (
        folds.group_by(["section", "target", "parent_model", "period_type", "gamma"])
        .agg(
            pl.col("sum_abs_error").sum(),
            pl.col("sum_abs_y").sum(),
            pl.col("sum_signed_error").sum(),
            pl.col("n_points").sum(),
            pl.col("rls_block").n_unique().alias("n_folds"),
            pl.col("wmape").median().alias("median_fold_wmape"),
            pl.col("wmape").max().alias("worst_fold_wmape"),
            pl.col("beats_gamma_1").mean().alias("positive_fold_share_vs_gamma_1"),
            pl.col("fold_wmape_improvement").median().alias("median_fold_improvement_vs_gamma_1"),
        )
        .with_columns(
            _safe_div("sum_abs_error", "sum_abs_y", "wmape"),
            _safe_div("sum_signed_error", "sum_abs_y", "bias"),
        )
        .sort(["section", "target", "parent_model", "period_type", "wmape", "gamma"])
    )

    keys = ["section", "target", "parent_model"]
    hist = (
        summary.filter(pl.col("period_type") == "in_sample")
        .with_columns((pl.col("gamma") - 1.0).abs().alias("_gamma_distance"))
        .sort(keys + ["wmape", "worst_fold_wmape", "_gamma_distance", "gamma"])
        .unique(subset=keys, keep="first", maintain_order=True)
        .select(
            *keys, "gamma",
            pl.col("wmape").alias("history_wmape"),
            pl.col("bias").alias("history_bias"),
            pl.col("worst_fold_wmape").alias("history_worst_fold_wmape"),
            pl.col("positive_fold_share_vs_gamma_1").alias("history_positive_fold_share"),
            "n_folds",
        )
    )
    if hist.height == 0:
        return {
            "driver_strength_candidates": folds,
            "driver_strength_summary": summary,
            "driver_strength_recommendation": _empty(),
        }

    hist_base = summary.filter(
        (pl.col("period_type") == "in_sample") & (pl.col("gamma") == 1.0)
    ).select(
        *keys,
        pl.col("wmape").alias("baseline_history_wmape"),
        pl.col("bias").alias("baseline_history_bias"),
    )
    hold = summary.filter(pl.col("period_type") == "out_sample").select(
        *keys, "gamma",
        pl.col("wmape").alias("holdout_wmape"),
        pl.col("bias").alias("holdout_bias"),
    )
    hold_base = summary.filter(
        (pl.col("period_type") == "out_sample") & (pl.col("gamma") == 1.0)
    ).select(
        *keys,
        pl.col("wmape").alias("baseline_holdout_wmape"),
        pl.col("bias").alias("baseline_holdout_bias"),
    )

    recommendation = (
        hist.join(hist_base, on=keys, how="left")
        .join(hold, on=keys + ["gamma"], how="left")
        .join(hold_base, on=keys, how="left")
        .with_columns(
            (pl.col("baseline_history_wmape") - pl.col("history_wmape")).alias("history_wmape_improvement"),
            (pl.col("baseline_holdout_wmape") - pl.col("holdout_wmape")).alias("holdout_wmape_improvement"),
            (pl.col("holdout_bias").abs() - pl.col("baseline_holdout_bias").abs()).alias("holdout_abs_bias_change"),
        )
        .with_columns(
            (
                ((pl.col("gamma") - 1.0).abs() > 1e-12)
                & (pl.col("n_folds") >= int(getattr(settings, "STAT_OPT_DRIVER_STRENGTH_MIN_FOLDS", 4)))
                & (pl.col("history_wmape_improvement") >= 0.005)
                & (pl.col("history_positive_fold_share") >= 0.60)
            ).alias("history_candidate")
        )
        .with_columns(
            pl.when(~pl.col("history_candidate"))
            .then(pl.lit("keep_gamma_1"))
            .when(pl.col("holdout_wmape").is_null())
            .then(pl.lit("history_candidate_no_holdout"))
            .when(
                (pl.col("holdout_wmape_improvement") >= 0.0)
                & (pl.col("holdout_abs_bias_change") <= 0.005)
            )
            .then(pl.lit("validated_calibration"))
            .when(
                (pl.col("holdout_wmape_improvement") < 0.0)
                | (pl.col("holdout_abs_bias_change") > 0.010)
            )
            .then(pl.lit("holdout_veto"))
            .otherwise(pl.lit("mixed_holdout"))
            .alias("validation_status")
        )
        .sort(keys)
    )
    return {
        "driver_strength_candidates": folds,
        "driver_strength_summary": summary,
        "driver_strength_recommendation": recommendation,
    }


def build_calendar_day_recurrence_diagnostics(
    forecast: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Compare every OOS calendar day with the same month/day in closed history.

    This is a generic recurrence audit, not a new calendar driver.  It is used
    to decide whether concentrated OOS residuals (for example around year-end)
    are recurrent historical patterns or a one-off regime/data issue.
    """
    empty = {
        "calendar_day_yearly": _empty(),
        "calendar_day_recurrence": _empty(),
    }
    if forecast.height == 0:
        return empty
    required = {
        "unique_id", "ds", "period_type", "rls_metric_eligible", "y", "yhat", "value", "valuehat"
    }
    if not required.issubset(forecast.columns):
        return empty

    base = (
        forecast.filter(
            (pl.col("unique_id").str.count_matches(r"\|\|") == 2)
            & pl.col("period_type").is_in(["in_sample", "out_sample"])
            & pl.col("rls_metric_eligible").fill_null(False)
        )
        .select(
            "unique_id", "ds", "period_type", "rls_metric_eligible",
            "y", "yhat", "value", "valuehat",
        )
        .with_columns(
            pl.col("ds").cast(pl.Date),
            pl.col("unique_id").str.extract(r"^([^|]+)", 1).alias("section"),
        )
    )
    if base.height == 0:
        return empty

    yearly_parts: list[pl.DataFrame] = []
    for target in ("Unidades", "Valor ($)"):
        actual, fcst = ("y", "yhat") if target == "Unidades" else ("value", "valuehat")
        rows = (
            base.select(
                "section", "unique_id", "ds", "period_type", "y",
                pl.col(actual).cast(pl.Float64).alias("actual"),
                pl.col(fcst).cast(pl.Float64).alias("forecast"),
            )
            .filter(pl.col("actual").is_finite() & pl.col("forecast").is_finite())
            .with_columns(
                pl.lit(target).alias("target"),
                pl.col("ds").dt.year().cast(pl.Int16).alias("year"),
                pl.col("ds").dt.strftime("%m-%d").alias("month_day"),
                (pl.col("forecast") - pl.col("actual")).abs().alias("abs_error"),
                (pl.col("forecast") - pl.col("actual")).alias("signed_error"),
                pl.col("actual").abs().alias("abs_actual"),
                (pl.col("y").is_finite() & (pl.col("y") > 0.0)).alias("metric_eligible"),
                (pl.col("y").is_null() | (pl.col("y") <= 0.0)).alias("zero_sale_day"),
            )
            .with_columns(
                pl.when(pl.col("metric_eligible")).then(pl.col("abs_error")).otherwise(0.0).alias("official_abs_error"),
                pl.when(pl.col("metric_eligible")).then(pl.col("abs_actual")).otherwise(0.0).alias("official_abs_y"),
                pl.when(pl.col("metric_eligible")).then(pl.col("signed_error")).otherwise(0.0).alias("official_signed_error"),
                pl.when(pl.col("zero_sale_day")).then(pl.col("abs_error")).otherwise(0.0).alias("zero_sale_abs_error"),
            )
        )
        yearly_parts.append(
            rows.group_by(["section", "target", "period_type", "year", "month_day"])
            .agg(
                pl.len().alias("n_points"),
                pl.col("unique_id").n_unique().alias("n_leaves"),
                pl.col("metric_eligible").sum().alias("n_metric_points"),
                pl.col("zero_sale_day").sum().alias("n_zero_sale_points"),
                pl.col("abs_error").sum().alias("sum_abs_error_all"),
                pl.col("abs_actual").sum().alias("sum_abs_actual_all"),
                pl.col("zero_sale_abs_error").sum().alias("zero_sale_abs_error"),
                pl.col("official_abs_error").sum().alias("sum_abs_error_official"),
                pl.col("official_abs_y").sum().alias("sum_abs_y_official"),
                pl.col("official_signed_error").sum().alias("sum_signed_error_official"),
                (pl.col("forecast") < pl.col("actual")).mean().alias("underforecast_share_all"),
            )
            .with_columns(
                _safe_div("sum_abs_error_all", "sum_abs_actual_all", "wmape_all_points"),
                _safe_div("sum_abs_error_official", "sum_abs_y_official", "wmape_official"),
                _safe_div("sum_signed_error_official", "sum_abs_y_official", "bias_official"),
                pl.when(pl.col("sum_abs_error_all") > 0)
                .then(pl.col("zero_sale_abs_error") / pl.col("sum_abs_error_all"))
                .otherwise(0.0)
                .alias("zero_sale_error_share"),
            )
        )

    yearly = pl.concat(yearly_parts, how="diagonal_relaxed") if yearly_parts else _empty()
    if yearly.height == 0:
        return empty

    hist = (
        yearly.filter(pl.col("period_type") == "in_sample")
        .group_by(["section", "target", "month_day"])
        .agg(
            pl.col("year").n_unique().alias("history_years"),
            pl.col("wmape_official").median().alias("history_median_wmape_official"),
            pl.col("wmape_all_points").median().alias("history_median_wmape_all_points"),
            pl.col("zero_sale_error_share").median().alias("history_median_zero_sale_error_share"),
            pl.col("underforecast_share_all").median().alias("history_median_underforecast_share"),
        )
    )
    hold = (
        yearly.filter(pl.col("period_type") == "out_sample")
        .group_by(["section", "target", "month_day"])
        .agg(
            pl.col("year").max().alias("holdout_year"),
            pl.col("sum_abs_error_all").sum().alias("holdout_abs_error_all"),
            pl.col("sum_abs_actual_all").sum().alias("holdout_abs_actual_all"),
            pl.col("zero_sale_abs_error").sum().alias("holdout_zero_sale_abs_error"),
            pl.col("sum_abs_error_official").sum().alias("holdout_abs_error_official"),
            pl.col("sum_abs_y_official").sum().alias("holdout_abs_y_official"),
            pl.col("sum_signed_error_official").sum().alias("holdout_signed_error_official"),
            pl.col("underforecast_share_all").mean().alias("holdout_underforecast_share"),
        )
        .with_columns(
            _safe_div("holdout_abs_error_all", "holdout_abs_actual_all", "holdout_wmape_all_points"),
            _safe_div("holdout_abs_error_official", "holdout_abs_y_official", "holdout_wmape_official"),
            _safe_div("holdout_signed_error_official", "holdout_abs_y_official", "holdout_bias_official"),
            pl.when(pl.col("holdout_abs_error_all") > 0)
            .then(pl.col("holdout_zero_sale_abs_error") / pl.col("holdout_abs_error_all"))
            .otherwise(0.0)
            .alias("holdout_zero_sale_error_share"),
        )
    )
    recurrence = (
        hold.join(hist, on=["section", "target", "month_day"], how="left")
        .with_columns(
            (pl.col("holdout_wmape_official") - pl.col("history_median_wmape_official")).alias("wmape_official_vs_history"),
            (pl.col("holdout_wmape_all_points") - pl.col("history_median_wmape_all_points")).alias("wmape_all_points_vs_history"),
            (pl.col("holdout_zero_sale_error_share") - pl.col("history_median_zero_sale_error_share")).alias("zero_sale_error_share_vs_history"),
        )
        .with_columns(
            pl.when(
                (pl.col("history_years").fill_null(0) < 1)
                | pl.col("history_median_wmape_all_points").is_null()
            )
            .then(pl.lit("no_history"))
            .when(
                (pl.col("wmape_all_points_vs_history") <= 0.10)
                & (pl.col("zero_sale_error_share_vs_history").abs() <= 0.10)
            )
            .then(pl.lit("historically_recurrent"))
            .when(
                (pl.col("wmape_all_points_vs_history") >= 0.25)
                | (pl.col("zero_sale_error_share_vs_history") >= 0.20)
            )
            .then(pl.lit("holdout_regime_shift"))
            .otherwise(pl.lit("mixed_recurrence"))
            .alias("recurrence_signal")
        )
        .sort("holdout_abs_error_all", descending=True)
    )
    return {
        "calendar_day_yearly": yearly.sort(["section", "target", "month_day", "period_type", "year"]),
        "calendar_day_recurrence": recurrence,
    }


def summarize_leaf_parent_candidates(candidates: pl.DataFrame) -> pl.DataFrame:
    """Normalize compact Phase-5 leaf×period×candidate summaries."""
    if candidates.height == 0:
        return _empty()
    required = {
        "unique_id", "target", "period_type", "candidate_parent",
        "sum_abs_error", "sum_abs_y", "sum_signed_error", "n_points",
        "n_folds", "positive_fold_share", "wmape", "bias",
    }
    if not required.issubset(candidates.columns):
        return _empty()
    return (
        candidates.with_columns(
            pl.col("unique_id").str.extract(r"^([^|]+)", 1).alias("section"),
            pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("store_id"),
            pl.col("unique_id").str.extract(r"\|\|S:(.+)$", 1).alias("sku_id"),
        )
        .sort(["section", "store_id", "sku_id", "target", "period_type", "candidate_parent"])
    )

def summarize_leaf_parent_by_section(candidates: pl.DataFrame) -> pl.DataFrame:
    """Bottom-up Section summary of compact Phase-5 leaf candidates."""
    base = summarize_leaf_parent_candidates(candidates)
    if base.height == 0:
        return _empty()
    return (
        base.group_by(["section", "target", "period_type", "candidate_parent"])
        .agg(
            pl.col("sum_abs_error").sum(),
            pl.col("sum_abs_y").sum(),
            pl.col("sum_signed_error").sum(),
            pl.col("n_points").sum(),
            pl.col("unique_id").n_unique().alias("n_leaves"),
            pl.col("n_folds").sum().alias("n_leaf_folds"),
        )
        .with_columns(
            _safe_div("sum_abs_error", "sum_abs_y", "wmape"),
            _safe_div("sum_signed_error", "sum_abs_y", "bias"),
        )
        .sort(["section", "target", "period_type", "wmape", "candidate_parent"])
    )

def recommend_leaf_parent_switches(candidates: pl.DataFrame) -> pl.DataFrame:
    """Create leaf-specific Section-vs-Store candidates from closed history.

    Input is already compact leaf×period×candidate output from the leaf engine.
    OOS can only validate/veto a candidate that passed the historical gates.
    ``ses_only`` is attached only as a diagnostic benchmark.
    """
    base = summarize_leaf_parent_candidates(candidates)
    if base.height == 0:
        return _empty()

    keys = ["section", "store_id", "sku_id", "unique_id", "target", "period_type"]
    current = base.filter(pl.col("candidate_parent") == "current_policy").select(
        *keys,
        pl.col("sum_abs_error").alias("current_sum_abs_error"),
        pl.col("sum_abs_y").alias("current_sum_abs_y"),
        pl.col("sum_signed_error").alias("current_sum_signed_error"),
        pl.col("wmape").alias("current_wmape"),
        pl.col("bias").alias("current_bias"),
        pl.col("n_points").alias("current_n_points"),
        pl.col("last_current_parent").alias("baseline_parent"),
    )
    alt = base.filter(pl.col("candidate_parent").is_in(["store", "section"]))
    if current.height == 0 or alt.height == 0:
        return _empty()
    joined = alt.join(current, on=keys, how="inner")
    if joined.height == 0:
        return _empty()

    hist = (
        joined.filter(pl.col("period_type") == "in_sample")
        .with_columns(
            (pl.col("current_wmape") - pl.col("wmape")).alias("history_wmape_improvement"),
            (pl.col("bias").abs() - pl.col("current_bias").abs()).alias("history_abs_bias_change"),
        )
        .with_columns(
            (
                (pl.col("n_points") == pl.col("current_n_points"))
                & (pl.col("n_folds") >= int(getattr(settings, "STAT_OPT_PARENT_MIN_FOLDS", 4)))
                & (pl.col("history_wmape_improvement") >= float(getattr(settings, "STAT_OPT_PARENT_MIN_WMAPE_IMPROVEMENT", 0.010)))
                & (pl.col("positive_fold_share") >= float(getattr(settings, "STAT_OPT_PARENT_MIN_POSITIVE_FOLD_SHARE", 0.60)))
                & (pl.col("history_abs_bias_change") <= float(getattr(settings, "STAT_OPT_PARENT_MAX_HISTORY_ABS_BIAS_DETERIORATION", 0.010)))
            ).alias("history_candidate")
        )
        .filter(pl.col("history_candidate"))
        .select(
            "section", "store_id", "sku_id", "unique_id", "target", "candidate_parent",
            pl.col("n_folds").alias("history_n_folds"),
            pl.col("positive_fold_share").alias("history_positive_fold_share"),
            pl.col("wmape").alias("history_candidate_wmape"),
            pl.col("current_wmape").alias("history_current_wmape"),
            pl.col("bias").alias("history_candidate_bias"),
            pl.col("current_bias").alias("history_current_bias"),
            "history_wmape_improvement", "history_abs_bias_change",
        )
        .sort(
            ["unique_id", "target", "history_wmape_improvement", "history_positive_fold_share"],
            descending=[False, False, True, True],
        )
        .unique(subset=["unique_id", "target"], keep="first", maintain_order=True)
    )
    if hist.height == 0:
        return hist

    hold = (
        joined.filter(pl.col("period_type") == "out_sample")
        .with_columns(
            (pl.col("current_wmape") - pl.col("wmape")).alias("holdout_wmape_improvement"),
            (pl.col("bias").abs() - pl.col("current_bias").abs()).alias("holdout_abs_bias_change"),
        )
        .select(
            "unique_id", "target", "candidate_parent",
            pl.col("baseline_parent").alias("holdout_baseline_parent"),
            pl.col("wmape").alias("holdout_candidate_wmape"),
            pl.col("current_wmape").alias("holdout_current_wmape"),
            pl.col("bias").alias("holdout_candidate_bias"),
            pl.col("current_bias").alias("holdout_current_bias"),
            (pl.col("n_points") == pl.col("current_n_points")).alias("holdout_support_match"),
            "holdout_wmape_improvement", "holdout_abs_bias_change",
        )
    )

    ses_hist = base.filter(
        (pl.col("period_type") == "in_sample") & (pl.col("candidate_parent") == "ses_only")
    ).select("unique_id", "target", pl.col("wmape").alias("history_ses_only_wmape"))
    ses_hold = base.filter(
        (pl.col("period_type") == "out_sample") & (pl.col("candidate_parent") == "ses_only")
    ).select("unique_id", "target", pl.col("wmape").alias("holdout_ses_only_wmape"))

    out = hist.join(hold, on=["unique_id", "target", "candidate_parent"], how="left")
    out = out.join(ses_hist, on=["unique_id", "target"], how="left").join(
        ses_hold, on=["unique_id", "target"], how="left"
    )
    validate_bias = float(
        getattr(settings, "STAT_OPT_PARENT_HOLDOUT_VALIDATE_MAX_ABS_BIAS_DETERIORATION", 0.005)
    )
    veto_bias = float(
        getattr(settings, "STAT_OPT_PARENT_HOLDOUT_VETO_ABS_BIAS_DETERIORATION", 0.010)
    )
    return (
        out.with_columns(
            pl.when(pl.col("holdout_wmape_improvement").is_null())
            .then(pl.lit("no_holdout"))
            .when(~pl.col("holdout_support_match").fill_null(False))
            .then(pl.lit("holdout_veto"))
            .when(pl.col("candidate_parent") == pl.col("holdout_baseline_parent"))
            .then(pl.lit("already_current_parent"))
            .when(
                (pl.col("holdout_wmape_improvement") >= 0.0)
                & (pl.col("holdout_abs_bias_change") <= validate_bias)
            )
            .then(pl.lit("validated_parent_switch"))
            .when(
                (pl.col("holdout_wmape_improvement") < 0.0)
                | (pl.col("holdout_abs_bias_change") > veto_bias)
            )
            .then(pl.lit("holdout_veto"))
            .otherwise(pl.lit("mixed_holdout"))
            .alias("validation_status")
        )
        .sort(["validation_status", "history_wmape_improvement"], descending=[False, True])
    )

def build_leaf_zero_rate_diagnostics(forecast: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Audit zero-sale incidence without affecting official model selection."""
    empty = {"leaf_zero_rate_summary": _empty(), "leaf_zero_rate_patterns": _empty()}
    if forecast.height == 0:
        return empty
    required = {"unique_id", "ds", "period_type", "y", "rls_metric_eligible"}
    if not required.issubset(forecast.columns):
        return empty
    cols = [*required]
    if "rls_block" in forecast.columns:
        cols.append("rls_block")
    leaf = (
        forecast.filter(
            (pl.col("unique_id").str.count_matches(r"\|\|") == 2)
            & pl.col("period_type").is_in(["in_sample", "out_sample"])
            & pl.col("rls_metric_eligible").fill_null(False)
            & pl.col("y").is_finite()
        )
        .select(cols)
        .with_columns(
            pl.col("ds").cast(pl.Date),
            pl.col("unique_id").str.extract(r"^([^|]+)", 1).alias("section"),
            pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("store_id"),
            pl.col("unique_id").str.extract(r"\|\|S:(.+)$", 1).alias("sku_id"),
            (pl.col("y") <= 0.0).alias("zero_sale"),
        )
    )
    if leaf.height == 0:
        return empty
    keys = ["section", "store_id", "sku_id", "unique_id", "period_type"]
    summary = (
        leaf.group_by(keys)
        .agg(
            pl.len().alias("n_points"),
            pl.col("zero_sale").sum().alias("n_zero_sale"),
            pl.col("y").filter(pl.col("y") > 0.0).sum().alias("positive_units"),
        )
        .with_columns((pl.col("n_zero_sale") / pl.col("n_points")).alias("zero_rate"))
        .sort(["section", "store_id", "zero_rate"], descending=[False, False, True])
    )

    dated = leaf.with_columns(
        pl.col("ds").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("ds").dt.month().cast(pl.Int8).alias("month"),
    )
    parts: list[pl.DataFrame] = []
    dimensions = [("weekday", "weekday"), ("month", "month")]
    if "rls_block" in dated.columns:
        dimensions.append(("block", "rls_block"))
    for label, col in dimensions:
        part = (
            dated.group_by([*keys, col])
            .agg(
                pl.len().alias("n_points"),
                pl.col("zero_sale").sum().alias("n_zero_sale"),
            )
            .with_columns(
                pl.lit(label).alias("dimension"),
                pl.col(col).cast(pl.Utf8).alias("dimension_value"),
                (pl.col("n_zero_sale") / pl.col("n_points")).alias("zero_rate"),
            )
            .drop(col)
        )
        parts.append(part)
    patterns = pl.concat(parts, how="diagonal_relaxed") if parts else _empty()
    return {"leaf_zero_rate_summary": summary, "leaf_zero_rate_patterns": patterns}

def _top_rows(df: pl.DataFrame, period: str, n: int = 20) -> list[dict]:
    if df.height == 0 or "period_type" not in df.columns:
        return []
    return df.filter(pl.col("period_type") == period).head(n).to_dicts()


def _top_per_group(
    df: pl.DataFrame,
    period: str,
    group_keys: list[str],
    n_per_group: int,
) -> list[dict]:
    """Return the best N rows for every group so no section/target disappears."""
    if df.height == 0 or "period_type" not in df.columns:
        return []
    part = df.filter(pl.col("period_type") == period)
    if part.height == 0:
        return []
    # Input summaries are already sorted by wMAPE within their keys.  Rank
    # explicitly anyway so this helper remains correct if upstream sorting changes.
    order_cols = [*group_keys, "wmape"] if "wmape" in part.columns else group_keys
    part = part.sort(order_cols).with_columns(
        pl.col("wmape").rank(method="ordinal").over(group_keys).alias("_group_rank")
    )
    return (
        part.filter(pl.col("_group_rank") <= int(n_per_group))
        .drop("_group_rank")
        .to_dicts()
    )


def _format_pct(x) -> str:
    try:
        return f"{100.0 * float(x):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def build_text_report(
    *,
    update_block_days: int,
    ses_summary: pl.DataFrame,
    rls_summary: pl.DataFrame,
    ses_recommendation: pl.DataFrame,
    rls_recommendation: pl.DataFrame,
    driver_summary: pl.DataFrame,
    driver_refit_summary: pl.DataFrame,
    driver_refit_validation: pl.DataFrame,
    promotion_reaudit: pl.DataFrame,
    leaf_residual_summary: pl.DataFrame,
    residual_signal_summary: pl.DataFrame,
    driver_strength_recommendation: pl.DataFrame,
    calendar_day_recurrence: pl.DataFrame,
    leaf_parent_section_summary: pl.DataFrame,
    leaf_parent_recommendation: pl.DataFrame,
    leaf_zero_rate_summary: pl.DataFrame,
) -> str:
    lines = [
        f"OPTIMIZACIÓN ESTADÍSTICA v{settings.APP_VERSION} — SES + RLS (mismo modelo; tuning causal + diagnóstico residual/calibración/parent)",
        f"Bloque de actualización: {int(update_block_days)}d",
        "Objetivo: wMAPE oficial y!=0; BIAS como control. OOS actual es holdout, no criterio de tuning.",
        "",
        "=== 1) SES — candidato histórico por Sección/target ===",
    ]
    for r in ses_recommendation.to_dicts() if ses_recommendation.height else []:
        lines.append(
            f"sec={r['section']} | {r['target']:<10} | alpha={float(r['alpha']):.4f}"
            f"{' [diag-only]' if r.get('productive_candidate') is False else ''} | "
            f"history wMAPE={_format_pct(r.get('history_wmape'))} | BIAS={_format_pct(r.get('history_bias'))} | "
            f"worst-fold={_format_pct(r.get('history_worst_fold_wmape'))} | "
            f"holdout={_format_pct(r.get('holdout_wmape'))}"
        )
    lines.extend(["", "=== 2) SES — holdout OOS (solo validación) ==="])
    for r in _top_per_group(
        ses_summary, "out_sample", ["section", "target"], 6
    ):
        lines.append(
            f"sec={r['section']} | {r['target']:<10} | alpha={float(r['alpha']):.3f} | "
            f"wMAPE={_format_pct(r.get('wmape'))} | BIAS={_format_pct(r.get('bias'))}"
        )

    lines.extend(["", "=== 3) RLS — candidato histórico por nodo/target ==="])
    for r in rls_recommendation.to_dicts() if rls_recommendation.height else []:
        lines.append(
            f"{r['unique_id']:<18} | {r['target']:<10} | "
            f"{r['dynamics']}/lambda={float(r['lambda']):.4f}"
            f"{' [diag-only]' if r.get('productive_candidate') is False else ''} | "
            f"history wMAPE={_format_pct(r.get('history_wmape'))} | BIAS={_format_pct(r.get('history_bias'))} | "
            f"worst-fold={_format_pct(r.get('history_worst_fold_wmape'))} | "
            f"holdout={_format_pct(r.get('holdout_wmape'))}"
        )
    lines.extend(["", "=== 4) RLS — holdout OOS (solo validación) ==="])
    for r in _top_per_group(
        rls_summary, "out_sample", ["section", "node_level", "target"], 4
    ):
        lines.append(
            f"sec={r['section']} | {r['node_level']:<7} | {r['target']:<10} | "
            f"{r['dynamics']}/lambda={float(r['lambda']):.3f} | wMAPE={_format_pct(r.get('wmape'))} | "
            f"BIAS={_format_pct(r.get('bias'))}"
        )

    lines.extend(
        [
            "",
            "=== 5) Drivers existentes — screening de contribución ===",
            "Delta > 0: al anular el grupo empeora wMAPE (señal útil). Delta < 0: revisar.",
            "El screening no refitea weekday/month/price/holiday; AR sí compara contra el RLS base refiteado con la misma lambda. Nada se modifica automáticamente.",
        ]
    )
    for r in _top_rows(driver_summary, "in_sample", 80):
        lines.append(
            f"sec={r['section']} | {r['node_level']:<7} | {r['target']:<10} | {r['driver_group']:<14} | "
            f"delta={_format_pct(r.get('weighted_wmape_delta_if_removed'))} | "
            f"positive-folds={_format_pct(r.get('positive_fold_share'))} | {r.get('screen_class')}"
        )

    lines.extend(
        [
            "",
            "=== 6) Drivers existentes — refit exacto del MISMO RLS (Phase 2) ===",
            "Improvement > 0: el challenger (remove/add) reduce wMAPE respecto al RLS actual.",
            "Valor($)/price prueba add donde el grupo no está y remove donde ya fue promovido; usa solo drivers existentes.",
        ]
    )
    for r in _top_rows(driver_refit_summary, "in_sample", 120):
        lines.append(
            f"{r.get('unique_id', r['section']):<18} | {r['target']:<10} | "
            f"{r['test_action']:<6} {r['driver_group']:<14} | "
            f"improvement={_format_pct(r.get('weighted_wmape_improvement'))} | "
            f"positive-folds={_format_pct(r.get('positive_fold_share'))} | "
            f"abs-BIAS-change={_format_pct(r.get('weighted_abs_bias_change'))} | "
            f"{r.get('decision_signal')}"
        )

    lines.extend([
        "",
        "=== 7) Candidatos de drivers — validación OOS (holdout; nunca tuning) ===",
        "Solo aparecen cambios que YA fueron candidatos por historia cerrada.",
        "validated_candidate: holdout no empeora wMAPE y |BIAS| empeora <=0.5 pp; holdout_veto: se descarta.",
    ])
    if driver_refit_validation.height:
        for r in driver_refit_validation.to_dicts():
            lines.append(
                f"{r['unique_id']:<18} | {r['target']:<10} | "
                f"{r['test_action']:<6} {r['driver_group']:<14} | "
                f"hist={_format_pct(r.get('history_wmape_improvement'))} "
                f"({ _format_pct(r.get('history_positive_fold_share')) } folds+) | "
                f"holdout={_format_pct(r.get('holdout_wmape_improvement'))} | "
                f"holdout Δ|BIAS|={_format_pct(r.get('holdout_abs_bias_change'))} | "
                f"{r.get('holdout_validation', 'n/a')}"
            )
    else:
        lines.append("Sin candidatos históricos que requieran validación holdout.")

    lines.extend([
        "",
        "=== 7b) Promociones activas — reauditoría tras re-selección de parámetros ===",
        "Add-back > 0: volver a incluir el grupo mejora el RLS actual; la reauditoría no revierte nada automáticamente.",
    ])
    if promotion_reaudit.height:
        for r in promotion_reaudit.to_dicts():
            lines.append(
                f"{r['unique_id']:<18} | {r['target']:<10} | add-back {r['driver_group']:<14} | "
                f"hist={_format_pct(r.get('history_addback_improvement'))} | "
                f"holdout={_format_pct(r.get('holdout_addback_improvement'))} | "
                f"Δ|BIAS| holdout={_format_pct(r.get('holdout_addback_abs_bias_change'))} | "
                f"{r.get('reaudit_status', 'n/a')}"
            )
    else:
        lines.append("Sin promociones activas con refit add-back disponible.")

    lines.extend([
        "",
        "=== 8) Residuos extremos SKU+Tienda — diagnóstico de información faltante (Phase 3) ===",
        "Regla extrema: factor>=4 y gap escalado>=0.15, o gap escalado>=3; escala causal=max(nivel SES, nivel inicial, 1).",
        "El diagnóstico NO crea drivers: mide cuánto efecto RLS adicional habría sido necesario para reconciliar nivel SES y actual.",
    ])
    if leaf_residual_summary.height:
        min_points = int(getattr(settings, "OOS_ACTIVE_MIN_NONZERO_DAYS", 7))
        top_store = (
            leaf_residual_summary.filter(
                (pl.col("period_type") == "out_sample")
                & (pl.col("n_metric_points") >= min_points)
            )
            .sort(["wmape_official", "extreme_abs_error_official"], descending=[True, True])
            .head(12)
        )
        if top_store.height:
            lines.append("Top store/target OOS por wMAPE (con participación de errores extremos):")
            for r in top_store.to_dicts():
                lines.append(
                    f"sec={r['section']} | store={r['store_id']} | {r['target']:<10} | "
                    f"wMAPE={_format_pct(r.get('wmape_official'))} | all-points={_format_pct(r.get('wmape_all_points'))} | "
                    f"extremos={int(r.get('n_extreme') or 0)}/{int(r.get('n_points') or 0)} | "
                    f"share-error-extremo={_format_pct(r.get('extreme_error_share'))} | "
                    f"error-y=0={_format_pct(r.get('zero_sale_error_share'))} | "
                    f"driver-ayuda-extremos={_format_pct(r.get('driver_help_metric_extreme_share'))} | "
                    f"median |efecto faltante| metric={float(r.get('median_abs_unexplained_effect_metric') or 0.0):.3f} | "
                    f"{r.get('driver_review_signal', 'n/a')} / {r.get('zero_sale_review_signal', 'n/a')}"
                )
    else:
        lines.append("Sin resumen residual disponible.")

    if residual_signal_summary.height:
        top_dates = (
            residual_signal_summary.filter(
                (pl.col("period_type") == "out_sample")
                & (pl.col("dimension") == "date")
            )
            .sort("extreme_abs_error_official", descending=True)
            .head(10)
        )
        if top_dates.height:
            lines.append("Fechas OOS con mayor concentración de error extremo:")
            for r in top_dates.to_dicts():
                lines.append(
                    f"sec={r['section']} | {r['target']:<10} | fecha={r['dimension_value']} | "
                    f"hojas={int(r.get('n_leaves') or 0)} | "
                    f"share-error-extremo={_format_pct(r.get('extreme_error_share_within_target'))} | "
                    f"y=0-extremos={_format_pct(r.get('zero_sale_extreme_share'))} | "
                    f"underforecast(metric)={_format_pct(r.get('underforecast_metric_share'))} | "
                    f"driver-ayuda={_format_pct(r.get('driver_help_extreme_share'))} | "
                    f"median |efecto faltante|={float(r.get('median_abs_unexplained_effect') or 0.0):.3f}"
                )

    lines.extend([
        "",
        "=== 9) Intensidad del efecto RLS en hojas — gamma diagnóstico (Phase 4) ===",
        "gamma=1.0 reproduce producción. Un gamma distinto solo puede ser candidato por historia cerrada; OOS valida/veta.",
    ])
    if driver_strength_recommendation.height:
        for r in driver_strength_recommendation.to_dicts():
            lines.append(
                f"sec={r['section']} | {r['target']:<10} | parent={r['parent_model']:<7} | "
                f"gamma={float(r['gamma']):.2f} | hist-improvement={_format_pct(r.get('history_wmape_improvement'))} | "
                f"folds+={_format_pct(r.get('history_positive_fold_share'))} | "
                f"holdout-improvement={_format_pct(r.get('holdout_wmape_improvement'))} | "
                f"Δ|BIAS|={_format_pct(r.get('holdout_abs_bias_change'))} | "
                f"{r.get('validation_status', 'n/a')}"
            )
    else:
        lines.append("Sin diagnóstico gamma disponible.")

    lines.extend([
        "",
        "=== 10) Recurrencia calendario — mismo mes/día en historia cerrada (diagnóstico, no driver) ===",
        "Compara cada día OOS contra el mismo MM-DD de años previos; sirve para separar patrón recurrente de cambio de régimen.",
    ])
    if calendar_day_recurrence.height:
        top_calendar = calendar_day_recurrence.sort(
            "holdout_abs_error_all", descending=True
        ).head(12)
        for r in top_calendar.to_dicts():
            lines.append(
                f"sec={r['section']} | {r['target']:<10} | día={r['month_day']} | "
                f"holdout all-points={_format_pct(r.get('holdout_wmape_all_points'))} | "
                f"hist-med={_format_pct(r.get('history_median_wmape_all_points'))} | "
                f"error-y=0 holdout={_format_pct(r.get('holdout_zero_sale_error_share'))} | "
                f"hist-med={_format_pct(r.get('history_median_zero_sale_error_share'))} | "
                f"años={int(r.get('history_years') or 0)} | {r.get('recurrence_signal', 'n/a')}"
            )
    else:
        lines.append("Sin recurrencia calendario disponible.")

    lines.extend([
        "",
        "=== 11) Parent RLS en contexto SKU+Tienda — Section vs Store (Phase 5) ===",
        "Mismo SES/alpha y mismos RLS existentes; solo cambia qué parent aporta el efecto. SES-only es referencia diagnóstica, nunca candidato.",
        "Candidato histórico: mejora leaf >=1 pp, >=4 folds, >=60% folds no peores y deterioro histórico de |BIAS| <=1 pp. OOS solo valida/veta.",
    ])
    if leaf_parent_section_summary.height:
        hist_parent = leaf_parent_section_summary.filter(
            pl.col("period_type") == "in_sample"
        ).sort(["section", "target", "wmape"])
        for r in hist_parent.to_dicts():
            lines.append(
                f"sec={r['section']} | {r['target']:<10} | {r['candidate_parent']:<14} | "
                f"history wMAPE={_format_pct(r.get('wmape'))} | BIAS={_format_pct(r.get('bias'))} | "
                f"hojas={int(r.get('n_leaves') or 0)}"
            )
    else:
        lines.append("Sin resumen Section/Store leaf disponible.")

    if leaf_parent_recommendation.height:
        status_counts = (
            leaf_parent_recommendation.group_by(["section", "target", "validation_status"])
            .agg(pl.len().alias("n"))
            .sort(["section", "target", "validation_status"])
        )
        lines.append("Candidatos leaf por estado:")
        for r in status_counts.to_dicts():
            lines.append(
                f"sec={r['section']} | {r['target']:<10} | {r['validation_status']:<24} | n={int(r['n'])}"
            )
        lines.append("Top candidatos por mejora histórica:")
        top = leaf_parent_recommendation.sort(
            "history_wmape_improvement", descending=True
        ).head(20)
        for r in top.to_dicts():
            lines.append(
                f"{r['unique_id']:<28} | {r['target']:<10} | parent={r['candidate_parent']:<7} | "
                f"hist+={_format_pct(r.get('history_wmape_improvement'))} "
                f"({ _format_pct(r.get('history_positive_fold_share')) } folds+) | "
                f"holdout+={_format_pct(r.get('holdout_wmape_improvement'))} | "
                f"Δ|BIAS|={_format_pct(r.get('holdout_abs_bias_change'))} | "
                f"SES-only hist={_format_pct(r.get('history_ses_only_wmape'))} | "
                f"{r.get('validation_status', 'n/a')}"
            )
    else:
        lines.append("Sin switches Section/Store que superen los gates históricos.")

    lines.extend([
        "",
        "=== 12) Venta cero SKU+Tienda — auditoría diagnóstica ===",
        "No participa en wMAPE/BIAS oficial ni en selección. Se usa para separar error de magnitud de error por ocurrencia y=0.",
    ])
    if leaf_zero_rate_summary.height:
        top_zero = (
            leaf_zero_rate_summary.filter(pl.col("period_type") == "out_sample")
            .sort(["zero_rate", "n_points"], descending=[True, True])
            .head(12)
        )
        for r in top_zero.to_dicts():
            lines.append(
                f"{r['unique_id']:<28} | zero-rate OOS={_format_pct(r.get('zero_rate'))} | "
                f"ceros={int(r.get('n_zero_sale') or 0)}/{int(r.get('n_points') or 0)}"
            )
    else:
        lines.append("Sin auditoría de venta cero disponible.")

    lines.extend(
        [
            "",
            "REGLA DE GOBIERNO:",
            "- No existe promoción automática: cualquier cambio productivo queda explícito en settings/CHANGELOG.",
            "- La selección se decide con historia cerrada; OOS actual se usa como holdout final.",
            "- Phase 2 refitea exactamente el MISMO RLS para decidir keep/remove/add sobre drivers existentes.",
            "- El OOS actual solo puede VALIDAR o VETAR un candidato histórico; nunca crearlo.",
            "- Phase 5 compara solo Section vs Store ya existentes usando el mismo SES/alpha; SES-only es referencia diagnóstica.",
            "- Si el refit no explica los residuos extremos, se diagnostica información faltante antes de proponer cualquier driver nuevo.",
            "- Cualquier driver nuevo requiere aprobación explícita antes de incorporarse.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_optimization_artifacts(
    *,
    out_dir: Path,
    update_block_days: int,
    diagnostics: dict[str, pl.DataFrame],
    forecast: pl.DataFrame | None = None,
) -> Path:
    """Persist compact optimization diagnostics beside the forecast artifact."""
    opt_dir = Path(out_dir) / OPTIMIZATION_DIRNAME
    opt_dir.mkdir(parents=True, exist_ok=True)

    ses = diagnostics.get("ses_candidates", _empty())
    leaf_parent = diagnostics.get("leaf_parent_candidates", _empty())
    rls = diagnostics.get("rls_candidates", _empty())
    drv = diagnostics.get("driver_screen", _empty())
    drv_refit = diagnostics.get("driver_refit", _empty())

    ses_summary = summarize_ses(ses)
    rls_summary = summarize_rls(rls)
    rls_node_summary = summarize_rls_by_node(rls)
    ses_rec = recommend_ses(ses_summary)
    rls_rec = recommend_rls(rls_node_summary)
    drv_summary = summarize_driver_screen(drv)
    drv_refit_summary = summarize_driver_refit(drv_refit)
    drv_refit_validation = validate_driver_refit_candidates(drv_refit_summary)
    promotion_reaudit = audit_promoted_exclusions(drv_refit_summary)
    residuals = build_leaf_residual_diagnostics(
        forecast if forecast is not None else _empty()
    )
    leaf_residual_extremes = residuals["leaf_residual_extremes"]
    leaf_residual_summary = residuals["leaf_residual_summary"]
    residual_signal_summary = residuals["residual_signal_summary"]
    strength = build_leaf_driver_strength_diagnostics(
        forecast if forecast is not None else _empty()
    )
    driver_strength_candidates = strength["driver_strength_candidates"]
    driver_strength_summary = strength["driver_strength_summary"]
    driver_strength_recommendation = strength["driver_strength_recommendation"]
    calendar = build_calendar_day_recurrence_diagnostics(
        forecast if forecast is not None else _empty()
    )
    calendar_day_yearly = calendar["calendar_day_yearly"]
    calendar_day_recurrence = calendar["calendar_day_recurrence"]
    leaf_parent_summary = summarize_leaf_parent_candidates(leaf_parent)
    leaf_parent_section_summary = summarize_leaf_parent_by_section(leaf_parent)
    leaf_parent_recommendation = recommend_leaf_parent_switches(leaf_parent)
    zero_rates = build_leaf_zero_rate_diagnostics(
        forecast if forecast is not None else _empty()
    )
    leaf_zero_rate_summary = zero_rates["leaf_zero_rate_summary"]
    leaf_zero_rate_patterns = zero_rates["leaf_zero_rate_patterns"]

    files = {
        "ses_candidates.parquet": ses,
        "ses_summary.parquet": ses_summary,
        "ses_recommendation.parquet": ses_rec,
        "leaf_parent_candidates.parquet": leaf_parent,
        "leaf_parent_summary.parquet": leaf_parent_summary,
        "leaf_parent_section_summary.parquet": leaf_parent_section_summary,
        "leaf_parent_recommendation.parquet": leaf_parent_recommendation,
        "leaf_zero_rate_summary.parquet": leaf_zero_rate_summary,
        "leaf_zero_rate_patterns.parquet": leaf_zero_rate_patterns,
        "rls_candidates.parquet": rls,
        "rls_summary.parquet": rls_summary,
        "rls_node_summary.parquet": rls_node_summary,
        "rls_recommendation.parquet": rls_rec,
        "driver_screen.parquet": drv,
        "driver_summary.parquet": drv_summary,
        "driver_refit.parquet": drv_refit,
        "driver_refit_summary.parquet": drv_refit_summary,
        "driver_refit_validation.parquet": drv_refit_validation,
        "promotion_reaudit.parquet": promotion_reaudit,
        "leaf_residual_extremes.parquet": leaf_residual_extremes,
        "leaf_residual_summary.parquet": leaf_residual_summary,
        "residual_signal_summary.parquet": residual_signal_summary,
        "driver_strength_candidates.parquet": driver_strength_candidates,
        "driver_strength_summary.parquet": driver_strength_summary,
        "driver_strength_recommendation.parquet": driver_strength_recommendation,
        "calendar_day_yearly.parquet": calendar_day_yearly,
        "calendar_day_recurrence.parquet": calendar_day_recurrence,
    }
    for name, frame in files.items():
        path = opt_dir / name
        if frame.height:
            frame.write_parquet(path, compression="zstd", statistics=True)
        elif path.exists():
            path.unlink()

    report = build_text_report(
        update_block_days=int(update_block_days),
        ses_summary=ses_summary,
        rls_summary=rls_summary,
        ses_recommendation=ses_rec,
        rls_recommendation=rls_rec,
        driver_summary=drv_summary,
        driver_refit_summary=drv_refit_summary,
        driver_refit_validation=drv_refit_validation,
        promotion_reaudit=promotion_reaudit,
        leaf_residual_summary=leaf_residual_summary,
        residual_signal_summary=residual_signal_summary,
        driver_strength_recommendation=driver_strength_recommendation,
        calendar_day_recurrence=calendar_day_recurrence,
        leaf_parent_section_summary=leaf_parent_section_summary,
        leaf_parent_recommendation=leaf_parent_recommendation,
        leaf_zero_rate_summary=leaf_zero_rate_summary,
    )
    (opt_dir / "optimization_summary.txt").write_text(report, encoding="utf-8")
    return opt_dir
