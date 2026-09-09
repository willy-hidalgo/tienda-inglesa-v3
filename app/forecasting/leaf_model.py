"""Small, testable statistical rules for the SKU+store model.

This module deliberately contains no Polars code.  The production runner applies
the same rules vectorially, while unit tests can validate the statistical
contract without loading the forecasting engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class RegimeDecision:
    kind: str
    structural_level: float
    history_blocks: int
    reset_level: bool
    reason: str


def _clean_levels(values: Iterable[float | int | None]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        x = float(value)
        if isfinite(x) and x >= 0.0:
            out.append(x)
    return out


def robust_structural_level(
    prior_block_means: Iterable[float | int | None],
    *,
    max_blocks: int = 5,
) -> tuple[float, int]:
    """Median of the most recent *prior* complete block means.

    B0 (the immediately closed block) must not be included here.  A median is
    used instead of a mean so one or two promotional/transient blocks cannot
    redefine the structural level.
    """
    vals = _clean_levels(prior_block_means)
    if max_blocks > 0:
        vals = vals[:max_blocks]
    if not vals:
        return 0.0, 0
    return float(median(vals)), len(vals)


def classify_regime(
    b0: float | None,
    b1: float | None,
    b2: float | None,
    b3: float | None,
    *,
    prior_block_means: Iterable[float | int | None],
    trend_up_ratio: float = 1.05,
    trend_down_ratio: float = 0.95,
    trend_max_step_ratio: float = 2.0,
    transient_ratio: float = 1.75,
    min_history_blocks: int = 4,
    eps: float = 1e-9,
) -> RegimeDecision:
    """Classify a forecast origin using a conservative rule.

    Trend requires THREE consecutive coherent transitions B3→B2→B1→B0.
    A one- or two-block jump therefore cannot become a structural trend.

    If there is enough history and B0 is far from the robust median of prior
    blocks, the event is a transient and SES should be reinitialized to the
    structural level before forecasting the next block.
    """
    structural, n_hist = robust_structural_level(prior_block_means)
    vals = [b0, b1, b2, b3]
    complete = all(
        v is not None and isfinite(float(v)) and float(v) >= 0.0
        for v in vals
    )
    if complete:
        x0, x1, x2, x3 = map(float, vals)

        trend_up = (
            x3 > eps
            and x2 > x3 * trend_up_ratio
            and x1 > x2 * trend_up_ratio
            and x0 > x1 * trend_up_ratio
            and x2 / max(x3, eps) <= trend_max_step_ratio
            and x1 / max(x2, eps) <= trend_max_step_ratio
            and x0 / max(x1, eps) <= trend_max_step_ratio
        )
        trend_down = (
            x0 >= 0.0
            and x1 > eps
            and x2 > eps
            and x3 > eps
            and x2 < x3 * trend_down_ratio
            and x1 < x2 * trend_down_ratio
            and x0 < x1 * trend_down_ratio
            and x3 / max(x2, eps) <= trend_max_step_ratio
            and x2 / max(x1, eps) <= trend_max_step_ratio
            and x1 / max(x0, eps) <= trend_max_step_ratio
        )
        if trend_up:
            return RegimeDecision(
                "trend_up", structural, n_hist, False,
                "three coherent rising block transitions",
            )
        if trend_down:
            return RegimeDecision(
                "trend_down", structural, n_hist, False,
                "three coherent falling block transitions",
            )

    if (
        n_hist >= min_history_blocks
        and structural > eps
        and b0 is not None
        and isfinite(float(b0))
    ):
        x0 = max(0.0, float(b0))
        if x0 > structural * transient_ratio:
            return RegimeDecision(
                "transient_up", structural, n_hist, True,
                "latest block is high versus robust prior median",
            )
        if x0 < structural / transient_ratio:
            return RegimeDecision(
                "transient_down", structural, n_hist, False,
                "latest block is low; use a reactive pure-SES trajectory",
            )

    return RegimeDecision(
        "stable", structural, n_hist, False,
        "no persistent trend or isolated structural break",
    )


def reset_ses_level(
    ses_level: float,
    decision: RegimeDecision,
) -> tuple[float, str]:
    """Return the level used at the forecast origin and its audit status."""
    level = max(0.0, float(ses_level))
    if decision.reset_level and decision.structural_level >= 0.0:
        suffix = "UP" if decision.kind.endswith("_up") else "DOWN"
        return decision.structural_level, f"SES_RESET_TRANSIENT_{suffix}"
    return level, "OK"




def alpha_for_regime(
    selected_alpha: float,
    decision: RegimeDecision,
    *,
    transient_down_alpha: float = 0.20,
) -> tuple[float, str]:
    """Keep normal alpha except for an abrupt downward regime transition.

    On transient_down the state must be allowed to fall.  Resetting upward to
    an older structural median would recreate the 261298 failure.  We therefore
    select a documented, reactive PURE SES trajectory instead.
    """
    alpha = float(selected_alpha)
    if decision.kind == "transient_down":
        return float(transient_down_alpha), "SES_REACTIVE_TRANSIENT_DOWN"
    return alpha, "OK"

def choose_parent(
    store_wmape: float | None,
    section_wmape: float | None,
    *,
    default: str = "store",
) -> str:
    """Choose exactly one RLS parent; there is intentionally no 'none' model."""
    pairs: list[tuple[float, str]] = []
    for name, score in (("store", store_wmape), ("section", section_wmape)):
        if score is not None:
            x = float(score)
            if isfinite(x):
                pairs.append((x, name))
    if not pairs:
        return default if default in {"store", "section"} else "store"
    return min(pairs, key=lambda item: (item[0], item[1] != default))[1]
