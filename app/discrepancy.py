"""Pure-Python Actual-vs-Forecast discrepancy rule used by dashboard/audits."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DiscrepancyRule:
    factor_threshold: float = 4.0
    material_scaled_gap: float = 0.15
    severe_scaled_gap: float = 3.0
    floor_share: float = 0.05


def _median_positive(values: list[float]) -> float:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)) and float(v) > 0)
    if not vals:
        return 0.0
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def discrepancy_points(dates: list, actuals: list, forecasts: list, *, rule: DiscrepancyRule | None = None) -> list[dict]:
    """Return materially extreme Actual-vs-Forecast discrepancies.

    Symmetric by construction. A ratio component detects multiplicative misses,
    while a series-scale component prevents tiny values from being flagged only
    because their ratio is large.
    """
    rule = rule or DiscrepancyRule()
    triples = []
    positive_actuals: list[float] = []
    positive_any: list[float] = []
    for d, a, f in zip(dates, actuals, forecasts):
        if a is None or f is None:
            continue
        try:
            av, fv = float(a), float(f)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(av) and math.isfinite(fv)):
            continue
        av = max(av, 0.0)
        fv = max(fv, 0.0)
        triples.append((d, av, fv))
        if av > 0:
            positive_actuals.append(av)
            positive_any.append(av)
        if fv > 0:
            positive_any.append(fv)
    scale = _median_positive(positive_actuals) or _median_positive(positive_any)
    if scale <= 0:
        return []
    floor = max(scale * float(rule.floor_share), 1e-9)
    out: list[dict] = []
    for d, av, fv in triples:
        hi, lo = max(av, fv), min(av, fv)
        factor_gap = (hi + floor) / (lo + floor)
        abs_error = abs(av - fv)
        scaled_gap = abs_error / max(scale, floor)
        if ((factor_gap >= rule.factor_threshold and scaled_gap >= rule.material_scaled_gap)
                or scaled_gap >= rule.severe_scaled_gap):
            out.append({
                "ds": d,
                "actual": av,
                "forecast": fv,
                "abs_error": abs_error,
                "factor_gap": factor_gap,
                "scaled_gap": scaled_gap,
                "series_scale": scale,
            })
    return out
