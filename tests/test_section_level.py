"""
Tests del modelo a nivel sección (Fase B): descomposición causal + top-down.

Cubre:
- Mapeo feature → factor causal (level/trend/seasonality/edp/discount/
  feature_display/volume/price).
- Mapa hijo→padre y tabla nivel/tendencia por serie.
- Coherencia aditiva exacta de `_allocate_level` (Σ hijos = padre por día).
- Integración de `_run_section_level` (1 fit RLS por sección + cascada
  sección→tienda→SKU + descomposición causal).

Requiere `rls_opt` (se salta si no está disponible).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from forecasts import ForecastConfig, RLSForecastPipeline, RLSForecastRunner

try:
    from rls_opt import RecursiveLeastSquaresRegression
except ImportError:  # pragma: no cover
    RecursiveLeastSquaresRegression = None

REQUIRES_RLS = pytest.mark.skipif(
    RecursiveLeastSquaresRegression is None, reason="rls_opt no disponible"
)


# ── 1. Mapeo de factores causales ─────────────────────────────────────────
def test_factor_group_mapping():
    cases = {
        "intercept": "level",
        "trend": "trend",
        "weekday_5": "seasonality",
        "month_7": "seasonality",
        "edp": "edp",
        "discount": "discount",
        "conteo_sku": "volume",
        "asp": "price",
        "navidad_0": "feature_display",
        "mothers_day_-3": "feature_display",
    }
    for feat, expected in cases.items():
        assert RLSForecastPipeline._factor_group(feat) == expected


# ── 2. Mapa hijo→padre y tabla nivel/tendencia ────────────────────────────
def test_parent_map_structure():
    train = pl.DataFrame(
        {
            "unique_id": ["1", "1||00122", "1||00154", "1||00122||A", "1||00154||C"],
            "ds": [dt.date(2024, 1, 1)] * 5,
            "y": [100.0, 60.0, 40.0, 60.0, 40.0],
        }
    )
    pm = RLSForecastPipeline._parent_map(train)
    pairs = set(zip(pm["child"].to_list(), pm["parent"].to_list()))
    assert pairs == {
        ("1||00122", "1"),
        ("1||00154", "1"),
        ("1||00122||A", "1||00122"),
        ("1||00154||C", "1||00154"),
    }


def test_level_trend_table_shares():
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(weeks=w) * 7 for w in range(6)]
    rows = []
    for d in dates:
        rows.append({"unique_id": "1", "ds": d, "y": 200.0})
        rows.append({"unique_id": "1||00122", "ds": d, "y": 60.0})
        rows.append({"unique_id": "1||00154", "ds": d, "y": 140.0})
    train = pl.DataFrame(rows).sort(["unique_id", "ds"])
    pm = RLSForecastPipeline._parent_map(train)
    tbl = RLSForecastPipeline._level_trend_table(train, pm, start)
    assert {"child", "parent", "share", "slope_c", "slope_p"} <= set(tbl.columns)
    sh = dict(zip(tbl["child"].to_list(), tbl["share"].to_list()))
    assert sh["1||00122"] == pytest.approx(60 / 200, rel=1e-9)
    assert sh["1||00154"] == pytest.approx(140 / 200, rel=1e-9)
    # trend plano → pendientes ≈ 0
    assert abs(tbl["slope_c"].sum()) < 1e-6


# ── 3. Coherencia aditiva de _allocate_level ──────────────────────────────
def test_allocate_level_coherent():
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(3)]
    children = pl.DataFrame(
        {
            "unique_id": ["1||00122", "1||00154"] * 3,
            "ds": [d for d in dates for _ in (0, 1)],
            "y": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            "value": [5.0, 6.0, 5.0, 6.0, 5.0, 6.0],
        }
    )
    parent_fcst = pl.DataFrame(
        {
            "parent": ["1"] * 3,
            "ds": dates,
            "_yhat_parent": [1000.0, 1000.0, 1000.0],
        }
    )
    tbl = pl.DataFrame(
        {
            "child": ["1||00122", "1||00154"],
            "parent": ["1", "1"],
            "share": [0.3, 0.7],
            "slope_c": [0.1, 0.1],
            "slope_p": [0.1, 0.1],
        }
    )
    lp = pl.DataFrame(
        {"unique_id": [], "_last_price": []},
        schema={"unique_id": pl.Utf8, "_last_price": pl.Float64},
    )
    out = RLSForecastPipeline._allocate_level(
        children, parent_fcst, tbl, lp, dt.date(2024, 1, 1), "out_sample"
    )
    assert {"yhat", "valuehat"} <= set(out.columns)
    sums = out.group_by("ds").agg(pl.col("yhat").sum().alias("s")).sort("ds")
    assert sums["s"].to_list() == pytest.approx([1000.0] * 3, rel=1e-9, abs=1e-6)
    # slopes iguales → reparto exacto por share (300 / 700)
    first = out.filter(
        (pl.col("ds") == dates[0]) & (pl.col("unique_id") == "1||00122")
    )["yhat"][0]
    assert first == pytest.approx(300.0, rel=1e-9, abs=1e-6)


# ── 4. Config por settings ────────────────────────────────────────────────
def test_section_level_config_from_settings():
    import settings

    cfg = ForecastConfig.from_settings()
    assert cfg.section_level_model is settings.SECTION_LEVEL_MODEL
    assert cfg.decompose_effects is settings.DECOMPOSE_EFFECTS
# ── 5. Integración _run_section_level ─────────────────────────────────────
@REQUIRES_RLS
def test_run_section_level_integration():
    import settings

    cfg = ForecastConfig.from_settings()
    cfg.section_level_model = True
    cfg.decompose_effects = True
    pipe = RLSForecastPipeline(cfg, n_jobs=1)

    rng = np.random.default_rng(7)
    start = dt.date(2024, 1, 1)
    n = 120
    dates = [start + dt.timedelta(days=i) for i in range(n)]

    def _tr(d):
        return (d - dt.date(1970, 1, 1)).days / 365.25

    uids = ["1", "1||00122", "1||00154", "1||00122||A", "1||00154||C"]
    sec_y = np.round(500 + np.arange(n) + rng.normal(0, 30, n)).clip(min=0)
    rows = []
    for i, d in enumerate(dates):
        wd2 = 1 if d.weekday() == 2 else 0
        base = {
            "ds": d,
            "intercept": 1,
            "trend": _tr(d),
            "weekday_2": wd2,
            "edp": 15.0,
            "discount": 0.0,
            "asp": 12.0,
            "conteo_sku": 40,
            "navidad_0": 1 if (d.month, d.day) == (12, 25) else 0,
        }
        rows.append({**base, "unique_id": "1", "y": float(sec_y[i]), "value": 12.0})
        rows.append(
            {
                **base,
                "unique_id": "1||00122",
                "y": float(max(0.0, round(sec_y[i] * 0.6 + rng.normal(0, 8)))),
                "value": 11.0,
            }
        )
        rows.append(
            {
                **base,
                "unique_id": "1||00154",
                "y": float(max(0.0, round(sec_y[i] * 0.4 + rng.normal(0, 8)))),
                "value": 10.5,
            }
        )
        rows.append(
            {
                **base,
                "unique_id": "1||00122||A",
                "y": float(max(0.0, round(sec_y[i] * 0.35 + rng.normal(0, 6)))),
                "value": 10.0,
            }
        )
        rows.append(
            {
                **base,
                "unique_id": "1||00154||C",
                "y": float(max(0.0, round(sec_y[i] * 0.4 + rng.normal(0, 7)))),
                "value": 10.0,
            }
        )
    df_train = pl.DataFrame(rows).sort(["unique_id", "ds"])

    fstart = dt.date(2025, 1, 1)
    fend = dt.date(2025, 1, 28)
    frows = []
    for d in pl.date_range(fstart, fend, interval="1d", eager=True):
        wd2 = 1 if d.weekday() == 2 else 0
        for uid in uids:
            frows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": 0.0,
                    "value": 0.0,
                    "intercept": 1,
                    "trend": _tr(d),
                    "weekday_2": wd2,
                    "edp": 15.0,
                    "discount": 0.0,
                    "asp": 12.0,
                    "conteo_sku": 40,
                    "navidad_0": 0,
                }
            )
    df_fcst = pl.DataFrame(frows)

    driver_cols = [
        "intercept",
        "trend",
        "weekday_2",
        "edp",
        "discount",
        "asp",
        "conteo_sku",
        "navidad_0",
    ]
    meta = {
        "train_start": start,
        "train_end": dates[-1],
        "test_start": None,
        "test_end": None,
        "forecast_start": fstart,
        "forecast_end": fend,
    }
    hz = dict(meta)
    runner = RLSForecastRunner(driver_cols=driver_cols, rmse_error=0.05, n_jobs=1)

    res, wm = pipe._run_section_level(
        "1", df_train, pl.DataFrame(), df_fcst, driver_cols, meta, hz, runner
    )
    assert res.height > 0
    assert wm.height == len(uids)
    assert {"yhat", "valuehat", "period_type"} <= set(res.columns)

    # Coherencia top-down exacta en el pronóstico (sección → tiendas → SKUs)
    fc = res.filter(pl.col("period_type") == "forecast_only")
    depth = pl.col("unique_id").str.count_matches(r"\|\|", literal=False)
    sec_fc = fc.filter(pl.col("unique_id") == "1").select(["ds", "yhat"]).sort("ds")
    stores_fc = fc.filter(depth == 1)
    sum_stores = stores_fc.group_by("ds").agg(pl.col("yhat").sum().alias("s")).sort("ds")
    j1 = sec_fc.join(sum_stores, on="ds")
    assert (j1["yhat"] - j1["s"]).abs().max() < 1e-6

    for store in ("1||00122", "1||00154"):
        st_fc = fc.filter(pl.col("unique_id") == store).select(["ds", "yhat"])
        sk_fc = fc.filter(pl.col("unique_id").str.starts_with(store + "||"))
        sk_sum = sk_fc.group_by("ds").agg(pl.col("yhat").sum().alias("s")).sort("ds")
        jj = st_fc.join(sk_sum, on="ds")
        assert (jj["yhat"] - jj["s"]).abs().max() < 1e-6

    # valuehat de hijos en forecast = último precio observado (>0)
    child_fc = fc.filter(depth >= 1)
    assert child_fc["valuehat"].drop_nulls().min() > 0

    # Descomposición causal: los 8 factores esperados
    assert len(pipe._decomposition_frames) == 1
    dec = pipe._decomposition_frames[0]
    assert dec["unique_id"].unique().to_list() == ["1"]
    expected = {
        "level",
        "trend",
        "seasonality",
        "edp",
        "discount",
        "feature_display",
        "volume",
        "price",
    }
    assert expected <= set(dec["factor"].unique().to_list())
# Columnas ef_* (descomposición por día) presentes SOLO en la sección
    sec_rows = res.filter(pl.col("unique_id") == "1")
    child_rows = res.filter(pl.col("unique_id").str.starts_with("1||"))
    expected_ef = {
        "ef_level",
        "ef_trend",
        "ef_seasonality",
        "ef_edp",
        "ef_discount",
        "ef_feature_display",
        "ef_volume",
        "ef_price",
        "ef_total",
        "ef_resid",
    }
    assert expected_ef <= set(sec_rows.columns)
    assert child_rows["ef_level"].null_count() == child_rows.height

        # Sin factor de corrección: ln(1+ŷ) = Σ efectos → residuo ≈ 0
    # (tolerancia 1e-2: yhat se redondea a entero dentro de _predict, lo que
    # introduce hasta ~1e-3 en escala log1p para yhat≈10³).
    check = sec_rows.filter(pl.col("period_type") == "forecast_only").with_columns(
        ((pl.col("yhat") + 1).log()).alias("ln1p_yhat")
    )
    assert (check["ln1p_yhat"] - check["ef_total"]).abs().max() < 1e-2
    assert check["ef_resid"].abs().max() < 1e-2