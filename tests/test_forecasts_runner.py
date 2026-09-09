"""Functional contracts for the v11 forecasting runner."""
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

from app.forecasting.runner import RLSForecastRunner

try:
    from rls_opt import RecursiveLeastSquaresRegression
except ImportError:  # pragma: no cover
    RecursiveLeastSquaresRegression = None


def test_v11_runner_has_single_leaf_path():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(
        encoding="utf-8"
    )
    assert "def fast_leaf_forecasts(" in src
    assert "def derive_sku_store_forecasts(" not in src
    assert "def _apply_leaf_guardrail(" not in src
    assert "def _apply_ses_tuned(" not in src
    assert "def _select_model_wmape(" not in src


def test_v11_parent_model_cannot_be_none():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(
        encoding="utf-8"
    )
    assert '{"_parent": "none"' not in src
    assert 'pl.lit("none")' not in src
    assert "Leaf parent invariant violated" in src
    assert '("store", "section")' in src


def test_v11_driver_strength_is_full():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(
        encoding="utf-8"
    )
    assert "LEAF_PARENT_DRIVER_STRENGTH" in src
    assert "v11 requires LEAF_PARENT_DRIVER_STRENGTH=1.0" in src
    assert 'pl.lit(1.0).alias("_effective_strength_y")' in src
    assert 'pl.lit(1.0).alias("_effective_strength_v")' in src


@pytest.mark.skipif(
    RecursiveLeastSquaresRegression is None, reason="rls_opt no disponible"
)
def test_fit_and_predict_sections_only_fits_given_ids():
    """Parent RLS must ignore ids not explicitly requested."""
    runner = RLSForecastRunner(
        driver_cols=["intercept", "driver_a"], rmse_error=0.2
    )
    n = 30
    dates = [
        dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)
    ]
    rng = np.random.default_rng(1)
    train = pl.concat(
        [
            pl.DataFrame(
                {
                    "unique_id": [uid] * n,
                    "ds": dates,
                    "y": (10 + rng.normal(0, 1, n)).tolist(),
                    "value": (100 + rng.normal(0, 5, n)).tolist(),
                    "intercept": [1] * n,
                    "driver_a": rng.uniform(0, 1, n).tolist(),
                }
            )
            for uid in ("1", "1||T:A||S:X")
        ],
        how="vertical",
    )
    targets = {"in_sample": train}
    res_df, coefs = runner.fit_and_predict_sections(
        train, targets, ["1"], desc="test"
    )
    assert set(coefs.keys()) == {"1"}
    assert res_df.height > 0
    assert set(res_df["unique_id"].unique().to_list()) == {"1"}
