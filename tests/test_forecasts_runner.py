"""Tests optimización runner (Fase 1/6) sin rls_opt obligatorio."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from forecasts import StageTimer, densify_section_panel
import datetime as dt


def test_stage_timer_runs():
    t = StageTimer("unit")
    elapsed = t.stop()
    assert elapsed >= 0


def test_densify_n_series_times_days():
    df = pl.DataFrame(
        {
            "unique_id": ["a", "b"],
            "ds": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "y": [1.0, 2.0],
            "value": [10.0, 20.0],
        }
    )
    out = densify_section_panel(df, dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert out.height == 2 * 5
