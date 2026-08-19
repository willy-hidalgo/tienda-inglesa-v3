"""Shared low-level utilities for the forecasting pipeline."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import polars as pl

logger = logging.getLogger(__name__)


@contextmanager
def stage_timer(label: str):
    """Log elapsed time for a pipeline stage."""
    t0 = time.perf_counter()
    logger.info("⏱ %s: iniciando…", label)
    try:
        yield
    finally:
        logger.info("⏱ %s: %.1fs", label, time.perf_counter() - t0)


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect using the best streaming API supported by the installed Polars."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        try:
            return lf.collect(streaming=True)
        except (TypeError, ValueError):
            return lf.collect()


def lf_columns(lf: pl.LazyFrame | pl.DataFrame) -> list[str]:
    """Return columns without forcing a LazyFrame to collect."""
    if isinstance(lf, pl.DataFrame):
        return list(lf.columns)
    try:
        return list(lf.collect_schema().names())
    except Exception:
        return list(lf.columns)
