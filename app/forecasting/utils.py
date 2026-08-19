"""Shared forecasting utilities."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import polars as pl

logger = logging.getLogger(__name__)


@contextmanager
def _stage_timer(label: str):
    """Instrumentación de tiempos por etapa (Fase 0). Uso: `with _stage_timer('x'): ...`"""
    t0 = time.perf_counter()
    logger.info("⏱ %s: iniciando…", label)
    try:
        yield
    finally:
        logger.info("⏱ %s: %.1fs", label, time.perf_counter() - t0)

def _collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect con engine streaming si está disponible (Polars 1.x / 0.20+)."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        try:
            return lf.collect(streaming=True)
        except (TypeError, ValueError):
            return lf.collect()

def _lf_columns(lf: pl.LazyFrame | pl.DataFrame) -> list[str]:
    if isinstance(lf, pl.DataFrame):
        return list(lf.columns)
    try:
        return list(lf.collect_schema().names())
    except Exception:
        return list(lf.columns)

