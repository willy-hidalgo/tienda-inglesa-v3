"""Public forecasting API with lazy imports.

Keeping this package initializer lightweight allows pure NumPy components and
tests to run without importing Polars/Streamlit eagerly.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DataAggregator": ("app.forecasting.aggregation", "DataAggregator"),
    "HolidayCalendar": ("app.forecasting.calendar", "HolidayCalendar"),
    "ForecastConfig": ("app.forecasting.config", "ForecastConfig"),
    "CalendarFeatureBuilder": ("app.forecasting.features", "CalendarFeatureBuilder"),
    "compute_wmape": ("app.forecasting.metrics", "compute_wmape"),
    "densify_section_panel": ("app.forecasting.panel", "densify_section_panel"),
    "RLSForecastRunner": ("app.forecasting.runner", "RLSForecastRunner"),
    "RLSForecastPipeline": ("app.forecasting.pipeline", "RLSForecastPipeline"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
