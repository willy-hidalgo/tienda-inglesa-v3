from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from app.forecasting.calendar import HolidayCalendar
from app.forecasting.features import CalendarFeatureBuilder


def test_holiday_calendar_returns_dates_for_requested_year():
    calendar = HolidayCalendar(years=[2025])
    dates = calendar.get_holidays()
    assert dates
    assert all(isinstance(value, dt.date) for value in dates)
    assert min(dates).year == 2025
    assert max(dates).year == 2025


def test_calendar_features_preserve_input_columns():
    calendar = HolidayCalendar(years=[2025])
    builder = CalendarFeatureBuilder(calendar)
    df = pl.DataFrame({"date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2)], "y": [10.0, 12.0]})
    result = builder.add_features(df)
    assert result.height == df.height
    assert set(df.columns).issubset(result.columns)


def test_calendar_features_reject_missing_date_column():
    calendar = HolidayCalendar(years=[2025])
    builder = CalendarFeatureBuilder(calendar)
    df = pl.DataFrame({"y": [10.0, 12.0]})
    with pytest.raises((KeyError, ValueError, pl.exceptions.ColumnNotFoundError)):
        builder.add_features(df)
