from __future__ import annotations
import datetime as dt
import polars as pl
from app.forecasting.calendar import HolidayCalendar
from app.forecasting.features import CalendarFeatureBuilder


def test_holiday_calendar_year_scope():
    cal = HolidayCalendar({}, 2026)
    assert isinstance(cal, HolidayCalendar)


def test_feature_builder_preserves_rows():
    cal = HolidayCalendar({}, 2026)
    builder = CalendarFeatureBuilder(cal)
    df = pl.DataFrame({"ds": [dt.date(2026,1,1), dt.date(2026,1,2)], "y":[1.0,2.0]})
    out = builder.extract_drivers(df)
    assert out.height == 2
    assert "ds" in out.columns
