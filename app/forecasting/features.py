"""Calendar and retail feature engineering."""
from __future__ import annotations
import datetime as dt
import polars as pl
from app.forecasting.calendar import HolidayCalendar

class CalendarFeatureBuilder:
    _RAMP_HOLIDAYS = {"mothers_day", "fathers_day"}

    def __init__(self, calendar: HolidayCalendar):
        self._calendar = calendar

    @staticmethod
    def _calendar_dummies(df: pl.DataFrame) -> pl.DataFrame:
        # Polars dt.weekday(): Mon=1 … Sun=7. Se omite la categoría base (_1).
        _weekday_names = {
            2: "Tue",
            3: "Wed",
            4: "Thu",
            5: "Fri",
            6: "Sat",
            7: "Sun",
        }
        _month_names = {
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        df = df.with_columns(pl.col("ds").dt.weekday().alias("weekday"))
        df = df.to_dummies(columns=["weekday"])
        df = df.with_columns(pl.col("ds").dt.month().alias("month"))
        df = df.to_dummies(columns=["month"])
        rename = {}
        for n, name in _weekday_names.items():
            old = f"weekday_{n}"
            if old in df.columns:
                rename[old] = name
        for n, name in _month_names.items():
            old = f"month_{n}"
            if old in df.columns:
                rename[old] = name
        if rename:
            df = df.rename(rename)
        # Quitar categorías base (Mon / Jan) y columnas numéricas intermedias
        drop = [
            c for c in ("weekday_1", "month_1", "weekday", "month") if c in df.columns
        ]
        if drop:
            df = df.drop(drop)
        return df

    @staticmethod
    def _join_offset_dummies(
        df: pl.DataFrame,
        holiday: str,
        dates: list[dt.date],
        before: int,
        after: int,
    ) -> pl.DataFrame:
        """Un solo join por festividad (todas las columnas offset a la vez)."""
        rows = []
        cols = []
        for offset in range(-before, after + 1):
            col = f"{holiday}_{offset:+d}"
            cols.append(col)
            for f in dates:
                rows.append({"ds": f + dt.timedelta(days=offset), "col": col, "v": 1})
        if not rows:
            return df
        long = pl.DataFrame(rows).unique(subset=["ds", "col"])
        # pivot compatible con Polars 0.20+ / 1.x
        try:
            wide = long.pivot(
                on="col", index="ds", values="v", aggregate_function="max"
            )
        except TypeError:
            wide = long.pivot(
                values="v", index="ds", columns="col", aggregate_function="max"
            )
        for c in cols:
            if c not in wide.columns:
                wide = wide.with_columns(pl.lit(0).alias(c))
        wide = wide.with_columns(
            [pl.col(c).fill_null(0).cast(pl.Int8) for c in cols if c in wide.columns]
        )
        return df.join(wide, on="ds", how="left").with_columns(
            [pl.col(c).fill_null(0) for c in cols]
        )

    @staticmethod
    def _join_ramp(
        df: pl.DataFrame,
        holiday: str,
        dates: list[dt.date],
        before: int,
        after: int,
    ) -> pl.DataFrame:
        span = max(before, after, 1)
        rows = [
            {
                "ds": f + dt.timedelta(days=offset),
                "weight": 1.0 - abs(offset) / span,
            }
            for f in dates
            for offset in range(-before, after + 1)
        ]
        ramp_df = (
            pl.DataFrame(rows, schema={"ds": pl.Date, "weight": pl.Float64})
            .group_by("ds")
            .agg(pl.col("weight").max())
            .rename({"weight": holiday})
        )
        return df.join(ramp_df, on="ds", how="left").with_columns(
            pl.col(holiday).fill_null(0.0)
        )

    def _holiday_dummies(
        self, df: pl.DataFrame, req_columns: list[str] | None
    ) -> pl.DataFrame:
        for holiday, info in self._calendar.processed.items():
            if not info["active"] or not info["fecha"]:
                continue
            before = info["original_data"]["window_before_days"]
            after = info["original_data"]["window_after_days"]
            if holiday in self._RAMP_HOLIDAYS:
                df = self._join_ramp(df, holiday, info["fecha"], before, after)
            else:
                df = self._join_offset_dummies(
                    df, holiday, info["fecha"], before, after
                )
        if req_columns:
            missing = [c for c in req_columns if c not in df.columns]
            if missing:
                df = df.with_columns([pl.lit(0).alias(c) for c in missing])
        return df

    def extract_drivers(
        self, df: pl.DataFrame, req_columns: list[str] | None = None
    ) -> pl.DataFrame:
        """Features de calendario/festivos sobre fechas únicas → join (evita N×series)."""
        if df.height == 0:
            return df
        dates = df.select("ds").unique().sort("ds")
        feats = self._calendar_dummies(dates)
        # columnas de drivers (excluir ids de negocio)
        _exclude = {
            "ds",
            "y",
            "unique_id",
            "value",
            "valuehat",
            "conteo_sku",
            "sku_desc",
            "store_name",
            "seccion",
            "intercept",
            "asp",
            "edp",
            "discount",
        }
        if req_columns is None:
            req_columns = [c for c in feats.columns if c not in _exclude]
        feats = self._holiday_dummies(feats, req_columns)
        feat_cols = [c for c in feats.columns if c != "ds"]
        # asegurar columnas pedidas
        missing = [c for c in (req_columns or []) if c not in feats.columns]
        if missing:
            feats = feats.with_columns([pl.lit(0).alias(c) for c in missing])
            feat_cols = [c for c in feats.columns if c != "ds"]
        return df.join(feats.select(["ds"] + feat_cols), on="ds", how="left")
