"""v12.5 read-only rolling backtest: explicit calendar + causal price-state drivers.

Why this diagnostic exists
--------------------------
The factorized autoregressive family (lag28/same-weekday/annual variants) did
not generalize to the production OOS, especially section 23 over Dec/Jan.  This
module therefore keeps the production v12.5 SKU-total selector as the baseline
and tests a small, interpretable residual layer driven only by information
available before each target block:

* month-of-year regime;
* configured retail/holiday windows from settings.HOLIDAYS;
* commercial category shrinkage (if category exists in selected.parquet);
* a causal SKU price-state proxy: recent 28d ASP / recent 84d ASP.

No OOS actual is used to fit/select anything.  For every pseudo-OOS, factor
parameters are fit from older CLOSED 28-day blocks.  Production OOS is used
once as a final holdout after a candidate is selected from PRE-OOS history.

The layer is deliberately multiplicative and auditable.  It is NOT a new
SKU×store model and does not change occurrence/store-share.  Leaf evaluation
uses the exact production v12.5 28d×84d share returned by _block_candidate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path
from statistics import median

import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings
from app.forecasting.calendar import HolidayCalendar
from app.forecasting.features import CalendarFeatureBuilder
from app.forecasting.leaf_v12 import (
    _block_candidate,
    _finite_nonnegative,
    _leaf_keys,
    _selected_sku_total_forecast,
)

EPS = 1e-12
CANDIDATES = (
    "month_sec_full",
    "month_cat_full",
    "calendar_sec_full",
    "calendar_cat_full",
    "calendar_sec_shape",
    "calendar_cat_shape",
    "calendar_sec_price_full",
    "calendar_cat_price_full",
)


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def _schema(path: Path) -> set[str]:
    return set(pl.scan_parquet(str(path)).collect_schema().names())


def _date_col(names: set[str]) -> str:
    for c in ("SALES_DAY", "SALES_DATE", getattr(settings, "DATE_COLUMN", "SALES_DAY")):
        if c in names:
            return c
    raise ValueError("No encuentro columna de fecha en selected.parquet")


def _category_col(names: set[str]) -> str | None:
    # Description is preferable for audit output; codes are valid fallbacks.
    for c in (
        "DESC_CATEGORIA", "CATEGORIA", "DESC_FAMILIA", "FAMILIA",
        "DESC_SUBCATEGORIA", "SUBCATEGORIA",
    ):
        if c in names:
            return c
    return None


def _pct(x: float) -> str:
    return "   nan" if not math.isfinite(x) else f"{100.0*x:7.2f}%"


def _pp(x: float) -> str:
    return "   nan" if not math.isfinite(x) else f"{100.0*x:+6.2f}pp"


def _oos_bounds(forecast_path: Path) -> dict[str, tuple[dt.date, dt.date]]:
    names = _schema(forecast_path)
    sec_expr = (
        pl.col("seccion").cast(pl.Utf8)
        if "seccion" in names
        else pl.col("unique_id").str.extract(r"^([^|]+)", 1)
    )
    df = _collect(
        pl.scan_parquet(str(forecast_path))
        .filter(
            (pl.col("period_type") == "out_sample")
            & pl.col("unique_id").str.contains(r"\|\|S:")
        )
        .select(sec_expr.alias("SECCION"), pl.col("ds").cast(pl.Date))
        .group_by("SECCION")
        .agg(pl.col("ds").min().alias("d0"), pl.col("ds").max().alias("d1"))
        .sort("SECCION")
    )
    return {str(r["SECCION"]): (r["d0"], r["d1"]) for r in df.iter_rows(named=True)}


def _history_from_selected(selected_path: Path, sec: str):
    names = _schema(selected_path)
    dcol = _date_col(names)
    cat_col = _category_col(names)

    base_cols = ["SECCION", "STORE_ID", "SKU_ID", dcol, "SLS_QTY", "SLS_VAL"]
    hist = _collect(
        pl.scan_parquet(str(selected_path))
        .select(*base_cols)
        .with_columns(
            pl.col("SECCION").cast(pl.Utf8),
            pl.col("STORE_ID").cast(pl.Utf8),
            pl.col("SKU_ID").cast(pl.Utf8),
            pl.col(dcol).cast(pl.Date).alias("ds"),
            pl.col("SLS_QTY").cast(pl.Float64).fill_null(0.0).alias("y"),
            pl.col("SLS_VAL").cast(pl.Float64).fill_null(0.0).alias("value"),
        )
        .filter(pl.col("SECCION") == sec)
        .with_columns(
            (
                pl.col("SECCION") + pl.lit("||T:") + pl.col("STORE_ID")
                + pl.lit("||S:") + pl.col("SKU_ID")
            ).alias("unique_id")
        )
        .group_by(["unique_id", "ds"])
        .agg(pl.col("y").sum().alias("y"), pl.col("value").sum().alias("value"))
        .sort(["unique_id", "ds"])
    )
    prepared = _leaf_keys(hist).with_columns(
        _finite_nonnegative("y").alias("_v12_y"),
        _finite_nonnegative("value").alias("_v12_v"),
        pl.col("ds").dt.weekday().cast(pl.Int8).alias("_v12_dow"),
    )
    sku_daily = (
        prepared.group_by(["_v12_sku", "ds"])
        .agg(
            pl.col("_v12_y").sum().alias("_sku_day_y"),
            pl.col("_v12_v").sum().alias("_sku_day_v"),
        )
        .sort(["_v12_sku", "ds"])
    )
    sku_first = prepared.group_by("_v12_sku").agg(
        pl.col("ds").min().alias("_sku_first_ds")
    )
    uid_first = prepared.group_by(["unique_id", "_v12_sku"]).agg(
        pl.col("ds").min().alias("_uid_first_ds")
    )
    ids = prepared.select("unique_id", "_v12_sku", "_v12_store_uid").unique()

    if cat_col:
        meta = _collect(
            pl.scan_parquet(str(selected_path))
            .select("SECCION", "SKU_ID", cat_col)
            .with_columns(
                pl.col("SECCION").cast(pl.Utf8),
                pl.col("SKU_ID").cast(pl.Utf8).alias("_v12_sku"),
                pl.col(cat_col).cast(pl.Utf8).fill_null("__UNKNOWN__").alias("_cat"),
            )
            .filter(pl.col("SECCION") == sec)
            .group_by("_v12_sku")
            .agg(pl.col("_cat").first().alias("_cat"))
        )
    else:
        meta = sku_first.select("_v12_sku").with_columns(pl.lit("__SECTION__").alias("_cat"))
    return prepared, sku_daily, sku_first, uid_first, ids, meta, (cat_col or "<none>")


def _incumbent_rows(forecast_path: Path, sec: str) -> pl.DataFrame:
    names = _schema(forecast_path)
    sec_expr = (
        pl.col("seccion").cast(pl.Utf8)
        if "seccion" in names
        else pl.col("unique_id").str.extract(r"^([^|]+)", 1)
    )
    cols = ["unique_id", "ds", "period_type", "y", "value", "yhat", "valuehat"]
    extra: list[str] = []
    if "v11_yhat_raw_before_v12" in names:
        extra.append("v11_yhat_raw_before_v12")
    if "v11_valuehat_raw_before_v12" in names:
        extra.append("v11_valuehat_raw_before_v12")
    df = _collect(
        pl.scan_parquet(str(forecast_path))
        .filter(pl.col("unique_id").str.contains(r"\|\|S:"))
        .select(*[pl.col(c) for c in cols + extra], sec_expr.alias("SECCION"))
        .filter(pl.col("SECCION") == sec)
        .with_columns(pl.col("ds").cast(pl.Date))
    )
    yexpr = (
        pl.coalesce([
            pl.col("v11_yhat_raw_before_v12").cast(pl.Float64).round(0),
            pl.col("yhat").cast(pl.Float64),
        ])
        if "v11_yhat_raw_before_v12" in df.columns
        else pl.col("yhat").cast(pl.Float64)
    )
    vexpr = (
        pl.coalesce([
            pl.col("v11_valuehat_raw_before_v12").cast(pl.Float64).round(2),
            pl.col("valuehat").cast(pl.Float64),
        ])
        if "v11_valuehat_raw_before_v12" in df.columns
        else pl.col("valuehat").cast(pl.Float64)
    )
    return df.with_columns(yexpr.alias("yhat"), vexpr.alias("valuehat")).select(
        "unique_id", "ds", "period_type", "y", "value", "yhat", "valuehat"
    )


def _calendar_table(lo: dt.date, hi: dt.date) -> tuple[pl.DataFrame, list[str]]:
    dates = pl.DataFrame({"ds": pl.date_range(lo, hi, interval="1d", eager=True)}).with_columns(
        pl.col("ds").cast(pl.Date)
    )
    cal = HolidayCalendar(getattr(settings, "HOLIDAYS", {}), max(lo.year, hi.year))
    feats = CalendarFeatureBuilder(cal).extract_drivers(dates)
    out = dates.with_columns(pl.col("ds").dt.month().cast(pl.Int8).alias("_month"))
    event_cols: list[str] = []
    for holiday in getattr(settings, "HOLIDAYS", {}).keys():
        hcols = [c for c in feats.columns if c == holiday or c.startswith(f"{holiday}_")]
        if not hcols:
            continue
        cname = f"_ev_{holiday}"
        event_cols.append(cname)
        out = out.join(
            feats.select(
                "ds",
                pl.max_horizontal(*[pl.col(c).cast(pl.Float64) for c in hcols])
                .fill_null(0.0)
                .alias(cname),
            ),
            on="ds",
            how="left",
        )
    if event_cols:
        out = out.with_columns([pl.col(c).fill_null(0.0) for c in event_cols])
    return out, event_cols


def _price_state(sku_daily: pl.DataFrame, origin: dt.date) -> pl.DataFrame:
    r28 = origin - dt.timedelta(days=28)
    r84 = origin - dt.timedelta(days=84)
    x28 = (
        sku_daily.filter((pl.col("ds") >= pl.lit(r28)) & (pl.col("ds") < pl.lit(origin)))
        .group_by("_v12_sku")
        .agg(
            pl.col("_sku_day_y").sum().alias("_q28"),
            pl.col("_sku_day_v").sum().alias("_v28"),
        )
    )
    x84 = (
        sku_daily.filter((pl.col("ds") >= pl.lit(r84)) & (pl.col("ds") < pl.lit(origin)))
        .group_by("_v12_sku")
        .agg(
            pl.col("_sku_day_y").sum().alias("_q84"),
            pl.col("_sku_day_v").sum().alias("_v84"),
        )
    )
    return (
        x84.join(x28, on="_v12_sku", how="left")
        .with_columns(
            pl.when(pl.col("_q28").fill_null(0.0) > EPS)
            .then(pl.col("_v28").fill_null(0.0) / pl.col("_q28"))
            .otherwise(None).alias("_asp28"),
            pl.when(pl.col("_q84").fill_null(0.0) > EPS)
            .then(pl.col("_v84").fill_null(0.0) / pl.col("_q84"))
            .otherwise(None).alias("_asp84"),
        )
        .with_columns(
            pl.when((pl.col("_asp28") > EPS) & (pl.col("_asp84") > EPS))
            .then((pl.col("_asp28") / pl.col("_asp84")).log().clip(-0.70, 0.70))
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("_price_state")
        )
        .select("_v12_sku", "_price_state")
    )


def _baseline_block(
    prepared: pl.DataFrame,
    sku_daily: pl.DataFrame,
    sku_first: pl.DataFrame,
    ids: pl.DataFrame,
    meta: pl.DataFrame,
    incumbent: pl.DataFrame,
    start: dt.date,
    end: dt.date,
    block_id: int,
) -> pl.DataFrame:
    sku_ids = ids.select("_v12_sku").unique()
    fc = _selected_sku_total_forecast(
        prepared_history=prepared,
        sku_daily_all=sku_daily,
        sku_first_all=sku_first,
        sku_ids=sku_ids,
        incumbent_rows=incumbent,
        origin=start,
        end=end,
    ).select(
        "_v12_sku", "ds",
        pl.col("v12_sku_forecast_y").cast(pl.Float64).fill_null(0.0).alias("base_y"),
        pl.col("v12_sku_forecast_value").cast(pl.Float64).fill_null(0.0).alias("base_v"),
    )
    actual = sku_daily.filter(
        (pl.col("ds") >= pl.lit(start)) & (pl.col("ds") <= pl.lit(end))
    ).select(
        "_v12_sku", "ds",
        pl.col("_sku_day_y").cast(pl.Float64).alias("actual_y"),
        pl.col("_sku_day_v").cast(pl.Float64).alias("actual_v"),
    )
    cal, _ = _calendar_table(start, end)
    return (
        fc.join(actual, on=["_v12_sku", "ds"], how="left")
        .join(meta, on="_v12_sku", how="left")
        .join(_price_state(sku_daily, start), on="_v12_sku", how="left")
        .join(cal, on="ds", how="left")
        .with_columns(
            pl.col("actual_y").fill_null(0.0),
            pl.col("actual_v").fill_null(0.0),
            pl.col("_cat").fill_null("__UNKNOWN__"),
            pl.col("_price_state").fill_null(0.0),
            pl.lit(block_id).cast(pl.Int16).alias("block_id"),
        )
    )


def _fit_month_tables(hist: pl.DataFrame, suffix: str):
    actual = f"actual_{suffix}"
    base = f"base_{suffix}"
    g = hist.filter((pl.col(actual) > 0) & (pl.col(base) >= 0))
    sec = (
        g.group_by("_month")
        .agg(
            pl.col(actual).sum().alias("num"),
            pl.col(base).sum().alias("den"),
            pl.col("ds").n_unique().cast(pl.Float64).alias("n"),
        )
        .with_columns(
            pl.when(pl.col("den") > EPS)
            .then((pl.col("num") / pl.col("den")).clip(0.50, 2.00))
            .otherwise(1.0).alias("raw")
        )
        .with_columns((pl.col("n") / (pl.col("n") + 7.0)).clip(0.0, 1.0).alias("w"))
        .with_columns((pl.col("raw").log() * pl.col("w")).exp().alias("month_sec"))
        .select("_month", "month_sec")
    )
    cat = (
        g.group_by(["_cat", "_month"])
        .agg(
            pl.col(actual).sum().alias("num"),
            pl.col(base).sum().alias("den"),
            pl.col("ds").n_unique().cast(pl.Float64).alias("n"),
        )
        .with_columns(
            pl.when(pl.col("den") > EPS)
            .then((pl.col("num") / pl.col("den")).clip(0.50, 2.00))
            .otherwise(1.0).alias("raw")
        )
        .join(sec, on="_month", how="left")
        .with_columns(
            pl.col("month_sec").fill_null(1.0),
            (pl.col("n") / (pl.col("n") + 14.0)).clip(0.0, 1.0).alias("w"),
        )
        .with_columns(
            (
                pl.col("month_sec").log() * (1.0 - pl.col("w"))
                + pl.col("raw").log() * pl.col("w")
            ).exp().alias("month_cat")
        )
        .select("_cat", "_month", "month_cat")
    )
    return sec, cat


def _attach_month(df: pl.DataFrame, sec: pl.DataFrame, cat: pl.DataFrame, suffix: str) -> pl.DataFrame:
    return (
        df.join(sec.rename({"month_sec": f"_msec_{suffix}"}), on="_month", how="left")
        .join(cat.rename({"month_cat": f"_mcat_{suffix}"}), on=["_cat", "_month"], how="left")
        .with_columns(
            pl.col(f"_msec_{suffix}").fill_null(1.0),
            pl.col(f"_mcat_{suffix}").fill_null(pl.col(f"_msec_{suffix}")),
        )
    )


def _fit_event_factors(hist_with_month: pl.DataFrame, suffix: str, event_cols: list[str]):
    actual = f"actual_{suffix}"
    base = f"base_{suffix}"
    msec = f"_msec_{suffix}"
    mcat = f"_mcat_{suffix}"
    sec_f: dict[str, float] = {}
    cat_f: dict[tuple[str, str], float] = {}

    cats = hist_with_month.select("_cat").unique().get_column("_cat").to_list()
    for ev in event_cols:
        x = hist_with_month.filter((pl.col(ev) > 0) & (pl.col(actual) > 0))
        if x.height == 0:
            sec_f[ev] = 1.0
            continue
        s = x.select(
            (pl.col(actual) * pl.col(ev)).sum().alias("num"),
            (pl.col(base) * pl.col(msec) * pl.col(ev)).sum().alias("den"),
            pl.col("ds").n_unique().cast(pl.Float64).alias("n"),
        ).row(0, named=True)
        den = float(s["den"] or 0.0); num = float(s["num"] or 0.0); n = float(s["n"] or 0.0)
        raw = min(2.50, max(0.50, num / den)) if den > EPS else 1.0
        w = n / (n + 2.0) if n > 0 else 0.0
        sf = math.exp(w * math.log(max(EPS, raw)))
        sec_f[ev] = sf

        cg = (
            x.group_by("_cat")
            .agg(
                (pl.col(actual) * pl.col(ev)).sum().alias("num"),
                (pl.col(base) * pl.col(mcat) * pl.col(ev)).sum().alias("den"),
                pl.col("ds").n_unique().cast(pl.Float64).alias("n"),
            )
        )
        for r in cg.iter_rows(named=True):
            den_c = float(r["den"] or 0.0); num_c = float(r["num"] or 0.0); n_c = float(r["n"] or 0.0)
            raw_c = min(2.50, max(0.50, num_c / den_c)) if den_c > EPS else sf
            wc = n_c / (n_c + 4.0) if n_c > 0 else 0.0
            cf = math.exp((1.0 - wc) * math.log(max(EPS, sf)) + wc * math.log(max(EPS, raw_c)))
            cat_f[(str(r["_cat"]), ev)] = cf
    # Ensure every category can fall back without special cases later.
    for cat in cats:
        for ev in event_cols:
            cat_f.setdefault((str(cat), ev), sec_f.get(ev, 1.0))
    return sec_f, cat_f


def _event_multiplier_tables(
    dates: pl.DataFrame,
    categories: list[str],
    event_cols: list[str],
    sec_f: dict[str, float],
    cat_f: dict[tuple[str, str], float],
    suffix: str,
):
    date_rows = dates.select("ds", *event_cols).iter_rows(named=True)
    sec_rows = []
    cached_date_rows = []
    for r in date_rows:
        cached_date_rows.append(r)
        logm = 0.0
        for ev in event_cols:
            w = float(r.get(ev) or 0.0)
            if w > 0:
                logm += w * math.log(max(EPS, sec_f.get(ev, 1.0)))
        sec_rows.append({"ds": r["ds"], f"_esec_{suffix}": min(2.50, max(0.50, math.exp(logm)))})
    cat_rows = []
    for cat in categories:
        for r in cached_date_rows:
            logm = 0.0
            for ev in event_cols:
                w = float(r.get(ev) or 0.0)
                if w > 0:
                    logm += w * math.log(max(EPS, cat_f.get((str(cat), ev), sec_f.get(ev, 1.0))))
            cat_rows.append({"_cat": str(cat), "ds": r["ds"], f"_ecat_{suffix}": min(2.50, max(0.50, math.exp(logm)))})
    sec_df = pl.DataFrame(sec_rows) if sec_rows else dates.select("ds").with_columns(pl.lit(1.0).alias(f"_esec_{suffix}"))
    cat_df = pl.DataFrame(cat_rows) if cat_rows else pl.DataFrame({"_cat": [], "ds": [], f"_ecat_{suffix}": []})
    return sec_df, cat_df


def _fit_price_beta(hist_cal: pl.DataFrame, suffix: str):
    actual = f"actual_{suffix}"
    base = f"base_{suffix}"
    cal = f"_calcat_{suffix}"
    b = (
        hist_cal.group_by(["block_id", "_v12_sku", "_cat", "_price_state"])
        .agg(
            pl.col(actual).sum().alias("a"),
            (pl.col(base) * pl.col(cal)).sum().alias("p"),
        )
        .filter((pl.col("a") > EPS) & (pl.col("p") > EPS) & (pl.col("_price_state").abs() > 0.01))
        .with_columns(
            (pl.col("a") / pl.col("p")).log().clip(-1.5, 1.5).alias("z"),
            pl.col("a").sqrt().alias("w"),
        )
    )
    if b.height == 0:
        return 0.0, {}
    den = b.select((pl.col("w") * pl.col("_price_state") * pl.col("_price_state")).sum().alias("d")).item()
    num = b.select((pl.col("w") * pl.col("_price_state") * pl.col("z")).sum().alias("n")).item()
    beta_sec = float(num or 0.0) / max(EPS, float(den or 0.0))
    if suffix == "y":
        beta_sec = min(0.0, max(-3.0, beta_sec))
    else:
        beta_sec = min(2.0, max(-1.0, beta_sec))

    cg = b.group_by("_cat").agg(
        (pl.col("w") * pl.col("_price_state") * pl.col("z")).sum().alias("nume"),
        (pl.col("w") * pl.col("_price_state") * pl.col("_price_state")).sum().alias("dene"),
        pl.len().cast(pl.Float64).alias("n"),
    )
    out: dict[str, float] = {}
    for r in cg.iter_rows(named=True):
        raw = float(r["nume"] or 0.0) / max(EPS, float(r["dene"] or 0.0))
        if suffix == "y":
            raw = min(0.0, max(-3.0, raw))
        else:
            raw = min(2.0, max(-1.0, raw))
        n = float(r["n"] or 0.0)
        w = n / (n + 50.0)
        out[str(r["_cat"])] = (1.0 - w) * beta_sec + w * raw
    return beta_sec, out


def _apply_driver_candidates(train_hist: pl.DataFrame, target: pl.DataFrame, suffix: str, event_cols: list[str]) -> pl.DataFrame:
    sec_month, cat_month = _fit_month_tables(train_hist, suffix)
    hist_m = _attach_month(train_hist, sec_month, cat_month, suffix)
    sec_f, cat_f = _fit_event_factors(hist_m, suffix, event_cols)

    cats = target.select("_cat").unique().get_column("_cat").to_list()
    date_flags = target.select("ds", *event_cols).unique().sort("ds")
    esec, ecat = _event_multiplier_tables(date_flags, [str(c) for c in cats], event_cols, sec_f, cat_f, suffix)

    # Build calendar multipliers for historical rows too, because price beta is
    # fitted on the remaining level residual after calendar adjustment.
    hist_dates = hist_m.select("ds", *event_cols).unique().sort("ds")
    hist_cats = hist_m.select("_cat").unique().get_column("_cat").to_list()
    hesec, hecat = _event_multiplier_tables(hist_dates, [str(c) for c in hist_cats], event_cols, sec_f, cat_f, suffix)
    hist_cal = (
        hist_m.join(hesec, on="ds", how="left")
        .join(hecat, on=["_cat", "ds"], how="left")
        .with_columns(
            (pl.col(f"_msec_{suffix}") * pl.col(f"_esec_{suffix}").fill_null(1.0)).clip(0.40, 2.50).alias(f"_calsec_{suffix}"),
            (pl.col(f"_mcat_{suffix}") * pl.col(f"_ecat_{suffix}").fill_null(pl.col(f"_esec_{suffix}")).fill_null(1.0)).clip(0.40, 2.50).alias(f"_calcat_{suffix}"),
        )
    )
    beta_sec, beta_cat = _fit_price_beta(hist_cal, suffix)
    beta_df = pl.DataFrame(
        {"_cat": list(beta_cat.keys()), f"_beta_{suffix}": list(beta_cat.values())}
    ) if beta_cat else pl.DataFrame({"_cat": [], f"_beta_{suffix}": []})

    x = (
        _attach_month(target, sec_month, cat_month, suffix)
        .join(esec, on="ds", how="left")
        .join(ecat, on=["_cat", "ds"], how="left")
        .join(beta_df, on="_cat", how="left")
        .with_columns(
            pl.col(f"_esec_{suffix}").fill_null(1.0),
            pl.col(f"_ecat_{suffix}").fill_null(pl.col(f"_esec_{suffix}")).fill_null(1.0),
            pl.col(f"_beta_{suffix}").fill_null(beta_sec),
        )
        .with_columns(
            (pl.col(f"_msec_{suffix}") * pl.col(f"_esec_{suffix}")).clip(0.40, 2.50).alias(f"_cal_sec_{suffix}"),
            (pl.col(f"_mcat_{suffix}") * pl.col(f"_ecat_{suffix}")).clip(0.40, 2.50).alias(f"_cal_cat_{suffix}"),
            (pl.col(f"_beta_{suffix}") * pl.col("_price_state")).exp().clip(0.70, 1.40).alias(f"_price_mult_{suffix}"),
        )
    )
    base = f"base_{suffix}"
    x = x.with_columns(
        (pl.col(base) * pl.col(f"_msec_{suffix}")).clip(lower_bound=0.0).alias(f"fc_month_sec_full_{suffix}"),
        (pl.col(base) * pl.col(f"_mcat_{suffix}")).clip(lower_bound=0.0).alias(f"fc_month_cat_full_{suffix}"),
        (pl.col(base) * pl.col(f"_cal_sec_{suffix}")).clip(lower_bound=0.0).alias(f"fc_calendar_sec_full_{suffix}"),
        (pl.col(base) * pl.col(f"_cal_cat_{suffix}")).clip(lower_bound=0.0).alias(f"fc_calendar_cat_full_{suffix}"),
        (pl.col(base) * pl.col(f"_cal_sec_{suffix}") * pl.col(f"_price_mult_{suffix}")).clip(lower_bound=0.0).alias(f"fc_calendar_sec_price_full_{suffix}"),
        (pl.col(base) * pl.col(f"_cal_cat_{suffix}") * pl.col(f"_price_mult_{suffix}")).clip(lower_bound=0.0).alias(f"fc_calendar_cat_price_full_{suffix}"),
    )

    # Shape-only versions preserve the production SKU 28-day total exactly.
    for scope in ("sec", "cat"):
        raw = f"fc_calendar_{scope}_full_{suffix}"
        sum_raw = f"_sum_raw_{scope}_{suffix}"
        sum_base = f"_sum_base_{scope}_{suffix}"
        x = x.with_columns(
            pl.col(raw).sum().over("_v12_sku").alias(sum_raw),
            pl.col(base).sum().over("_v12_sku").alias(sum_base),
        ).with_columns(
            pl.when(pl.col(sum_raw) > EPS)
            .then(pl.col(raw) * pl.col(sum_base) / pl.col(sum_raw))
            .otherwise(pl.col(base))
            .clip(lower_bound=0.0)
            .alias(f"fc_calendar_{scope}_shape_{suffix}")
        ).drop(sum_raw, sum_base)
    return x


def _eval_sku(daily: pl.DataFrame, fc_col: str, suffix: str):
    actual = f"actual_{suffix}"
    r = daily.select(
        pl.when(pl.col(actual) > 0).then(pl.col(actual)).otherwise(0.0).sum().alias("den"),
        pl.when(pl.col(actual) > 0).then((pl.col(fc_col) - pl.col(actual)).abs()).otherwise(0.0).sum().alias("ae"),
        pl.when(pl.col(actual) > 0).then(pl.col(fc_col) - pl.col(actual)).otherwise(0.0).sum().alias("se"),
    ).row(0, named=True)
    den = float(r["den"] or 0.0); ae = float(r["ae"] or 0.0); se = float(r["se"] or 0.0)
    return ae / max(EPS, den), se / max(EPS, den), ae, den


def _eval_leaf(
    share_rows: pl.DataFrame,
    daily: pl.DataFrame,
    fc_col: str,
    prepared: pl.DataFrame,
    start: dt.date,
    end: dt.date,
    suffix: str,
    active_min: int,
):
    share_col = "v12_store_share_y" if suffix == "y" else "v12_store_share_value"
    actual_col = "_v12_y" if suffix == "y" else "_v12_v"
    sku_fc = daily.select("_v12_sku", "ds", pl.col(fc_col).alias("sku_fc"))
    fc = share_rows.select("unique_id", "_v12_sku", "ds", share_col).join(
        sku_fc, on=["_v12_sku", "ds"], how="left"
    ).with_columns(
        (pl.col("sku_fc").fill_null(0.0) * pl.col(share_col).fill_null(0.0)).alias("fc")
    )
    actual = prepared.filter(
        (pl.col("ds") >= pl.lit(start)) & (pl.col("ds") <= pl.lit(end))
    ).select("unique_id", "ds", pl.col(actual_col).alias("actual"))
    x = fc.join(actual, on=["unique_id", "ds"], how="left").with_columns(pl.col("actual").fill_null(0.0))
    eligible = (
        x.group_by("unique_id")
        .agg((pl.col("actual") > 0).sum().alias("n"))
        .filter(pl.col("n") >= active_min)
        .select("unique_id")
    )
    x = x.join(eligible, on="unique_id", how="semi")
    r = x.select(
        pl.when(pl.col("actual") > 0).then(pl.col("actual")).otherwise(0.0).sum().alias("den"),
        pl.when(pl.col("actual") > 0).then((pl.col("fc") - pl.col("actual")).abs()).otherwise(0.0).sum().alias("ae"),
        pl.when(pl.col("actual") > 0).then(pl.col("fc") - pl.col("actual")).otherwise(0.0).sum().alias("se"),
    ).row(0, named=True)
    den = float(r["den"] or 0.0); ae = float(r["ae"] or 0.0); se = float(r["se"] or 0.0)
    return ae / max(EPS, den), se / max(EPS, den), ae, den, eligible.height


def _summary(rows: list[dict], sec: str, suffix: str, metric: str):
    base = {
        r["eval_rank"]: r
        for r in rows
        if r["sec"] == sec and r["suffix"] == suffix and r["candidate"] == "current_selector"
    }
    out = []
    for cand in CANDIDATES:
        rs = [r for r in rows if r["sec"] == sec and r["suffix"] == suffix and r["candidate"] == cand]
        if not rs:
            continue
        ae = sum(float(r[f"{metric}_ae"]) for r in rs); den = sum(float(r[f"{metric}_den"]) for r in rs)
        bae = sum(float(base[r["eval_rank"]][f"{metric}_ae"]) for r in rs); bden = sum(float(base[r["eval_rank"]][f"{metric}_den"]) for r in rs)
        pooled = ae / max(EPS, den); bpooled = bae / max(EPS, bden)
        gains = [base[r["eval_rank"]][f"{metric}_wmape"] - r[f"{metric}_wmape"] for r in rs]
        recent = [g for r, g in zip(rs, gains) if r["eval_rank"] <= 4]
        out.append({
            "candidate": cand,
            "pooled": pooled,
            "gain": bpooled - pooled,
            "median": median(gains),
            "recent4": sum(recent) / len(recent) if recent else float("nan"),
            "win": sum(g > 0 for g in gains) / len(gains),
            "worst": min(gains),
        })
    return sorted(out, key=lambda r: (-r["gain"], r["pooled"]))


def _print_summary(title: str, rows: list[dict]):
    print(f"\n{title}")
    print("  candidate                      pooled    gain    median  recent4  win%   worst")
    for r in rows:
        print(
            f"  {r['candidate']:30s} {_pct(r['pooled'])} {_pp(r['gain'])} {_pp(r['median'])} "
            f"{_pp(r['recent4'])} {100*r['win']:5.0f}% {_pp(r['worst'])}"
        )


def _robust_pick(summary: list[dict]) -> dict | None:
    eligible = [
        r for r in summary
        if r["gain"] > 0 and r["recent4"] > 0 and r["win"] >= 0.67 and r["worst"] >= -0.02
    ]
    return max(eligible, key=lambda r: (r["gain"], r["recent4"], r["win"])) if eligible else None


def _run_section(
    selected_path: Path,
    forecast_path: Path,
    sec: str,
    oos_origin: dt.date,
    blocks: int,
    history_max: int,
    active_min: int,
    out_dir: Path,
):
    prepared, sku_daily, sku_first, uid_first, ids, meta, cat_name = _history_from_selected(selected_path, sec)
    incumbent = _incumbent_rows(forecast_path, sec)
    block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))
    max_block = blocks + history_max
    daily_by_id: dict[int, pl.DataFrame] = {}

    # Calendar columns are stable over the whole diagnostic date range.
    all_lo = oos_origin - dt.timedelta(days=block_days * max_block)
    all_hi = oos_origin + dt.timedelta(days=block_days - 1)
    _, event_cols = _calendar_table(all_lo, all_hi)

    print(f"\n=== SECCIÓN {sec} | calendar/price rolling PRE-OOS={blocks} | history={history_max} | categoria={cat_name} ===")
    print(f"event drivers={len(event_cols)}: {', '.join(c.replace('_ev_','') for c in event_cols)}")
    print(f"precomputando {max_block} bloques current SKU-total ...")
    for bid in range(1, max_block + 1):
        start = oos_origin - dt.timedelta(days=block_days * bid)
        end = start + dt.timedelta(days=block_days - 1)
        daily = _baseline_block(prepared, sku_daily, sku_first, ids, meta, incumbent, start, end, bid)
        daily_by_id[bid] = daily
        print(f"  block {bid:02d}/{max_block}: {start}→{end} SKU={daily.select('_v12_sku').n_unique()}")

    rows: list[dict] = []
    for eval_rank in range(blocks, 0, -1):
        start = oos_origin - dt.timedelta(days=block_days * eval_rank)
        end = start + dt.timedelta(days=block_days - 1)
        target = daily_by_id[eval_rank]
        hist_parts = [daily_by_id[h] for h in range(eval_rank + 1, min(max_block, eval_rank + history_max) + 1)]
        if not hist_parts:
            continue
        hist = pl.concat(hist_parts, how="vertical_relaxed")
        share_rows = _block_candidate(
            prepared, sku_daily, sku_first, uid_first, ids, incumbent, start, end
        ).select("unique_id", "_v12_sku", "ds", "v12_store_share_y", "v12_store_share_value")

        for suffix in ("y", "v"):
            x = _apply_driver_candidates(hist, target, suffix, event_cols)
            bsku = _eval_sku(x, f"base_{suffix}", suffix)
            bleaf = _eval_leaf(share_rows, x, f"base_{suffix}", prepared, start, end, suffix, active_min)
            rows.append({
                "sec": sec, "eval_rank": eval_rank, "start": start, "end": end,
                "suffix": suffix, "candidate": "current_selector",
                "sku_wmape": bsku[0], "sku_bias": bsku[1], "sku_ae": bsku[2], "sku_den": bsku[3],
                "leaf_wmape": bleaf[0], "leaf_bias": bleaf[1], "leaf_ae": bleaf[2], "leaf_den": bleaf[3], "active": bleaf[4],
            })
            for cand in CANDIDATES:
                fc = f"fc_{cand}_{suffix}"
                csku = _eval_sku(x, fc, suffix)
                cleaf = _eval_leaf(share_rows, x, fc, prepared, start, end, suffix, active_min)
                rows.append({
                    "sec": sec, "eval_rank": eval_rank, "start": start, "end": end,
                    "suffix": suffix, "candidate": cand,
                    "sku_wmape": csku[0], "sku_bias": csku[1], "sku_ae": csku[2], "sku_den": csku[3],
                    "leaf_wmape": cleaf[0], "leaf_bias": cleaf[1], "leaf_ae": cleaf[2], "leaf_den": cleaf[3], "active": cleaf[4],
                })
        by = next(r for r in rows if r["sec"] == sec and r["eval_rank"] == eval_rank and r["suffix"] == "y" and r["candidate"] == "current_selector")
        bv = next(r for r in rows if r["sec"] == sec and r["eval_rank"] == eval_rank and r["suffix"] == "v" and r["candidate"] == "current_selector")
        print(f"  pseudo-OOS {start}→{end} | current leaf U/V={_pct(by['leaf_wmape'])}/{_pct(bv['leaf_wmape'])}")

    pl.DataFrame(rows).write_csv(out_dir / f"calendar_driver_rolling_sec_{sec}.csv")

    picks: dict[str, dict | None] = {}
    print(f"\n--- RESUMEN PRE-OOS sec={sec} ---")
    for suffix, label in (("y", "Unidades"), ("v", "Valor ($)")):
        sku_sum = _summary(rows, sec, suffix, "sku")
        leaf_sum = _summary(rows, sec, suffix, "leaf")
        _print_summary(f"SKU-TOTAL | sec={sec} | {label}", sku_sum)
        _print_summary(f"LEAF share v12.5 fijo | sec={sec} | {label}", leaf_sum)
        pick = _robust_pick(leaf_sum)
        picks[suffix] = pick
        if pick:
            print(
                f"  ==> PRE-OOS winner causal: {pick['candidate']} | gain={_pp(pick['gain'])} "
                f"recent4={_pp(pick['recent4'])} win={100*pick['win']:.0f}% worst={_pp(pick['worst'])}"
            )
        else:
            print("  ==> sin driver challenger que cumpla guardas robustas")

    # True production OOS holdout.  Fit factor parameters using blocks 1..history_max only.
    oos_end = oos_origin + dt.timedelta(days=block_days - 1)
    oos_target = _baseline_block(prepared, sku_daily, sku_first, ids, meta, incumbent, oos_origin, oos_end, 0)
    oos_hist = pl.concat([daily_by_id[h] for h in range(1, min(history_max, max_block) + 1)], how="vertical_relaxed")
    oos_share = _block_candidate(
        prepared, sku_daily, sku_first, uid_first, ids, incumbent, oos_origin, oos_end
    ).select("unique_id", "_v12_sku", "ds", "v12_store_share_y", "v12_store_share_value")

    print(f"\n=== HOLDOUT OOS PRODUCCIÓN sec={sec} | {oos_origin}→{oos_end} ===")
    holdout_rows: list[dict] = []
    for suffix, label in (("y", "Unidades"), ("v", "Valor ($)")):
        x = _apply_driver_candidates(oos_hist, oos_target, suffix, event_cols)
        bsku = _eval_sku(x, f"base_{suffix}", suffix)
        bleaf = _eval_leaf(oos_share, x, f"base_{suffix}", prepared, oos_origin, oos_end, suffix, active_min)
        pick = picks[suffix]
        if pick is None:
            print(f"  {label:10s}: current SKU={_pct(bsku[0])} LEAF={_pct(bleaf[0])} | sin challenger causal")
            holdout_rows.append({
                "sec": sec, "suffix": suffix, "candidate": "current_selector",
                "sku_wmape": bsku[0], "sku_bias": bsku[1], "leaf_wmape": bleaf[0], "leaf_bias": bleaf[1], "gain_leaf": 0.0,
            })
            continue
        cand = pick["candidate"]; fc = f"fc_{cand}_{suffix}"
        csku = _eval_sku(x, fc, suffix)
        cleaf = _eval_leaf(oos_share, x, fc, prepared, oos_origin, oos_end, suffix, active_min)
        gain = bleaf[0] - cleaf[0]
        print(
            f"  {label:10s}: PRE-OOS pick={cand:26s} | current LEAF={_pct(bleaf[0])} → candidate={_pct(cleaf[0])} "
            f"gain={_pp(gain)} | BIAS {_pct(bleaf[1])}→{_pct(cleaf[1])}"
        )
        holdout_rows.append({
            "sec": sec, "suffix": suffix, "candidate": cand,
            "sku_wmape": csku[0], "sku_bias": csku[1], "leaf_wmape": cleaf[0], "leaf_bias": cleaf[1], "gain_leaf": gain,
            "baseline_sku_wmape": bsku[0], "baseline_leaf_wmape": bleaf[0], "baseline_leaf_bias": bleaf[1],
        })
    pl.DataFrame(holdout_rows).write_csv(out_dir / f"calendar_driver_holdout_sec_{sec}.csv")
    return rows, holdout_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--history-max", type=int, default=16,
                    help="bloques cerrados usados para aprender calendar/price; 13+ incluye el análogo anual")
    ap.add_argument("--active-min", type=int, default=int(getattr(settings, "OOS_ACTIVE_MIN_NONZERO_DAYS", 7)))
    args = ap.parse_args()
    if args.blocks < 8:
        raise ValueError("--blocks debe ser >=8")
    if args.history_max < 13:
        raise ValueError("--history-max debe ser >=13 para incluir al menos un ciclo anual 13×28")

    selected_path = Path(settings.SELECTED_PATH)
    forecast_path = Path(getattr(settings, "FORECAST_PATH", Path(getattr(settings, "OUT_DIR", "data/output")) / "forecast.parquet"))
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if not forecast_path.exists():
        raise FileNotFoundError(forecast_path)
    out_dir = Path(getattr(settings, "OUT_DIR", "data/output")) / "diagnostics" / "v12_sku_total_calendar_driver"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("DIAGNÓSTICO v12.5 — SKU-TOTAL DRIVERS CALENDARIO + PRICE-STATE")
    print(f"Selected: {selected_path}")
    print(f"Forecast: {forecast_path}")
    print(f"Salida  : {out_dir}")
    print("Read-only. PRE-OOS aprende drivers solo con bloques cerrados; OOS producción es holdout final.")
    print(f"blocks={args.blocks} | history_max={args.history_max} | candidates={CANDIDATES}")
    print("Price-state causal = log(ASP últimos 28d / ASP últimos 84d); NO usa precio/actual futuro.")

    bounds = _oos_bounds(forecast_path)
    all_rows: list[dict] = []; holdouts: list[dict] = []
    for sec in sorted(bounds):
        origin, _ = bounds[sec]
        r, h = _run_section(
            selected_path, forecast_path, sec, origin,
            int(args.blocks), int(args.history_max), int(args.active_min), out_dir,
        )
        all_rows.extend(r); holdouts.extend(h)
    if all_rows:
        pl.DataFrame(all_rows).write_csv(out_dir / "calendar_driver_rolling_all.csv")
    if holdouts:
        pl.DataFrame(holdouts).write_csv(out_dir / "calendar_driver_holdout_all.csv")

    print("\n=== DECISIÓN HOLDOUT ===")
    good = [r for r in holdouts if r.get("candidate") != "current_selector" and float(r.get("gain_leaf", 0.0)) > 0]
    for r in holdouts:
        print(
            f"  sec={r['sec']} {'U' if r['suffix']=='y' else 'V'} | {r['candidate']} | "
            f"holdout leaf={_pct(float(r['leaf_wmape']))} gain={_pp(float(r.get('gain_leaf',0.0)))}"
        )
    if holdouts and len(good) == len(holdouts):
        print("- Los 4/4 picks PRE-OOS mejoran el OOS real: drivers explícitos candidatos fuertes para v12.6.")
    else:
        print(f"- Mejoran {len(good)}/{len(holdouts)} paneles OOS. No promover globalmente si existen regresiones.")
    print("- Si calendar_*_full gana pero *_shape no, la señal es principalmente UPLIFT/NIVEL de calendario.")
    print("- Si *_shape gana, calendario corrige principalmente la forma diaria manteniendo el total SKU.")
    print("- Si *_price_full agrega ganancia, el estado causal de precio aporta nivel explicable; si no, se elimina.")
    print("- Si tampoco generaliza, cerrar heurísticas lineales y pasar a modelo pooled SKU-total con drivers supervisados.")
    print("- OOS nunca participa en el ajuste de factores ni en la elección del candidato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
