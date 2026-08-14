"""
Pipeline de forecasting jerárquico con Regresión RLS
=====================================================
Niveles: sección, tienda, sku-tienda
  (filtros independientes tienda/SKU — ver settings.make_unique_id).

Modelo por periodo y nivel:
  - In-sample (train) — los 3 niveles usan RLS:
      · sección / tienda: modelo RLS propio
      · sku+tienda: coeficientes de sección o tienda (mejor WMAPE in-sample)
  - OOS (test) y forecast-only:
      · sección / tienda: sigue RLS (mismos coeficientes de train)
      · sku+tienda: efecto(coefs seleccionados) + SES no causal del residuo

Ventanas por sección:
  - train / OOS / forecast-only (ver settings.section_horizons)

Rendimiento:
  - Carga lazy + collect streaming; agregación lazy.
  - Derivación SKU+tienda vectorizada por tienda.
  - Checkpoint por sección (`forecast_seccion_<n>_partial.parquet`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from dateutil.easter import easter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings

try:
    from rls_opt import RecursiveLeastSquaresRegression, RLSConstantPrior, RLSPrior
    from rls_opt.edp import decompose_price
except ImportError:  # pragma: no cover
    RecursiveLeastSquaresRegression = None  # type: ignore
    RLSConstantPrior = None  # type: ignore
    RLSPrior = None  # type: ignore
    decompose_price = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    class _NoOp:
        def __init__(self, iterable=None, *a, **k):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable if self._iterable is not None else [])

        def update(self, n: int = 1) -> None:
            pass

        def set_postfix_str(self, *a, **k) -> None:
            pass

        def close(self) -> None:
            pass

    def tqdm(iterable=None, *a, **k):
        return _NoOp(iterable, *a, **k)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
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
        except TypeError, ValueError:
            return lf.collect()


def _lf_columns(lf: pl.LazyFrame | pl.DataFrame) -> list[str]:
    if isinstance(lf, pl.DataFrame):
        return list(lf.columns)
    try:
        return list(lf.collect_schema().names())
    except Exception:
        return list(lf.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ForecastConfig:
    selected_path: Path
    metric_horizon_days: int
    aggregation_levels: dict
    forecast_levels: list
    rmse_error: float
    correction_factor: bool
    forgetting_factor: float
    min_y_to_update: float
    out_dir: Path
    date_column: str
    quantity_column: str
    price_column: str
    holidays: dict
    current_zone: str
    id_ejecucion: str
    now_year: int = field(init=False)

    def __post_init__(self) -> None:
        self.now_year = dt.datetime.now(tz=ZoneInfo(self.current_zone)).year

    @classmethod
    def from_settings(cls) -> ForecastConfig:
        return cls(
            selected_path=Path(settings.SELECTED_PATH),
            metric_horizon_days=settings.METRIC_HORIZON_DAYS,
            aggregation_levels=settings.AGGREGATION_LEVELS,
            forecast_levels=settings.FORECAST_LEVELS,
            rmse_error=settings.RMSE_ERROR,
            correction_factor=settings.CORRECTION_FACTOR,
            forgetting_factor=getattr(settings, "FORGETTING_FACTOR", 0.995),
            min_y_to_update=getattr(settings, "MIN_Y_TO_UPDATE", 1.0),
            out_dir=Path(settings.OUT_DIR),
            date_column=settings.DATE_COLUMN,
            quantity_column=settings.QUANTITY_COLUMN,
            price_column=settings.PRICE_COLUMN,
            holidays=settings.HOLIDAYS,
            current_zone=settings.CURRENT_ZONE,
            id_ejecucion=settings.ID_EJECUCION,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Festividades
# ─────────────────────────────────────────────────────────────────────────────
class HolidayCalendar:
    _YEARS_BACK = 6
    _YEARS_FWD = 3
    _EASTER_BASED = {
        "pascuas_y_semana_de_turismo": lambda y: easter(y),
        "carnaval": lambda y: easter(y) - dt.timedelta(days=48),
    }

    def __init__(self, holidays_cfg: dict, now_year: int):
        self._cfg = holidays_cfg
        self._year_range = range(
            now_year - self._YEARS_BACK, now_year + self._YEARS_FWD
        )
        self.processed: dict[str, dict] = self._build()

    def _windows(
        self, info: dict, dates: list[dt.date]
    ) -> list[tuple[dt.date, dt.date]]:
        before = dt.timedelta(days=info["window_before_days"])
        after = dt.timedelta(days=info["window_after_days"])
        return [(f - before, f + after) for f in dates]

    def _fixed_dates(self, info: dict) -> list[dt.date]:
        dates: list[dt.date] = []
        if "month" in info and "day" in info:
            dates = [dt.date(y, info["month"], info["day"]) for y in self._year_range]
        absolute_month = info.get("absolute_month")
        if absolute_month:
            for year in self._year_range:
                first = dt.date(year, absolute_month, 1)
                offset = ((info["absolute_weekday"] - 1) - first.weekday()) % 7
                dates.append(
                    first
                    + dt.timedelta(days=offset)
                    + dt.timedelta(weeks=info["relative_ocurrence"] - 1)
                )
        return dates

    def _movable_dates(self, name: str, info: dict) -> list[dt.date]:
        if not info.get("relative_ocurrence"):
            return []
        fn = self._EASTER_BASED.get(name)
        return [fn(y) for y in self._year_range] if fn else []

    def _build(self) -> dict[str, dict]:
        processed: dict[str, dict] = {}
        for name, info in self._cfg.items():
            try:
                fixed = self._fixed_dates(info)
            except KeyError as exc:
                logger.warning("No se pudo parsear festividad '%s': %s", name, exc)
                continue
            if info.get("relative_ocurrence") and name in self._EASTER_BASED:
                fixed = self._movable_dates(name, info)
            if not fixed:
                continue
            processed[name] = {
                "fecha": fixed,
                "fecha_rango": self._windows(info, fixed),
                "active": info["active"],
                "original_data": info,
            }
        return processed


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Agregación jerárquica (sección → SKU → store) + descripciones
# ─────────────────────────────────────────────────────────────────────────────
class DataAggregator:
    """
    unique_id — filtros independientes tienda/sku (ver settings.make_unique_id):
      - "1"                      (sección)
      - "1||T:00122"             (tienda, todos los SKU)
      - "1||S:SKU123"            (sku, todas las tiendas)
      - "1||T:00122||S:SKU123"   (tienda + sku)
    Columnas extra: sku_desc, store_name, seccion
    """

    def __init__(
        self,
        date_column: str,
        quantity_column: str,
        price_column: str,
        aggregation_levels: dict,
    ):
        self._date_col = date_column
        self._qty_col = quantity_column
        self._prc_col = price_column
        self._levels = aggregation_levels

    @staticmethod
    def _base_aggs() -> list[pl.Expr]:
        return [
            pl.col("y").sum(),
            pl.col("value").sum(),
            pl.count("y").alias("conteo_sku"),
        ]

    def aggregate(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """
        Agrega solo los niveles de pronóstico: sección, tienda, sku-tienda.
        Acepta DataFrame o LazyFrame; el plan se ejecuta con collect streaming.
        """
        lf = df.lazy() if isinstance(df, pl.DataFrame) else df
        cols_in = set(_lf_columns(lf))

        rename_map = {
            self._date_col: "ds",
            self._qty_col: "y",
            self._prc_col: "value",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in cols_in}
        if rename_map:
            lf = lf.rename(rename_map)
            cols_in = (cols_in - set(rename_map)) | set(rename_map.values())

        lf = lf.filter(pl.col("y") >= 0)

        cast_cols = [
            pl.col(c).cast(pl.Utf8).str.strip_chars()
            for c in ("SECCION", "SKU_ID", "STORE_ID")
            if c in cols_in
        ]
        if cast_cols:
            lf = lf.with_columns(cast_cols)

        # Dimensiones pequeñas en eager (lookup tables)
        store_rows = [
            {"SECCION": sec, "STORE_ID": loc, "store_name": name}
            for sec, cfg in settings.SECCIONES.items()
            for loc, name in cfg["local_names"].items()
        ]
        store_name_df = (
            pl.DataFrame(store_rows)
            if store_rows
            else pl.DataFrame(
                schema={"SECCION": pl.Utf8, "STORE_ID": pl.Utf8, "store_name": pl.Utf8}
            )
        )

        if "DESCRIPCION" in cols_in:
            sku_desc_lf = (
                lf.select(["SKU_ID", "DESCRIPCION"])
                .drop_nulls()
                .unique(subset=["SKU_ID"], maintain_order=False)
                .rename({"DESCRIPCION": "sku_desc"})
            )
        else:
            sku_desc_lf = pl.DataFrame(
                schema={"SKU_ID": pl.Utf8, "sku_desc": pl.Utf8}
            ).lazy()

        base_aggs = self._base_aggs()

        # 1) sección
        lvl_sec = (
            lf.group_by(["ds", "SECCION"])
            .agg(base_aggs)
            .with_columns(
                pl.col("SECCION").alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
                pl.lit("").alias("sku_desc"),
                pl.lit("").alias("store_name"),
            )
        )

        # 2) sección + tienda
        lvl_store = (
            lf.group_by(["ds", "SECCION", "STORE_ID"])
            .agg(base_aggs)
            .with_columns(
                pl.concat_str(
                    [pl.col("SECCION"), pl.lit("||T:"), pl.col("STORE_ID")]
                ).alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
                pl.lit("").alias("sku_desc"),
            )
            .join(store_name_df.lazy(), on=["SECCION", "STORE_ID"], how="left")
            .with_columns(pl.col("store_name").fill_null(""))
        )

        # 3) sección + tienda + sku
        lvl_store_sku = (
            lf.group_by(["ds", "SECCION", "STORE_ID", "SKU_ID"])
            .agg(base_aggs)
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("SECCION"),
                        pl.lit("||T:"),
                        pl.col("STORE_ID"),
                        pl.lit("||S:"),
                        pl.col("SKU_ID"),
                    ]
                ).alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
            )
            .join(store_name_df.lazy(), on=["SECCION", "STORE_ID"], how="left")
            .join(sku_desc_lf, on="SKU_ID", how="left")
            .with_columns(
                pl.col("store_name").fill_null(""),
                pl.col("sku_desc").fill_null(""),
            )
        )

        out_cols = [
            "unique_id",
            "ds",
            "y",
            "value",
            "conteo_sku",
            "seccion",
            "sku_desc",
            "store_name",
        ]
        out_lf = (
            pl.concat(
                [
                    lvl_sec.select(out_cols),
                    lvl_store.select(out_cols),
                    lvl_store_sku.select(out_cols),
                ],
                how="vertical",
            )
            .with_columns(pl.lit(1).cast(pl.Int8).alias("intercept"))
            .sort(["unique_id", "ds"])
        )
        return _collect_streaming(out_lf)


# ─────────────────────────────────────────────────────────────────────────────
# RLS runner
# ─────────────────────────────────────────────────────────────────────────────
class RLSForecastRunner:
    """
    RLS a nivel sección y tienda; deriva SKU+tienda sin re-fit.

      - `fit_and_predict_sections`: RLS real (sección o tienda) en train/OOS/fcst.
        El RLS se ajusta sobre log1p(y); las predicciones se reconstruyen con
        `expm1()`.
      - `derive_sku_store_forecasts`: selección sección vs tienda por WMAPE
        in-sample; yhat RLS en train; en OOS/fcst reconstruye en log-space
        `expm1(intercept + efecto + SES(log-residuo))`.
    """

    def __init__(
        self,
        driver_cols: list[str],
        rmse_error: float,
        forgetting_factor: float = 0.995,
        min_y_to_update: float = 1.0,
        use_correction_factor: bool = False,
        n_jobs: int | None = None,
    ):
        self._driver_cols = driver_cols
        _price_exclude = {"asp", "edp", "discount"}
        self._driver_cols_price = [c for c in driver_cols if c not in _price_exclude]
        self._rmse_error = rmse_error
        self._forgetting_factor = forgetting_factor
        self._min_y_to_update = min_y_to_update
        self._use_correction_factor = use_correction_factor
        # n_jobs ahora es Nº de THREADS (ver docstring de clase), no procesos.
        self._n_jobs = n_jobs

    @staticmethod
    def _compute_wmape(res_df: pl.DataFrame) -> pl.DataFrame:
        """
        WMAPE bottom-up, separado por in_sample / out_sample.

        1. Errores solo en hojas SKU+tienda (y ≠ 0, no forecast_only):
             e = |y − ŷ|
        2. WMAPE hoja = Σe / Σ|y| por unique_id y period_type
        3. Tienda / sección = suma de numeradores y denominadores de sus hojas
           (no usa el yhat del RLS de esos niveles).
        """
        empty = pl.DataFrame(
            schema={
                "unique_id": pl.Utf8,
                "period_type": pl.Utf8,
                "nivel": pl.Utf8,
                "wmape": pl.Float64,
                "bias": pl.Float64,
                "n_points": pl.UInt32,
                "sum_y": pl.Float64,
                "sum_yhat": pl.Float64,
                "sum_abs_error": pl.Float64,
            }
        )
        if res_df.height == 0 or "yhat" not in res_df.columns:
            return empty

        # Solo hojas: sec||T:xx||S:yy
        leaves = res_df.filter(
            pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2
        )
        leaves = leaves.filter(
            pl.col("y").is_not_null()
            & pl.col("yhat").is_not_null()
            & pl.col("y").is_finite()
            & (pl.col("y") != 0)
        )
        if "period_type" in leaves.columns:
            leaves = leaves.filter(
                pl.col("period_type").is_in(["in_sample", "out_sample"])
            )
        else:
            leaves = leaves.with_columns(pl.lit("in_sample").alias("period_type"))

        if leaves.height == 0:
            return empty

        # Parse store_uid / seccion desde unique_id
        leaves = leaves.with_columns(
            pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("_store_uid"),
            pl.col("unique_id").str.split("||").list.first().alias("_seccion"),
            (pl.col("y") - pl.col("yhat")).abs().alias("_abs_error"),
            pl.col("y").abs().alias("_abs_y"),
        )

        def _finalize(df: pl.DataFrame, nivel: str) -> pl.DataFrame:
            return df.with_columns(
                (pl.col("sum_abs_error") / pl.col("sum_abs_y")).alias("wmape"),
                (
                    (pl.col("sum_yhat") - pl.col("sum_y"))
                    / pl.col("sum_y").replace(0, None)
                ).alias("bias"),
                pl.lit(nivel).alias("nivel"),
            ).select(
                [
                    "unique_id",
                    "period_type",
                    "nivel",
                    "wmape",
                    "bias",
                    "n_points",
                    "sum_y",
                    "sum_yhat",
                    "sum_abs_error",
                ]
            )

        # ── Hojas SKU+tienda ───────────────────────────────────────────────
        leaf_agg = leaves.group_by(["unique_id", "period_type"]).agg(
            pl.col("_abs_error").sum().alias("sum_abs_error"),
            pl.col("_abs_y").sum().alias("sum_abs_y"),
            pl.col("y").sum().alias("sum_y"),
            pl.col("yhat").sum().alias("sum_yhat"),
            pl.len().alias("n_points"),
        )
        out_leaf = _finalize(leaf_agg, "sku_tienda")

        # ── Tienda (bottom-up desde hojas) ─────────────────────────────────
        store_agg = (
            leaves.group_by(["_store_uid", "period_type"])
            .agg(
                pl.col("_abs_error").sum().alias("sum_abs_error"),
                pl.col("_abs_y").sum().alias("sum_abs_y"),
                pl.col("y").sum().alias("sum_y"),
                pl.col("yhat").sum().alias("sum_yhat"),
                pl.len().alias("n_points"),
            )
            .rename({"_store_uid": "unique_id"})
        )
        out_store = _finalize(store_agg, "tienda")

        # ── Sección (bottom-up desde hojas) ────────────────────────────────
        sec_agg = (
            leaves.group_by(["_seccion", "period_type"])
            .agg(
                pl.col("_abs_error").sum().alias("sum_abs_error"),
                pl.col("_abs_y").sum().alias("sum_abs_y"),
                pl.col("y").sum().alias("sum_y"),
                pl.col("yhat").sum().alias("sum_yhat"),
                pl.len().alias("n_points"),
            )
            .rename({"_seccion": "unique_id"})
        )
        out_sec = _finalize(sec_agg, "seccion")

        return pl.concat([out_leaf, out_store, out_sec], how="vertical")

    @staticmethod
    def _correction_factor(errors: np.ndarray) -> float:
        sigma2 = errors.var(ddof=1)
        return float(np.exp(sigma2 / 2))

    def _default_priors(self, n_features: int):
        return [RLSConstantPrior(standard_error=0.5, rmse_error=self._rmse_error)] + [
            RLSPrior(coefficient=0, standard_error=0.5, rmse_error=self._rmse_error)
            for _ in range(max(0, n_features - 1))
        ]

    def _new_rls(self, min_y: float, return_all_coefs: bool = False):
        return RecursiveLeastSquaresRegression(
            forgetting_factor=self._forgetting_factor,
            min_y_to_update=min_y,
            return_all_coefs=return_all_coefs,
        )

    # ── Fit único por serie (Fase 1) ────────────────────────────────────────
    def _fit_models(self, train_g: pl.DataFrame):
        """Ajusta model_y (log1p(y)) y model_p (log1p(price)) UNA sola vez."""
        if RecursiveLeastSquaresRegression is None:
            raise RuntimeError("rls_opt no disponible")

        X_y = train_g.select(self._driver_cols).to_numpy().astype(np.float64, order="C")
        y = train_g["y"].to_numpy()
        log_y = np.log1p(y)
        model_y = self._new_rls(self._min_y_to_update)
        priors_y = self._default_priors(len(self._driver_cols))
        model_y.fit(x=X_y, y=log_y, priors=priors_y)

        X_p = (
            train_g.select(self._driver_cols_price)
            .to_numpy()
            .astype(np.float64, order="C")
        )
        price = train_g["value"].to_numpy()
        log_price = np.log1p(np.clip(price, 0.0, None))
        model_p = self._new_rls(1e-8)
        priors_p = self._default_priors(len(self._driver_cols_price))
        model_p.fit(x=X_p, y=log_price, priors=priors_p)

        return model_y, model_p

    def _predict_with_models(
        self,
        unique_id: str,
        model_y,
        model_p,
        train_g: pl.DataFrame,
        test_g: pl.DataFrame,
        meta: dict | None = None,
    ) -> pl.DataFrame | None:
        if test_g.height == 0:
            return None

        X_y_test = (
            test_g.select(self._driver_cols).to_numpy().astype(np.float64, order="C")
        )
        log_yhat_test = model_y.predict(X_y_test)
        if self._use_correction_factor:
            corr = self._correction_factor(np.asarray(model_y.errors))
            yhat_test = np.round(np.exp(log_yhat_test) * corr).ravel()
        else:
            yhat_test = np.round(np.expm1(log_yhat_test)).ravel()

        X_p_test = (
            test_g.select(self._driver_cols_price)
            .to_numpy()
            .astype(np.float64, order="C")
        )
        log_pricehat = model_p.predict(X_p_test)
        if self._use_correction_factor:
            corr_p = self._correction_factor(np.asarray(model_p.errors))
            pricehat = np.expm1(log_pricehat) * corr_p
        else:
            pricehat = np.expm1(log_pricehat)
        pricehat = np.round(pricehat, 2).ravel()

        y_real = (
            test_g["y"].to_numpy()
            if "y" in test_g.columns
            else np.zeros(len(yhat_test))
        )
        price_real = (
            test_g["value"].to_numpy()
            if "value" in test_g.columns
            else np.zeros(len(pricehat))
        )

        result = {
            "unique_id": unique_id,
            "ds": test_g["ds"],
            "value": price_real,
            "valuehat": pricehat,
            "y": y_real,
            "yhat": yhat_test,
        }
        if meta:
            for k, v in meta.items():
                result[k] = v
        for col in ("sku_desc", "store_name", "seccion"):
            if col in train_g.columns:
                result[col] = train_g[col][0]
        return pl.DataFrame(result)

    @staticmethod
    def _normalize_partition_dict(d: dict) -> dict:
        return {(k[0] if isinstance(k, tuple) else k): v for k, v in d.items()}

    def _workers(self) -> int:
        return self._n_jobs if self._n_jobs and self._n_jobs > 1 else 1

    # ── RLS sección / tienda ──────────────────────────────────────────────
    def fit_and_predict_sections(
        self,
        train: pl.DataFrame,
        targets: dict[str, pl.DataFrame],
        section_ids: list[str],
        desc: str = "RLS sección",
        meta: dict | None = None,
    ) -> tuple[pl.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """
        Ajusta RLS para los `section_ids` dados (sección o tienda).
        Devuelve (res_df, coefs) con coefs[id] = (coef_y, coef_p).
        """
        train_parts = self._normalize_partition_dict(
            train.partition_by("unique_id", as_dict=True)
        )
        target_parts = {
            name: self._normalize_partition_dict(
                df.partition_by("unique_id", as_dict=True)
            )
            for name, df in targets.items()
            if df.height
        }

        results: list[pl.DataFrame] = []
        coefs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for uid in section_ids:
            train_g = train_parts.get(uid)
            if train_g is None or train_g.height == 0:
                continue
            try:
                model_y, model_p = self._fit_models(train_g)
            except Exception as exc:
                logger.warning("%s: fit falló para %s: %s", desc, uid, exc)
                continue
            coefs[uid] = (
                np.asarray(model_y.final_coef_[0], dtype=np.float64).ravel(),
                np.asarray(model_p.final_coef_[0], dtype=np.float64).ravel(),
            )
            for name, parts in target_parts.items():
                test_g = parts.get(uid)
                if test_g is None or test_g.height == 0:
                    continue
                try:
                    frame = self._predict_with_models(
                        uid, model_y, model_p, train_g, test_g, meta
                    )
                except Exception as exc:
                    logger.warning(
                        "%s: predict falló para %s/%s: %s", desc, uid, name, exc
                    )
                    continue
                if frame is not None and frame.height:
                    results.append(
                        frame.with_columns(pl.lit(name).alias("period_type"))
                    )

        res_df = (
            pl.concat(results, how="diagonal_relaxed") if results else pl.DataFrame()
        )
        return res_df, coefs

    # ── SKU+tienda: efecto RLS seleccionado + SES no causal ───────────────
    @staticmethod
    def _apply_ses(
        df: pl.DataFrame, src_col: str, out_col: str, alpha: float
    ) -> pl.DataFrame:
        """
        SES **no causal** por unique_id: s(t) = ewm_mean incluyendo y_neto(t)
        (sin shift). La causalidad del pipeline la aportan los coeficientes
        RLS (ajustados solo en train), no el suavizado del residuo.

        Filas forecast_only (sin actuals): se propaga el último estado SES
        de la parte con actuals.
        """
        df = df.sort(["unique_id", "ds"])
        has_period = "period_type" in df.columns
        is_actual = (
            (pl.col("period_type") != "forecast_only") if has_period else pl.lit(True)
        )

        actual = df.filter(is_actual).with_columns(
            pl.col(src_col)
            .ewm_mean(alpha=alpha, adjust=False)
            .over("unique_id")
            .alias(out_col)
        )
        last_state = actual.group_by("unique_id").agg(
            pl.col(out_col).last().alias("_last_s")
        )

        if has_period:
            forecast = df.filter(~is_actual)
            if forecast.height:
                forecast = (
                    forecast.join(last_state, on="unique_id", how="left")
                    .with_columns(pl.col("_last_s").fill_null(0.0).alias(out_col))
                    .drop("_last_s")
                )
                return pl.concat([actual, forecast], how="diagonal_relaxed").sort(
                    ["unique_id", "ds"]
                )
        return actual.sort(["unique_id", "ds"])

    @staticmethod
    def _select_model_wmape(
        y: np.ndarray,
        yhat_sec: np.ndarray,
        yhat_sto: np.ndarray,
        uid: np.ndarray,
        is_train: np.ndarray,
    ) -> dict[str, str]:
        """WMAPE/BIAS solo in-sample → {unique_id: 'seccion'|'tienda'}."""
        selection: dict[str, str] = {}
        # Agrupar índices por uid (solo filas train con y != 0)
        mask = is_train & np.isfinite(y) & (y != 0)
        if not np.any(mask):
            for u in np.unique(uid):
                selection[str(u)] = "seccion"
            return selection

        # Ordenar por uid para barrido lineal
        order = np.argsort(uid, kind="mergesort")
        uid_s = uid[order]
        y_s = y[order]
        ys_s = yhat_sec[order]
        yt_s = yhat_sto[order]
        m_s = mask[order]

        n = len(uid_s)
        i = 0
        while i < n:
            j = i + 1
            while j < n and uid_s[j] == uid_s[i]:
                j += 1
            m = m_s[i:j]
            if not np.any(m):
                selection[str(uid_s[i])] = "seccion"
                i = j
                continue
            yy = y_s[i:j][m]
            denom = float(np.abs(yy).sum())
            if denom == 0:
                selection[str(uid_s[i])] = "seccion"
                i = j
                continue
            err_sec = float(np.abs(yy - ys_s[i:j][m]).sum())
            err_sto = float(np.abs(yy - yt_s[i:j][m]).sum())
            if err_sto < err_sec:
                selection[str(uid_s[i])] = "tienda"
            elif err_sto > err_sec:
                selection[str(uid_s[i])] = "seccion"
            else:
                sum_y = float(yy.sum())
                if sum_y == 0:
                    selection[str(uid_s[i])] = "seccion"
                else:
                    bias_sec = float((ys_s[i:j][m] - yy).sum()) / sum_y
                    bias_sto = float((yt_s[i:j][m] - yy).sum()) / sum_y
                    selection[str(uid_s[i])] = (
                        "tienda" if abs(bias_sto) <= abs(bias_sec) else "seccion"
                    )
            i = j
        return selection

    def derive_sku_store_forecasts(
        self,
        panel: pl.DataFrame,
        section_id: str,
        section_coefs: dict,
        store_coefs: dict,
        alpha: float | None = None,
    ) -> pl.DataFrame:
        """
        Derivación SKU+tienda **tienda a tienda** (bajo uso de memoria).

        El modelo RLS se ajusta sobre log1p(y) (ver `_fit_models`), por lo que
        la reconstrucción del pronóstico ocurre EN LOG-SPACE:

          - Todos los períodos (in_sample / out_sample / forecast_only):
              yhat = expm1(intercept + efecto + SES(log-residuo))

        donde `residuo_log = log1p(y) − (intercept + efecto)`, e `intercept` +
        `efecto` provienen del modelo (sección o tienda) elegido por
        `_select_model_wmape`. `_apply_ses` calcula el SES sobre TODO el
        período con actuals (in_sample + out_sample) como una única serie
        continua, así que `_y_neto_hat`/`_v_neto_hat` ya son válidos también
        para in_sample: no hace falta (ni es correcto) usar la predicción
        cruda `_yhat_rls`/`_valuehat_rls` del RLS de sección/tienda para esas
        filas. Esa predicción cruda vive en la escala del AGREGADO (sección o
        tienda), no en la de la hoja SKU+tienda, y usarla directamente en
        in_sample producía un salto de escala de varios órdenes de magnitud
        entre in_sample y OOS/forecast (yhat/valuehat in-sample en la escala
        del agregado, OOS ya corregido por el SES). `_yhat_rls`/
        `_valuehat_rls` se conservan solo como entrada de
        `_select_model_wmape` (comparación sección vs. tienda), no como
        salida final.

        Nota histórica: antes se restaba `efecto` (log-space) de `y`
        (lineal), produciendo un residuo incoherente y pronósticos 0/negativos
        en OOS/forecast_only (causa del ranking SKU+tienda vacío). ya
        corregido junto con el punto anterior.

        No materializa el panel completo con columnas intermedias duplicadas.
        """
        import gc

        if panel.height == 0:
            return pl.DataFrame()

        coef_section = section_coefs.get(section_id)
        if coef_section is None:
            return pl.DataFrame()

        if alpha is None:
            alpha = float(getattr(settings, "SES_ALPHA", 0.3))

        driver_cols = self._driver_cols
        driver_cols_price = self._driver_cols_price
        missing = [c for c in driver_cols + driver_cols_price if c not in panel.columns]
        if missing:
            logger.warning("derive_sku_store_forecasts: faltan drivers %s", missing[:8])
            return pl.DataFrame()

        idx_y = [i for i, c in enumerate(driver_cols) if c != "intercept"]
        cols_y = [driver_cols[i] for i in idx_y]
        idx_p = [i for i, c in enumerate(driver_cols_price) if c != "intercept"]
        cols_p = [driver_cols_price[i] for i in idx_p]
        # Índice del término constante (columna "intercept") dentro del vector
        # de coeficientes. El modelo RLS se ajusta sobre log1p(y), así que
        # log(y) = intercept + efecto(drivers). El intercept depende de dónde
        # aparezca "intercept" en driver_cols (no se asume posición 0).
        intercept_idx_y = driver_cols.index("intercept")
        intercept_idx_p = driver_cols_price.index("intercept")

        # Solo columnas necesarias (reduce pico de RAM)
        meta_keep = [
            c
            for c in (
                "unique_id",
                "ds",
                "y",
                "value",
                "period_type",
                "sku_desc",
                "store_name",
                "seccion",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "forecast_start",
                "forecast_end",
            )
            if c in panel.columns
        ]
        keep_cols = list(dict.fromkeys(meta_keep + driver_cols + driver_cols_price))
        panel = panel.select(keep_cols).with_columns(
            pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("_store_uid")
        )

        coef_y_sec = np.ascontiguousarray(coef_section[0], dtype=np.float64).ravel()
        coef_p_sec = np.ascontiguousarray(coef_section[1], dtype=np.float64).ravel()
        coef_y_sec_fx = coef_y_sec[idx_y]
        coef_p_sec_fx = coef_p_sec[idx_p]

        store_uids = panel.get_column("_store_uid").unique().to_list()
        frames: list[pl.DataFrame] = []
        has_period = "period_type" in panel.columns

        for si, store_uid in enumerate(store_uids):
            coef_store = store_coefs.get(store_uid)
            if coef_store is None:
                continue

            sub = panel.filter(pl.col("_store_uid") == store_uid)
            if sub.height == 0:
                continue

            # Un solo to_numpy por bloque de drivers (Float64)
            X_y = np.ascontiguousarray(
                sub.select(driver_cols).to_numpy(), dtype=np.float64
            )
            X_p = np.ascontiguousarray(
                sub.select(driver_cols_price).to_numpy(), dtype=np.float64
            )
            y = sub["y"].to_numpy().astype(np.float64, copy=False)
            value = (
                sub["value"].to_numpy().astype(np.float64, copy=False)
                if "value" in sub.columns
                else np.zeros_like(y)
            )
            uids = sub["unique_id"].to_numpy()
            if has_period:
                periods = sub["period_type"].to_numpy()
                is_train = periods == "in_sample"
            else:
                periods = None
                is_train = np.ones(len(y), dtype=bool)

            coef_y_sto = np.ascontiguousarray(coef_store[0], dtype=np.float64).ravel()
            coef_p_sto = np.ascontiguousarray(coef_store[1], dtype=np.float64).ravel()

            # Predicciones completas RLS (auditoría + selección)
            yhat_sec = np.round(np.expm1(X_y @ coef_y_sec)).ravel()
            yhat_sto = np.round(np.expm1(X_y @ coef_y_sto)).ravel()
            valuehat_sec = np.round(np.expm1(X_p @ coef_p_sec), 2).ravel()
            valuehat_sto = np.round(np.expm1(X_p @ coef_p_sto), 2).ravel()

            selection = self._select_model_wmape(y, yhat_sec, yhat_sto, uids, is_train)
            use_sto = np.fromiter(
                (selection.get(str(u), "seccion") == "tienda" for u in uids),
                dtype=bool,
                count=len(uids),
            )
            yhat_rls = np.where(use_sto, yhat_sto, yhat_sec)
            valuehat_rls = np.where(use_sto, valuehat_sto, valuehat_sec)

            # Efecto sin intercepto del modelo elegido (vistas, sin copiar X)
            effect_y = np.where(
                use_sto,
                X_y[:, idx_y] @ coef_y_sto[idx_y],
                X_y[:, idx_y] @ coef_y_sec_fx,
            )
            effect_v = np.where(
                use_sto,
                X_p[:, idx_p] @ coef_p_sto[idx_p],
                X_p[:, idx_p] @ coef_p_sec_fx,
            )
            # Interceptos del modelo log seccionado por serie (uno por fila).
            intercept_y = np.where(
                use_sto,
                coef_y_sto[intercept_idx_y],
                coef_y_sec[intercept_idx_y],
            )
            intercept_v = np.where(
                use_sto,
                coef_p_sto[intercept_idx_p],
                coef_p_sec[intercept_idx_p],
            )
            del X_y, X_p

            # Residuo EN LOG-SPACE (modelo RLS ajustado sobre log1p):
            #   log1p(y) − (intercept + efecto de drivers)
            # Antes se restaba effect (log-space) de y (lineal), lo que producía
            # un "y_neto" incoherente y pronósticos 0/negativos en OOS/fcst.
            with np.errstate(divide="ignore", invalid="ignore"):
                log_y = np.log1p(np.clip(y, 0.0, None))
                log_v = np.log1p(np.clip(value, 0.0, None))
            log_resid_y = np.where(
                np.isfinite(log_y), log_y - (intercept_y + effect_y), np.nan
            )
            log_resid_v = np.where(
                np.isfinite(log_v), log_v - (intercept_v + effect_v), np.nan
            )
            modelo = np.where(use_sto, "tienda", "seccion")

            block = sub.select([c for c in meta_keep if c in sub.columns]).with_columns(
                pl.Series("yhat_seccion", yhat_sec),
                pl.Series("valuehat_seccion", valuehat_sec),
                pl.Series("yhat_tienda", yhat_sto),
                pl.Series("valuehat_tienda", valuehat_sto),
                pl.Series("modelo_seleccionado", modelo),
                pl.Series("driver_effect", effect_y),
                pl.Series("driver_effect_value", effect_v),
                pl.Series("_intercept_y", intercept_y),
                pl.Series("_intercept_v", intercept_v),
                pl.Series("_y_neto", log_resid_y),
                pl.Series("_v_neto", log_resid_v),
                pl.Series("_yhat_rls", yhat_rls),
                pl.Series("_valuehat_rls", valuehat_rls),
            )
            del yhat_sec, yhat_sto, valuehat_sec, valuehat_sto
            del yhat_rls, valuehat_rls, effect_y, effect_v, intercept_y, intercept_v

            block = self._apply_ses(block, "_y_neto", "_y_neto_hat", alpha)
            block = self._apply_ses(block, "_v_neto", "_v_neto_hat", alpha)

            if has_period:
                # Reconstrucción en log-space: expm1(intercept + efecto + SES(residuo)).
                # Se aplica IGUAL para in_sample/out_sample/forecast_only.
                #
                # Antes, in_sample usaba directamente `_yhat_rls` (predicción cruda
                # del RLS de sección/tienda, ajustado sobre el y/value AGREGADO de
                # ese nivel). Esa predicción vive en la escala del agregado
                # (sección o tienda), no en la escala de la hoja SKU+tienda, y por
                # eso el gráfico mostraba yhat/valuehat in-sample varios órdenes de
                # magnitud por encima de los actuals, con un salto abrupto al pasar
                # a OOS (que sí usaba la reconstrucción con SES).
                # `_y_neto_hat`/`_v_neto_hat` ya están definidos para in_sample y
                # out_sample por igual (`_apply_ses` trata todo el período con
                # actuals como una sola serie continua), así que no hace falta
                # ninguna rama especial: usar la misma fórmula corrige la escala.
                block = block.with_columns(
                    (
                        (
                            pl.col("_intercept_y")
                            + pl.col("driver_effect")
                            + pl.col("_y_neto_hat")
                        ).exp()
                        - 1
                    )
                    .clip(lower_bound=0.0)
                    .round(0)
                    .alias("yhat"),
                    (
                        (
                            pl.col("_intercept_v")
                            + pl.col("driver_effect_value")
                            + pl.col("_v_neto_hat")
                        ).exp()
                        - 1
                    )
                    .clip(lower_bound=0.0)
                    .round(2)
                    .alias("valuehat"),
                )
            else:
                block = block.with_columns(
                    pl.col("_yhat_rls").alias("yhat"),
                    pl.col("_valuehat_rls").alias("valuehat"),
                )

            drop_tmp = [
                c
                for c in (
                    "_y_neto",
                    "_v_neto",
                    "_y_neto_hat",
                    "_v_neto_hat",
                    "_intercept_y",
                    "_intercept_v",
                    "_yhat_rls",
                    "_valuehat_rls",
                    "_store_uid",
                )
                if c in block.columns
            ]
            frames.append(block.drop(drop_tmp))
            del sub, block, y, value, use_sto
            if (si + 1) % 5 == 0:
                gc.collect()

        if not frames:
            return pl.DataFrame()

        out = pl.concat(frames, how="vertical")
        del frames
        gc.collect()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador – split por sección
# ─────────────────────────────────────────────────────────────────────────────


def densify_section_panel(
    df: pl.DataFrame,
    date_start: dt.date,
    date_end: dt.date,
    *,
    fill_y: float = 0.0,
    fill_value: float = 0.0,
    extra_uids: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Panel denso: (unique_id × cada día en [date_start, date_end]).
    Sin venta → y=0, value=0. Métricas excluyen ceros aparte.

    Grid vía np.repeat/tile (más barato que join cross de Polars).
    """
    if date_end < date_start:
        return df
    if df.height == 0 and (extra_uids is None or extra_uids.height == 0):
        return df

    dates = pl.date_range(date_start, date_end, interval="1d", eager=True)
    n_days = len(dates)

    if df.height:
        df = df.with_columns(pl.col("ds").cast(pl.Date))
        uids_s = df.get_column("unique_id").unique()
    else:
        uids_s = pl.Series("unique_id", [], dtype=pl.Utf8)
    if extra_uids is not None and extra_uids.height:
        extra_s = extra_uids.get_column("unique_id")
        uids_s = pl.concat([uids_s, extra_s]).unique()

    n_uid = uids_s.len()
    if n_uid == 0:
        return df

    # Grid denso sin cross-join: O(n) construcción
    uid_arr = uids_s.to_numpy()
    ds_arr = dates.to_numpy()
    grid = pl.DataFrame(
        {
            "unique_id": np.repeat(uid_arr, n_days),
            "ds": np.tile(ds_arr, n_uid),
        }
    ).with_columns(pl.col("ds").cast(pl.Date))

    meta_cols = [
        c
        for c in ("sku_desc", "store_name", "seccion", "conteo_sku")
        if df.height and c in df.columns
    ]
    if meta_cols:
        meta = (
            df.select(["unique_id"] + meta_cols)
            .group_by("unique_id")
            .agg([pl.col(c).drop_nulls().first().alias(c) for c in meta_cols])
        )
        grid = grid.join(meta, on="unique_id", how="left")

    if df.height:
        data_cols = [
            c for c in df.columns if c not in ("unique_id", "ds") and c not in meta_cols
        ]
        join_df = df.select(["unique_id", "ds"] + data_cols)
        out = grid.join(join_df, on=["unique_id", "ds"], how="left")
    else:
        out = grid

    fills = []
    if "y" in out.columns:
        fills.append(pl.col("y").fill_null(fill_y))
    else:
        fills.append(pl.lit(fill_y).alias("y"))
    if "value" in out.columns:
        fills.append(pl.col("value").fill_null(fill_value))
    else:
        fills.append(pl.lit(fill_value).alias("value"))
    if "conteo_sku" in out.columns:
        fills.append(pl.col("conteo_sku").fill_null(0))
    if fills:
        out = out.with_columns(fills)

    logger.info(
        "densify: %d series × %d días = %d filas",
        n_uid,
        n_days,
        out.height,
    )
    # Orden natural del grid: uid bloqueado × días → ya casi ordenado
    return out.sort(["unique_id", "ds"])


class RLSForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig,
        n_jobs: int | None = None,
        limit_series: int | None = None,
    ):
        self._cfg = config
        # n_jobs = Nº de threads (ver docstring RLSForecastRunner)
        self._n_jobs = n_jobs
        self._limit_series = limit_series
        self._calendar: HolidayCalendar | None = None
        self._feature_builder: CalendarFeatureBuilder | None = None
        self._aggregator = DataAggregator(
            config.date_column,
            config.quantity_column,
            config.price_column,
            config.aggregation_levels,
        )

    def _load(self) -> pl.LazyFrame:
        """
        Scan lazy del parquet de entrada.
        No materializa: filtros por sección/fecha se empujan al scan.
        """
        path = self._cfg.selected_path
        logger.info("Scan lazy selected desde %s", path)

        lf = pl.scan_parquet(str(path))
        schema_names = set(_lf_columns(lf))

        # Proyección temprana: solo columnas necesarias
        wanted = [
            self._cfg.date_column,
            self._cfg.quantity_column,
            self._cfg.price_column,
            "SECCION",
            "SKU_ID",
            "STORE_ID",
        ]
        if "DESCRIPCION" in schema_names:
            wanted.append("DESCRIPCION")
        available = [c for c in wanted if c in schema_names]
        if available:
            lf = lf.select(available)

        lf = lf.with_columns(pl.col(self._cfg.date_column).cast(pl.Date))

        # Log de schema (sin materializar filas)
        logger.info(
            "Selected lazy listo | cols=%s",
            _lf_columns(lf),
        )
        return lf

    def _first_data_by_section(
        self, selected: pl.LazyFrame | pl.DataFrame
    ) -> dict[str, dt.date]:
        col = self._cfg.date_column
        lf = selected.lazy() if isinstance(selected, pl.DataFrame) else selected
        summary = _collect_streaming(
            lf.group_by("SECCION").agg(pl.col(col).min().alias("min_d"))
        )
        out: dict[str, dt.date] = {}
        for r in summary.iter_rows(named=True):
            d = r["min_d"]
            if isinstance(d, dt.datetime):
                d = d.date()
            out[str(r["SECCION"])] = d
        return out

    def _apply_limit_series(self, df_train: pl.DataFrame) -> pl.DataFrame:
        """Debug/benchmark: muestrea N SKUs, conservando niveles sección/tienda."""
        if not self._limit_series or df_train.height == 0:
            return df_train
        depth = pl.col("unique_id").str.count_matches(r"\|\|", literal=False)
        sku_ids = df_train.filter(depth == 2)["unique_id"].unique().to_list()
        if len(sku_ids) <= self._limit_series:
            return df_train
        rng = np.random.default_rng(42)
        keep = set(rng.choice(sku_ids, size=self._limit_series, replace=False).tolist())
        logger.warning(
            "⚠ --limit-series activo: %d SKUs de %d (solo para debug/benchmark, "
            "NO usar en producción)",
            len(keep),
            len(sku_ids),
        )
        return df_train.filter((depth < 2) | pl.col("unique_id").is_in(list(keep)))

    def _build_calendar_frame(
        self,
        unique_ids: list[str],
        start: dt.date,
        end: dt.date,
        template: pl.DataFrame,
    ) -> pl.DataFrame:
        """Genera filas ds para [start, end] por unique_id (y=0). Grid vía np.repeat/tile."""
        n_days = (end - start).days + 1
        n_uid = len(unique_ids)
        if n_days <= 0 or n_uid == 0:
            return pl.DataFrame()

        dates = pl.date_range(start, end, interval="1d", eager=True)
        uid_arr = np.asarray(unique_ids, dtype=object)
        ds_arr = dates.to_numpy()
        grid = pl.DataFrame(
            {
                "unique_id": np.repeat(uid_arr, n_days),
                "ds": np.tile(ds_arr, n_uid),
            }
        ).with_columns(pl.col("ds").cast(pl.Date))

        meta_cols = [
            c for c in ("sku_desc", "store_name", "seccion") if c in template.columns
        ]
        if meta_cols:
            keep = set(unique_ids)
            meta = (
                template.select(["unique_id"] + meta_cols)
                .unique(subset=["unique_id"], maintain_order=False)
                .filter(pl.col("unique_id").is_in(list(keep)))
            )
            grid = grid.join(meta, on="unique_id", how="left")

        return grid.with_columns(
            pl.lit(0.0).alias("y"),
            pl.lit(0.0).alias("value"),
            pl.lit(1).cast(pl.Int8).alias("intercept"),
            pl.lit(0).cast(pl.UInt32).alias("conteo_sku"),
        )

    @staticmethod
    def _calculate_edp_per_series(df: pl.DataFrame) -> pl.DataFrame:
        """Fallback: 1 llamada numba por serie (loop Python). Usar solo si el
        batch con `indexors` no está disponible o falla."""
        n_series = df["unique_id"].n_unique()
        logger.info("EDP: decompose_price por serie (loop) sobre %d series…", n_series)
        parts = []
        groups = df.sort(["unique_id", "ds"]).partition_by(
            "unique_id", maintain_order=True
        )
        for part in tqdm(
            groups,
            desc="EDP (loop)",
            unit="serie",
            ncols=50,  # Controla el ancho total
            ascii="░█",  # Define los caracteres de llenado (vacío/lleno)
            bar_format="{bar} [{n_fmt}/{total_fmt}] {desc}...",  # Estructura del texto
        ):
            uid = part["unique_id"][0]
            asp, edp, discount = decompose_price(
                sales_dollars=part["value"].to_numpy(),
                sales_units=part["y"].to_numpy(),
            )
            computed = {
                "asp": np.asarray(asp, dtype=np.float64).ravel(),
                "edp": np.asarray(edp, dtype=np.float64).ravel(),
                "discount": np.asarray(discount, dtype=np.float64).ravel(),
            }
            for name, arr in computed.items():
                if arr.shape[0] != part.height:
                    raise ValueError(
                        f"decompose_price '{name}' len={arr.shape[0]} "
                        f"vs unique_id={uid} height={part.height}"
                    )
            parts.append(
                part.with_columns(
                    pl.Series("asp", computed["asp"]),
                    pl.Series("edp", computed["edp"]),
                    pl.Series("discount", computed["discount"]),
                )
            )
        return pl.concat(parts, how="vertical")

    @staticmethod
    def _calculate_edp(df: pl.DataFrame) -> pl.DataFrame:
        """
        ASP/EDP/discount.

        Fase 2: 1 sola llamada numba con `indexors` (batched) sobre TODO el
        panel ordenado, en vez de 1 llamada numba por serie vía loop Python.
        `decompose_price` ya soporta `indexors: Sequence[slice]`; solo había
        que construir los slices por grupo y llamarlo una vez.

        Fallback vectorizado aproximado si no hay rls_opt o hay demasiadas
        series (umbral); fallback a loop por serie si el batch falla.
        """
        if df.height == 0:
            return df
        n_series = df["unique_id"].n_unique()
        use_fast = decompose_price is None or n_series > 5000
        if use_fast:
            if decompose_price is not None and n_series > 5000:
                logger.warning(
                    "EDP: %d series → modo rápido vectorizado (umbral 5000)",
                    n_series,
                )
            return df.with_columns(
                (pl.col("value") / pl.col("y").clip(lower_bound=1e-8))
                .fill_nan(0.0)
                .alias("asp"),
                (pl.col("value") / pl.col("y").clip(lower_bound=1e-8))
                .fill_nan(0.0)
                .alias("edp"),
                pl.lit(0.0).alias("discount"),
            )

        logger.info(
            "EDP: decompose_price batched (indexors) sobre %d series…", n_series
        )
        df_sorted = df.sort(["unique_id", "ds"])
        counts = (
            df_sorted.select("unique_id")
            .with_row_index("_idx")
            .group_by("unique_id", maintain_order=True)
            .agg(pl.col("_idx").min().alias("_start"), pl.len().alias("_n"))
        )
        starts = counts["_start"].to_list()
        lens = counts["_n"].to_list()
        indexors = [slice(s, s + n) for s, n in zip(starts, lens)]

        sales_dollars = df_sorted["value"].to_numpy()
        sales_units = df_sorted["y"].to_numpy()

        try:
            asp, edp, discount = decompose_price(
                sales_dollars=sales_dollars,
                sales_units=sales_units,
                indexors=indexors,
            )
        except Exception as exc:
            logger.warning(
                "EDP batched con indexors falló (%s); fallback a loop por serie.",
                exc,
            )
            return RLSForecastPipeline._calculate_edp_per_series(df_sorted)

        return df_sorted.with_columns(
            pl.Series("asp", np.asarray(asp, dtype=np.float64)),
            pl.Series("edp", np.asarray(edp, dtype=np.float64)),
            pl.Series("discount", np.asarray(discount, dtype=np.float64)),
        )

    def _run_section(
        self,
        selected: pl.LazyFrame | pl.DataFrame,
        seccion: str,
        first_data: dt.date,
        driver_cols: list[str] | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
        """Ejecuta train / OOS / forecast-only para una sección."""
        hz = settings.section_horizons(seccion, first_data)
        logger.info(
            "Sección %s | train [%s → %s] | OOS [%s → %s] | fcst [%s → %s]",
            seccion,
            hz["train_start"],
            hz["train_end"],
            hz["test_start"],
            hz["test_end"],
            hz["forecast_start"],
            hz["forecast_end"],
        )

        date_col = self._cfg.date_column
        # Filtros lazy: pushdown a scan_parquet (sección + ventana de fechas)
        selected_lf = (
            selected.lazy() if isinstance(selected, pl.DataFrame) else selected
        )
        sec_lf = selected_lf.filter(pl.col("SECCION") == seccion)

        raw_train_lf = sec_lf.filter(
            (pl.col(date_col) >= hz["train_start"])
            & (pl.col(date_col) <= hz["train_end"])
        )
        raw_oos_lf = sec_lf.filter(
            (pl.col(date_col) >= hz["test_start"])
            & (pl.col(date_col) <= hz["test_end"])
        )

        # ── Agregación + densify + EDP + features (TRAIN) ─────────────────────
        with _stage_timer(f"{seccion}: agregación train (lazy+streaming)"):
            df_train = self._aggregator.aggregate(raw_train_lf)
        logger.info(
            "Sección %s: train agregado shape=%s | series=%d",
            seccion,
            df_train.shape,
            df_train["unique_id"].n_unique() if df_train.height else 0,
        )

        df_train = self._apply_limit_series(df_train)

        n_before = df_train.height
        with _stage_timer(f"{seccion}: densify train"):
            df_train = densify_section_panel(
                df_train, hz["train_start"], hz["train_end"]
            )
        logger.info(
            "Sección %s: train densificado %d → %d filas (spine %s→%s)",
            seccion,
            n_before,
            df_train.height,
            hz["train_start"],
            hz["train_end"],
        )

        with _stage_timer(f"{seccion}: EDP train"):
            df_train = self._calculate_edp(df_train)

        with _stage_timer(f"{seccion}: features train"):
            df_train = self._feature_builder.extract_drivers(
                df_train, req_columns=driver_cols
            ).sort("ds")
        logger.info("Sección %s: features train OK shape=%s", seccion, df_train.shape)

        if driver_cols is None:
            driver_cols = [
                c
                for c in df_train.columns
                if c
                not in (
                    "ds",
                    "y",
                    "value",
                    "valuehat",
                    "unique_id",
                    "conteo_sku",
                    "sku_desc",
                    "store_name",
                    "seccion",
                )
            ]

        # ── Agregación + densify + EDP + features (OOS) ────────────────────────
        with _stage_timer(f"{seccion}: agregación+densify OOS (lazy+streaming)"):
            # aggregate materializa; si no hay filas OOS devuelve vacío
            df_oos = self._aggregator.aggregate(raw_oos_lf)
            if df_oos.height:
                # densify incluye series de train ausentes en OOS vía extra_uids
                train_uids = df_train.select("unique_id").unique()
                df_oos = densify_section_panel(
                    df_oos,
                    hz["test_start"],
                    hz["test_end"],
                    extra_uids=train_uids,
                )
                # Meta completa desde train (cubre series solo presentes en train)
                meta_cols = [
                    c
                    for c in ("sku_desc", "store_name", "seccion", "conteo_sku")
                    if c in df_train.columns
                ]
                if meta_cols:
                    meta = (
                        df_train.select(["unique_id"] + meta_cols)
                        .group_by("unique_id")
                        .agg(
                            [pl.col(c).drop_nulls().first().alias(c) for c in meta_cols]
                        )
                    )
                    existing_meta = [c for c in meta_cols if c in df_oos.columns]
                    if existing_meta:
                        df_oos = df_oos.drop(existing_meta)
                    df_oos = df_oos.join(meta, on="unique_id", how="left")

        if df_oos.height:
            with _stage_timer(f"{seccion}: EDP OOS"):
                df_oos = self._calculate_edp(df_oos)
            with _stage_timer(f"{seccion}: features OOS"):
                df_oos = self._feature_builder.extract_drivers(
                    df_oos, req_columns=driver_cols
                ).sort("ds")
            logger.info("Sección %s: OOS densificado shape=%s", seccion, df_oos.shape)

        # ── Forecast-only (calendario sintético) ───────────────────────────────
        uids = df_train["unique_id"].unique().to_list()
        fcst_start = hz["forecast_start"]
        fcst_end = hz["forecast_end"]
        logger.info(
            "Sección %s: calendario forecast-only (%d series × %d días)…",
            seccion,
            len(uids),
            (fcst_end - fcst_start).days + 1,
        )
        with _stage_timer(f"{seccion}: forecast-only build+features"):
            df_fcst_raw = self._build_calendar_frame(
                uids, fcst_start, fcst_end, df_train
            )
            if df_fcst_raw.height:
                df_fcst = self._feature_builder.extract_drivers(
                    df_fcst_raw.with_columns(pl.lit(1).alias("intercept")),
                    req_columns=driver_cols,
                ).sort("ds")
                logger.info(
                    "Sección %s: forecast-only shape=%s", seccion, df_fcst.shape
                )
            else:
                df_fcst = pl.DataFrame()

        # ── Runner ─────────────────────────────────────────────────────────────
        runner = RLSForecastRunner(
            driver_cols=driver_cols,
            rmse_error=self._cfg.rmse_error,
            forgetting_factor=self._cfg.forgetting_factor,
            min_y_to_update=self._cfg.min_y_to_update,
            use_correction_factor=self._cfg.correction_factor,
            n_jobs=self._n_jobs,
        )

        meta = {
            "train_start": hz["train_start"],
            "train_end": hz["train_end"],
            "test_start": hz["test_start"],
            "test_end": hz["test_end"],
            "forecast_start": hz["forecast_start"],
            "forecast_end": hz["forecast_end"],
        }

        targets: dict[str, pl.DataFrame] = {"in_sample": df_train}
        if df_oos.height:
            targets["out_sample"] = df_oos
        if df_fcst.height:
            targets["forecast_only"] = df_fcst

        # ── 1) RLS SOLO a nivel sección ────────────────────────────────────────
        train_section = df_train.filter(pl.col("unique_id") == seccion)
        targets_section = {
            name: df.filter(pl.col("unique_id") == seccion)
            for name, df in targets.items()
        }

        with _stage_timer(f"{seccion}: RLS sección (1 fit por variable)"):
            res_section, section_coefs_raw = runner.fit_and_predict_sections(
                train_section,
                targets_section,
                [seccion],
                desc=f"{seccion} RLS sección",
                meta=meta,
            )

        section_coefs = (
            {seccion: section_coefs_raw[seccion]}
            if section_coefs_raw and seccion in section_coefs_raw
            else {}
        )

        if res_section.height:
            res_section = res_section.with_columns(
                pl.when(pl.col("period_type") == "forecast_only")
                .then(0.0)
                .otherwise(pl.col("y"))
                .alias("y"),
                pl.when(pl.col("period_type") == "forecast_only")
                .then(0.0)
                .otherwise(pl.col("value"))
                .alias("value"),
                pl.lit(None).cast(pl.Float64).alias("driver_effect"),
                pl.lit(None).cast(pl.Float64).alias("driver_effect_value"),
            )

        if not section_coefs:
            logger.warning(
                "Sección %s: no se pudo ajustar RLS a nivel sección; "
                "no hay coeficientes para derivar tienda/sku.",
                seccion,
            )
            return res_section, pl.DataFrame(), driver_cols

        # ── 2) RLS a nivel tienda (sin duplicación) ────────────────────────────
        locales = settings.SECCIONES[seccion]["locales"]
        store_results: list[pl.DataFrame] = []
        store_coefs: dict[str, tuple] = {}

        def _process_one_store(store_id: str):
            """Ajusta RLS para una tienda. Devuelve (res_df | None, coefs_dict)."""
            try:
                store_uid = settings.make_unique_id(seccion, store=store_id)
                train_store = df_train.filter(pl.col("unique_id") == store_uid)
                if train_store.height == 0:
                    logger.warning(
                        "Sección %s: sin datos para tienda %s", seccion, store_id
                    )
                    return None, {}

                store_targets = {
                    name: df.filter(pl.col("unique_id") == store_uid)
                    for name, df in targets.items()
                }
                store_res, store_coefs_raw = runner.fit_and_predict_sections(
                    train_store,
                    store_targets,
                    [store_uid],
                    desc=f"{seccion} tienda {store_id} RLS",
                    meta=meta,
                )

                coefs_out = {}
                if store_coefs_raw and store_uid in store_coefs_raw:
                    coefs_out[store_uid] = store_coefs_raw[store_uid]

                res_out = None
                if store_res.height:
                    res_out = store_res.with_columns(
                        pl.when(pl.col("period_type") == "forecast_only")
                        .then(0.0)
                        .otherwise(pl.col("y"))
                        .alias("y"),
                        pl.when(pl.col("period_type") == "forecast_only")
                        .then(0.0)
                        .otherwise(pl.col("value"))
                        .alias("value"),
                        pl.lit(None).cast(pl.Float64).alias("driver_effect"),
                        pl.lit(None).cast(pl.Float64).alias("driver_effect_value"),
                    )
                return res_out, coefs_out

            except Exception as exc:
                logger.warning(
                    "Sección %s: error RLS tienda %s: %s", seccion, store_id, exc
                )
                return None, {}

        if self._n_jobs and self._n_jobs > 1 and len(locales) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self._n_jobs, len(locales))
            ) as executor:
                futures = {
                    executor.submit(_process_one_store, sid): sid for sid in locales
                }
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{seccion} procesando tiendas",
                    unit="tienda",
                    leave=False,
                    ncols=50,  # Controla el ancho total
                    ascii="░█",  # Define los caracteres de llenado (vacío/lleno)
                    bar_format="{bar} [{n_fmt}/{total_fmt}] {desc}...",  # Estructura del texto
                ):
                    res_df, coefs_dict = fut.result()
                    if res_df is not None:
                        store_results.append(res_df)
                    if coefs_dict:
                        store_coefs.update(coefs_dict)
        else:
            for sid in locales:
                res_df, coefs_dict = _process_one_store(sid)
                if res_df is not None:
                    store_results.append(res_df)
                if coefs_dict:
                    store_coefs.update(coefs_dict)

        res_store = (
            pl.concat(store_results, how="diagonal_relaxed")
            if store_results
            else pl.DataFrame()
        )
        logger.info(
            "Sección %s: %d filas de pronósticos a nivel tienda",
            seccion,
            res_store.height,
        )

        # ── 3) Derivación SKU+tienda (train+OOS+fcst; tienda a tienda) ────────
        res_derived = pl.DataFrame()
        is_sku_store = (
            pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2
        )
        # Filtrar SKU+tienda ANTES de concat para no inflar el panel
        parts_sku: list[pl.DataFrame] = []
        tr = df_train.filter(is_sku_store)
        if tr.height:
            parts_sku.append(tr.with_columns(pl.lit("in_sample").alias("period_type")))
        if df_oos.height:
            oo = df_oos.filter(is_sku_store)
            if oo.height:
                parts_sku.append(
                    oo.with_columns(pl.lit("out_sample").alias("period_type"))
                )
        if df_fcst.height:
            fc = df_fcst.filter(is_sku_store)
            if fc.height:
                parts_sku.append(
                    fc.with_columns(pl.lit("forecast_only").alias("period_type"))
                )

        if parts_sku and section_coefs and store_coefs:
            sku_level = pl.concat(parts_sku, how="diagonal_relaxed")
            del parts_sku
            if meta:
                sku_level = sku_level.with_columns(
                    [pl.lit(v).alias(k) for k, v in meta.items()]
                )
            logger.info(
                "Sección %s: %d filas SKU+tienda (train+OOS+fcst)",
                seccion,
                sku_level.height,
            )
            with _stage_timer(f"{seccion}: derivación SKU+tienda (por tienda)"):
                res_derived = runner.derive_sku_store_forecasts(
                    sku_level,
                    seccion,
                    section_coefs,
                    store_coefs,
                )
            del sku_level
            logger.info(
                "Sección %s: %d filas derivadas SKU+tienda",
                seccion,
                res_derived.height,
            )
        else:
            del parts_sku

        # ── Combinar resultados ────────────────────────────────────────────────
        res_df = pl.concat(
            [f for f in (res_section, res_store, res_derived) if f.height],
            how="diagonal_relaxed",
        )
        if not res_df.height:
            return pl.DataFrame(), pl.DataFrame(), driver_cols

        wmapes_df = RLSForecastRunner._compute_wmape(res_df)
        return res_df, wmapes_df, driver_cols

    def run(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        stages = tqdm(
            total=3,
            desc="Pipeline RLS",
            unit="etapa",
            bar_format="{bar} [{n_fmt}/{total_fmt}] {desc}...",  # Estructura del texto
            ncols=50,  # Controla el ancho total
            ascii="░█",  # Define los caracteres de llenado (vacío/lleno)
        )

        def _advance(label: str) -> None:
            stages.set_postfix_str(label)
            stages.update(1)

        pipeline_t0 = time.perf_counter()

        self._calendar = HolidayCalendar(self._cfg.holidays, self._cfg.now_year)
        self._feature_builder = CalendarFeatureBuilder(self._calendar)

        selected = self._load()
        first_by_sec = self._first_data_by_section(selected)
        _advance("datos cargados")

        all_res, all_wm = [], []
        driver_cols = None
        for seccion in settings.FOCUS_SECTIONS:
            if seccion not in first_by_sec:
                logger.warning("Sin datos para sección %s; se omite.", seccion)
                continue
            sec_t0 = time.perf_counter()
            res, wm, driver_cols = self._run_section(
                selected, seccion, first_by_sec[seccion], driver_cols
            )
            logger.info(
                "⏱ Sección %s: TOTAL %.1fs", seccion, time.perf_counter() - sec_t0
            )
            if res.height:
                all_res.append(res)
                # Checkpoint (Fase 5): no perder trabajo si el proceso se corta.
                self._write_checkpoint(seccion, res)
            if wm.height:
                all_wm.append(wm)
        _advance("forecast por sección listo")

        res_df = (
            pl.concat(all_res, how="diagonal_relaxed") if all_res else pl.DataFrame()
        )
        wmapes_df = (
            pl.concat(all_wm, how="diagonal_relaxed") if all_wm else pl.DataFrame()
        )
        _advance("resultados consolidados")
        stages.close()
        logger.info("⏱ Pipeline RLS: TOTAL %.1fs", time.perf_counter() - pipeline_t0)
        return res_df, wmapes_df

    def _write_checkpoint(self, seccion: str, res: pl.DataFrame) -> None:
        try:
            self._cfg.out_dir.mkdir(parents=True, exist_ok=True)
            partial_path = (
                self._cfg.out_dir / f"forecast_seccion_{seccion}_partial.parquet"
            )
            res.write_parquet(
                partial_path, compression="zstd", compression_level=3, statistics=True
            )
            logger.info("✓ Checkpoint sección %s: %s", seccion, partial_path)
        except Exception:
            logger.exception("No se pudo escribir checkpoint de sección %s", seccion)

    def save(self, res_df: pl.DataFrame, wmapes_df: pl.DataFrame) -> None:
        self._cfg.out_dir.mkdir(parents=True, exist_ok=True)
        forecast_path = self._cfg.out_dir / "forecast.parquet"
        res_df.write_parquet(
            forecast_path, compression="zstd", compression_level=3, statistics=True
        )
        logger.info("✓ Forecast guardado en: %s", forecast_path)
        wmape_path = self._cfg.out_dir / "wmape.parquet"
        wmapes_df.write_parquet(
            wmape_path, compression="zstd", compression_level=3, statistics=True
        )
        logger.info("✓ WMAPE guardado en: %s", wmape_path)
        self._export_forecast_excel(res_df)

    def _export_forecast_excel(self, res_df: pl.DataFrame) -> None:
        """
        Por sección: Excel con SKU, Local, Forecast sumarizado
        en [forecast_start, forecast_end] (sin actuals).
        """
        if res_df.height == 0:
            return
        # Solo nivel SKU (profundidad 2: sec||store||sku) y periodo forecast_only
        sku_level = res_df.filter(
            pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2
        )
        if "period_type" in sku_level.columns:
            sku_level = sku_level.filter(pl.col("period_type") == "forecast_only")
        elif (
            "forecast_start" in sku_level.columns
            and "forecast_end" in sku_level.columns
        ):
            sku_level = sku_level.filter(
                (pl.col("ds").cast(pl.Date) >= pl.col("forecast_start").cast(pl.Date))
                & (pl.col("ds").cast(pl.Date) <= pl.col("forecast_end").cast(pl.Date))
            )
        if sku_level.height == 0:
            logger.warning("Sin filas forecast-only a nivel SKU para export Excel")
            return

        # Parse unique_id → seccion, local, sku (esquema T:/S: — ver settings.split_unique_id)
        uids = sku_level["unique_id"].to_list()
        parsed = [settings.split_unique_id(u) for u in uids]
        parts = sku_level.with_columns(
            pl.Series("SECCION", [p["seccion"] for p in parsed]),
            pl.Series("Local", [p["store"] for p in parsed]),
            pl.Series("SKU", [p["sku"] for p in parsed]),
        )

        for seccion in settings.FOCUS_SECTIONS:
            sec_df = parts.filter(pl.col("SECCION") == seccion)
            if sec_df.height == 0:
                continue
            summary = (
                sec_df.group_by(["SKU", "Local"])
                .agg(pl.col("yhat").sum().alias("Forecast sumarizado"))
                .sort(["Local", "SKU"])
                .select(["SKU", "Local", "Forecast sumarizado"])
            )
            out_path = self._cfg.out_dir / f"forecast_seccion_{seccion}.xlsx"
            try:
                summary.write_excel(out_path)
            except Exception:
                # fallback openpyxl / xlsxwriter no disponible → csv
                csv_path = out_path.with_suffix(".csv")
                summary.write_csv(csv_path)
                logger.warning("write_excel falló; exportado CSV: %s", csv_path)
                continue
            logger.info(
                "✓ Excel sección %s (%d filas): %s",
                seccion,
                summary.height,
                out_path,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline RLS sección→SKU→local")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Threads paralelos (p. ej. RLS por tienda). Default: secuencial.",
    )
    parser.add_argument(
        "--limit-series",
        type=int,
        default=None,
        help=(
            "Debug/benchmark: muestrea N SKUs por sección (conserva niveles "
            "sección/tienda) para iterar rápido. NO usar en producción."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ForecastConfig.from_settings()
    pipeline = RLSForecastPipeline(
        config, n_jobs=args.n_jobs, limit_series=args.limit_series
    )
    res_df, wmapes_df = pipeline.run()
    print(res_df.head())
    print(wmapes_df.head())
    pipeline.save(res_df, wmapes_df)


if __name__ == "__main__":
    main()
