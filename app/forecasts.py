"""
Pipeline de forecasting jerárquico con Regresión RLS
=====================================================
Niveles: sección (1 / 23) → SKU → local (tienda).

Ventanas por sección (SECCIONES):
  - train: [train_start, train_end]  train_end = test_start original
            train_start alineado a bloques de 28 días desde train_end,
            acotado por domingo ≥ primer dato
  - test (OOS): lunes ≥ test_start → test_end  → métricas out-of-sample
  - forecast-only: test_end+1 → test_end+28  (sin actuals, línea punteada)

100% Polars + numpy. OOP, paralelizable por unique_id.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
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


class StageTimer:
    """Cronómetro por etapa (Fase 0)."""

    def __init__(self, name: str):
        self.name = name
        self.t0 = time.perf_counter()

    def stop(self) -> float:
        dt = time.perf_counter() - self.t0
        logger.info("⏱ %s: %.2fs", self.name, dt)
        return dt



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
            min_y_to_update=getattr(settings, "MIN_Y_TO_update", 1.0),
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
        df = df.with_columns(pl.col("ds").dt.weekday().alias("weekday"))
        df = df.to_dummies(columns=["weekday"])
        df = df.with_columns(pl.col("ds").dt.month().alias("month"))
        df = df.to_dummies(columns=["month"])
        keep = [c for c in df.columns if not c.endswith("_1")]
        return df.select(keep)

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
            "ds", "y", "unique_id", "value", "valuehat", "conteo_sku",
            "sku_desc", "store_name", "seccion", "intercept",
            "asp", "edp", "discount",
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
    unique_id (sección → tienda → SKU):
      - "1"                  (sección)
      - "1||00122"           (tienda)
      - "1||00122||SKU123"   (SKU en tienda)
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

    def aggregate(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_map = {
            self._date_col: "ds",
            self._qty_col: "y",
            self._prc_col: "value",
        }
        df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})
        df = df.filter(pl.col("y") >= 0)

        for col in ("SECCION", "SKU_ID", "STORE_ID"):
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Utf8).str.strip_chars())

        # Mapas de descripción
        sku_desc_map: dict[str, str] = {}
        if "DESCRIPCION" in df.columns:
            tmp = (
                df.select(["SKU_ID", "DESCRIPCION"])
                .drop_nulls()
                .unique(subset=["SKU_ID"])
            )
            sku_desc_map = dict(
                zip(tmp["SKU_ID"].to_list(), tmp["DESCRIPCION"].to_list())
            )

        store_name_map: dict[tuple[str, str], str] = {}
        for sec, cfg in settings.SECCIONES.items():
            for loc, name in cfg["local_names"].items():
                store_name_map[(sec, loc)] = name

        frames: list[pl.DataFrame] = []
        level_keys = list(self._levels.keys())  # SECCION, SKU_ID, STORE_ID

        for depth in range(1, len(level_keys) + 1):
            keys = level_keys[:depth]
            id_expr = pl.concat_str([pl.col(k) for k in keys], separator="||")
            agg_exprs = [
                pl.col("y").sum(),
                pl.col("value").sum(),
                pl.count("y").alias("conteo_sku"),
            ]
            sum_df = (
                df.group_by(["ds"] + keys)
                .agg(agg_exprs)
                .with_columns(
                    id_expr.alias("unique_id"),
                    pl.col("SECCION").alias("seccion"),
                )
            )
            frames.append(sum_df)

        out = (
            pl.concat(frames, how="diagonal_relaxed")
            .with_columns(pl.lit(1).alias("intercept"))
            .sort(["unique_id", "ds"])
        )

        # Enriquecer descripciones: unique_id = sec || store || sku
        uids_df = out.select("unique_id").unique()
        parts = uids_df.with_columns(
            pl.col("unique_id").str.split_exact("||", 2).alias("_p")
        ).unnest("_p")
        # field_0=seccion, field_1=store, field_2=sku
        rename = {}
        cols = parts.columns
        for i, name in enumerate(["_sec", "_store", "_sku"]):
            src = f"field_{i}"
            if src in cols:
                rename[src] = name
        parts = parts.rename(rename) if rename else parts

        sku_df = (
            pl.DataFrame(
                {
                    "_sku": list(sku_desc_map.keys()),
                    "sku_desc": list(sku_desc_map.values()),
                }
            )
            if sku_desc_map
            else pl.DataFrame(schema={"_sku": pl.Utf8, "sku_desc": pl.Utf8})
        )
        store_rows = [
            {"_sec": sec, "_store": loc, "store_name": name}
            for (sec, loc), name in store_name_map.items()
        ]
        store_df = (
            pl.DataFrame(store_rows)
            if store_rows
            else pl.DataFrame(
                schema={"_sec": pl.Utf8, "_store": pl.Utf8, "store_name": pl.Utf8}
            )
        )

        meta = parts
        if "_sku" in meta.columns:
            meta = meta.join(sku_df, on="_sku", how="left")
        else:
            meta = meta.with_columns(pl.lit("").alias("sku_desc"))
        if "_sec" in meta.columns and "_store" in meta.columns:
            meta = meta.join(store_df, on=["_sec", "_store"], how="left")
        else:
            meta = meta.with_columns(pl.lit("").alias("store_name"))

        meta = meta.select(
            [
                "unique_id",
                pl.col("sku_desc").fill_null(""),
                pl.col("store_name").fill_null(""),
            ]
        )
        out = out.join(meta, on="unique_id", how="left")
        return out.select(
            [
                c
                for c in [
                    "unique_id",
                    "ds",
                    "y",
                    "value",
                    "intercept",
                    "conteo_sku",
                    "seccion",
                    "sku_desc",
                    "store_name",
                ]
                if c in out.columns
            ]
        )


# ─────────────────────────────────────────────────────────────────────────────
# RLS runner
# ─────────────────────────────────────────────────────────────────────────────
class RLSForecastRunner:
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
        self._n_jobs = n_jobs

    @staticmethod
    def _correction_factor(errors: np.ndarray) -> float:
        sigma2 = errors.var(ddof=1)
        return float(np.exp(sigma2 / 2))

    def _default_priors_static(self, n_features: int):
        return [
            RLSConstantPrior(standard_error=0.5, rmse_error=self._rmse_error)
        ] + [
            RLSPrior(coefficient=0, standard_error=0.5, rmse_error=self._rmse_error)
            for _ in range(max(0, n_features - 1))
        ]

    def _fit_models(self, train_g: pl.DataFrame):
        """Un solo fit de model_y y model_p sobre train_g (Fase 1)."""
        X_y = train_g.select(self._driver_cols).to_numpy().astype(np.float64, order="C")
        y = train_g["y"].to_numpy()
        log_y = np.log1p(y)
        model_y = RecursiveLeastSquaresRegression(
            forgetting_factor=self._forgetting_factor,
            min_y_to_update=self._min_y_to_update,
            return_all_coefs=False,
        )
        model_y.fit(
            x=X_y, y=log_y, priors=self._default_priors_static(X_y.shape[1])
        )

        X_p = (
            train_g.select(self._driver_cols_price)
            .to_numpy()
            .astype(np.float64, order="C")
        )
        price = train_g["value"].to_numpy() if "value" in train_g.columns else np.zeros(len(y))
        log_price = np.log1p(np.clip(price, 0.0, None))
        model_p = RecursiveLeastSquaresRegression(
            forgetting_factor=self._forgetting_factor,
            min_y_to_update=1e-8,
            return_all_coefs=False,
        )
        model_p.fit(
            x=X_p, y=log_price, priors=self._default_priors_static(X_p.shape[1])
        )
        return model_y, model_p

    def _predict_with_models(
        self,
        model_y,
        model_p,
        unique_id: str,
        train_g: pl.DataFrame,
        test_g: pl.DataFrame,
        meta: dict | None = None,
    ) -> pl.DataFrame | None:
        if test_g.height == 0:
            return None
        X_y_test = (
            test_g.select(self._driver_cols).to_numpy().astype(np.float64, order="C")
        )
        log_yhat = model_y.predict(X_y_test)
        if self._use_correction_factor:
            try:
                corr = self._correction_factor(np.asarray(model_y.errors))
                yhat = np.round(np.exp(log_yhat) * corr).ravel()
            except Exception:
                yhat = np.round(np.expm1(log_yhat)).ravel()
        else:
            yhat = np.round(np.expm1(log_yhat)).ravel()

        X_p_test = (
            test_g.select(self._driver_cols_price)
            .to_numpy()
            .astype(np.float64, order="C")
        )
        log_ph = model_p.predict(X_p_test)
        if self._use_correction_factor:
            try:
                corr_p = self._correction_factor(np.asarray(model_p.errors))
                valuehat = np.round(np.expm1(log_ph) * corr_p, 2).ravel()
            except Exception:
                valuehat = np.round(np.expm1(log_ph), 2).ravel()
        else:
            valuehat = np.round(np.expm1(log_ph), 2).ravel()

        y_real = (
            test_g["y"].to_numpy()
            if "y" in test_g.columns
            else np.zeros(len(yhat))
        )
        price_real = (
            test_g["value"].to_numpy()
            if "value" in test_g.columns
            else np.zeros(len(valuehat))
        )
        result = {
            "unique_id": unique_id,
            "ds": test_g["ds"],
            "value": price_real,
            "valuehat": valuehat,
            "y": y_real,
            "yhat": yhat,
        }
        if meta:
            for k, v in meta.items():
                result[k] = v
        for col in ("sku_desc", "store_name", "seccion"):
            if col in train_g.columns:
                result[col] = train_g[col][0]
            elif col in test_g.columns:
                result[col] = test_g[col][0]
        return pl.DataFrame(result)

    def _fit_predict_one(
        self,
        unique_id: str,
        train_g: pl.DataFrame,
        test_g: pl.DataFrame,
        meta: dict | None = None,
    ) -> pl.DataFrame | None:
        """Compat: fit único + un target (usado por run simple)."""
        if train_g.height == 0 or test_g.height == 0:
            return None
        if RecursiveLeastSquaresRegression is None:
            raise RuntimeError("rls_opt no disponible")
        try:
            model_y, model_p = self._fit_models(train_g)
            return self._predict_with_models(
                model_y, model_p, unique_id, train_g, test_g, meta
            )
        except Exception as exc:
            logger.warning("fit/predict falló %s: %s", unique_id, exc)
            return None

    def fit_and_predict_multi(
        self,
        unique_id: str,
        train_g: pl.DataFrame,
        targets: dict[str, pl.DataFrame],
        meta: dict | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fase 1: 1 fit y + 1 fit value, N predicts (in/OOS/forecast).
        targets: period_type -> DataFrame test.
        """
        out: dict[str, pl.DataFrame] = {}
        if train_g.height == 0 or RecursiveLeastSquaresRegression is None:
            return out
        try:
            model_y, model_p = self._fit_models(train_g)
        except Exception as exc:
            logger.warning("fit multi %s: %s", unique_id, exc)
            return out
        for name, tg in targets.items():
            if tg is None or tg.height == 0:
                continue
            try:
                pred = self._predict_with_models(
                    model_y, model_p, unique_id, train_g, tg, meta
                )
                if pred is not None:
                    out[name] = pred
            except Exception as exc:
                logger.warning("predict %s/%s: %s", unique_id, name, exc)
        return out

    def run_multi(
        self,
        train: pl.DataFrame,
        targets: dict[str, pl.DataFrame],
        forecast_levels: list[str],
        desc: str = "RLS multi-target",
        meta: dict | None = None,
        limit_series: int | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fit una vez por serie; predice todos los targets (Fase 1)."""
        unique_ids = self._build_tasks(train, forecast_levels)
        if limit_series and limit_series > 0:
            unique_ids = unique_ids[:limit_series]
            logger.info("%s: limit_series=%d", desc, limit_series)
        logger.info("%s: %d series", desc, len(unique_ids))
        needed = set(unique_ids)
        train_parts = self._normalize_partition_dict(
            train.filter(pl.col("unique_id").is_in(list(needed))).partition_by(
                "unique_id", as_dict=True
            )
        )
        target_parts: dict[str, dict] = {}
        for name, tdf in targets.items():
            if tdf is None or tdf.height == 0:
                target_parts[name] = {}
                continue
            target_parts[name] = self._normalize_partition_dict(
                tdf.filter(pl.col("unique_id").is_in(list(needed))).partition_by(
                    "unique_id", as_dict=True
                )
            )

        def _one(uid: str) -> dict[str, pl.DataFrame]:
            if uid not in train_parts:
                return {}
            tg = {
                name: parts[uid]
                for name, parts in target_parts.items()
                if uid in parts
            }
            if not tg:
                return {}
            return self.fit_and_predict_multi(uid, train_parts[uid], tg, meta)

        results_by_period: dict[str, list[pl.DataFrame]] = {k: [] for k in targets}
        if self._n_jobs and self._n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self._n_jobs) as ex:
                futs = {ex.submit(_one, uid): uid for uid in unique_ids}
                for fut in tqdm(
                    as_completed(futs), total=len(futs), desc=desc, unit="serie"
                ):
                    try:
                        got = fut.result()
                    except Exception as exc:
                        logger.warning("serie falló: %s", exc)
                        continue
                    for name, df in got.items():
                        results_by_period.setdefault(name, []).append(df)
        else:
            for uid in tqdm(unique_ids, desc=desc, unit="serie"):
                got = _one(uid)
                for name, df in got.items():
                    results_by_period.setdefault(name, []).append(df)

        frames = []
        for name, lst in results_by_period.items():
            if not lst:
                continue
            part = pl.concat(lst, how="diagonal_relaxed").with_columns(
                pl.lit(name).alias("period_type")
            )
            frames.append(part)
        res_df = (
            pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
        )
        wmapes_df = self._compute_wmape(res_df) if res_df.height else pl.DataFrame()
        return res_df, wmapes_df

    def _build_tasks(

        self, train: pl.DataFrame, forecast_levels: list[str], min_obs: int = 28
    ) -> list[str]:
        """Series con al menos min_obs observaciones en train (evita series vacías/ruido)."""
        counts = (
            train.group_by("unique_id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_obs)
        )
        eligible = set(counts["unique_id"].to_list())
        unique_ids: list[str] = []
        for i, _lvl in enumerate(forecast_levels):
            level_ids = (
                train.filter(
                    pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == i
                )["unique_id"]
                .unique()
                .to_list()
            )
            unique_ids.extend([u for u in level_ids if u in eligible])
        return unique_ids

    @staticmethod
    def _normalize_partition_dict(d: dict) -> dict:
        return {(k[0] if isinstance(k, tuple) else k): v for k, v in d.items()}


    # ── Rolling 28d (yhat28 / valuehat28) ───────────────────────────────────
    def _default_priors(self, n_features: int):
        return [
            RLSConstantPrior(standard_error=0.5, rmse_error=self._rmse_error)
        ] + [
            RLSPrior(coefficient=0, standard_error=0.5, rmse_error=self._rmse_error)
            for _ in range(max(0, n_features - 1))
        ]

    def _extract_priors(self, model, n_features: int):
        """Opción B: coeficientes del modelo final → priors RLS."""
        coefs = None
        for attr in ("coef_", "coefficients", "coefs", "beta", "theta"):
            if hasattr(model, attr):
                val = getattr(model, attr)
                if val is not None:
                    coefs = np.asarray(val, dtype=np.float64).ravel()
                    break
        if coefs is None or coefs.size == 0:
            return self._default_priors(n_features)
        # Ajustar longitud
        if coefs.size < n_features:
            coefs = np.pad(coefs, (0, n_features - coefs.size))
        elif coefs.size > n_features:
            coefs = coefs[:n_features]
        priors = [
            RLSPrior(
                coefficient=float(coefs[0]),
                standard_error=0.5,
                rmse_error=self._rmse_error,
            )
        ]
        # Prefer ConstantPrior for intercept if constructor fails on coefficient
        try:
            priors[0] = RLSConstantPrior(
                standard_error=0.5, rmse_error=self._rmse_error
            )
            # Keep coefficient via RLSPrior for intercept when supported
            priors[0] = RLSPrior(
                coefficient=float(coefs[0]),
                standard_error=0.5,
                rmse_error=self._rmse_error,
            )
        except Exception:
            priors[0] = RLSConstantPrior(
                standard_error=0.5, rmse_error=self._rmse_error
            )
        for i in range(1, n_features):
            priors.append(
                RLSPrior(
                    coefficient=float(coefs[i]),
                    standard_error=0.5,
                    rmse_error=self._rmse_error,
                )
            )
        return priors

    def _new_rls(self, min_y: float):
        return RecursiveLeastSquaresRegression(
            forgetting_factor=self._forgetting_factor,
            min_y_to_update=min_y,
            return_all_coefs=False,
        )

    def _fit_rls(self, X, y_log, priors, min_y: float):
        model = self._new_rls(min_y)
        if X.shape[0] == 0:
            return model, priors
        model.fit(x=X, y=y_log, priors=priors)
        return model, self._extract_priors(model, X.shape[1])

    def _predict_rls(self, model, X, use_corr: bool) -> np.ndarray:
        if X.shape[0] == 0:
            return np.array([])
        log_hat = model.predict(X)
        if use_corr and hasattr(model, "errors") and model.errors is not None:
            try:
                corr = self._correction_factor(np.asarray(model.errors))
                return np.round(np.exp(log_hat) * corr).ravel()
            except Exception:
                pass
        return np.round(np.expm1(log_hat)).ravel()

    @staticmethod
    def _first_monday_on_or_after(d: dt.date) -> dt.date:
        # weekday: Mon=0 … Sun=6
        return d + dt.timedelta(days=(7 - d.weekday()) % 7)

    def _rolling_28_one(self, unique_id: str, g: pl.DataFrame) -> pl.DataFrame | None:
        """
        Rolling 28d O(n): un fit sobre actuals (opción B = modelo final).
        Predicción con model.predict (misma vía que yhat/valuehat) sobre todo el
        panel — evita coeficientes manuales X@β que producían valuehat28≈0 cuando
        return_all_coefs no expone trayectoria usable.
        """
        if RecursiveLeastSquaresRegression is None or g.height == 0:
            return None
        g = g.sort("ds")
        first_ds = g["ds"][0]
        if isinstance(first_ds, dt.datetime):
            first_ds = first_ds.date()
        start_mon = self._first_monday_on_or_after(first_ds)
        g = g.filter(pl.col("ds").cast(pl.Date) >= start_mon)
        if g.height == 0:
            return None

        dcols = [c for c in self._driver_cols if c in g.columns]
        pdcols = [c for c in self._driver_cols_price if c in g.columns]
        if not dcols:
            return None

        X_all = g.select(dcols).to_numpy().astype(np.float64, order="C")
        y_all = g["y"].to_numpy().astype(np.float64)
        has_value = "value" in g.columns
        if has_value and pdcols:
            Xp_all = g.select(pdcols).to_numpy().astype(np.float64, order="C")
            v_all = np.clip(g["value"].to_numpy().astype(np.float64), 0.0, None)
        else:
            Xp_all = X_all
            v_all = np.zeros_like(y_all)

        n = len(y_all)
        actual_mask = np.isfinite(y_all) & (y_all != 0)
        idx_act = np.where(actual_mask)[0]
        if idx_act.size < 2:
            return None

        yhat28 = np.full(n, np.nan)
        valuehat28 = np.full(n, np.nan)

        try:
            model_y = RecursiveLeastSquaresRegression(
                forgetting_factor=self._forgetting_factor,
                min_y_to_update=self._min_y_to_update,
                return_all_coefs=False,
            )
            model_y.fit(
                x=X_all[idx_act],
                y=np.log1p(y_all[idx_act]),
                priors=self._default_priors(X_all.shape[1]),
            )
            log_yhat = np.asarray(model_y.predict(X_all), dtype=np.float64).ravel()
            if self._use_correction_factor:
                try:
                    corr = self._correction_factor(np.asarray(model_y.errors))
                    yhat28 = np.round(np.exp(log_yhat) * corr).ravel()
                except Exception:
                    yhat28 = np.round(np.expm1(log_yhat)).ravel()
            else:
                yhat28 = np.round(np.expm1(log_yhat)).ravel()
        except Exception as exc:
            logger.warning("rolling28 y %s: %s", unique_id, exc)
            return None

        if has_value:
            actual_v = np.isfinite(v_all) & (v_all != 0)
            idx_v = np.where(actual_v)[0]
            if idx_v.size >= 2:
                try:
                    model_p = RecursiveLeastSquaresRegression(
                        forgetting_factor=self._forgetting_factor,
                        min_y_to_update=1e-8,
                        return_all_coefs=False,
                    )
                    model_p.fit(
                        x=Xp_all[idx_v],
                        y=np.log1p(v_all[idx_v]),
                        priors=self._default_priors(Xp_all.shape[1]),
                    )
                    log_vhat = np.asarray(
                        model_p.predict(Xp_all), dtype=np.float64
                    ).ravel()
                    if self._use_correction_factor:
                        try:
                            corr_p = self._correction_factor(
                                np.asarray(model_p.errors)
                            )
                            valuehat28 = np.round(np.expm1(log_vhat) * corr_p, 2).ravel()
                        except Exception:
                            valuehat28 = np.round(np.expm1(log_vhat), 2).ravel()
                    else:
                        valuehat28 = np.round(np.expm1(log_vhat), 2).ravel()
                except Exception as exc:
                    logger.warning("rolling28 value %s: %s", unique_id, exc)

        return pl.DataFrame(
            {
                "unique_id": unique_id,
                "ds": g["ds"],
                "yhat28": yhat28,
                "valuehat28": valuehat28,
            }
        )

    def run_rolling_28(


        self,
        panel: pl.DataFrame,
        forecast_levels: list[str],
        desc: str = "Rolling 28d",
    ) -> pl.DataFrame:
        """Genera yhat28/valuehat28 para todas las series del panel."""
        if panel.height == 0 or RecursiveLeastSquaresRegression is None:
            return pl.DataFrame(
                schema={
                    "unique_id": pl.Utf8,
                    "ds": pl.Date,
                    "yhat28": pl.Float64,
                    "valuehat28": pl.Float64,
                }
            )
        unique_ids = self._build_tasks(panel, forecast_levels, min_obs=28)
        parts = self._normalize_partition_dict(
            panel.filter(pl.col("unique_id").is_in(unique_ids)).partition_by(
                "unique_id", as_dict=True
            )
        )
        frames = []
        uids_ok = [u for u in unique_ids if u in parts]

        def _roll_one(uid: str):
            try:
                return self._rolling_28_one(uid, parts[uid])
            except Exception as exc:
                logger.warning("rolling28 %s: %s", uid, exc)
                return None

        if self._n_jobs and self._n_jobs > 1 and len(uids_ok) > 1:
            with ThreadPoolExecutor(max_workers=self._n_jobs) as ex:
                futs = [ex.submit(_roll_one, u) for u in uids_ok]
                for fut in tqdm(
                    as_completed(futs), total=len(futs), desc=desc, unit="serie"
                ):
                    one = fut.result()
                    if one is not None and one.height:
                        frames.append(one)
        else:
            for uid in tqdm(uids_ok, desc=desc, leave=False):
                one = _roll_one(uid)
                if one is not None and one.height:
                    frames.append(one)
        if not frames:
            return pl.DataFrame(
                schema={
                    "unique_id": pl.Utf8,
                    "ds": pl.Date,
                    "yhat28": pl.Float64,
                    "valuehat28": pl.Float64,
                }
            )
        return pl.concat(frames, how="diagonal_relaxed")

    @staticmethod
    def compute_wmape28(panel: pl.DataFrame) -> pl.DataFrame:
        """WMAPE/BIAS de yhat28 vs y (excluye y==0 y nulos)."""
        if panel.height == 0 or "yhat28" not in panel.columns:
            return pl.DataFrame(
                schema={
                    "unique_id": pl.Utf8,
                    "wmape_28": pl.Float64,
                    "bias_28": pl.Float64,
                    "n_points_28": pl.UInt32,
                }
            )
        scored = panel.filter(
            pl.col("y").is_not_null()
            & (pl.col("y") != 0)
            & pl.col("yhat28").is_not_null()
        )
        if scored.height == 0:
            return pl.DataFrame(
                schema={
                    "unique_id": pl.Utf8,
                    "wmape_28": pl.Float64,
                    "bias_28": pl.Float64,
                    "n_points_28": pl.UInt32,
                }
            )
        return (
            scored.group_by("unique_id")
            .agg(
                (pl.col("y") - pl.col("yhat28")).abs().sum().alias("_ae"),
                pl.col("y").abs().sum().alias("_ay"),
                pl.col("y").sum().alias("_sy"),
                pl.col("yhat28").sum().alias("_sh"),
                pl.len().alias("n_points_28"),
            )
            .filter(pl.col("_ay") != 0)
            .with_columns(
                (pl.col("_ae") / pl.col("_ay")).alias("wmape_28"),
                ((pl.col("_sh") - pl.col("_sy")) / pl.col("_sy")).alias("bias_28"),
            )
            .select(["unique_id", "wmape_28", "bias_28", "n_points_28"])
        )


    def run(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        forecast_levels: list[str],
        desc: str = "Ajustando RLS",
        meta: dict | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        unique_ids = self._build_tasks(train, forecast_levels)
        logger.info("%s: %d series a ajustar", desc, len(unique_ids))

        # Particionar solo las series necesarias (evita dict gigante con ruido)
        needed = set(unique_ids)
        train_f = train.filter(pl.col("unique_id").is_in(list(needed)))
        test_f = test.filter(pl.col("unique_id").is_in(list(needed)))

        train_parts = self._normalize_partition_dict(
            train_f.partition_by("unique_id", as_dict=True)
        )
        test_parts = self._normalize_partition_dict(
            test_f.partition_by("unique_id", as_dict=True)
        )
        tasks = [
            (uid, train_parts[uid], test_parts[uid])
            for uid in unique_ids
            if uid in train_parts and uid in test_parts
        ]
        logger.info("%s: %d tasks con train+test", desc, len(tasks))

        results: list[pl.DataFrame] = []
        if self._n_jobs and self._n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self._n_jobs) as executor:
                futures = [
                    executor.submit(self._fit_predict_one, *task, meta)
                    for task in tasks
                ]
                for fut in tqdm(
                    as_completed(futures), total=len(futures), desc=desc, unit="serie"
                ):
                    res = fut.result()
                    if res is not None:
                        results.append(res)
        else:
            for uid, tr, te in tqdm(tasks, desc=desc, unit="serie"):
                res = self._fit_predict_one(uid, tr, te, meta)
                if res is not None:
                    results.append(res)

        if not results:
            return pl.DataFrame(), pl.DataFrame()

        res_df = pl.concat(results, how="diagonal_relaxed")
        wmapes_df = self._compute_wmape(res_df)
        return res_df, wmapes_df

    @staticmethod
    def _compute_wmape(res_df: pl.DataFrame) -> pl.DataFrame:
        """
        WMAPE = Σ|y − ŷ| / Σ|y|  (fracción).
        Excluye y == 0 / nulos y period_type == forecast_only.
        """
        scored = res_df.filter(pl.col("y").is_not_null() & (pl.col("y") != 0))
        if "period_type" in scored.columns:
            scored = scored.filter(pl.col("period_type") != "forecast_only")
        if scored.height == 0:
            return pl.DataFrame(
                schema={
                    "unique_id": pl.Utf8,
                    "wmape": pl.Float64,
                    "bias": pl.Float64,
                    "n_points": pl.UInt32,
                    "sum_y": pl.Float64,
                    "sum_yhat": pl.Float64,
                }
            )
        return (
            scored.group_by("unique_id")
            .agg(
                (pl.col("y") - pl.col("yhat")).abs().sum().alias("_abs_error"),
                pl.col("y").abs().sum().alias("_abs_actual"),
                pl.col("y").sum().alias("sum_y"),
                pl.col("yhat").sum().alias("sum_yhat"),
                pl.len().alias("n_points"),
            )
            .with_columns(
                (pl.col("_abs_error") / pl.col("_abs_actual")).alias("wmape"),
                (
                    (pl.col("sum_yhat") - pl.col("sum_y"))
                    / pl.col("sum_y").replace(0, None)
                ).alias("bias"),
            )
            .select(["unique_id", "wmape", "bias", "n_points", "sum_y", "sum_yhat"])
        )



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
    """
    if date_end < date_start:
        return df
    if df.height == 0 and (extra_uids is None or extra_uids.height == 0):
        return df

    spine = pl.DataFrame(
        {"ds": pl.date_range(date_start, date_end, interval="1d", eager=True)}
    )
    n_days = spine.height

    if df.height:
        df = df.with_columns(pl.col("ds").cast(pl.Date))
        uids = df.select("unique_id").unique()
    else:
        uids = pl.DataFrame({"unique_id": pl.Series([], dtype=pl.Utf8)})
    if extra_uids is not None and extra_uids.height:
        uids = pl.concat([uids, extra_uids.select("unique_id")], how="diagonal_relaxed").unique()

    if uids.height == 0:
        return df

    grid = uids.join(spine, how="cross")

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
            c
            for c in df.columns
            if c not in ("unique_id", "ds") and c not in meta_cols
        ]
        join_df = df.select(["unique_id", "ds"] + data_cols)
        out = grid.join(join_df, on=["unique_id", "ds"], how="left")
    else:
        out = grid

    fills = []
    if "y" in out.columns:
        fills.append(pl.col("y").fill_null(fill_y))
    elif df.height == 0 or "y" not in (df.columns if df.height else []):
        fills.append(pl.lit(fill_y).alias("y"))
    if "value" in out.columns:
        fills.append(pl.col("value").fill_null(fill_value))
    elif "value" not in out.columns:
        fills.append(pl.lit(fill_value).alias("value"))
    if "conteo_sku" in out.columns:
        fills.append(pl.col("conteo_sku").fill_null(0))
    if fills:
        out = out.with_columns(fills)

    logger.info(
        "densify: %d series × %d días = %d filas",
        uids.height,
        n_days,
        out.height,
    )
    return out.sort(["unique_id", "ds"])


class RLSForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig,
        n_jobs: int | None = None,
        limit_series: int | None = None,
    ):
        self._cfg = config
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

    def _load(self) -> pl.DataFrame:
        logger.info("Cargando selected desde %s", self._cfg.selected_path)
        # scan + collect: permite que el engine elija proyecciones posteriores
        selected = (
            pl.scan_parquet(self._cfg.selected_path)
            .with_columns(
                pl.col(self._cfg.date_column).cast(pl.Date),
            )
            .collect()
        )
        logger.info("Selected shape: %s", selected.shape)
        return selected

    def _first_data_by_section(self, selected: pl.DataFrame) -> dict[str, dt.date]:
        col = self._cfg.date_column
        rows = (
            selected.group_by("SECCION")
            .agg(pl.col(col).min().alias("min_d"))
            .iter_rows(named=True)
        )
        out = {}
        for r in rows:
            d = r["min_d"]
            if isinstance(d, dt.datetime):
                d = d.date()
            out[str(r["SECCION"])] = d
        return out

    def _build_calendar_frame(
        self, unique_ids: list[str], start: dt.date, end: dt.date, template: pl.DataFrame
    ) -> pl.DataFrame:
        """Genera filas ds para [start, end] por unique_id (y=0) sin loops por celda en Polars."""
        n_days = (end - start).days + 1
        if n_days <= 0 or not unique_ids:
            return pl.DataFrame()

        dates = pl.date_range(start, end, interval="1d", eager=True)

        # Meta una sola pasada (unique_id → sku_desc / store_name / seccion)
        meta_cols = [c for c in ("sku_desc", "store_name", "seccion") if c in template.columns]
        if meta_cols:
            meta = (
                template.select(["unique_id"] + meta_cols)
                .unique(subset=["unique_id"])
                .filter(pl.col("unique_id").is_in(unique_ids))
            )
        else:
            meta = pl.DataFrame({"unique_id": unique_ids})

        # Cross join fechas × series (vectorizado)
        ids_df = pl.DataFrame({"unique_id": unique_ids})
        grid = ids_df.join(pl.DataFrame({"ds": dates}), how="cross")
        grid = grid.join(meta, on="unique_id", how="left").with_columns(
            pl.lit(0.0).alias("y"),
            pl.lit(0.0).alias("value"),
            pl.lit(1).alias("intercept"),
            pl.lit(0).alias("conteo_sku"),
        )
        return grid

    @staticmethod
    def _calculate_edp(df: pl.DataFrame) -> pl.DataFrame:
        """
        ASP / EDP / discount.

        `decompose_price(..., indexors=...)` falla en nopython (tipos de slice
        incompatibles en rls_opt actual). Estrategia:
          1) Fórmula vectorizada Polars (rápida, estable) — default.
          2) Opcional: decompose_price por serie SOLO si EDP_USE_LOOP=1 y
             n_series es bajo (< 500).
        """
        if df.height == 0:
            return df

        def _vectorized(frame: pl.DataFrame) -> pl.DataFrame:
            return frame.with_columns(
                (pl.col("value") / pl.col("y").clip(lower_bound=1e-8))
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("asp"),
                (pl.col("value") / pl.col("y").clip(lower_bound=1e-8))
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("edp"),
                pl.lit(0.0).alias("discount"),
            )

        if decompose_price is None:
            return _vectorized(df)

        import os

        use_loop = os.environ.get("EDP_USE_LOOP", "0") == "1"
        n_series = df["unique_id"].n_unique()
        if not use_loop or n_series > 500:
            if n_series > 500 and use_loop:
                logger.info(
                    "EDP: %d series → vectorizado (loop desactivado por tamaño)",
                    n_series,
                )
            return _vectorized(df)

        # Loop controlado (solo debug / series pocas)
        logger.info("EDP: loop decompose_price sobre %d series…", n_series)
        ordered = df.sort(["unique_id", "ds"])
        parts: list[pl.DataFrame] = []
        for part in ordered.partition_by("unique_id", maintain_order=True):
            try:
                a, e, d = decompose_price(
                    sales_dollars=part["value"].to_numpy().astype(np.float64),
                    sales_units=part["y"].to_numpy().astype(np.float64),
                )
                parts.append(
                    part.with_columns(
                        pl.Series("asp", np.asarray(a, dtype=np.float64).ravel()),
                        pl.Series("edp", np.asarray(e, dtype=np.float64).ravel()),
                        pl.Series(
                            "discount", np.asarray(d, dtype=np.float64).ravel()
                        ),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "EDP serie %s falló (%s); vectorizado local",
                    part["unique_id"][0],
                    exc,
                )
                parts.append(_vectorized(part))
        return pl.concat(parts, how="vertical")

    def _run_section(


        self,
        selected: pl.DataFrame,
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
        sec_df = selected.filter(pl.col("SECCION") == seccion)

        raw_train = sec_df.filter(
            (pl.col(date_col) >= hz["train_start"])
            & (pl.col(date_col) <= hz["train_end"])
        )
        raw_oos = sec_df.filter(
            (pl.col(date_col) >= hz["test_start"])
            & (pl.col(date_col) <= hz["test_end"])
        )
        logger.info(
            "Sección %s raw train=%d filas | oos=%d filas",
            seccion,
            raw_train.height,
            raw_oos.height,
        )

        t_agg = StageTimer(f"{seccion} aggregate train")
        logger.info("Sección %s: agregando train…", seccion)
        df_train = self._aggregator.aggregate(raw_train)
        t_agg.stop()
        logger.info(
            "Sección %s: train agregado shape=%s | series=%d",
            seccion,
            df_train.shape,
            df_train["unique_id"].n_unique() if df_train.height else 0,
        )

        # Panel denso train: spine settings [train_start, train_end]
        n_before = df_train.height
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

        t_edp = StageTimer(f"{seccion} EDP")
        logger.info("Sección %s: EDP…", seccion)
        df_train = self._calculate_edp(df_train)
        t_edp.stop()

        logger.info("Sección %s: features train…", seccion)
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

        logger.info("Sección %s: agregando OOS…", seccion)
        df_oos = self._aggregator.aggregate(raw_oos) if raw_oos.height else pl.DataFrame()
        if df_oos.height:
            # Densificar OOS al spine de la sección [test_start, test_end]
            # y alinear unique_ids con train (mismas series)
            train_uids = df_train.select("unique_id").unique()
            df_oos = densify_section_panel(
                df_oos, hz["test_start"], hz["test_end"]
            )
            # asegurar todas las series de train también en OOS (ceros si no hubo venta)
            oos_uids = df_oos.select("unique_id").unique()
            missing = train_uids.join(oos_uids, on="unique_id", how="anti")
            if missing.height:
                spine_oos = pl.DataFrame(
                    {
                        "ds": pl.date_range(
                            hz["test_start"], hz["test_end"], interval="1d", eager=True
                        )
                    }
                )
                extra = missing.join(spine_oos, how="cross").with_columns(
                    pl.lit(0.0).alias("y"),
                    pl.lit(0.0).alias("value"),
                )
                # copiar meta desde train
                meta_cols = [
                    c
                    for c in ("sku_desc", "store_name", "seccion", "conteo_sku")
                    if c in df_train.columns
                ]
                if meta_cols:
                    meta = (
                        df_train.select(["unique_id"] + meta_cols)
                        .group_by("unique_id")
                        .agg([pl.col(c).drop_nulls().first().alias(c) for c in meta_cols])
                    )
                    extra = extra.join(meta, on="unique_id", how="left")
                df_oos = pl.concat([df_oos, extra], how="diagonal_relaxed")
            df_oos = self._calculate_edp(df_oos)
            df_oos = self._feature_builder.extract_drivers(
                df_oos, req_columns=driver_cols
            ).sort("ds")
            logger.info("Sección %s: OOS densificado shape=%s", seccion, df_oos.shape)

        # Frame de solo-forecast (calendario sintético)
        uids = df_train["unique_id"].unique().to_list()
        # Solo-forecast: día siguiente a test_end → forecast_end (settings)
        fcst_start = hz["forecast_start"]
        fcst_end = hz["forecast_end"]
        logger.info(
            "Sección %s: calendario forecast-only (%d series × %d días)…",
            seccion,
            len(uids),
            (fcst_end - fcst_start).days + 1,
        )
        df_fcst_raw = self._build_calendar_frame(
            uids, fcst_start, fcst_end, df_train
        )
        if df_fcst_raw.height:
            df_fcst = self._feature_builder.extract_drivers(
                df_fcst_raw.with_columns(pl.lit(1).alias("intercept")),
                req_columns=driver_cols,
            ).sort("ds")
            logger.info("Sección %s: forecast-only shape=%s", seccion, df_fcst.shape)
        else:
            df_fcst = pl.DataFrame()

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

        # Fase 1: un fit por serie → predicción in / OOS / forecast
        targets: dict[str, pl.DataFrame] = {"in_sample": df_train}
        if df_oos.height:
            targets["out_sample"] = df_oos
        if df_fcst.height:
            targets["forecast_only"] = df_fcst

        t_fit = StageTimer(f"{seccion} RLS multi-target")
        res_df, wmapes_df = runner.run_multi(
            df_train,
            targets,
            self._cfg.forecast_levels,
            desc=f"{seccion} RLS",
            meta=meta,
            limit_series=getattr(self, "_limit_series", None),
        )
        t_fit.stop()
        if res_df.height and "forecast_only" in targets:
            # y/value del tramo solo-forecast no son actuals
            res_df = res_df.with_columns(
                pl.when(pl.col("period_type") == "forecast_only")
                .then(pl.lit(0.0))
                .otherwise(pl.col("y"))
                .alias("y"),
                pl.when(pl.col("period_type") == "forecast_only")
                .then(pl.lit(0.0))
                .otherwise(pl.col("value"))
                .alias("value"),
            )
        if not res_df.height:
            return pl.DataFrame(), pl.DataFrame(), driver_cols

        # ── Rolling 28d sobre toda la historia (train+OOS+forecast panel) ──
        panel_parts = [df_train]
        if df_oos.height:
            panel_parts.append(df_oos)
        if df_fcst.height:
            panel_parts.append(df_fcst)
        panel = pl.concat(panel_parts, how="diagonal_relaxed").sort(
            ["unique_id", "ds"]
        )
        logger.info(
            "Sección %s: rolling 28d sobre panel shape=%s…", seccion, panel.shape
        )
        t_roll = StageTimer(f"{seccion} rolling28")
        roll = runner.run_rolling_28(
            panel, self._cfg.forecast_levels, desc=f"{seccion} rolling28"
        )
        t_roll.stop()
        if roll.height and res_df.height:
            # alinear tipos de ds
            if res_df["ds"].dtype != roll["ds"].dtype:
                roll = roll.with_columns(pl.col("ds").cast(res_df["ds"].dtype))
            res_df = res_df.join(
                roll.select(["unique_id", "ds", "yhat28", "valuehat28"]),
                on=["unique_id", "ds"],
                how="left",
            )
            wm28 = RLSForecastRunner.compute_wmape28(
                res_df.filter(
                    pl.col("y").is_not_null()
                    & pl.col("yhat28").is_not_null()
                )
            )
            if wm28.height:
                if "wmape" in wmapes_df.columns and "unique_id" in wmapes_df.columns:
                    wmapes_df = wmapes_df.join(wm28, on="unique_id", how="left")
                else:
                    wmapes_df = wm28
            logger.info(
                "Sección %s: yhat28 unido (%d filas roll, %d series wm28)",
                seccion,
                roll.height,
                wm28.height if wm28.height else 0,
            )
        elif res_df.height:
            res_df = res_df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("yhat28"),
                pl.lit(None).cast(pl.Float64).alias("valuehat28"),
            )

        return res_df, wmapes_df, driver_cols

    def run(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        stages = tqdm(
            total=3,
            desc="Pipeline RLS",
            unit="etapa",
            bar_format="{l_bar}{bar}| {postfix}",
        )

        def _advance(label: str) -> None:
            stages.set_postfix_str(label)
            stages.update(1)

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
            t_sec = StageTimer(f"sección {seccion} total")
            res, wm, driver_cols = self._run_section(
                selected, seccion, first_by_sec[seccion], driver_cols
            )
            t_sec.stop()
            if res.height:
                all_res.append(res)
                # Checkpoint parcial (Fase 5)
                ck = self._cfg.out_dir / f"forecast_seccion_{seccion}.parquet"
                self._cfg.out_dir.mkdir(parents=True, exist_ok=True)
                res.write_parquet(ck, compression="zstd", compression_level=3)
                logger.info("✓ checkpoint %s (%d filas)", ck.name, res.height)
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
        return res_df, wmapes_df

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
        elif "forecast_start" in sku_level.columns and "forecast_end" in sku_level.columns:
            sku_level = sku_level.filter(
                (pl.col("ds").cast(pl.Date) >= pl.col("forecast_start").cast(pl.Date))
                & (pl.col("ds").cast(pl.Date) <= pl.col("forecast_end").cast(pl.Date))
            )
        if sku_level.height == 0:
            logger.warning("Sin filas forecast-only a nivel SKU para export Excel")
            return

        # Parse unique_id → seccion, local, sku
        parts = sku_level.with_columns(
            pl.col("unique_id").str.split_exact("||", 2).alias("_p")
        ).unnest("_p")
        rename = {}
        for i, name in enumerate(["SECCION", "Local", "SKU"]):
            src = f"field_{i}"
            if src in parts.columns:
                rename[src] = name
        parts = parts.rename(rename) if rename else parts

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
                logger.warning(
                    "write_excel falló; exportado CSV: %s", csv_path
                )
                continue
            logger.info(
                "✓ Excel sección %s (%d filas): %s",
                seccion,
                summary.height,
                out_path,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline RLS sección→tienda→SKU")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Workers (threads) por unique_id. Default: secuencial.",
    )
    parser.add_argument(
        "--limit-series",
        type=int,
        default=None,
        help="Máx. series por sección (dev/debug). Default: todas.",
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
