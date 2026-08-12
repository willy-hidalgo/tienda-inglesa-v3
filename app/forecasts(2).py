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

100% Polars + numpy. OOP, paralelizable por unique_id (threads: numba libera
el GIL en `_rls` / `decompose_price`, así que ThreadPoolExecutor da
paralelismo real sin el costo de pickling de DataFrames Polars entre
procesos).

Rendimiento (ver README § Rendimiento para detalle y benchmarks):
  - 1 solo fit por serie/variable (antes: 3 fits redundantes sobre train
    para in-sample / OOS / forecast-only).
  - EDP vectorizado en una sola llamada numba con `indexors` (antes: 1
    llamada numba por serie vía loop Python).
  - Rolling 28d: 1 fit por serie con `return_all_coefs=True`, walk-forward
    O(n) leyendo el vector de coeficientes vigente en cada bloque (antes:
    hasta 2 reajustes completos por bloque de 28 días → O(n²)).
  - `--limit-series N` para iterar rápido en desarrollo sin correr el
    dataset completo.
  - Checkpoint por sección (`forecast_seccion_<n>_partial.parquet`) para no
    perder trabajo si el proceso se corta.
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
    compute_rolling28: bool = False
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
            compute_rolling28=settings.COMPUTE_ROLLING28,
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
        existing = set(df.columns)
        if req_columns is None:
            req_columns = [
                c for c in feats.columns if c not in _exclude and c not in existing
            ]
        else:
            # Si el DataFrame ya tiene alguna de las columnas pedidas, no las
            # solicitamos de las features de calendario para evitar duplicados.
            req_columns = [c for c in req_columns if c not in existing]

        feats = self._holiday_dummies(feats, req_columns)
        feat_cols = [c for c in feats.columns if c != "ds"]
        feat_cols = [c for c in dict.fromkeys(feat_cols) if c not in existing]

        # asegurar columnas pedidas que no existan ya en df
        missing = [
            c
            for c in (req_columns or [])
            if c not in feats.columns and c not in existing
        ]
        if missing:
            feats = feats.with_columns([pl.lit(0).alias(c) for c in missing])
            feat_cols = [c for c in feats.columns if c != "ds" and c not in existing]

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
    """
    Ajusta y predice modelos RLS por serie (unique_id).

    Diseño de rendimiento (Fase 1 + 4):
      - `fit_and_predict_multi` hace **1 solo fit** por serie (variable y,
        variable precio) y predice sobre todos los targets pedidos
        (in_sample / out_sample / forecast_only), en vez de 1 fit por
        target como antes.
      - El loop por serie corre en `ThreadPoolExecutor`: los kernels
        numba (`_rls`, `decompose_price`) son `nopython=True` y liberan el
        GIL, por lo que threads dan paralelismo real sin el costo de
        pickling de DataFrames Polars que exige `ProcessPoolExecutor`.
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
    def _correction_factor(errors: np.ndarray) -> float:
        sigma2 = errors.var(ddof=1)
        return float(np.exp(sigma2 / 2))

    def _default_priors(self, n_features: int):
        if RLSConstantPrior is None or RLSPrior is None:
            raise RuntimeError("No se ha instanciado RLSConstantPrior ó RLSPrior")
        else:
            return [
                RLSConstantPrior(standard_error=0.5, rmse_error=self._rmse_error)
            ] + [
                RLSPrior(coefficient=0, standard_error=0.5, rmse_error=self._rmse_error)
                for _ in range(max(0, n_features - 1))
            ]

    def _new_rls(self, min_y: float, return_all_coefs: bool = False):
        if RecursiveLeastSquaresRegression is None:
            raise RuntimeError("No se ha instanciado RecursiveLeastSquaresRegression")
        else:
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

    def _workers(self) -> int:
        return self._n_jobs if self._n_jobs and self._n_jobs > 1 else 1

    # ── Fit + predicción multi-target (Fase 1 + 4) ──────────────────────────
    def fit_and_predict_multi(
        self,
        train: pl.DataFrame,
        targets: dict[str, pl.DataFrame],
        forecast_levels: list[str],
        desc: str = "Ajustando RLS",
        meta: dict | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Ajusta el modelo UNA vez por serie (sobre `train`) y predice sobre
        todos los `targets` (dict period_type -> DataFrame de test).

        Antes: 1 fit por target (in_sample/out_sample/forecast_only) → 3
        fits redundantes sobre los mismos datos de train. Ahora: 1 fit,
        N predicciones.
        """
        unique_ids = self._build_tasks(train, forecast_levels)
        logger.info("%s: %d series a ajustar", desc, len(unique_ids))
        if not unique_ids:
            return pl.DataFrame(), pl.DataFrame()

        needed = set(unique_ids)
        train_f = train.filter(pl.col("unique_id").is_in(list(needed)))
        train_parts = self._normalize_partition_dict(
            train_f.partition_by("unique_id", as_dict=True)
        )
        target_parts: dict[str, dict] = {}
        for name, df in targets.items():
            if df.height == 0:
                target_parts[name] = {}
                continue
            df_f = df.filter(pl.col("unique_id").is_in(list(needed)))
            target_parts[name] = self._normalize_partition_dict(
                df_f.partition_by("unique_id", as_dict=True)
            )

        def _task(uid: str) -> list[pl.DataFrame]:
            train_g = train_parts.get(uid)
            if train_g is None or train_g.height == 0:
                return []
            try:
                model_y, model_p = self._fit_models(train_g)
            except Exception as exc:  # serie degenerada: se loguea y se salta
                logger.warning("%s: fit falló para %s: %s", desc, uid, exc)
                return []
            frames: list[pl.DataFrame] = []
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
                    frames.append(frame.with_columns(pl.lit(name).alias("period_type")))
            return frames

        results: list[pl.DataFrame] = []
        with ThreadPoolExecutor(max_workers=self._workers()) as executor:
            futures = {executor.submit(_task, uid): uid for uid in unique_ids}
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=desc, unit="serie"
            ):
                results.extend(fut.result())

        if not results:
            return pl.DataFrame(), pl.DataFrame()

        res_df = pl.concat(results, how="diagonal_relaxed")
        wm_frames = []
        for name in targets:
            sub = res_df.filter(pl.col("period_type") == name)
            if sub.height:
                wm = self._compute_wmape(sub)
                if wm.height:
                    wm_frames.append(wm)
        wmapes_df = (
            pl.concat(wm_frames, how="diagonal_relaxed")
            if wm_frames
            else pl.DataFrame()
        )
        return res_df, wmapes_df

    def run(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        forecast_levels: list[str],
        desc: str = "Ajustando RLS",
        meta: dict | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Compat: fit + predict sobre un único target. Ver `fit_and_predict_multi`."""
        res_df, wmapes_df = self.fit_and_predict_multi(
            train, {"_default": test}, forecast_levels, desc=desc, meta=meta
        )
        if res_df.height and "period_type" in res_df.columns:
            res_df = res_df.drop("period_type")
        return res_df, wmapes_df

    # ── Rolling 28d (yhat28 / valuehat28) — Fase 3(b): O(n) por serie ───────
    @staticmethod
    def _first_monday_on_or_after(d: dt.date) -> dt.date:
        # weekday: Mon=0 … Sun=6
        return d + dt.timedelta(days=(7 - d.weekday()) % 7)

    def _rolling_28_one(self, unique_id: str, g: pl.DataFrame) -> pl.DataFrame | None:
        """
        Walk-forward 28d sobre toda la historia de la serie.

        Antes: hasta 2 reajustes RLS completos por bloque de 28 días
        (O(n²/28) por serie). Ahora: 1 solo fit por variable con
        `return_all_coefs=True`; para predecir el bloque que empieza en
        `pos` se usa el vector de coeficientes tal como quedó tras el
        último actual estrictamente anterior a `pos`, sin reajustar nada
        por bloque (se evita el O(n²) de refits).

        Precisión importante sobre causalidad: la trayectoria de
        coeficientes SÍ es causal una vez que el walk-forward de RLS
        arranca (el estado en la posición j depende solo de las
        observaciones 0..j, nunca de observaciones futuras — así funciona
        `_rls` internamente). Pero el *seed* inicial (`_coefficient_seeds`,
        "opción B") se calcula a partir de TODO el historial de actuals
        disponible, igual que en el diseño original ("se ajusta el modelo
        final con todos los actuals → priors"). Esto significa que los
        primeros bloques (antes de que haya suficientes actuals para que
        el walk-forward domine sobre el seed) heredan ese sesgo de mirar
        el historial completo — comportamiento igual al de la versión
        anterior, no una regresión introducida por este cambio.

        Nota: esto cambia levemente los valores numéricos de `yhat28` /
        `wmape_28` respecto de la versión anterior, porque antes se
        reseteaba la covarianza (`standard_error` fijo) en cada bloque en
        vez de dejarla evolucionar de forma continua. La versión O(n) es
        matemáticamente más correcta (RLS genuinamente recursivo) y es la
        que se documenta en README § Rendimiento.
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
        if has_value:
            Xp_all = (
                g.select(pdcols).to_numpy().astype(np.float64, order="C")
                if pdcols
                else X_all
            )
            v_all = g["value"].to_numpy().astype(np.float64)
        else:
            Xp_all = X_all
            v_all = np.zeros_like(y_all)

        n = g.height
        H = int(getattr(settings, "ROLLING_HORIZON_DAYS", 28))

        actual_mask = np.isfinite(y_all) & (y_all != 0)
        actual_v = (np.isfinite(v_all) & (v_all != 0)) if has_value else actual_mask

        idx_act = np.where(actual_mask)[0]
        if idx_act.size < 2:
            return None

        # ── Fit único (variable y) con historial completo de coeficientes ──
        priors_y = self._default_priors(X_all.shape[1])
        model_y = self._new_rls(self._min_y_to_update, return_all_coefs=True)
        try:
            model_y.fit(x=X_all[idx_act], y=np.log1p(y_all[idx_act]), priors=priors_y)
        except Exception as exc:
            logger.warning("rolling28 %s: fit y falló: %s", unique_id, exc)
            return None
        all_coefs_y = model_y.all_coef[0]
        seed_y = model_y._coefficient_seeds(
            X_all[idx_act], np.log1p(y_all[idx_act]), priors_y
        )
        corr_y = None
        if self._use_correction_factor:
            try:
                corr_y = self._correction_factor(np.asarray(model_y.errors))
            except Exception:
                corr_y = None

        # ── Fit único (variable precio), si hay datos ──
        all_coefs_p = None
        seed_p = None
        corr_p = None
        idx_v = np.array([], dtype=int)
        if has_value and np.any(actual_v):
            idx_v = np.where(actual_v)[0]
            try:
                priors_p = self._default_priors(Xp_all.shape[1])
                model_p = self._new_rls(1e-8, return_all_coefs=True)
                model_p.fit(
                    x=Xp_all[idx_v],
                    y=np.log1p(np.clip(v_all[idx_v], 0.0, None)),
                    priors=priors_p,
                )
                all_coefs_p = model_p.all_coef[0]
                seed_p = model_p._coefficient_seeds(
                    Xp_all[idx_v],
                    np.log1p(np.clip(v_all[idx_v], 0.0, None)),
                    priors_p,
                )
                if self._use_correction_factor:
                    try:
                        corr_p = self._correction_factor(np.asarray(model_p.errors))
                    except Exception:
                        corr_p = None
            except Exception as exc:
                logger.debug("rolling28 %s: fit price falló: %s", unique_id, exc)
                all_coefs_p = None

        yhat28 = np.full(n, np.nan)
        valuehat28 = np.full(n, np.nan)

        pos = 0
        while pos < n:
            end = min(pos + H, n)

            j_y = int(np.searchsorted(idx_act, pos, side="left")) - 1
            coef_y = seed_y if j_y < 0 else all_coefs_y[j_y]
            log_hat = X_all[pos:end] @ coef_y
            if corr_y is not None:
                yhat28[pos:end] = np.round(np.exp(log_hat) * corr_y).ravel()
            else:
                yhat28[pos:end] = np.round(np.expm1(log_hat)).ravel()

            if all_coefs_p is not None:
                j_p = int(np.searchsorted(idx_v, pos, side="left")) - 1
                coef_p = seed_p if j_p < 0 else all_coefs_p[j_p]
                log_hat_p = Xp_all[pos:end] @ coef_p
                if corr_p is not None:
                    valuehat28[pos:end] = np.round(np.exp(log_hat_p) * corr_p, 2)
                else:
                    valuehat28[pos:end] = np.round(np.expm1(log_hat_p), 2)

            pos = end

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
        """Genera yhat28/valuehat28 para todas las series del panel (threaded)."""
        empty_schema = {
            "unique_id": pl.Utf8,
            "ds": pl.Date,
            "yhat28": pl.Float64,
            "valuehat28": pl.Float64,
        }
        if panel.height == 0 or RecursiveLeastSquaresRegression is None:
            return pl.DataFrame(schema=empty_schema)

        # Fase 5: alineado con min_obs=28 del resto del pipeline (antes: 2).
        unique_ids = self._build_tasks(panel, forecast_levels, min_obs=28)
        if not unique_ids:
            return pl.DataFrame(schema=empty_schema)

        logger.info(
            "rolling28: %d unique_id en tasks, panel filas=%d, workers=%d",
            len(unique_ids),
            panel.height,
            self._workers(),
        )
        logger.info(
            "rolling28: construyendo particiones de %d unique_id...",
            len(unique_ids),
        )
        parts = self._normalize_partition_dict(
            panel.filter(pl.col("unique_id").is_in(unique_ids)).partition_by(
                "unique_id", as_dict=True
            )
        )

        frames = []
        with ThreadPoolExecutor(max_workers=self._workers()) as executor:
            futures = {
                executor.submit(self._rolling_28_one, uid, parts[uid]): uid
                for uid in unique_ids
                if uid in parts
            }
            completed = 0
            log_every = 200 if len(futures) > 200 else len(futures) + 1
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc,
                unit="serie",
                leave=False,
            ):
                completed += 1
                uid = futures[fut]
                if completed % log_every == 0 or completed == len(futures):
                    logger.info(
                        "rolling28: %d/%d series completadas",
                        completed,
                        len(futures),
                    )
                try:
                    one = fut.result()
                except Exception as exc:
                    logger.warning("rolling28 %s: %s", uid, exc)
                    continue
                if one is not None and one.height:
                    frames.append(one)

        if not frames:
            return pl.DataFrame(schema=empty_schema)
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
        uids = pl.concat(
            [uids, extra_uids.select("unique_id")], how="diagonal_relaxed"
        ).unique()

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
            c for c in df.columns if c not in ("unique_id", "ds") and c not in meta_cols
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
        """Genera filas ds para [start, end] por unique_id (y=0) sin loops por celda en Polars."""
        n_days = (end - start).days + 1
        if n_days <= 0 or not unique_ids:
            return pl.DataFrame()

        dates = pl.date_range(start, end, interval="1d", eager=True)

        # Meta una sola pasada (unique_id → sku_desc / store_name / seccion)
        meta_cols = [
            c for c in ("sku_desc", "store_name", "seccion") if c in template.columns
        ]
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
    def _calculate_edp_per_series(df: pl.DataFrame) -> pl.DataFrame:
        """Fallback: 1 llamada numba por serie (loop Python). Usar solo si el
        batch con `indexors` no está disponible o falla."""
        n_series = df["unique_id"].n_unique()
        logger.info("EDP: decompose_price por serie (loop) sobre %d series…", n_series)
        parts = []
        groups = df.sort(["unique_id", "ds"]).partition_by(
            "unique_id", maintain_order=True
        )
        for part in tqdm(groups, desc="EDP (loop)", unit="serie"):
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

    def _fit_section_level_models(
        self,
        section_data: pl.DataFrame,
        driver_cols: list[str],
        rmse_error: float,
        forgetting_factor: float,
        min_y_to_update: float,
        use_correction_factor: bool,
    ):
        """
        Ajusta modelos RLS a nivel de sección para las variables y y value.

        Returns:
            tuple: (model_y, model_p) - los modelos RLS ajustados
        """
        if RecursiveLeastSquaresRegression is None:
            raise RuntimeError("rls_opt no disponible")

        # Prepare y variable (log1p(y))
        X_y = section_data.select(driver_cols).to_numpy().astype(np.float64, order="C")
        y = section_data["y"].to_numpy()
        log_y = np.log1p(y)
        model_y = RecursiveLeastSquaresRegression(
            forgetting_factor=forgetting_factor,
            min_y_to_update=min_y_to_update,
            return_all_coefs=False,
        )
        priors_y = [RLSConstantPrior(standard_error=0.5, rmse_error=rmse_error)] + [
            RLSPrior(coefficient=0, standard_error=0.5, rmse_error=rmse_error)
            for _ in range(max(0, len(driver_cols) - 1))
        ]
        model_y.fit(x=X_y, y=log_y, priors=priors_y)

        # Prepare value variable (log1p(value))
        # Excluir asp, edp, discount para el modelo de precio (como en el código original)
        _price_exclude = {"asp", "edp", "discount"}
        driver_cols_price = [c for c in driver_cols if c not in _price_exclude]
        X_p = (
            section_data.select(driver_cols_price)
            .to_numpy()
            .astype(np.float64, order="C")
        )
        price = section_data["value"].to_numpy()
        log_price = np.log1p(np.clip(price, 0.0, None))
        model_p = RecursiveLeastSquaresRegression(
            forgetting_factor=forgetting_factor,
            min_y_to_update=1e-8,  # Mismo valor que en el código original
            return_all_coefs=False,
        )
        priors_p = [RLSConstantPrior(standard_error=0.5, rmse_error=rmse_error)] + [
            RLSPrior(coefficient=0, standard_error=0.5, rmse_error=rmse_error)
            for _ in range(max(0, len(driver_cols_price) - 1))
        ]
        model_p.fit(x=X_p, y=log_price, priors=priors_p)

        return model_y, model_p

    def _predict_using_section_procedure(
        self,
        target_df: pl.DataFrame,
        section_coeffs_y: np.ndarray,
        section_coeffs_p: np.ndarray,
        driver_cols: list[str],
        period_type: str,
        meta: dict,
    ) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
        """
        Aplica el procedimiento de predicción usando modelos de sección:
        1. Extraer coeficientes de sección (ya proporcionados)
        2. Para cada unique_id en target_df:
           a. Obtener datos históricos para esa combinación
           b. Separar actuals (y) de drivers (excluyendo intercept)
           c. Calcular efectos de drivers: drivers × coeficientes_sección
           d. Calcular net y: actuals - efectos_de_drivers
           e. Aplicar smoothing exponencial a net y para obtener net_y_hat (parámetro 0.1)
           f. Predicción final: yhat = net_y_hat + suma_de_efectos_de_drivers
        3. Lo mismo para la variable value

        Returns:
            tuple: (prediction_frames, wmape_frames)
        """
        if target_df.height == 0:
            return [], []

        # Particionar una sola vez por serie para evitar múltiples filtros costosos
        sorted_target = target_df.sort(["unique_id", "ds"])
        groups = sorted_target.partition_by("unique_id", maintain_order=True)

        n_series = len(groups)
        logger.info(
            "Predicción de sección: procesando %d series para %s", n_series, period_type
        )

        prediction_frames = []
        wmape_frames = []

        # Procesar cada unique_id
        for uid_data in groups:
            if uid_data.height == 0:
                continue
            uid = uid_data["unique_id"][0]

            # Preparar datos para y variable
            effective_driver_cols_y = [c for c in driver_cols if c != "intercept"]
            X_y = (
                uid_data.select(effective_driver_cols_y)
                .to_numpy()
                .astype(np.float64, order="C")
            )
            y_actual = uid_data["y"].to_numpy()

            # Calcular efectos de drivers: X_y × section_coeffs_y
            # section_coeffs_y ya excluye el intercept, por lo que también removemos
            # la columna intercept de los drivers antes de multiplicar.
            driver_effects_y = (
                X_y @ section_coeffs_y
                if len(section_coeffs_y) > 0
                else np.zeros(len(y_actual))
            )

            # Calcular net y: actuals - driver effects
            net_y = y_actual - driver_effects_y

            # Aplicar smoothing exponencial a net y para obtener net_y_hat (parámetro 0.1)
            # net_y_hat[t] = 0.1 * net_y[t] + 0.9 * net_y_hat[t-1]
            net_y_hat = np.zeros_like(net_y)
            if len(net_y) > 0:
                net_y_hat[0] = net_y[0]  # Primera observación sin suavizado
                for i in range(1, len(net_y)):
                    net_y_hat[i] = 0.1 * net_y[i] + 0.9 * net_y_hat[i - 1]

            # Predicción final para y: yhat = net_y_hat + driver effects
            yhat = net_y_hat + driver_effects_y

            # Preparar datos para value variable
            # Excluir asp, edp, discount para el modelo de precio
            _price_exclude = {"asp", "edp", "discount"}
            driver_cols_price = [
                c for c in driver_cols if c not in _price_exclude and c != "intercept"
            ]
            X_p = (
                uid_data.select(driver_cols_price)
                .to_numpy()
                .astype(np.float64, order="C")
            )
            value_actual = uid_data["value"].to_numpy()

            # Calcular efectos de drivers para value: X_p × section_coeffs_p
            driver_effects_p = (
                X_p @ section_coeffs_p
                if len(section_coeffs_p) > 0
                else np.zeros(len(value_actual))
            )

            # Calcular net value: actuals - driver effects
            net_value = value_actual - driver_effects_p

            # Aplicar smoothing exponencial a net value con parámetro 0.1
            net_value_hat = np.zeros_like(net_value)
            if len(net_value) > 0:
                net_value_hat[0] = net_value[0]  # Primera observación sin suavizado
                for i in range(1, len(net_value)):
                    net_value_hat[i] = 0.1 * net_value[i] + 0.9 * net_value_hat[i - 1]

            # Predicción final para value: valuehat = net_value_hat + driver effects
            valuehat = net_value_hat + driver_effects_p

            # Construir el DataFrame de resultado para este unique_id
            result_data = {
                "unique_id": [uid] * len(uid_data),
                "ds": uid_data["ds"].to_list(),
                "y": y_actual,
                "yhat": yhat,
                "value": value_actual,
                "valuehat": valuehat,
            }

            # Añadir metadata si existe
            if meta:
                for k, v in meta.items():
                    result_data[k] = [v] * len(uid_data)

            # Copiar columnas metafrom uid_data si existen
            for col in ("sku_desc", "store_name", "seccion", "conteo_sku"):
                if col in uid_data.columns:
                    result_data[col] = uid_data[col].to_list()

            # Añadir columna period_type
            result_data["period_type"] = [period_type] * len(uid_data)

            frame = pl.DataFrame(result_data)
            prediction_frames.append(frame)

            # Para WMAPE, necesitamos calcular errores solo para out_sample
            if period_type == "out_sample":
                # Calcular errores absolutos relativos
                abs_error = np.abs(y_actual - yhat)
                # Evitar división por cero
                mask = y_actual != 0
                if np.any(mask):
                    wmape_value = np.sum(abs_error[mask]) / np.sum(
                        np.abs(y_actual[mask])
                    )
                else:
                    wmape_value = 0.0

                wmape_frame = pl.DataFrame(
                    {
                        "unique_id": [uid],
                        "wmape": [wmape_value],
                        "sum_y": [np.sum(np.abs(y_actual))],  # Suma de |y| para WMAPE
                        "n_with_sales": [np.sum(y_actual != 0)],  # Puntos con ventas
                        "n_ds": [len(y_actual)],  # Total de puntos
                    }
                )
                wmape_frames.append(wmape_frame)

        return prediction_frames, wmape_frames

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

        with _stage_timer(f"{seccion}: agregación train"):
            df_train = self._aggregator.aggregate(raw_train)
        logger.info(
            "Sección %s: train agregado shape=%s | series=%d",
            seccion,
            df_train.shape,
            df_train["unique_id"].n_unique() if df_train.height else 0,
        )

        df_train = self._apply_limit_series(df_train)

        # Panel denso train: spine settings [train_start, train_end]
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

        with _stage_timer(f"{seccion}: agregación+densify OOS"):
            df_oos = (
                self._aggregator.aggregate(raw_oos)
                if raw_oos.height
                else pl.DataFrame()
            )
            if df_oos.height:
                # Densificar OOS al spine de la sección [test_start, test_end]
                # y alinear unique_ids con train (mismas series)
                train_uids = df_train.select("unique_id").unique()
                df_oos = densify_section_panel(df_oos, hz["test_start"], hz["test_end"])
                # asegurar todas las series de train también en OOS (ceros si no hubo venta)
                oos_uids = df_oos.select("unique_id").unique()
                missing = train_uids.join(oos_uids, on="unique_id", how="anti")
                if missing.height:
                    spine_oos = pl.DataFrame(
                        {
                            "ds": pl.date_range(
                                hz["test_start"],
                                hz["test_end"],
                                interval="1d",
                                eager=True,
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
                            .agg(
                                [
                                    pl.col(c).drop_nulls().first().alias(c)
                                    for c in meta_cols
                                ]
                            )
                        )
                        extra = extra.join(meta, on="unique_id", how="left")
                    df_oos = pl.concat([df_oos, extra], how="diagonal_relaxed")

        if df_oos.height:
            with _stage_timer(f"{seccion}: EDP OOS"):
                df_oos = self._calculate_edp(df_oos)
            with _stage_timer(f"{seccion}: features OOS"):
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

        # Extract section-level data for fitting (where unique_id has no "||")
        section_train = df_train.filter(
            pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 0
        )

        if section_train.height == 0:
            logger.warning("Sección %s: No hay datos de sección para ajustar", seccion)
            return pl.DataFrame(), pl.DataFrame(), driver_cols

        # Apply feature engineering to section data (must match what's done to target data)
        section_train = self._feature_builder.extract_drivers(
            section_train, req_columns=driver_cols
        ).sort("ds")

        # Fit section-level RLS models for y and value variables
        logger.info("Sección %s: ajustando modelos RLS a nivel de sección", seccion)
        section_model_y, section_model_p = self._fit_section_level_models(
            section_train,
            driver_cols,
            self._cfg.rmse_error,
            self._cfg.forgetting_factor,
            self._cfg.min_y_to_update,
            self._cfg.correction_factor,
        )

        # Extract section-level coefficients (excluding intercept)
        section_coeffs_y = section_model_y.final_coef_[-1][1:]  # Excluir intercept
        section_coeffs_p = section_model_p.final_coef_[-1][1:]  # Excluir intercept

        logger.info(
            "Sección %s: coeficientes sección obtenidos (y: %d, value: %d)",
            seccion,
            len(section_coeffs_y),
            len(section_coeffs_p),
        )

        meta = {
            "train_start": hz["train_start"],
            "train_end": hz["train_end"],
            "test_start": hz["test_start"],
            "test_end": hz["test_end"],
            "forecast_start": hz["forecast_start"],
            "forecast_end": hz["forecast_end"],
        }

        # Prepare targets
        targets: dict[str, pl.DataFrame] = {"in_sample": df_train}
        if df_oos.height:
            targets["out_sample"] = df_oos
        if df_fcst.height:
            targets["forecast_only"] = df_fcst

        # Process each target using section-level models and prediction procedure
        with _stage_timer(f"{seccion}: predicción usando modelos de sección"):
            res_frames = []
            wm_frames = []

            for period_type, target_df in targets.items():
                if target_df.height == 0:
                    continue

                # Apply prediction procedure for each unique_id in target
                period_frames = self._predict_using_section_procedure(
                    target_df=target_df,
                    section_coeffs_y=section_coeffs_y,
                    section_coeffs_p=section_coeffs_p,
                    driver_cols=driver_cols,
                    period_type=period_type,
                    meta=meta,
                )

                if period_frames:
                    res_frames.extend(period_frames[0])  # prediction frames
                    if period_frames[1]:  # wmape frames
                        wm_frames.extend(period_frames[1])

            res_df = (
                pl.concat(res_frames, how="diagonal_relaxed")
                if res_frames
                else pl.DataFrame()
            )
            wmapes_df = (
                pl.concat(wm_frames, how="diagonal_relaxed")
                if wm_frames
                else pl.DataFrame()
            )

        if res_df.height:
            # forecast-only: sin actuals → y/value en 0 (línea punteada aguas abajo)
            res_df = res_df.with_columns(
                pl.when(pl.col("period_type") == "forecast_only")
                .then(0.0)
                .otherwise(pl.col("y"))
                .alias("y"),
                pl.when(pl.col("period_type") == "forecast_only")
                .then(0.0)
                .otherwise(pl.col("value"))
                .alias("value"),
            )

        if not res_df.height:
            return pl.DataFrame(), pl.DataFrame(), driver_cols

        with _stage_timer(f"{seccion}: rolling28 (opt-in)"):
            # construir runner aquí y pasarlo a _attach_rolling28 (tests monkeypatchan el runner)
            runner = RLSForecastRunner(
                driver_cols=driver_cols or [],
                rmse_error=self._cfg.rmse_error,
                forgetting_factor=self._cfg.forgetting_factor,
                min_y_to_update=self._cfg.min_y_to_update,
                use_correction_factor=self._cfg.correction_factor,
                n_jobs=self._n_jobs,
            )
            res_df, wmapes_df = self._attach_rolling28(
                runner, res_df, wmapes_df, df_train, df_oos, df_fcst
            )

        return res_df, wmapes_df, driver_cols

    def _attach_rolling28(
        self,
        runner: RLSForecastRunner,
        res_df: pl.DataFrame,
        wmapes_df: pl.DataFrame,
        df_train: pl.DataFrame,
        df_oos: pl.DataFrame,
        df_fcst: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Rolling 28d (yhat28 / valuehat28) sobre toda la historia
        (train+OOS+forecast). Cuando `compute_rolling28=False` (config) hace
        early-return y `res_df` NO gana columnas yhat28/valuehat28 (parquet
        más liviano y dashboard sin controles rolling). No modifica yhat/valuehat.
        """
        if not self._cfg.compute_rolling28:
            logger.info("rolling 28d desactivado (COMPUTE_ROLLING28=False)")
            return res_df, wmapes_df

        panel_parts = [p for p in (df_train, df_oos, df_fcst) if p.height]
        if not panel_parts:
            # sin panel no hay rolling; si ya había res_df, se deja sin col. rolling
            if res_df.height:
                res_df = res_df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("yhat28"),
                    pl.lit(None).cast(pl.Float64).alias("valuehat28"),
                )
            return res_df, wmapes_df

        panel = pl.concat(panel_parts, how="diagonal_relaxed").sort(["unique_id", "ds"])
        logger.info("rolling 28d: panel shape=%s…", panel.shape)
        # runner is provided by the caller (tests may monkeypatch its methods)
        roll = runner.run_rolling_28(panel, self._cfg.forecast_levels, desc="rolling28")
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
                    pl.col("y").is_not_null() & pl.col("yhat28").is_not_null()
                )
            )
            if wm28.height:
                if "wmape" in wmapes_df.columns and "unique_id" in wmapes_df.columns:
                    wmapes_df = wmapes_df.join(wm28, on="unique_id", how="left")
                else:
                    wmapes_df = wm28
            logger.info(
                "yhat28 unido (%d filas roll, %d series wm28)",
                roll.height,
                wm28.height if wm28.height else 0,
            )
        elif res_df.height:
            res_df = res_df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("yhat28"),
                pl.lit(None).cast(pl.Float64).alias("valuehat28"),
            )
        return res_df, wmapes_df

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
        help="Threads paralelos por unique_id (fit+predict y rolling28). Default: secuencial.",
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
    parser.add_argument(
        "--rolling28",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override de settings.COMPUTE_ROLLING28: --rolling28 habilita el "
            "walk-forward yhat28/valuehat28; --no-rolling28 lo deshabilita."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ForecastConfig.from_settings()
    if args.rolling28 is not None:
        config.compute_rolling28 = args.rolling28
    pipeline = RLSForecastPipeline(
        config, n_jobs=args.n_jobs, limit_series=args.limit_series
    )
    res_df, wmapes_df = pipeline.run()
    print(res_df.head())
    print(wmapes_df.head())
    pipeline.save(res_df, wmapes_df)


if __name__ == "__main__":
    main()
