"""
Pipeline de forecasting jerárquico con Regresión RLS
=====================================================
Niveles: sección (1 / 23), con dos FILTROS INDEPENDIENTES (no jerárquicos)
por debajo: tienda y SKU — cualquiera puede estar vacío, uno solo, o ambos
a la vez (ver settings.make_unique_id / README § Filtros independientes).

Modelo jerárquico (ver README § Modelo jerárquico):
  - RLS se ajusta ÚNICAMENTE a nivel sección (2 modelos: variable y,
    variable precio).
  - Tienda / SKU / tienda+SKU se DERIVAN sin RLS: efecto de drivers (con
    los coeficientes de la sección) + suavización exponencial simple (SES)
    causal sobre el residuo ("y_neto"). Totalmente vectorizado con Polars,
    sin loop por serie — ver `RLSForecastRunner.compute_derived_forecasts`.

Ventanas por sección (SECCIONES):
  - train: [train_start, train_end]  train_end = test_start original
            train_start alineado a bloques de 28 días desde train_end,
            acotado por domingo ≥ primer dato
  - test (OOS): lunes ≥ test_start → test_end  → métricas out-of-sample
  - forecast-only: test_end+1 → test_end+28  (sin actuals, línea punteada)

100% Polars + numpy. Los 1-2 fits RLS por sección corren secuencial (son
pocas series); el resto de las series (tienda/sku/tienda+sku) es una sola
operación vectorizada, sin paralelismo por serie necesario.

Rendimiento (ver README § Rendimiento para detalle y benchmarks):
  - RLS solo a nivel sección: de miles de fits (uno por serie) a solo 2-4
    fits totales por sección — el resto se deriva vectorizado.
  - EDP vectorizado en una sola llamada numba con `indexors` (antes: 1
    llamada numba por serie vía loop Python).
  - Rolling 28d: 1 fit por serie con `return_all_coefs=True`, walk-forward
    O(n) leyendo el vector de coeficientes vigente en cada bloque. Es
    **opcional** vía `settings.COMPUTE_ROLLING_28` (default: `False`) y,
    desde el modelo jerárquico, limitado al nivel sección (único nivel con
    RLS real); si está desactivado no se agregan `yhat28`/`valuehat28`.
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

        # Mapas de descripción (se adjuntan ANTES de agregar, para no tener
        # que reconstruirlos parseando el unique_id después).
        sku_desc_df = pl.DataFrame(
            schema={"SKU_ID": pl.Utf8, "sku_desc": pl.Utf8}
        )
        if "DESCRIPCION" in df.columns:
            sku_desc_df = (
                df.select(["SKU_ID", "DESCRIPCION"])
                .drop_nulls()
                .unique(subset=["SKU_ID"])
                .rename({"DESCRIPCION": "sku_desc"})
            )

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

        base_aggs = self._base_aggs()

        # 1) sección
        lvl_sec = (
            df.group_by(["ds", "SECCION"])
            .agg(base_aggs)
            .with_columns(
                pl.col("SECCION").alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
                pl.lit("").alias("sku_desc"),
                pl.lit("").alias("store_name"),
            )
        )

        # 2) sección + tienda (todos los SKU)
        lvl_store = (
            df.group_by(["ds", "SECCION", "STORE_ID"])
            .agg(base_aggs)
            .with_columns(
                pl.concat_str(
                    [pl.col("SECCION"), pl.lit("||T:"), pl.col("STORE_ID")]
                ).alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
                pl.lit("").alias("sku_desc"),
            )
            .join(store_name_df, on=["SECCION", "STORE_ID"], how="left")
            .with_columns(pl.col("store_name").fill_null(""))
        )

        # 3) sección + sku (todas las tiendas)
        lvl_sku = (
            df.group_by(["ds", "SECCION", "SKU_ID"])
            .agg(base_aggs)
            .with_columns(
                pl.concat_str(
                    [pl.col("SECCION"), pl.lit("||S:"), pl.col("SKU_ID")]
                ).alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
                pl.lit("").alias("store_name"),
            )
            .join(sku_desc_df, on="SKU_ID", how="left")
            .with_columns(pl.col("sku_desc").fill_null(""))
        )

        # 4) sección + tienda + sku
        lvl_store_sku = (
            df.group_by(["ds", "SECCION", "STORE_ID", "SKU_ID"])
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
            .join(store_name_df, on=["SECCION", "STORE_ID"], how="left")
            .join(sku_desc_df, on="SKU_ID", how="left")
            .with_columns(
                pl.col("store_name").fill_null(""),
                pl.col("sku_desc").fill_null(""),
            )
        )

        out = (
            pl.concat(
                [lvl_sec, lvl_store, lvl_sku, lvl_store_sku], how="diagonal_relaxed"
            )
            .with_columns(pl.lit(1).alias("intercept"))
            .sort(["unique_id", "ds"])
        )
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
    Ajusta RLS SOLO a nivel sección; deriva tienda/sku/tienda+sku sin RLS.

    Modelo jerárquico (ver README § Modelo jerárquico para el detalle):
      - `fit_and_predict_sections`: RLS real, únicamente para los 1-2 ids
        de nivel sección. Son pocas series, corre secuencial.
      - `compute_derived_forecasts`: para TODO lo demás (tienda, sku,
        tienda+sku) — sin ajustar ningún RLS. Efecto de drivers (con los
        coeficientes ya ajustados a nivel sección) + suavización
        exponencial simple (SES) causal sobre el residuo. Totalmente
        vectorizado con Polars (matmul + `ewm_mean`), sin loop por serie y
        sin threads (no hay nada que paralelizar por serie en este camino).
      - `run_rolling_28` / `_rolling_28_one`: opcional (`settings.
        COMPUTE_ROLLING_28`), limitado al nivel sección (único nivel con
        RLS real) — usa `ThreadPoolExecutor` porque el kernel numba `_rls`
        libera el GIL.
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
        return [
            RLSConstantPrior(standard_error=0.5, rmse_error=self._rmse_error)
        ] + [
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

        y_real = test_g["y"].to_numpy() if "y" in test_g.columns else np.zeros(len(yhat_test))
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

    # ── RLS SOLO a nivel sección ─────────────────────────────────────────
    def fit_and_predict_sections(
        self,
        train: pl.DataFrame,
        targets: dict[str, pl.DataFrame],
        section_ids: list[str],
        desc: str = "RLS sección",
        meta: dict | None = None,
    ) -> tuple[pl.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """
        Ajusta RLS ÚNICAMENTE para los `section_ids` dados (nivel sección —
        con el nuevo modelo jerárquico, RLS ya no se ajusta a nivel tienda/
        sku/tienda+sku, ver `compute_derived_forecasts`). Son 1-2 series por
        sección, así que corre secuencial (no hace falta ThreadPoolExecutor
        acá; el paralelismo real está en `compute_derived_forecasts`, que es
        una sola operación vectorizada sobre todas las demás series).

        Devuelve (res_df, coefs) donde `coefs[section_id] = (coef_y, coef_p)`
        son los arrays de coeficientes ajustados (mismo orden que
        `self._driver_cols` / `self._driver_cols_price`) — insumo para
        `compute_derived_forecasts`.
        """
        train_parts = self._normalize_partition_dict(
            train.partition_by("unique_id", as_dict=True)
        )
        target_parts = {
            name: self._normalize_partition_dict(df.partition_by("unique_id", as_dict=True))
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
                    results.append(frame.with_columns(pl.lit(name).alias("period_type")))

        res_df = pl.concat(results, how="diagonal_relaxed") if results else pl.DataFrame()
        return res_df, coefs

    # ── Tienda / sku / tienda+sku: SIN RLS — efecto de drivers + SES ───────
    @staticmethod
    def _apply_causal_ses(df: pl.DataFrame, src_col: str, out_col: str, alpha: float) -> pl.DataFrame:
        """
        SES causal (`s(t-1)`, no `s(t)`) por unique_id sobre `src_col`,
        vectorizado con `pl.Expr.ewm_mean` (nativo de Polars, sin loop por
        serie y sin numba):
          - Filas "actuales" (period_type != forecast_only): la predicción
            de cada fecha usa el estado suavizado HASTA el día anterior
            (`shift(1)`), nunca el valor de la propia fecha — mismo
            principio de causalidad que ya se usa en el rolling28.
          - Filas forecast_only: se propaga CONSTANTE el último estado
            suavizado conocido de la parte actual (propiedad estándar de una
            SES a cualquier horizonte de forecast).
        La primera fila de cada serie (sin día anterior) usa el propio
        estado inicial de la SES (que Polars siembra con el primer valor),
        equivalente a "sin modelo, se usa el propio dato" para ese único
        punto sin historia.
        """
        df = df.sort(["unique_id", "ds"])
        has_period = "period_type" in df.columns
        is_actual_expr = (
            (pl.col("period_type") != "forecast_only") if has_period else pl.lit(True)
        )

        actual = df.filter(is_actual_expr).with_columns(
            pl.col(src_col).ewm_mean(alpha=alpha, adjust=False).over("unique_id").alias("_s")
        )
        last_state = actual.group_by("unique_id").agg(pl.col("_s").last().alias("_last_s"))
        actual = actual.with_columns(
            pl.col("_s").shift(1).over("unique_id").fill_null(pl.col("_s")).alias(out_col)
        ).drop("_s")

        if has_period:
            forecast = df.filter(~is_actual_expr)
            if forecast.height:
                forecast = forecast.join(last_state, on="unique_id", how="left").with_columns(
                    pl.col("_last_s").fill_null(0.0).alias(out_col)
                ).drop("_last_s")
                return pl.concat([actual, forecast], how="diagonal_relaxed").sort(
                    ["unique_id", "ds"]
                )
        return actual.sort(["unique_id", "ds"])

    def compute_derived_forecasts(
        self,
        panel: pl.DataFrame,
        coefs: dict[str, tuple[np.ndarray, np.ndarray]],
        alpha: float,
    ) -> pl.DataFrame:
        """
        yhat/valuehat para TODOS los nodos que NO son de sección (tienda,
        sku, tienda+sku) — sin ajustar ningún RLS. Procedimiento (ver README
        § Modelo jerárquico):

          1. efecto(t)  = drivers_propios_del_nodo(t) · coef_sección  (sin
             intercepto; "coef_sección" son los coeficientes del RLS ya
             ajustado a nivel sección, en `coefs`).
          2. y_neto(t)  = y(t) − efecto(t)               (mismo para value).
          3. y_neto_hat(t) = SES causal sobre y_neto      (`_apply_causal_ses`).
          4. yhat(t)    = y_neto_hat(t) + efecto(t).

        Totalmente vectorizado con Polars (matmul de drivers + `ewm_mean`)
        sobre TODAS las series a la vez — sin loop por serie, sin numba, sin
        threads: no hay nada que paralelizar por serie en este camino.
        """
        if panel.height == 0:
            return pl.DataFrame()

        non_section = panel.filter(pl.col("unique_id").str.contains(r"\|\|"))
        if non_section.height == 0:
            return pl.DataFrame()

        # Drivers sin intercepto, identificados por NOMBRE (no por posición
        # — no asumimos que "intercept" sea la primera columna).
        idx_y = [i for i, c in enumerate(self._driver_cols) if c != "intercept"]
        cols_y = [self._driver_cols[i] for i in idx_y]
        idx_p = [i for i, c in enumerate(self._driver_cols_price) if c != "intercept"]
        cols_p = [self._driver_cols_price[i] for i in idx_p]

        frames = []
        for seccion, (coef_y_full, coef_p_full) in coefs.items():
            sub = non_section.filter(pl.col("unique_id").str.starts_with(f"{seccion}||"))
            if sub.height == 0:
                continue
            coef_y = coef_y_full[idx_y]
            coef_p = coef_p_full[idx_p]

            X_y = sub.select(cols_y).to_numpy().astype(np.float64)
            X_p = sub.select(cols_p).to_numpy().astype(np.float64)
            effect_y = X_y @ coef_y
            effect_v = X_p @ coef_p

            y_all = sub["y"].to_numpy().astype(np.float64)
            v_all = (
                sub["value"].to_numpy().astype(np.float64)
                if "value" in sub.columns
                else np.zeros_like(y_all)
            )

            sub = sub.with_columns(
                pl.Series("_effect_y", effect_y),
                pl.Series("_effect_v", effect_v),
                pl.Series("_y_neto", y_all - effect_y),
                pl.Series("_v_neto", v_all - effect_v),
            )
            sub = self._apply_causal_ses(sub, "_y_neto", "_y_neto_hat", alpha)
            sub = self._apply_causal_ses(sub, "_v_neto", "_v_neto_hat", alpha)
            sub = sub.with_columns(
                (pl.col("_y_neto_hat") + pl.col("_effect_y")).round(0).alias("yhat"),
                (pl.col("_v_neto_hat") + pl.col("_effect_v")).round(2).alias("valuehat"),
                pl.col("_effect_y").alias("driver_effect"),
                pl.col("_effect_v").alias("driver_effect_value"),
            ).drop(["_y_neto", "_v_neto", "_y_neto_hat", "_v_neto_hat", "_effect_y", "_effect_v"])
            frames.append(sub)

        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

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
            Xp_all = g.select(pdcols).to_numpy().astype(np.float64, order="C") if pdcols else X_all
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
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=desc, unit="serie", leave=False
            ):
                uid = futures[fut]
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
        keep = set(
            rng.choice(sku_ids, size=self._limit_series, replace=False).tolist()
        )
        logger.warning(
            "⚠ --limit-series activo: %d SKUs de %d (solo para debug/benchmark, "
            "NO usar en producción)",
            len(keep),
            len(sku_ids),
        )
        return df_train.filter((depth < 2) | pl.col("unique_id").is_in(list(keep)))

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

        logger.info("EDP: decompose_price batched (indexors) sobre %d series…", n_series)
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
                logger.info("Sección %s: forecast-only shape=%s", seccion, df_fcst.shape)
            else:
                df_fcst = pl.DataFrame()

        # ── RLS SOLO a nivel sección (nuevo modelo jerárquico) ──────────────
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

        train_section = df_train.filter(pl.col("unique_id") == seccion)
        targets_section = {
            name: df.filter(pl.col("unique_id") == seccion) for name, df in targets.items()
        }

        with _stage_timer(f"{seccion}: RLS sección (1 fit por variable)"):
            res_section, coefs = runner.fit_and_predict_sections(
                train_section,
                targets_section,
                [seccion],
                desc=f"{seccion} RLS sección",
                meta=meta,
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

        if not coefs:
            logger.warning(
                "Sección %s: no se pudo ajustar RLS a nivel sección; "
                "no hay coeficientes para derivar tienda/sku.",
                seccion,
            )
            return res_section, pl.DataFrame(), driver_cols

        # ── NUEVO: Ajustar RLS a nivel tienda (misma lógica que sección) ────────
        # Obtener locales de la sección desde settings
        locales = settings.SECCIONES[seccion]["locales"]

        # Lista para almacenar resultados de tiendas
        store_results = []

        # Ajustar RLS para cada tienda en la sección
        for store_id in locales:
            # Construir unique_id para la tienda
            store_unique_id = settings.make_unique_id(seccion, store=store_id)

            # Filtrar datos de entrenamiento para esta tienda
            train_store = df_train.filter(pl.col("unique_id") == store_unique_id)

            # Solo proceder si hay datos suficientes
            if train_store.height == 0:
                logger.warning(f"Sección {seccion}: Sin datos para tienda {store_id}")
                continue

            try:
                # Ajustar RLS para esta tienda (misma configuración que sección)
                store_res, _ = runner.fit_and_predict_sections(
                    train_store,
                    {name: df.filter(pl.col("unique_id") == store_unique_id) for name, df in targets.items()},
                    [store_unique_id],
                    desc=f"{seccion} tienda {store_id} RLS",
                    meta=meta,
                )

                if store_res.height:
                    # Aplicar el mismo post-procesamiento que para la sección
                    store_res = store_res.with_columns(
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
                    store_results.append(store_res)

            except Exception as exc:
                logger.warning(f"Sección {seccion}: Error ajustando RLS para tienda {store_id}: {exc}")
                continue

        # Combinar resultados de todas las tiendas
        res_store = pl.concat(store_results, how="diagonal_relaxed") if store_results else pl.DataFrame()
        logger.info(f"Sección {seccion}: obtenidas {res_store.height} filas de pronósticos a nivel tienda")

        # ── PARA EL MOMENTO: No calcular estimados a nivel tienda/sku ────────
        # Esto significa que no hacemos derivación adicional (ni para SKU ni para tienda+SKU)
        res_derived = pl.DataFrame()  # DataFrame vacío

        # Combinar resultados de sección y tiendas
        res_df = pl.concat(
            [f for f in (res_section, res_store) if f.height], how="diagonal_relaxed"
        )
        if not res_df.height:
            return pl.DataFrame(), pl.DataFrame(), driver_cols

        wmapes_df = RLSForecastRunner._compute_wmape(res_df)

        # ── Rolling 28d — SOLO nivel sección (RLS real; ver settings.COMPUTE_ROLLING_28) ──
        # Si está desactivado, NO se agregan columnas yhat28/valuehat28 en
        # absoluto (ni siquiera nulas) — el dashboard detecta si el
        # rolling28 fue calculado por la sola presencia de la columna.
        if getattr(settings, "COMPUTE_ROLLING_28", False):
            # Crear panel específico para la sección (train+OOS+forecast)
            panel_parts_section = [df_train.with_columns(pl.lit("in_sample").alias("period_type"))]
            if df_oos.height:
                panel_parts_section.append(df_oos.with_columns(pl.lit("out_sample").alias("period_type")))
            if df_fcst.height:
                panel_parts_section.append(df_fcst.with_columns(pl.lit("forecast_only").alias("period_type")))
            panel_all_section = pl.concat(panel_parts_section, how="diagonal_relaxed").sort(["unique_id", "ds"])
            if meta:
                panel_all_section = panel_all_section.with_columns([pl.lit(v).alias(k) for k, v in meta.items()])

            panel_section = panel_all_section.filter(pl.col("unique_id") == seccion)
            logger.info(
                "Sección %s: rolling 28d (solo nivel sección) sobre panel shape=%s…",
                seccion,
                panel_section.shape,
            )
            with _stage_timer(f"{seccion}: rolling28 sección (O(n))"):
                roll = runner.run_rolling_28(
                    panel_section, self._cfg.forecast_levels, desc=f"{seccion} rolling28"
                )
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
        else:
            logger.info(
                "Sección %s: rolling28 desactivado (settings.COMPUTE_ROLLING_28=False)",
                seccion,
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
        logger.info(
            "⏱ Pipeline RLS: TOTAL %.1fs", time.perf_counter() - pipeline_t0
        )
        return res_df, wmapes_df

    def _write_checkpoint(self, seccion: str, res: pl.DataFrame) -> None:
        try:
            self._cfg.out_dir.mkdir(parents=True, exist_ok=True)
            partial_path = self._cfg.out_dir / f"forecast_seccion_{seccion}_partial.parquet"
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
        elif "forecast_start" in sku_level.columns and "forecast_end" in sku_level.columns:
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
