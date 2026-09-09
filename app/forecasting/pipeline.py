"""Forecast pipeline orchestration."""
from __future__ import annotations
import datetime as dt
import gc
import logging
import shutil
import time
from pathlib import Path
import numpy as np
import polars as pl
import settings
from rls_opt.edp import decompose_price
from tqdm import tqdm
from app.forecasting.aggregation import DataAggregator
from app.forecasting.calendar import HolidayCalendar
from app.forecasting.config import ForecastConfig
from app.forecasting.features import CalendarFeatureBuilder
from app.forecasting.panel import densify_section_panel
from app.forecasting.metrics import compute_wmape
from app.forecasting.runner import RLSForecastRunner
from app.forecasting.utils import _stage_timer, _collect_streaming, _lf_columns

logger = logging.getLogger(__name__)


class RLSForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig,
        n_jobs: int | None = None,
        limit_series: int | None = None,
        optimization_diagnostics: bool = False,
        optimization_phase2: bool = False,
    ):
        self._cfg = config
        # n_jobs = Nº de threads (ver docstring RLSForecastRunner)
        self._n_jobs = n_jobs
        self._limit_series = limit_series
        self._optimization_diagnostics = bool(optimization_diagnostics)
        self._optimization_phase2 = bool(optimization_phase2)
        self._optimization_ses_frames: list[pl.DataFrame] = []
        self._optimization_leaf_parent_frames: list[pl.DataFrame] = []
        self._optimization_rls_frames: list[pl.DataFrame] = []
        self._optimization_driver_frames: list[pl.DataFrame] = []
        self._optimization_driver_refit_frames: list[pl.DataFrame] = []
        self._calendar: HolidayCalendar | None = None
        self._feature_builder: CalendarFeatureBuilder | None = None
        self._aggregator = DataAggregator(
            config.date_column,
            config.quantity_column,
            config.price_column,
            config.aggregation_levels,
        )

    def optimization_diagnostics(self) -> dict[str, pl.DataFrame]:
        """Return compact statistical-tuning artifacts collected during a run."""
        def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
            return (
                pl.concat([f for f in frames if f.height], how="diagonal_relaxed")
                if any(f.height for f in frames)
                else pl.DataFrame()
            )

        return {
            "ses_candidates": _concat(self._optimization_ses_frames),
            "leaf_parent_candidates": _concat(self._optimization_leaf_parent_frames),
            "rls_candidates": _concat(self._optimization_rls_frames),
            "driver_screen": _concat(self._optimization_driver_frames),
            "driver_refit": _concat(self._optimization_driver_refit_frames),
        }

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

    def _data_bounds_by_section(
        self, selected: pl.LazyFrame | pl.DataFrame
    ) -> dict[str, tuple[dt.date, dt.date]]:
        """Return min/max actual dates by section without materializing rows."""
        col = self._cfg.date_column
        lf = selected.lazy() if isinstance(selected, pl.DataFrame) else selected
        summary = _collect_streaming(
            lf.group_by("SECCION").agg(
                pl.col(col).min().alias("min_d"),
                pl.col(col).max().alias("max_d"),
            )
        )
        out: dict[str, tuple[dt.date, dt.date]] = {}
        for r in summary.iter_rows(named=True):
            d0, d1 = r["min_d"], r["max_d"]
            if isinstance(d0, dt.datetime):
                d0 = d0.date()
            if isinstance(d1, dt.datetime):
                d1 = d1.date()
            out[str(r["SECCION"])] = (d0, d1)
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
        """Genera filas ds para [start, end] por unique_id (y=0, value=0).

        No incluye asp/edp/discount: se rellenan después con carry-forward
        desde train/OOS vía `_carry_forward_prices`.
        """
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
    def _carry_forward_prices(
        history: pl.DataFrame, grid: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Propaga el último estado CONOCIDO de precio (asp / edp / discount)
        por unique_id a un target causal (OOS, extensión o forecast-only).

        Sin esto el RLS recibe precio=0 en el tramo sin actuals y el nivel de
        yhat se infla (elasticidad a precio → demanda artificialmente alta).

        Regla: último día con y>0 de cada serie; si no hay ventas, último día
        con asp/edp no nulo; si tampoco, 0.
        """
        price_cols = [c for c in ("asp", "edp", "discount") if c in history.columns]
        if grid.height == 0:
            return grid
        if not price_cols or history.height == 0:
            base_grid = grid.drop(
                [c for c in ("asp", "edp", "discount") if c in grid.columns]
            )
            return base_grid.with_columns(
                [pl.lit(0.0).alias(c) for c in ("asp", "edp", "discount")]
            )

        cols = ["unique_id", "ds"] + price_cols
        if "y" in history.columns:
            cols.append("y")
        hist = history.select([c for c in cols if c in history.columns])

        if "y" in hist.columns:
            with_sales = hist.filter(pl.col("y") > 0)
            base = with_sales if with_sales.height else hist
        else:
            base = hist

        # Preferir filas con algún precio observado > 0
        priced = base.filter(
            pl.any_horizontal(
                [(pl.col(c).is_not_null() & (pl.col(c) > 0)) for c in price_cols]
            )
        )
        if priced.height:
            base = priced

        last = (
            base.sort("ds")
            .group_by("unique_id")
            .agg([pl.col(c).last().alias(c) for c in price_cols])
        )
        # The target may already contain placeholder or actual-derived price
        # columns. Drop them BEFORE the join so OOS can never retain information
        # derived from its own y/value.
        base_grid = grid.drop(
            [c for c in ("asp", "edp", "discount") if c in grid.columns]
        )
        out = base_grid.join(last, on="unique_id", how="left")
        out = out.with_columns([pl.col(c).fill_null(0.0) for c in price_cols])
        # Columnas de precio que no estaban en history (defensa)
        for c in ("asp", "edp", "discount"):
            if c not in out.columns:
                out = out.with_columns(pl.lit(0.0).alias(c))
        return out

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
        decomp = decompose_price
        for part in tqdm(
            groups,
            desc="EDP (loop)",
            unit="serie",
            ncols=50,  # Controla el ancho total
            ascii="░█",  # Define los caracteres de llenado (vacío/lleno)
            bar_format="{bar} [{n_fmt}/{total_fmt}] {desc}...",  # Estructura del texto
        ):
            uid = part["unique_id"][0]
            asp, edp, discount = decomp(
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

        Fallback vectorizado aproximado cuando hay demasiadas series
        (umbral); fallback a loop por serie si el batch falla.
        """
        if df.height == 0:
            return df
        n_series = df["unique_id"].n_unique()
        decomp = decompose_price
        if n_series > 5000:
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
            asp, edp, discount = decomp(
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
            # pl.Series("asp", np.asarray(asp, dtype=np.float32)),
            # pl.Series("edp", np.asarray(edp, dtype=np.float32)),
            # pl.Series("discount", np.asarray(discount, dtype=np.float32)),
        )


    @staticmethod
    def _sanitize_numeric_drivers(df: pl.DataFrame) -> pl.DataFrame:
        """Replace NaN/±inf/null in numerical drivers before feature extraction.

        EDP/ASP/discount can become non-finite on intermittent or zero-sales
        rows.  RLS requires a finite design matrix, so sanitise at the source
        while keeping the rows (dropping them would bias sparse retail series).
        """
        if df.height == 0:
            return df
        cols = [c for c in ("asp", "edp", "discount") if c in df.columns]
        if not cols:
            return df
        exprs = []
        for c in cols:
            exprs.append(
                pl.when(pl.col(c).is_not_null() & pl.col(c).is_finite())
                .then(pl.col(c))
                .otherwise(0.0)
                .cast(pl.Float64)
                .alias(c)
            )
        return df.with_columns(exprs)

    def _run_section(
        self,
        selected: pl.LazyFrame | pl.DataFrame,
        seccion: str,
        first_data: dt.date,
        last_actual: dt.date,
        driver_cols: list[str] | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
        """Ejecuta train / OOS / forecast-only para una sección."""
        hz = settings.section_horizons(seccion, first_data, last_actual)
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
        # OOS is the latest 28 actual days. By contract there is no
        # actual_extension or observed_tail after OOS.

        # ── Agregación TRAIN: separar nodos RLS de hojas SKU+tienda ───────────
        # Las hojas NO se densifican ni pasan por EDP/features sobre toda la
        # historia. Ese era el principal cuello de botella: O(n_hojas*n_días).
        with _stage_timer(f"{seccion}: agregación train (lazy+streaming)"):
            df_train_all = self._aggregator.aggregate(raw_train_lf)
        df_train_all = self._apply_limit_series(df_train_all)
        _depth_train = pl.col("unique_id").str.count_matches(r"\|\|", literal=False)
        train_leaves = df_train_all.filter(_depth_train == 2)
        df_train = df_train_all.filter(_depth_train < 2)
        del df_train_all
        logger.info(
            "Sección %s: train | nodos RLS=%d filas/%d series | hojas observadas=%d filas/%d series",
            seccion,
            df_train.height,
            df_train["unique_id"].n_unique() if df_train.height else 0,
            train_leaves.height,
            train_leaves["unique_id"].n_unique() if train_leaves.height else 0,
        )

        n_before = df_train.height
        with _stage_timer(f"{seccion}: densify train SOLO sección/tienda"):
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
            df_train = self._sanitize_numeric_drivers(
                self._calculate_edp(df_train)
            )

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

        # ── OOS: separar nodos RLS y hojas ────────────────────────────────────
        with _stage_timer(f"{seccion}: agregación OOS (lazy+streaming)"):
            df_oos_all = self._aggregator.aggregate(raw_oos_lf)
            _depth_oos = pl.col("unique_id").str.count_matches(r"\|\|", literal=False)
            oos_leaves = df_oos_all.filter(_depth_oos == 2)
            # Solo hojas vistas en train: evita introducir series sin historial.
            if train_leaves.height and oos_leaves.height:
                leaf_uids = train_leaves.select("unique_id").unique()
                oos_leaves = oos_leaves.join(leaf_uids, on="unique_id", how="semi")
            df_oos = df_oos_all.filter(_depth_oos < 2)
            del df_oos_all
            if df_oos.height:
                # densify solo sección/tienda: unas pocas series, no miles.
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
            # OOS is a genuine forecast target: price drivers may NOT be
            # calculated from OOS actual y/value. Carry the last known TRAIN
            # price state forward, then build only calendar/known drivers.
            with _stage_timer(f"{seccion}: drivers OOS causales"):
                df_oos = self._sanitize_numeric_drivers(
                    self._carry_forward_prices(df_train, df_oos)
                )
                df_oos = self._feature_builder.extract_drivers(
                    df_oos, req_columns=driver_cols
                ).sort("ds")
            logger.info(
                "Sección %s: OOS causal (sin drivers derivados de actual OOS) shape=%s",
                seccion,
                df_oos.shape,
            )

        # No existe un período observado posterior a OOS; OOS termina en last_actual.

        # ── Forecast-only (calendario sintético + carry-forward de precios) ───
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
                # Para forecast-only sí conocemos todos los actuals hasta
                # last_actual. Recalculamos SOLO el estado de precio de los
                # nodos RLS con esa historia conocida; esto no actualiza SES ni
                # coeficientes RLS con un tail parcial.
                price_parts = [df_train]
                if df_oos.height:
                    price_parts.append(df_oos)
                price_actual_history = pl.concat(
                    price_parts, how="diagonal_relaxed"
                )
                price_actual_history = self._sanitize_numeric_drivers(
                    self._calculate_edp(price_actual_history)
                )
                df_fcst_raw = self._sanitize_numeric_drivers(
                    self._carry_forward_prices(
                        price_actual_history, df_fcst_raw
                    )
                )
                n_with_price = (
                    int(
                        df_fcst_raw.filter((pl.col("edp") > 0) | (pl.col("asp") > 0))[
                            "unique_id"
                        ].n_unique()
                    )
                    if "edp" in df_fcst_raw.columns
                    else 0
                )
                logger.info(
                    "Sección %s: carry-forward precios → %d/%d series con edp/asp>0",
                    seccion,
                    n_with_price,
                    len(uids),
                )
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
            n_jobs=self._n_jobs,
            optimization_diagnostics=self._optimization_diagnostics,
            optimization_phase2=self._optimization_phase2,
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
                pl.when(pl.col("period_type") == "forecast_only").then(0.0).otherwise(pl.col("y")).alias("y"),
                pl.when(pl.col("period_type") == "forecast_only").then(0.0).otherwise(pl.col("value")).alias("value"),
            )

        if not section_coefs:
            logger.warning(
                "Sección %s: no se pudo ajustar RLS a nivel sección; "
                "no hay coeficientes para derivar tienda/sku.",
                seccion,
            )
            return res_section, pl.DataFrame(), driver_cols

        # ── 2) RLS a nivel tienda: batch único ────────────────────────────
        # El runner evita el bucle externo store-by-store y particiona
        # train/targets una sola vez y paraleliza UIDs independientes.
        locales = settings.SECCIONES[seccion]["locales"]
        store_uids = [
            settings.make_unique_id(seccion, store=store_id)
            for store_id in locales
        ]
        train_stores = df_train.filter(pl.col("unique_id").is_in(store_uids))
        targets_stores = {
            name: df.filter(pl.col("unique_id").is_in(store_uids))
            for name, df in targets.items()
        }

        with _stage_timer(f"{seccion}: RLS tiendas batch ({len(store_uids)} series)"):
            res_store, store_coefs = runner.fit_and_predict_sections(
                train_stores,
                targets_stores,
                store_uids,
                desc=f"{seccion} RLS tiendas",
                meta=meta,
            )

        if res_store.height:
            res_store = res_store.with_columns(
                pl.when(pl.col("period_type") == "forecast_only").then(0.0).otherwise(pl.col("y")).alias("y"),
                pl.when(pl.col("period_type") == "forecast_only").then(0.0).otherwise(pl.col("value")).alias("value"),
            )

        logger.info(
            "Sección %s: %d filas de pronósticos a nivel tienda (%d/%d fits)",
            seccion,
            res_store.height,
            len(store_coefs),
            len(store_uids),
        )

        # ── 3) SKU+tienda — única ruta productiva SES+RLS ──────────────────
        # El SES estima la magnitud positiva de la hoja; el RLS elegido
        # (tienda o sección) aporta exactamente el mismo efecto de drivers que
        # usa su nodo padre. No existe una segunda familia de modelo leaf.
        res_derived = pl.DataFrame()

        with _stage_timer(f"{seccion}: SKU+tienda SES+RLS"):
            parent_forecasts = (
                pl.concat(
                    [f for f in (res_section, res_store) if f.height],
                    how="diagonal_relaxed",
                )
                if (res_section.height or res_store.height)
                else pl.DataFrame()
            )
            res_derived = runner.predict_leaf_series(
                train_leaves,
                oos_leaves,
                seccion,
                hz,
                meta=meta,
                parent_forecasts=parent_forecasts,
                diagnostics_out=(
                    self._optimization_ses_frames
                    if self._optimization_diagnostics
                    else None
                ),
                parent_diagnostics_out=(
                    self._optimization_leaf_parent_frames
                    if self._optimization_diagnostics and self._optimization_phase2
                    else None
                ),
            )

        logger.info(
            "Sección %s: %d filas SKU+tienda SES+RLS",
            seccion,
            res_derived.height,
        )

        if self._optimization_diagnostics:
            rls_diag = runner.optimization_candidate_frame()
            driver_diag = runner.optimization_driver_frame()
            driver_refit_diag = runner.optimization_driver_refit_frame()
            if rls_diag.height:
                self._optimization_rls_frames.append(rls_diag)
            if driver_diag.height:
                self._optimization_driver_frames.append(driver_diag)
            if driver_refit_diag.height:
                self._optimization_driver_refit_frames.append(driver_refit_diag)

        # Una sola familia de modelo en los tres períodos. No se aplican
        # correcciones post-hoc que modifiquen únicamente OOS/forecast-only.
        parent_res = (
            pl.concat(
                [f for f in (res_section, res_store) if f.height],
                how="diagonal_relaxed",
            )
            if (res_section.height or res_store.height)
            else pl.DataFrame()
        )
        res_df = pl.concat(
            [f for f in (parent_res, res_derived) if f.height],
            how="diagonal_relaxed",
        )
        if not res_df.height:
            return pl.DataFrame(), pl.DataFrame(), driver_cols

        wmapes_df = compute_wmape(
            res_df,
            period_types=("out_sample",)
            if bool(getattr(settings, "PIPELINE_WMAPE_OOS_ONLY", True))
            else None,
        )
        return res_df, wmapes_df, driver_cols

    def run(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        active_sections = list(settings.FOCUS_SECTIONS)
        total_steps = 2 + len(active_sections)  # carga + cada sección + consolidación
        stages = tqdm(
            total=total_steps,
            desc="Pipeline RLS",
            unit="etapa",
            bar_format="{bar} [{n_fmt}/{total_fmt}] {desc}...",
            ncols=70,
            ascii="░█",
        )
        completed = 0

        def _advance(label: str) -> None:
            nonlocal completed
            completed += 1
            stages.set_postfix_str(label)
            stages.update(1)
            pct = 100.0 * completed / total_steps
            logger.info("PROGRESO %.0f%% | %s", pct, label)

        pipeline_t0 = time.perf_counter()
        logger.info("PROGRESO 0%% | iniciando pipeline RLS")

        self._calendar = HolidayCalendar(self._cfg.holidays, self._cfg.now_year)
        self._feature_builder = CalendarFeatureBuilder(self._calendar)

        selected = self._load()
        bounds_by_sec = self._data_bounds_by_section(selected)
        _advance("datos cargados")

        memory_safe = (
            bool(getattr(settings, "MULTIBLOCK_MEMORY_SAFE", True))
            and int(getattr(settings, "RLS_BLOCK_DAYS", 28))
            <= int(getattr(settings, "MULTIBLOCK_MEMORY_SAFE_THRESHOLD_DAYS", 28))
        )
        section_spill_dir = self._cfg.out_dir / "_section_spill"
        if memory_safe:
            shutil.rmtree(section_spill_dir, ignore_errors=True)
            section_spill_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Memory-safe pipeline: resultados de sección se persisten antes de continuar"
            )
        all_res, all_wm = [], []
        section_res_paths: list[Path] = []
        section_wm_paths: list[Path] = []
        driver_cols = None
        for seccion in settings.FOCUS_SECTIONS:
            if seccion not in bounds_by_sec:
                logger.warning("Sin datos para sección %s; se omite.", seccion)
                _advance(f"sección {seccion} omitida")
                continue
            sec_t0 = time.perf_counter()
            first_data, last_actual = bounds_by_sec[seccion]
            res, wm, driver_cols = self._run_section(
                selected, seccion, first_data, last_actual, driver_cols
            )
            logger.info(
                "⏱ Sección %s: TOTAL %.1fs", seccion, time.perf_counter() - sec_t0
            )
            if res.height:
                if memory_safe:
                    part = section_spill_dir / f"forecast_section_{seccion}.parquet"
                    res.write_parquet(part, compression="zstd", compression_level=3, statistics=True)
                    section_res_paths.append(part)
                    logger.info("✓ Spill sección %s: %s", seccion, part)
                else:
                    all_res.append(res)
                # Full-section checkpoints are large (millions of rows). Keep
                # them opt-in for diagnostics/recovery, not on the production
                # hot path. The final forecast is still written normally.
                if bool(getattr(settings, "WRITE_SECTION_CHECKPOINTS", False)):
                    self._write_checkpoint(seccion, res)
            if wm.height:
                if memory_safe:
                    part_wm = section_spill_dir / f"wmape_section_{seccion}.parquet"
                    wm.write_parquet(part_wm, compression="zstd", compression_level=3, statistics=True)
                    section_wm_paths.append(part_wm)
                else:
                    all_wm.append(wm)
            if memory_safe:
                del res, wm
                gc.collect()
            _advance(f"sección {seccion} completada")

        if memory_safe:
            res_df = (
                pl.concat([pl.scan_parquet(str(p)) for p in section_res_paths], how="diagonal_relaxed")
                .collect(engine="streaming")
                if section_res_paths else pl.DataFrame()
            )
            wmapes_df = (
                pl.concat([pl.scan_parquet(str(p)) for p in section_wm_paths], how="diagonal_relaxed")
                .collect(engine="streaming")
                if section_wm_paths else pl.DataFrame()
            )
            shutil.rmtree(section_spill_dir, ignore_errors=True)
            gc.collect()
        else:
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

    def save(
        self,
        res_df: pl.DataFrame,
        wmapes_df: pl.DataFrame,
        *,
        build_dashboard: bool = True,
    ) -> None:
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
        # Artefactos del dashboard (index + metrics + series slim). In the
        # memory-safe multi-cadence path the caller builds them only AFTER
        # releasing res_df/wmapes_df, avoiding two full forecast copies in RAM.
        if build_dashboard:
            try:
                from app.dashboard_artifacts import build_artifacts

                adir = build_artifacts(forecast_path)
                logger.info("✓ Artefactos dashboard en: %s", adir)
            except Exception:
                logger.exception(
                    "No se pudieron construir artefactos del dashboard "
                    "(el dashboard usará path legacy hasta que se ejecute "
                    "`python -m app.dashboard_artifacts`)"
                )

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

        # Parse unique_id completamente en Polars para evitar conversiones masivas
        # forecast-only a listas Python y llamaba split_unique_id por fila.
        parts = sku_level.with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("SECCION"),
            pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("Local"),
            pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("SKU"),
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
