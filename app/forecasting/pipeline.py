"""Forecast pipeline orchestration."""
from __future__ import annotations
import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from app.forecasting.runner import RLSForecastRunner
from app.forecasting.utils import _stage_timer, _collect_streaming, _lf_columns

logger = logging.getLogger(__name__)

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
        Propaga el último estado de precio (asp / edp / discount) por unique_id
        al grid de forecast-only.

        Sin esto el RLS recibe precio=0 en el tramo sin actuals y el nivel de
        yhat se infla (elasticidad a precio → demanda artificialmente alta).

        Regla: último día con y>0 de cada serie; si no hay ventas, último día
        con asp/edp no nulo; si tampoco, 0.
        """
        price_cols = [c for c in ("asp", "edp", "discount") if c in history.columns]
        if grid.height == 0:
            return grid
        if not price_cols or history.height == 0:
            return grid.with_columns(
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
        out = grid.join(last, on="unique_id", how="left")
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
        if not decompose_price:
            raise TypeError(
                "El componente decompose_price no ha sido definido, instalar tqdm y probar nuevamente..."
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
                # "asp": np.asarray(asp, dtype=np.float32).ravel(),
                # "edp": np.asarray(edp, dtype=np.float32).ravel(),
                # "discount": np.asarray(discount, dtype=np.float32).ravel(),
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
                df_oos = self._sanitize_numeric_drivers(
                    self._calculate_edp(df_oos)
                )
            with _stage_timer(f"{seccion}: features OOS"):
                df_oos = self._feature_builder.extract_drivers(
                    df_oos, req_columns=driver_cols
                ).sort("ds")
            logger.info("Sección %s: OOS densificado shape=%s", seccion, df_oos.shape)

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
                # Historial de precios: train + OOS (si hay). Sin esto asp/edp/
                # discount quedan en 0 y el RLS infla el nivel de yhat.
                price_history = df_train
                if df_oos.height:
                    price_history = pl.concat(
                        [df_train, df_oos], how="diagonal_relaxed"
                    )
                df_fcst_raw = self._sanitize_numeric_drivers(
                    self._carry_forward_prices(price_history, df_fcst_raw)
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
        print("\n")
        logger.info(
            "Sección %s: %d filas de pronósticos a nivel tienda",
            seccion,
            res_store.height,
        )

        # ── 3) SKU+tienda ─────────────────────────────────────────────────────
        # USE_SES=True  → efecto(coefs sección/tienda elegidos) + SES causal
        # USE_SES=False → RLS puro por serie sku+tienda (igual que sección/tienda)
        res_derived = pl.DataFrame()
        is_sku_store = (
            pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2
        )
        use_ses = bool(getattr(settings, "USE_SES", True))

        if use_ses:
            # Filtrar SKU+tienda ANTES de concat para no inflar el panel
            parts_sku: list[pl.DataFrame] = []
            tr = df_train.filter(is_sku_store)
            if tr.height:
                parts_sku.append(
                    tr.with_columns(pl.lit("in_sample").alias("period_type"))
                )
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
                    "Sección %s: %d filas SKU+tienda (train+OOS+fcst) [USE_SES=True]",
                    seccion,
                    sku_level.height,
                )
                with _stage_timer(f"{seccion}: derivación SKU+tienda SES (por tienda)"):
                    res_derived = runner.derive_sku_store_forecasts(
                        sku_level,
                        seccion,
                        section_coefs,
                        store_coefs,
                    )
                del sku_level
                logger.info(
                    "Sección %s: %d filas derivadas SKU+tienda (SES)",
                    seccion,
                    res_derived.height,
                )
            else:
                del parts_sku
        else:
            # RLS puro a nivel sku+tienda (mismo camino que sección / tienda)
            train_sku = df_train.filter(is_sku_store)
            sku_ids = (
                train_sku.get_column("unique_id").unique().to_list()
                if train_sku.height
                else []
            )
            if sku_ids:
                targets_sku = {
                    name: df.filter(is_sku_store) for name, df in targets.items()
                }
                logger.info(
                    "Sección %s: RLS puro SKU+tienda (%d series) [USE_SES=False]",
                    seccion,
                    len(sku_ids),
                )
                with _stage_timer(f"{seccion}: RLS sku+tienda ({len(sku_ids)} series)"):
                    res_sku, _sku_coefs = runner.fit_and_predict_sections(
                        train_sku,
                        targets_sku,
                        sku_ids,
                        desc=f"{seccion} RLS sku+tienda",
                        meta=meta,
                    )
                if res_sku.height:
                    res_derived = res_sku.with_columns(
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
                        pl.lit("rls_propio").alias("modelo_seleccionado"),
                    )
                logger.info(
                    "Sección %s: %d filas RLS puro SKU+tienda",
                    seccion,
                    res_derived.height,
                )
            else:
                logger.warning(
                    "Sección %s: sin series sku+tienda para RLS puro", seccion
                )

        # ── Combinar resultados ────────────────────────────────────────────────
        res_df = pl.concat(
            [f for f in (res_section, res_store, res_derived) if f.height],
            how="diagonal_relaxed",
        )
        if not res_df.height:
            return pl.DataFrame(), pl.DataFrame(), driver_cols

        # Corrección de sesgo OOS/forecast (factor in-sample por unique_id)
        if getattr(settings, "BIAS_CORRECTION", True):
            with _stage_timer(f"{seccion}: bias correction OOS"):
                res_df = RLSForecastRunner.apply_bias_correction(res_df)
            logger.info(
                "Sección %s: bias correction OOS aplicada (min_points=%s, clip=%s)",
                seccion,
                getattr(settings, "BIAS_CORRECTION_MIN_POINTS", 7),
                getattr(settings, "BIAS_CORRECTION_CLIP", (0.5, 2.0)),
            )

        wmapes_df = RLSForecastRunner._compute_wmape(res_df)
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
        first_by_sec = self._first_data_by_section(selected)
        _advance("datos cargados")

        all_res, all_wm = [], []
        driver_cols = None
        for seccion in settings.FOCUS_SECTIONS:
            if seccion not in first_by_sec:
                logger.warning("Sin datos para sección %s; se omite.", seccion)
                _advance(f"sección {seccion} omitida")
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
            _advance(f"sección {seccion} completada")

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
        # Artefactos del dashboard (index + metrics + series slim)
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
