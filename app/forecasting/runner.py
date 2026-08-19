"""RLS forecasting runner: fit, prediction and SKU-store derivation."""
from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl

import settings
from app.forecasting.metrics import compute_wmape
from app.forecasting.utils import stage_timer as _stage_timer
from rls_opt import RecursiveLeastSquaresRegression, RLSConstantPrior, RLSPrior

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class _NoOp:
        def __init__(self, iterable=None, *a, **k): self._iterable = iterable
        def __iter__(self): return iter(self._iterable if self._iterable is not None else [])
        def update(self, n=1): pass
        def set_postfix_str(self, *a, **k): pass
        def close(self): pass
    def tqdm(iterable=None, *a, **k): return _NoOp(iterable, *a, **k)

logger = logging.getLogger(__name__)

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
        """Backward-compatible wrapper around the pure WMAPE metric."""
        return compute_wmape(res_df)

    @staticmethod
    def _correction_factor(errors: np.ndarray) -> float:
        sigma2 = errors.var(ddof=1)
        return float(np.exp(sigma2 / 2))

    def _default_priors(self, n_features: int):
        if RLSConstantPrior is None or RLSPrior is None:
            raise RuntimeError("rls_opt no disponible")

        return [RLSConstantPrior(standard_error=0.5, rmse_error=self._rmse_error)] + [
            RLSPrior(coefficient=0, standard_error=0.5, rmse_error=self._rmse_error)
            for _ in range(max(0, n_features - 1))
        ]

    def _new_rls(self, min_y: float, return_all_coefs: bool = False):
        if RecursiveLeastSquaresRegression is None:
            raise RuntimeError("rls_opt no disponible")

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
        # X_y = train_g.select(self._driver_cols).to_numpy().astype(np.float32, order="C")
        y = train_g["y"].to_numpy()
        log_y = np.log1p(y)
        model_y = self._new_rls(self._min_y_to_update)
        priors_y = self._default_priors(len(self._driver_cols))
        model_y.fit(x=X_y, y=log_y, priors=priors_y)

        X_p = (
            train_g.select(self._driver_cols_price)
            .to_numpy()
            .astype(np.float64, order="C")
            # .astype(np.float32, order="C")
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
            # test_g.select(self._driver_cols).to_numpy().astype(np.float32, order="C")
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
            # .astype(np.float32, order="C")
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
                # np.asarray(model_y.final_coef_[0], dtype=np.float32).ravel(),
                # np.asarray(model_p.final_coef_[0], dtype=np.float32).ravel(),
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

        coef_y_sec = np.ascontiguousarray(coef_section[0], dtype=np.float32).ravel()
        coef_p_sec = np.ascontiguousarray(coef_section[1], dtype=np.float32).ravel()

        coef_y_sec_fx = coef_y_sec[idx_y]
        coef_p_sec_fx = coef_p_sec[idx_p]

        store_uids = panel.get_column("_store_uid").unique().to_list()
        frames: list[pl.DataFrame] = []
        has_period = "period_type" in panel.columns

        for si, store_uid in enumerate(store_uids):
            # Liberar variables temporales explícitamente e invocar garbage collection si la tienda es muy grande
            coef_store = store_coefs.get(store_uid)
            if coef_store is None:
                continue

            sub = panel.filter(pl.col("_store_uid") == store_uid)
            if sub.height == 0:
                continue
            elif sub.height > 500_000:
                gc.collect()

            # Un solo to_numpy por bloque de drivers (Float32)
            X_y = np.ascontiguousarray(
                sub.select(driver_cols).to_numpy(), dtype=np.float32
            )
            X_p = np.ascontiguousarray(
                sub.select(driver_cols_price).to_numpy(), dtype=np.float32
            )
            y = sub["y"].to_numpy().astype(np.float64, copy=False)
            # y = sub["y"].to_numpy().astype(np.float32, copy=False)
            value = (
                sub["value"].to_numpy().astype(np.float64, copy=False)
                # sub["value"].to_numpy().astype(np.float32, copy=False)
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

            coef_y_sto = np.ascontiguousarray(coef_store[0], dtype=np.float32).ravel()
            coef_p_sto = np.ascontiguousarray(coef_store[1], dtype=np.float32).ravel()

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
