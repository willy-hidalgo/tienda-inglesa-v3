"""RLS forecasting runner and leaf-level derivation."""
from __future__ import annotations
import datetime as dt
import gc
import logging
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import polars as pl
import settings
from rls_opt import RecursiveLeastSquaresRegression, RLSConstantPrior, RLSPrior
from rls_opt.kernels import (
    _rls as _rls_numba_kernel,
    _predict_loglink_base,
    _predict_loglink_ar,
)
from app.forecasting.metrics import compute_wmape
from app.forecasting.leaf_fallback import apply_robust_leaf_fallbacks
from app.forecasting.leaf_v12 import apply_v12_occurrence_share_challenger

logger = logging.getLogger(__name__)

class RLSForecastRunner:
    """
    RLS a nivel sección/tienda + incumbent v11 y challenger leaf v12.

      - `fit_and_predict_sections`: RLS para sección y tienda.
      - `fast_leaf_forecasts`: construye el incumbent SES+RLS v11 y luego el
        challenger v12.6 SKU-total + pooled LightGBM shape + occurrence/store-share, seleccionado solo
        con bloques cerrados anteriores al horizonte objetivo.
    """

    _numba_kernel_warmed: bool = False

    @classmethod
    def _ensure_numba_rls_kernel(cls) -> None:
        """Fail fast if RLS lost Numba acceleration; warm one cached signature."""
        if cls._numba_kernel_warmed:
            return
        if bool(getattr(settings, "RLS_NUMBA_KERNELS_REQUIRED", True)):
            try:
                from numba.core.registry import CPUDispatcher
            except Exception as exc:  # pragma: no cover - dependency contract
                raise RuntimeError("Numba no disponible para el kernel RLS") from exc
            if not isinstance(_rls_numba_kernel, CPUDispatcher):
                raise RuntimeError(
                    "RLS performance invariant violated: _rls is not a Numba "
                    "CPUDispatcher. Refusing Python fallback."
                )
        t0 = time.perf_counter()
        x = np.ascontiguousarray(np.ones((4, 2), dtype=np.float64))
        y = np.ascontiguousarray(np.ones(4, dtype=np.float64))
        priors = np.zeros(2, dtype=np.float64)
        inv = np.eye(2, dtype=np.float64)
        _rls_numba_kernel(x, y, priors, inv, 0.995, None, True, 1e-8)
        # Warm recursive prediction kernels serially before the store thread
        # pool starts. Otherwise multiple workers may contend on first JIT.
        _predict_loglink_base(x, priors)
        _predict_loglink_ar(
            x,
            np.zeros(x.shape[1] + 4, dtype=np.float64),
            np.zeros(28, dtype=np.float64),
        )
        cls._numba_kernel_warmed = True
        logger.info(
            "RLS Numba kernel listo (nopython+cache+nogil): %.2fs",
            time.perf_counter() - t0,
        )

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

    def _new_rls(
        self,
        min_y: float,
        return_all_coefs: bool = False,
        forgetting_factor: float | None = None,
    ):
        if RecursiveLeastSquaresRegression is None:
            raise RuntimeError("rls_opt no disponible")

        return RecursiveLeastSquaresRegression(
            forgetting_factor=(
                self._forgetting_factor
                if forgetting_factor is None
                else float(forgetting_factor)
            ),
            min_y_to_update=min_y,
            return_all_coefs=return_all_coefs,
        )

    # ── Fit único por serie (Fase 1) ────────────────────────────────────────
    def _fit_models(self, train_g: pl.DataFrame):
        """Ajusta model_y (log1p(y)) y model_p (log1p(price)) UNA sola vez."""
        if RecursiveLeastSquaresRegression is None:
            raise RuntimeError("rls_opt no disponible")

        uid = str(train_g["unique_id"][0]) if "unique_id" in train_g.columns else "<unknown>"
        X_y = self._finite_matrix(train_g, self._driver_cols, uid=uid, stage="RLS fit y")
        y = train_g["y"].to_numpy()
        log_y = np.log1p(y)
        model_y = self._new_rls(self._min_y_to_update)
        priors_y = self._default_priors(len(self._driver_cols))
        model_y.fit(x=X_y, y=log_y, priors=priors_y)

        X_p = self._finite_matrix(
            train_g, self._driver_cols_price, uid=uid, stage="RLS fit value"
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

        X_y_test = self._finite_matrix(
            test_g, self._driver_cols, uid=unique_id, stage="RLS predict y"
        )
        log_yhat_test = model_y.predict(X_y_test)
        if self._use_correction_factor:
            corr = self._correction_factor(np.asarray(model_y.errors))
            yhat_test = np.round(np.exp(log_yhat_test) * corr).ravel()
        else:
            yhat_test = np.round(np.expm1(log_yhat_test)).ravel()

        X_p_test = self._finite_matrix(
            test_g, self._driver_cols_price, uid=unique_id, stage="RLS predict value"
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


    @staticmethod
    def _finite_matrix(
        df: pl.DataFrame,
        columns: list[str],
        *,
        uid: str,
        stage: str,
    ) -> np.ndarray:
        """Build a finite float64 design matrix for RLS with diagnostics."""
        if not columns:
            return np.empty((df.height, 0), dtype=np.float64)
        matrix = df.select(columns).to_numpy().astype(np.float64, order="C")
        finite = np.isfinite(matrix)
        if finite.all():
            return matrix
        bad = ~finite
        bad_by_col = {
            col: int(bad[:, j].sum())
            for j, col in enumerate(columns)
            if bad[:, j].any()
        }
        logger.warning(
            "%s %s: %d valores NaN/inf en drivers; imputando 0.0 | columnas=%s",
            stage, uid, int(bad.sum()), bad_by_col,
        )
        matrix[bad] = 0.0
        return matrix

    # ── RLS sección / tienda ──────────────────────────────────────────────
    @staticmethod
    def _ar_training_matrix(log_target: np.ndarray) -> np.ndarray:
        """Causal AR features: lag7, lag28, mean7, mean28 in log-space."""
        n = len(log_target)
        out = np.zeros((n, 4), dtype=np.float64)
        for i in range(n):
            if i >= 7:
                out[i, 0] = log_target[i - 7]
            if i >= 28:
                out[i, 1] = log_target[i - 28]
            if i > 0:
                out[i, 2] = float(np.mean(log_target[max(0, i - 7):i]))
                out[i, 3] = float(np.mean(log_target[max(0, i - 28):i]))
        return out

    @staticmethod
    def _ar_recursive_row(history: list[float]) -> np.ndarray:
        """AR row available at forecast origin; predicted rows feed recursion."""
        n = len(history)
        lag7 = history[-7] if n >= 7 else 0.0
        lag28 = history[-28] if n >= 28 else 0.0
        mean7 = float(np.mean(history[-7:])) if n else 0.0
        mean28 = float(np.mean(history[-28:])) if n else 0.0
        return np.asarray([lag7, lag28, mean7, mean28], dtype=np.float64)

    def _fit_and_predict_expanding_blocks(
        self, uid: str, train_g: pl.DataFrame, target_parts: dict[str, dict],
        *, meta: dict | None, desc: str, block_days: int,
    ) -> tuple[list[pl.DataFrame], tuple[np.ndarray, np.ndarray] | None]:
        """Expanding-28 with prior-block selection of RLS dynamics.

        The AR specification is treated as a candidate. For every 28-day
        target block we choose only
        from errors accumulated in PREVIOUS blocks:

          * base: calendar/commercial drivers only;
          * ar:   base + lag7/lag28/rolling7/rolling28, recursively forecast.

        This preserves the client's expanding-28 contract while preventing a
        recursively drifting AR path from being forced into OOS/forecast-only.
        """
        t_rls_total = time.perf_counter()
        train_g = train_g.sort("ds")
        oos_g = (target_parts.get("out_sample") or {}).get(uid)
        fcst_g = (target_parts.get("forecast_only") or {}).get(uid)

        parts = [train_g.with_columns(pl.lit("in_sample").alias("_source_period"))]
        if oos_g is not None and oos_g.height:
            parts.append(
                oos_g.sort("ds").with_columns(
                    pl.lit("out_sample").alias("_source_period")
                )
            )
        actual = pl.concat(parts, how="diagonal_relaxed").sort("ds")
        n = actual.height
        if n == 0:
            return [], None

        seed_n = min(max(block_days, int(getattr(settings, "RLS_INITIAL_SEED_DAYS", 28))), n)
        base = actual.drop("_source_period")
        y = base["y"].to_numpy().astype(np.float64, copy=False)
        v = base["value"].to_numpy().astype(np.float64, copy=False)
        log_y = np.log1p(np.clip(y, 0.0, None))
        log_v = np.log1p(np.clip(v, 0.0, None))
        Xy_base = self._finite_matrix(base, self._driver_cols, uid=uid, stage="RLS rolling fit y")
        Xv_base = self._finite_matrix(base, self._driver_cols_price, uid=uid, stage="RLS rolling fit value")

        ar_enabled = bool(getattr(settings, "RLS_AUTOREGRESSIVE_DRIVERS", True))
        mode_candidates = tuple(getattr(settings, "RLS_DYNAMICS_CANDIDATES", ("base", "ar" if ar_enabled else "base")))
        mode_candidates = tuple(dict.fromkeys(m for m in mode_candidates if m in {"base", "ar"} and (m != "ar" or ar_enabled))) or ("base",)
        lambdas = tuple(sorted({float(x) for x in getattr(settings, "RLS_FORGETTING_FACTOR_CANDIDATES", (self._forgetting_factor,)) if 0.0 < float(x) <= 1.0} | {float(self._forgetting_factor)}))
        default_lambda = float(self._forgetting_factor)
        default_mode = str(getattr(settings, "RLS_DEFAULT_DYNAMICS", "base"))
        if default_mode not in mode_candidates:
            default_mode = mode_candidates[0]
        candidates = tuple((mode, lam) for mode in mode_candidates for lam in lambdas)
        default_candidate = (default_mode, default_lambda)

        paths_y: dict[tuple[str, float], np.ndarray] = {}
        paths_v: dict[tuple[str, float], np.ndarray] = {}
        for mode, lam in candidates:
            if mode == "ar":
                Xy_fit = np.column_stack([Xy_base, self._ar_training_matrix(log_y)])
                Xv_fit = np.column_stack([Xv_base, self._ar_training_matrix(log_v)])
            else:
                Xy_fit, Xv_fit = Xy_base, Xv_base
            my = self._new_rls(self._min_y_to_update, return_all_coefs=True, forgetting_factor=lam)
            my.fit(x=Xy_fit, y=log_y, priors=self._default_priors(Xy_fit.shape[1]), seed_n_obs=seed_n)
            paths_y[(mode, lam)] = np.asarray(my.all_coef_[0], dtype=np.float64)
            mv = self._new_rls(1e-8, return_all_coefs=True, forgetting_factor=lam)
            mv.fit(x=Xv_fit, y=log_v, priors=self._default_priors(Xv_fit.shape[1]), seed_n_obs=seed_n)
            paths_v[(mode, lam)] = np.asarray(mv.all_coef_[0], dtype=np.float64)

        def predict_block(Xbase, coef, history_actual, start, end, mode):
            xrows = np.ascontiguousarray(Xbase[start:end], dtype=np.float64)
            c = np.ascontiguousarray(coef, dtype=np.float64)
            if mode == "ar":
                tail = np.ascontiguousarray(
                    history_actual[max(0, start - 28):start], dtype=np.float64
                )
                return _predict_loglink_ar(xrows, c, tail)
            return _predict_loglink_base(xrows, c)

        yh = np.full(n, np.nan); vh = np.full(n, np.nan)
        eligible = np.zeros(n, dtype=bool); block = np.zeros(n, dtype=np.int32)
        train_days = np.full(n, seed_n, dtype=np.int32)
        lambda_y = np.full(n, np.nan); lambda_v = np.full(n, np.nan)
        mode_y = np.full(n, "", dtype=object); mode_v = np.full(n, "", dtype=object)
        origin = np.full(n, None, dtype=object)
        cum_ae_y = {c: 0.0 for c in candidates}; cum_den_y = {c: 0.0 for c in candidates}
        cum_ae_v = {c: 0.0 for c in candidates}; cum_den_v = {c: 0.0 for c in candidates}

        def choose(cae, cden):
            scored = [(cae[c] / cden[c], c) for c in candidates if cden[c] > 0]
            return min(scored, key=lambda z: (z[0], z[1][0], z[1][1]))[1] if scored else default_candidate

        bno = 1
        for s in range(seed_n, n, block_days):
            e = min(s + block_days, n); boundary = s - 1
            chosen_y = choose(cum_ae_y, cum_den_y); chosen_v = choose(cum_ae_v, cum_den_v)
            cand_y = {}; cand_v = {}
            for c in candidates:
                mode, lam = c
                cand_y[c] = predict_block(Xy_base, paths_y[c][boundary], log_y, s, e, mode)
                cand_v[c] = predict_block(Xv_base, paths_v[c][boundary], log_v, s, e, mode)
            yh[s:e] = np.round(cand_y[chosen_y], 0); vh[s:e] = np.round(cand_v[chosen_v], 2)
            eligible[s:e] = True; block[s:e] = bno; train_days[s:e] = s
            mode_y[s:e], lambda_y[s:e] = chosen_y; mode_v[s:e], lambda_v[s:e] = chosen_v
            origin_date = base["ds"][boundary]
            origin[s:e] = [origin_date] * (e - s)
            valid_y = np.isfinite(y[s:e]); valid_v = np.isfinite(v[s:e])
            for c in candidates:
                if valid_y.any():
                    cum_ae_y[c] += float(np.abs(y[s:e][valid_y] - cand_y[c][valid_y]).sum()); cum_den_y[c] += float(np.abs(y[s:e][valid_y]).sum())
                if valid_v.any():
                    cum_ae_v[c] += float(np.abs(v[s:e][valid_v] - cand_v[c][valid_v]).sum()); cum_den_v[c] += float(np.abs(v[s:e][valid_v]).sum())
            bno += 1

        out = pl.DataFrame({
            "unique_id": [uid] * n, "ds": actual["ds"], "value": actual["value"], "valuehat": vh,
            "y": actual["y"], "yhat": yh, "period_type": actual["_source_period"],
            "rls_metric_eligible": eligible, "rls_block": block, "rls_train_days": train_days,
            "rls_lambda_y": lambda_y, "rls_lambda_value": lambda_v,
            "rls_dynamics_y": mode_y.tolist(), "rls_dynamics_value": mode_v.tolist(),
            "rls_forecast_origin": origin.tolist(),
        })
        if meta:
            out = out.with_columns([pl.lit(vv).alias(k) for k, vv in meta.items()])
        for c in ("sku_desc", "store_name", "seccion"):
            if c in train_g.columns and c not in out.columns:
                out = out.with_columns(pl.lit(train_g[c][0]).alias(c))

        final_cy = choose(cum_ae_y, cum_den_y); final_cv = choose(cum_ae_v, cum_den_v)
        final_py = paths_y[final_cy][-1].ravel(); final_pv = paths_v[final_cv][-1].ravel()
        if fcst_g is not None and fcst_g.height:
            fcst_g = fcst_g.sort("ds")
            Xyf_base = self._finite_matrix(fcst_g, self._driver_cols, uid=uid, stage="RLS forecast y")
            Xvf_base = self._finite_matrix(fcst_g, self._driver_cols_price, uid=uid, stage="RLS forecast value")
            cy = choose(cum_ae_y, cum_den_y); cv = choose(cum_ae_v, cum_den_v)
            def predict_future(Xbase, coef, hist_actual, mode):
                xrows = np.ascontiguousarray(Xbase, dtype=np.float64)
                c = np.ascontiguousarray(coef, dtype=np.float64)
                if mode == "ar":
                    tail = np.ascontiguousarray(hist_actual[-28:], dtype=np.float64)
                    return _predict_loglink_ar(xrows, c, tail)
                return _predict_loglink_base(xrows, c)
            fy = np.round(predict_future(Xyf_base, paths_y[cy][-1], log_y, cy[0]), 0)
            fv = np.round(predict_future(Xvf_base, paths_v[cv][-1], log_v, cv[0]), 2)
            forecast_origin = base["ds"][-1]
            f = pl.DataFrame({
                "unique_id": [uid] * fcst_g.height, "ds": fcst_g["ds"],
                "value": fcst_g["value"] if "value" in fcst_g.columns else [0.0] * fcst_g.height, "valuehat": fv,
                "y": fcst_g["y"] if "y" in fcst_g.columns else [0.0] * fcst_g.height, "yhat": fy,
                "period_type": ["forecast_only"] * fcst_g.height, "rls_metric_eligible": [False] * fcst_g.height,
                "rls_block": [bno] * fcst_g.height, "rls_train_days": [n] * fcst_g.height,
                "rls_lambda_y": [cy[1]] * fcst_g.height, "rls_lambda_value": [cv[1]] * fcst_g.height,
                "rls_dynamics_y": [cy[0]] * fcst_g.height, "rls_dynamics_value": [cv[0]] * fcst_g.height,
                "rls_forecast_origin": [forecast_origin] * fcst_g.height,
            })
            if meta:
                f = f.with_columns([pl.lit(vv).alias(k) for k, vv in meta.items()])
            for c in ("sku_desc", "store_name", "seccion"):
                if c in train_g.columns and c not in f.columns:
                    f = f.with_columns(pl.lit(train_g[c][0]).alias(c))
            out = pl.concat([out, f], how="diagonal_relaxed")
            final_py = paths_y[cy][-1].ravel(); final_pv = paths_v[cv][-1].ravel()

        logger.info(
            "%s %s: expanding-%dd | dynamics=%s | lambdas=%s | final y=%s/%.3f v=%s/%.3f",
            desc, uid, block_days, mode_candidates, lambdas, final_cy[0], final_cy[1], final_cv[0], final_cv[1],
        )
        logger.info(
            "⏱ %s %s: RLS candidatos + bloques: %.1fs",
            desc,
            uid,
            time.perf_counter() - t_rls_total,
        )
        return [out.sort("ds")], (final_py, final_pv)

    def fit_and_predict_sections(
        self,
        train: pl.DataFrame,
        targets: dict[str, pl.DataFrame],
        section_ids: list[str],
        desc: str = "RLS sección",
        meta: dict | None = None,
    ) -> tuple[pl.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """Fit/predict independent RLS series with one partition pass.

        v11.2 avoids the previous store-by-store orchestration, which repeatedly
        filtered and partitioned the same frames. Frames are partitioned once
        and independent UIDs can then run in parallel.
        """
        if train.height == 0 or not section_ids:
            return pl.DataFrame(), {}

        self._ensure_numba_rls_kernel()

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
        mode = str(getattr(settings, "RLS_FIT_MODE", "expanding_28")).lower()
        block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))

        def _one(uid: str):
            g = train_parts.get(uid)
            if g is None or g.height == 0:
                return uid, [], None
            if mode == "expanding_28":
                try:
                    parts, cf = self._fit_and_predict_expanding_blocks(
                        uid,
                        g,
                        target_parts,
                        meta=meta,
                        desc=desc,
                        block_days=block_days,
                    )
                    return uid, parts, cf
                except Exception as exc:
                    logger.warning(
                        "%s: expanding-%d falló para %s: %s",
                        desc, block_days, uid, exc,
                    )
                    return uid, [], None

            try:
                my, mv = self._fit_models(g)
            except Exception as exc:
                logger.warning("%s: fit falló para %s: %s", desc, uid, exc)
                return uid, [], None
            cf = (
                np.asarray(my.final_coef_[0], dtype=np.float64).ravel(),
                np.asarray(mv.final_coef_[0], dtype=np.float64).ravel(),
            )
            parts: list[pl.DataFrame] = []
            for name, by_uid in target_parts.items():
                tg = by_uid.get(uid)
                if tg is None or tg.height == 0:
                    continue
                try:
                    fr = self._predict_with_models(uid, my, mv, g, tg, meta)
                except Exception as exc:
                    logger.warning(
                        "%s: predict falló para %s/%s: %s",
                        desc, uid, name, exc,
                    )
                    continue
                if fr is not None and fr.height:
                    parts.append(
                        fr.with_columns(
                            pl.lit(name).alias("period_type"),
                            pl.lit(True).alias("rls_metric_eligible"),
                            pl.lit(None).cast(pl.Int32).alias("rls_block"),
                            pl.lit(g.height).cast(pl.Int32).alias("rls_train_days"),
                        )
                    )
            return uid, parts, cf

        ordered_ids = [str(uid) for uid in section_ids]
        results_by_uid: dict[str, list[pl.DataFrame]] = {}
        coefs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        workers = min(max(int(self._n_jobs or 1), 1), len(ordered_ids))

        if workers > 1 and len(ordered_ids) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_one, uid): uid for uid in ordered_ids}
                for fut in as_completed(futures):
                    uid, parts, cf = fut.result()
                    if parts:
                        results_by_uid[uid] = parts
                    if cf is not None:
                        coefs[uid] = cf
        else:
            for uid in ordered_ids:
                uid, parts, cf = _one(uid)
                if parts:
                    results_by_uid[uid] = parts
                if cf is not None:
                    coefs[uid] = cf

        results = [
            frame
            for uid in ordered_ids
            for frame in results_by_uid.get(uid, [])
        ]
        return (
            pl.concat(results, how="diagonal_relaxed")
            if results
            else pl.DataFrame(),
            coefs,
        )

    # ── Corrección de sesgo OOS / forecast ────────────────────────────────
    @staticmethod
    def apply_bias_correction(
        res_df: pl.DataFrame,
        *,
        min_points: int | None = None,
        clip: tuple[float, float] | None = None,
        enabled: bool | None = None,
    ) -> pl.DataFrame:
        """
        Escala ŷ en out_sample / forecast_only por el sesgo in-sample de cada serie.

          factor_y = Σ y / Σ ŷ     (in_sample, y>0, ŷ>0, finitos)
          factor_v = Σ value / Σ valuehat

        - Solo se corrigen out_sample y forecast_only (in_sample queda crudo).
        - Series con < min_points observaciones válidas → factor = 1.
        - factor se recorta a `clip` para evitar explosiones en series sparse.
        """
        if enabled is None:
            enabled = bool(getattr(settings, "BIAS_CORRECTION", True))
        if not enabled or res_df.height == 0:
            return res_df
        if "period_type" not in res_df.columns or "unique_id" not in res_df.columns:
            return res_df
        if "yhat" not in res_df.columns or "y" not in res_df.columns:
            return res_df

        if min_points is None:
            min_points = int(getattr(settings, "BIAS_CORRECTION_MIN_POINTS", 7))
        if clip is None:
            clip = tuple(getattr(settings, "BIAS_CORRECTION_CLIP", (0.5, 2.0)))
        lo, hi = float(clip[0]), float(clip[1])

        # SKU+tienda is the statistical source of truth for bottom-up
        # metrics. Its forecast contract is exactly SES_level × driver_factor;
        # NO post-hoc bias scaling is allowed. Detect leaves structurally so a
        # future model rename can never re-enable correction by accident.
        _uid = pl.col("unique_id").cast(pl.Utf8)
        is_leaf_uid = (
            _uid.str.contains(r"\|\|T:[^|]+")
            & _uid.str.contains(r"\|\|S:[^|]+")
        )
        train = res_df.filter(
            (pl.col("period_type") == "in_sample") & ~is_leaf_uid
        )
        if train.height == 0:
            return res_df

        has_value = "value" in res_df.columns and "valuehat" in res_df.columns

        # factor_y por unique_id. El factor solo se conserva si también reduce
        # el error absoluto ponderado (numerador del WMAPE) en train.
        scored_y = train.filter(
            pl.col("y").is_not_null()
            & pl.col("yhat").is_not_null()
            & pl.col("y").is_finite()
            & pl.col("yhat").is_finite()
            & (pl.col("y") > 0)
            & (pl.col("yhat") > 0)
        )
        candidate_y = (
            scored_y.group_by("unique_id")
            .agg(
                pl.col("y").sum().alias("_sum_y"),
                pl.col("yhat").sum().alias("_sum_yhat"),
                (pl.col("y") - pl.col("yhat")).abs().sum().alias("_ae_raw"),
                pl.len().alias("_n"),
            )
            .with_columns(
                pl.when((pl.col("_n") >= min_points) & (pl.col("_sum_yhat") > 0))
                .then((pl.col("_sum_y") / pl.col("_sum_yhat")).clip(lo, hi))
                .otherwise(1.0)
                .alias("_candidate_factor_y")
            )
        )
        corrected_y = (
            scored_y.join(
                candidate_y.select(["unique_id", "_candidate_factor_y"]),
                on="unique_id",
                how="left",
            )
            .group_by("unique_id")
            .agg(
                (
                    pl.col("y")
                    - pl.col("yhat") * pl.col("_candidate_factor_y")
                )
                .abs()
                .sum()
                .alias("_ae_corr")
            )
        )
        factors_y = (
            candidate_y.join(corrected_y, on="unique_id", how="left")
            .with_columns(
                pl.when(
                    (pl.col("_n") >= min_points)
                    & pl.col("_ae_corr").is_not_null()
                    & (pl.col("_ae_corr") < pl.col("_ae_raw"))
                )
                .then(pl.col("_candidate_factor_y"))
                .otherwise(1.0)
                .alias("bias_factor_y")
            )
            .select(["unique_id", "bias_factor_y"])
        )

        if has_value:
            scored_v = train.filter(
                pl.col("value").is_not_null()
                & pl.col("valuehat").is_not_null()
                & pl.col("value").is_finite()
                & pl.col("valuehat").is_finite()
                & (pl.col("value") > 0)
                & (pl.col("valuehat") > 0)
            )
            candidate_v = (
                scored_v.group_by("unique_id")
                .agg(
                    pl.col("value").sum().alias("_sum_v"),
                    pl.col("valuehat").sum().alias("_sum_vhat"),
                    (pl.col("value") - pl.col("valuehat")).abs().sum().alias("_ae_raw"),
                    pl.len().alias("_n"),
                )
                .with_columns(
                    pl.when((pl.col("_n") >= min_points) & (pl.col("_sum_vhat") > 0))
                    .then((pl.col("_sum_v") / pl.col("_sum_vhat")).clip(lo, hi))
                    .otherwise(1.0)
                    .alias("_candidate_factor_v")
                )
            )
            corrected_v = (
                scored_v.join(
                    candidate_v.select(["unique_id", "_candidate_factor_v"]),
                    on="unique_id",
                    how="left",
                )
                .group_by("unique_id")
                .agg(
                    (
                        pl.col("value")
                        - pl.col("valuehat") * pl.col("_candidate_factor_v")
                    )
                    .abs()
                    .sum()
                    .alias("_ae_corr")
                )
            )
            factors_v = (
                candidate_v.join(corrected_v, on="unique_id", how="left")
                .with_columns(
                    pl.when(
                        (pl.col("_n") >= min_points)
                        & pl.col("_ae_corr").is_not_null()
                        & (pl.col("_ae_corr") < pl.col("_ae_raw"))
                    )
                    .then(pl.col("_candidate_factor_v"))
                    .otherwise(1.0)
                    .alias("bias_factor_v")
                )
                .select(["unique_id", "bias_factor_v"])
            )
        else:
            factors_v = pl.DataFrame(
                schema={"unique_id": pl.Utf8, "bias_factor_v": pl.Float64}
            )

        # Base = todos los unique_id del resultado → left join de factores
        factors = (
            res_df.select("unique_id")
            .unique()
            .join(factors_y, on="unique_id", how="left")
            .join(factors_v, on="unique_id", how="left")
            .with_columns(
                pl.col("bias_factor_y").fill_null(1.0),
                pl.col("bias_factor_v").fill_null(1.0),
            )
        )

        out = res_df.join(factors, on="unique_id", how="left").with_columns(
            pl.col("bias_factor_y").fill_null(1.0),
            pl.col("bias_factor_v").fill_null(1.0),
        )

        _uid_out = pl.col("unique_id").cast(pl.Utf8)
        is_leaf_out = (
            _uid_out.str.contains(r"\|\|T:[^|]+")
            & _uid_out.str.contains(r"\|\|S:[^|]+")
        )
        is_corr = (
            pl.col("period_type").is_in(["out_sample", "forecast_only"])
            & ~is_leaf_out
        )
        if "modelo_seleccionado" in out.columns:
            _model = pl.col("modelo_seleccionado").cast(pl.Utf8)
            is_corr = is_corr & ~_model.str.starts_with("baseline:")
        exprs = [
            pl.when(is_corr)
            .then(
                (pl.col("yhat") * pl.col("bias_factor_y"))
                .clip(lower_bound=0.0)
                .round(0)
            )
            .otherwise(pl.col("yhat"))
            .alias("yhat"),
        ]
        if has_value:
            exprs.append(
                pl.when(is_corr)
                .then(
                    (pl.col("valuehat") * pl.col("bias_factor_v"))
                    .clip(lower_bound=0.0)
                    .round(2)
                )
                .otherwise(pl.col("valuehat"))
                .alias("valuehat")
            )
        out = out.with_columns(exprs).drop(
            [c for c in ("bias_factor_y", "bias_factor_v") if c in out.columns]
        )
        return out

    # ── SKU+tienda: nivel causal + forma RLS ──────────────────
    def fast_leaf_forecasts(
        self,
        train_leaves: pl.DataFrame,
        oos_leaves: pl.DataFrame,
        section_id: str,
        horizons: dict,
        meta: dict | None = None,
        parent_forecasts: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Production leaf model: original-scale SES level + normalized RLS shape.

        SES owns the baseline level of each SKU+store; fallback levels are conditioned on positive-sale days to match the official metric. Parent RLS forecasts are converted
        to bounded multiplicative factors with arithmetic mean exactly 1 per
        28-day block, so drivers can change shape but never the block mean level.

        The 28-day client contract is strict. Alpha is chosen only from PURE
        SES trajectories scored on earlier blocks; only after the SES level is
        fixed is the parent shape (store/section) selected. The current block
        never tunes itself.
        """
        if train_leaves.height == 0:
            return pl.DataFrame()

        t_leaf_total = time.perf_counter()
        uid_col = "unique_id"
        block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))
        memory_safe = (
            bool(getattr(settings, "MULTIBLOCK_MEMORY_SAFE", True))
            and block_days < int(getattr(settings, "MULTIBLOCK_MEMORY_SAFE_THRESHOLD_DAYS", 28))
        )
        if memory_safe:
            spill_root = Path(getattr(settings, "OUT_DIR", Path.cwd())) / "_spill"
            spill_root.mkdir(parents=True, exist_ok=True)
        else:
            spill_root = None
        spill_tmp = (
            tempfile.TemporaryDirectory(
                prefix=f"ti_leaf_{section_id}_{block_days}d_",
                dir=str(spill_root),
            )
            if memory_safe else None
        )
        spill_path = Path(spill_tmp.name) if spill_tmp is not None else None
        if memory_safe:
            logger.info(
                "Memory-safe leaf: sección %s bloque=%dd | estados intermedios -> disco",
                section_id, block_days,
            )
        warmup_days = int(getattr(settings, "LEAF_INITIAL_LEVEL_DAYS", 28))
        default_alpha = float(getattr(settings, "LEAF_SES_ALPHA", 0.70))
        alpha_candidates = tuple(
            sorted(
                {
                    float(a)
                    for a in getattr(
                        settings,
                        "LEAF_SES_ALPHA_CANDIDATES",
                        (0.20, 0.40, 0.60, 0.70, 0.80),
                    )
                    if 0.0 < float(a) <= 1.0
                }
                | {default_alpha}
            )
        )

        train_start = horizons["train_start"]
        test_start = horizons["test_start"]
        test_end = horizons["test_end"]
        forecast_start = horizons["forecast_start"]
        forecast_end = horizons["forecast_end"]
        warmup_end = train_start + dt.timedelta(days=warmup_days - 1)
        metric_start = warmup_end + dt.timedelta(days=1)

        train_obs = train_leaves.with_columns(pl.col("ds").cast(pl.Date))
        oos_obs = (
            oos_leaves.with_columns(pl.col("ds").cast(pl.Date))
            if oos_leaves.height
            else pl.DataFrame()
        )
        meta_cols = [
            c
            for c in ("sku_desc", "store_name", "seccion", "conteo_sku")
            if c in train_obs.columns
        ]
        ids = (
            train_obs.select([uid_col] + meta_cols)
            .group_by(uid_col)
            .agg([pl.col(c).drop_nulls().first().alias(c) for c in meta_cols])
            if meta_cols
            else train_obs.select(uid_col).unique()
        )

        def block_expr() -> pl.Expr:
            return (
                ((pl.col("ds") - pl.lit(train_start)).dt.total_days() // block_days)
                .cast(pl.Int32)
                .alias("_block")
            )

        # Initial structural level is LEAF-SPECIFIC.  A SKU that enters the
        # assortment later must not inherit the section's calendar warm-up.
        # Each leaf uses its own first 28 calendar days of available history.
        leaf_bounds = (
            train_obs.group_by(uid_col)
            .agg(pl.col("ds").min().alias("_leaf_start"))
            .with_columns(
                (pl.col("_leaf_start") + pl.duration(days=warmup_days - 1))
                .alias("_leaf_warmup_end")
            )
        )
        initial = (
            train_obs.join(leaf_bounds, on=uid_col, how="left")
            .filter(
                (pl.col("ds") >= pl.col("_leaf_start"))
                & (pl.col("ds") <= pl.col("_leaf_warmup_end"))
            )
            .group_by(uid_col)
            .agg(
                pl.when(pl.col("y").is_finite())
                .then(pl.col("y").clip(lower_bound=0.0))
                .otherwise(0.0)
                .sum()
                .alias("_sum_y0"),
                pl.when(pl.col("value").is_finite())
                .then(pl.col("value").clip(lower_bound=0.0))
                .otherwise(0.0)
                .sum()
                .alias("_sum_v0"),
                pl.when(pl.col("y").is_finite())
                .then(1)
                .otherwise(0)
                .sum()
                .alias("_n_obs_y0"),
                pl.when(pl.col("value").is_finite())
                .then(1)
                .otherwise(0)
                .sum()
                .alias("_n_obs_v0"),
            )
            .with_columns(
                # Contract: first 28 CALENDAR days. Missing sale dates are
                # actual zero, so the denominator is always warmup_days.
                (pl.col("_sum_y0") / pl.lit(float(warmup_days)))
                .alias("_mean_y0"),
                (pl.col("_sum_v0") / pl.lit(float(warmup_days)))
                .alias("_mean_v0"),
            )
        )
        state0 = (
            ids.select(uid_col)
            .join(initial, on=uid_col, how="left")
            .with_columns(
                pl.col("_mean_y0")
                .fill_nan(0.0)
                .fill_null(0.0)
                .clip(lower_bound=0.0)
                .alias("_level_y"),
                pl.col("_mean_v0")
                .fill_nan(0.0)
                .fill_null(0.0)
                .clip(lower_bound=0.0)
                .alias("_level_v"),
            )
            .select(uid_col, "_level_y", "_level_v")
        )

        # Parent RLS can contribute either SHAPE ONLY or LEVEL+SHAPE.
        #
        # v11.0-v11.4 forced every 28-day parent effect to arithmetic mean 1.
        # That is safe against level explosions, but it also deletes genuine
        # horizon-wide commercial uplifts (for example Christmas).  v11.5
        # keeps the old mean-one shape as one candidate and adds a causal
        # level+shape candidate:
        #
        #   level_factor(block b) = mean(parent RLS forecast in b)
        #                           / mean(parent actual in b-1)
        #
        # The factor uses only information available at the forecast origin.
        # The leaf then chooses shape_only vs level_shape from PRIOR block
        # errors, together with store vs section parent.  Nothing is selected
        # from the target block itself.
        parent = (
            parent_forecasts.with_columns(
                pl.col("ds").cast(pl.Date),
                block_expr(),
            )
            .with_columns(
                pl.when(pl.col("period_type") == "forecast_only")
                .then(pl.lit(-1))
                .otherwise(pl.col("_block"))
                .cast(pl.Int32)
                .alias("_shape_group")
            )
            if parent_forecasts is not None and parent_forecasts.height
            else pl.DataFrame()
        )
        parent_effects = pl.DataFrame()
        if parent.height:
            factor_lo, factor_hi = tuple(
                float(x)
                for x in getattr(
                    settings,
                    "FAST_LEAF_DRIVER_FACTOR_CLIP",
                    (0.50, 2.00),
                )
            )
            max_level_step = float(
                getattr(settings, "LEAF_REGIME_TREND_MAX_STEP_RATIO", 2.0)
            )
            if max_level_step < 1.0:
                raise ValueError("LEAF_REGIME_TREND_MAX_STEP_RATIO must be >= 1")

            # Causal parent baseline for block b = actual mean of block b-1.
            # Parent nodes are already dense, so this is a true calendar-day
            # level and is directly compatible with the leaf's daily SES level.
            prev_parent_level = (
                parent.filter(pl.col("period_type") != "forecast_only")
                .group_by(["unique_id", "_block"])
                .agg(
                    pl.col("y").fill_null(0.0).clip(lower_bound=0.0).mean().alias("_prev_y"),
                    pl.col("value").fill_null(0.0).clip(lower_bound=0.0).mean().alias("_prev_v"),
                )
                .with_columns((pl.col("_block") + 1).cast(pl.Int32).alias("_block"))
            )
            parent_block_mean = (
                parent.filter(pl.col("_block") >= 1)
                .group_by(["unique_id", "_block"])
                .agg(
                    pl.col("yhat").fill_null(0.0).clip(lower_bound=0.0).mean().alias("_mean_py_block"),
                    pl.col("valuehat").fill_null(0.0).clip(lower_bound=0.0).mean().alias("_mean_pv_block"),
                )
                .join(prev_parent_level, on=["unique_id", "_block"], how="left")
                .with_columns(
                    pl.when(pl.col("_prev_y") > 1e-12)
                    .then(pl.col("_mean_py_block") / pl.col("_prev_y"))
                    .otherwise(1.0)
                    .clip(1.0 / max_level_step, max_level_step)
                    .alias("_level_factor_y"),
                    pl.when(pl.col("_prev_v") > 1e-12)
                    .then(pl.col("_mean_pv_block") / pl.col("_prev_v"))
                    .otherwise(1.0)
                    .clip(1.0 / max_level_step, max_level_step)
                    .alias("_level_factor_v"),
                )
                .select(
                    "unique_id", "_block",
                    "_level_factor_y", "_level_factor_v",
                )
            )

            parent_effects = (
                parent.filter(pl.col("_block") >= 1)
                .join(parent_block_mean, on=["unique_id", "_block"], how="left")
                .with_columns(
                    pl.col("_level_factor_y").fill_null(1.0),
                    pl.col("_level_factor_v").fill_null(1.0),
                    pl.col("yhat").fill_null(0.0).clip(lower_bound=0.0).alias("_py"),
                    pl.col("valuehat").fill_null(0.0).clip(lower_bound=0.0).alias("_pv"),
                )
                .with_columns(
                    pl.col("_py").mean().over(["unique_id", "_shape_group"]).alias("_mean_py"),
                    pl.col("_pv").mean().over(["unique_id", "_shape_group"]).alias("_mean_pv"),
                )
                .with_columns(
                    pl.when(pl.col("_mean_py") > 1e-12)
                    .then(pl.col("_py") / pl.col("_mean_py"))
                    .otherwise(1.0)
                    .alias("_ratio_y"),
                    pl.when(pl.col("_mean_pv") > 1e-12)
                    .then(pl.col("_pv") / pl.col("_mean_pv"))
                    .otherwise(1.0)
                    .alias("_ratio_v"),
                )
                .with_columns(
                    (pl.col("_ratio_y") - 1.0).alias("_dev_y"),
                    (pl.col("_ratio_v") - 1.0).alias("_dev_v"),
                )
                .with_columns(
                    pl.col("_dev_y").max().over(["unique_id", "_shape_group"]).alias("_max_dy"),
                    pl.col("_dev_y").min().over(["unique_id", "_shape_group"]).alias("_min_dy"),
                    pl.col("_dev_v").max().over(["unique_id", "_shape_group"]).alias("_max_dv"),
                    pl.col("_dev_v").min().over(["unique_id", "_shape_group"]).alias("_min_dv"),
                )
                # Keep the old safe shape normalization exactly: one scalar
                # shrinks all daily deviations, so shape mean remains 1.
                .with_columns(
                    pl.min_horizontal(
                        pl.lit(1.0),
                        pl.when(pl.col("_max_dy") > 0)
                        .then((factor_hi - 1.0) / pl.col("_max_dy"))
                        .otherwise(1.0),
                        pl.when(pl.col("_min_dy") < 0)
                        .then((1.0 - factor_lo) / (-pl.col("_min_dy")))
                        .otherwise(1.0),
                    ).alias("_shape_scale_y"),
                    pl.min_horizontal(
                        pl.lit(1.0),
                        pl.when(pl.col("_max_dv") > 0)
                        .then((factor_hi - 1.0) / pl.col("_max_dv"))
                        .otherwise(1.0),
                        pl.when(pl.col("_min_dv") < 0)
                        .then((1.0 - factor_lo) / (-pl.col("_min_dv")))
                        .otherwise(1.0),
                    ).alias("_shape_scale_v"),
                )
                .with_columns(
                    (1.0 + pl.col("_shape_scale_y") * pl.col("_dev_y")).alias("_shape_factor_y"),
                    (1.0 + pl.col("_shape_scale_v") * pl.col("_dev_v")).alias("_shape_factor_v"),
                )
                .with_columns(
                    pl.col("_shape_factor_y").log().alias("_effect_y_shape"),
                    pl.col("_shape_factor_v").log().alias("_effect_v_shape"),
                )
                .select(
                    "unique_id", "ds", "_block",
                    "_effect_y_shape", "_effect_v_shape",
                    "_level_factor_y", "_level_factor_v",
                )
            )

            # Forecast-only: if the future parent path is perfectly flat,
            # reuse the prior OOS SHAPE but keep the CURRENT causal level
            # factor from the parent RLS forecast.
            shape_source_end = test_end
            shape_source_start = shape_source_end - dt.timedelta(days=block_days - 1)
            shape_shift_days = int((forecast_start - shape_source_start).days)
            prev_shape = (
                parent_effects.filter(
                    (pl.col("ds") >= pl.lit(shape_source_start))
                    & (pl.col("ds") <= pl.lit(shape_source_end))
                )
                .with_columns((pl.col("ds") + pl.duration(days=shape_shift_days)).alias("ds"))
                .select(
                    "unique_id", "ds",
                    pl.col("_effect_y_shape").alias("_prev_effect_y_shape"),
                    pl.col("_effect_v_shape").alias("_prev_effect_v_shape"),
                )
            )
            shape_stats = (
                parent_effects.filter(
                    (pl.col("ds") >= pl.lit(forecast_start))
                    & (pl.col("ds") <= pl.lit(forecast_end))
                )
                .group_by("unique_id")
                .agg(
                    pl.col("_effect_y_shape").std().fill_null(0.0).alias("_fc_std_y"),
                    pl.col("_effect_v_shape").std().fill_null(0.0).alias("_fc_std_v"),
                )
            )
            parent_effects = (
                parent_effects.join(shape_stats, on="unique_id", how="left")
                .join(prev_shape, on=["unique_id", "ds"], how="left")
                .with_columns(
                    pl.when(
                        (pl.col("ds") >= pl.lit(forecast_start))
                        & (pl.col("ds") <= pl.lit(forecast_end))
                        & (pl.col("_fc_std_y").fill_null(0.0) < 1e-10)
                        & pl.col("_prev_effect_y_shape").is_not_null()
                    )
                    .then(pl.col("_prev_effect_y_shape"))
                    .otherwise(pl.col("_effect_y_shape"))
                    .alias("_effect_y_shape"),
                    pl.when(
                        (pl.col("ds") >= pl.lit(forecast_start))
                        & (pl.col("ds") <= pl.lit(forecast_end))
                        & (pl.col("_fc_std_v").fill_null(0.0) < 1e-10)
                        & pl.col("_prev_effect_v_shape").is_not_null()
                    )
                    .then(pl.col("_prev_effect_v_shape"))
                    .otherwise(pl.col("_effect_v_shape"))
                    .alias("_effect_v_shape"),
                )
                .with_columns(
                    (pl.col("_effect_y_shape") + pl.col("_level_factor_y").log()).alias("_effect_y_level"),
                    (pl.col("_effect_v_shape") + pl.col("_level_factor_v").log()).alias("_effect_v_level"),
                )
                .select(
                    "unique_id", "ds", "_block",
                    "_effect_y_shape", "_effect_v_shape",
                    "_effect_y_level", "_effect_v_level",
                    "_level_factor_y", "_level_factor_v",
                )
            )

        # Explicit parent availability. A missing store RLS must not be
        # represented as a silent factor=1 candidate.
        require_parent = bool(
            getattr(settings, "LEAF_REQUIRE_PARENT_DRIVERS", True)
        )
        parent_uid_set = (
            parent.select("unique_id").unique()
            if parent.height
            else pl.DataFrame(schema={"unique_id": pl.Utf8})
        )
        section_parent_available = (
            bool(
                parent_uid_set.filter(
                    pl.col("unique_id") == str(section_id)
                ).height
            )
            if parent_uid_set.height
            else False
        )
        store_parent_ids = (
            parent_uid_set.filter(
                pl.col("unique_id")
                .str.count_matches(r"\|\|", literal=False)
                == 1
            )
            .rename({"unique_id": "_store_uid"})
            .with_columns(pl.lit(True).alias("_store_parent_available"))
            if parent_uid_set.height
            else pl.DataFrame(
                schema={
                    "_store_uid": pl.Utf8,
                    "_store_parent_available": pl.Boolean,
                }
            )
        )
        leaf_parent_availability = (
            ids.select(uid_col)
            .with_columns(
                pl.col(uid_col)
                .str.replace(r"\|\|S:.*$", "")
                .alias("_store_uid")
            )
            .join(store_parent_ids, on="_store_uid", how="left")
            .with_columns(
                pl.col("_store_parent_available")
                .fill_null(False),
                pl.lit(section_parent_available)
                .alias("_section_parent_available"),
            )
            .select(
                uid_col,
                "_store_parent_available",
                "_section_parent_available",
            )
        )
        if require_parent:
            missing_parent = leaf_parent_availability.filter(
                ~pl.col("_store_parent_available")
                & ~pl.col("_section_parent_available")
            )
            if missing_parent.height:
                raise RuntimeError(
                    "Leaf RLS parent invariant violated: leaves without store "
                    f"or section parent; examples={missing_parent.head(5).to_dicts()}"
                )

        # Observed leaf rows through OOS only.  No post-OOS leakage.
        actual_parts = [
            train_obs.select(
                [c for c in (uid_col, "ds", "y", "value") if c in train_obs.columns]
            )
        ]
        if oos_obs.height:
            actual_parts.append(
                oos_obs.select(
                    [c for c in (uid_col, "ds", "y", "value") if c in oos_obs.columns]
                )
            )
        actual_obs = (
            pl.concat(actual_parts, how="diagonal_relaxed")
            .with_columns(block_expr())
        )

        # Dense only for OOS + forecast-only (56 days), never full leaf history.
        uid_arr = ids[uid_col].to_numpy()
        oos_dates = pl.date_range(test_start, test_end, interval="1d", eager=True).to_numpy()
        fc_dates = pl.date_range(forecast_start, forecast_end, interval="1d", eager=True).to_numpy()

        oos_grid = pl.DataFrame(
            {uid_col: np.repeat(uid_arr, len(oos_dates)), "ds": np.tile(oos_dates, len(uid_arr))}
        ).with_columns(pl.col("ds").cast(pl.Date))
        if meta_cols:
            oos_grid = oos_grid.join(ids, on=uid_col, how="left")
        if oos_obs.height:
            oos_grid = oos_grid.join(
                oos_obs.select(uid_col, "ds", "y", "value"),
                on=[uid_col, "ds"], how="left",
            )
        oos_grid = oos_grid.with_columns(
            (pl.col("y").fill_null(0.0) if "y" in oos_grid.columns else pl.lit(0.0).alias("y")),
            (pl.col("value").fill_null(0.0) if "value" in oos_grid.columns else pl.lit(0.0).alias("value")),
            pl.lit("out_sample").alias("period_type"),
        )


        forecast_grid = pl.DataFrame(
            {
                uid_col: np.repeat(uid_arr, len(fc_dates)),
                "ds": np.tile(fc_dates, len(uid_arr)),
                "y": np.zeros(len(uid_arr) * len(fc_dates)),
                "value": np.zeros(len(uid_arr) * len(fc_dates)),
            }
        ).with_columns(
            pl.col("ds").cast(pl.Date),
            pl.lit("forecast_only").alias("period_type"),
        )
        if meta_cols:
            forecast_grid = forecast_grid.join(ids, on=uid_col, how="left")

        historical = (
            train_obs.join(leaf_bounds, on=uid_col, how="left")
            .filter(pl.col("ds") > pl.col("_leaf_warmup_end"))
            .select(
                [c for c in (uid_col, "ds", "y", "value") + tuple(meta_cols)
                 if c in train_obs.columns]
            )
            .with_columns(pl.lit("in_sample").alias("period_type"))
        )

        row_parts = [historical, oos_grid, forecast_grid]
        rows = pl.concat(
            row_parts, how="diagonal_relaxed"
        ).with_columns(
            block_expr(),
            pl.col(uid_col).str.replace(r"\|\|S:.*$", "").alias("_store_uid"),
        )

        # Join store and section parent effects.  Keep both candidates:
        # mean-one shape and causal level+shape.
        if parent_effects.height:
            store_eff = parent_effects.filter(
                pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 1
            ).select(
                pl.col("unique_id").alias("_store_uid"), "ds",
                pl.col("_effect_y_shape").alias("_store_ey_shape"),
                pl.col("_effect_v_shape").alias("_store_ev_shape"),
                pl.col("_effect_y_level").alias("_store_ey_level"),
                pl.col("_effect_v_level").alias("_store_ev_level"),
                pl.col("_level_factor_y").alias("_store_level_factor_y"),
                pl.col("_level_factor_v").alias("_store_level_factor_v"),
            )
            sec_eff = parent_effects.filter(
                pl.col("unique_id") == str(section_id)
            ).select(
                "ds",
                pl.col("_effect_y_shape").alias("_sec_ey_shape"),
                pl.col("_effect_v_shape").alias("_sec_ev_shape"),
                pl.col("_effect_y_level").alias("_sec_ey_level"),
                pl.col("_effect_v_level").alias("_sec_ev_level"),
                pl.col("_level_factor_y").alias("_sec_level_factor_y"),
                pl.col("_level_factor_v").alias("_sec_level_factor_v"),
            )
            rows = rows.join(store_eff, on=["_store_uid", "ds"], how="left")
            rows = rows.join(sec_eff, on="ds", how="left")

        effect_cols = (
            "_store_ey_shape", "_store_ev_shape", "_store_ey_level", "_store_ev_level",
            "_sec_ey_shape", "_sec_ev_shape", "_sec_ey_level", "_sec_ev_level",
        )
        level_cols = (
            "_store_level_factor_y", "_store_level_factor_v",
            "_sec_level_factor_y", "_sec_level_factor_v",
        )
        for c in effect_cols:
            if c not in rows.columns:
                rows = rows.with_columns(pl.lit(0.0).alias(c))
            else:
                rows = rows.with_columns(pl.col(c).fill_nan(0.0).fill_null(0.0))
        for c in level_cols:
            if c not in rows.columns:
                rows = rows.with_columns(pl.lit(1.0).alias(c))
            else:
                rows = rows.with_columns(pl.col(c).fill_nan(1.0).fill_null(1.0))

        # Actual observations receive the same causal parent candidates used
        # by their forecast block; candidate scoring happens only after close.
        actual_resid = (
            actual_obs.join(leaf_bounds, on=uid_col, how="left")
            .filter(pl.col("ds") > pl.col("_leaf_warmup_end"))
            .with_columns(
                pl.col(uid_col).str.replace(r"\|\|S:.*$", "").alias("_store_uid")
            )
        )
        if parent_effects.height:
            actual_resid = actual_resid.join(
                store_eff, on=["_store_uid", "ds"], how="left"
            ).join(sec_eff, on="ds", how="left")
        for c in effect_cols:
            if c not in actual_resid.columns:
                actual_resid = actual_resid.with_columns(pl.lit(0.0).alias(c))
            else:
                actual_resid = actual_resid.with_columns(pl.col(c).fill_nan(0.0).fill_null(0.0))
        for c in level_cols:
            if c not in actual_resid.columns:
                actual_resid = actual_resid.with_columns(pl.lit(1.0).alias(c))
            else:
                actual_resid = actual_resid.with_columns(pl.col(c).fill_nan(1.0).fill_null(1.0))

        max_block = int((forecast_end - train_start).days // block_days)
        actual_last_block = int(
            (test_end - train_start).days // block_days
        )

        # ── TWO-STAGE SKU+tienda engine ───────────────────────────────────
        # Stage 1 (PURE SES): choose alpha using SES-only block forecasts scored on positive-sale days.
        # Stage 2 (DRIVERS): with the SES level already fixed, choose whether
        # store or section RLS shape gives the lower prior cumulative wMAPE.
        #
        # This separation is deliberate: parent/RLS performance can NEVER
        # influence which alpha defines the structural leaf level.
        t_candidates = time.perf_counter()

        alpha_params = pl.DataFrame(
            [{"_alpha": float(a)} for a in alpha_candidates]
        )
        parent_names = tuple(
            str(x)
            for x in getattr(
                settings,
                "LEAF_PARENT_CANDIDATES",
                ("store", "section"),
            )
            if str(x) in {"store", "section"}
        )
        if set(parent_names) != {"store", "section"}:
            raise ValueError(
                "LEAF_PARENT_CANDIDATES must contain exactly store and section; "
                f"got {parent_names}"
            )
        parent_strength = float(
            getattr(settings, "LEAF_PARENT_DRIVER_STRENGTH", 1.0)
        )
        if abs(parent_strength - 1.0) > 1e-12:
            raise ValueError(
                "v11 requires LEAF_PARENT_DRIVER_STRENGTH=1.0; "
                f"got {parent_strength}"
            )
        driver_modes = tuple(
            str(x)
            for x in getattr(
                settings,
                "LEAF_PARENT_DRIVER_MODES",
                ("shape_only", "level_shape"),
            )
            if str(x) in {"shape_only", "level_shape"}
        )
        if not driver_modes:
            raise ValueError(
                "LEAF_PARENT_DRIVER_MODES must contain at least one valid mode; "
                f"got {driver_modes}"
            )
        if not bool(getattr(settings, "LEAF_PARENT_LEVEL_SHAPE_SELECTABLE", True)):
            driver_modes = tuple(m for m in driver_modes if m == "shape_only")
            if not driver_modes:
                driver_modes = ("shape_only",)
        default_driver_mode = str(
            getattr(settings, "LEAF_PARENT_DEFAULT_DRIVER_MODE", "shape_only")
        )
        if default_driver_mode not in driver_modes:
            raise ValueError(
                "LEAF_PARENT_DEFAULT_DRIVER_MODE must be in LEAF_PARENT_DRIVER_MODES"
            )
        parent_params = pl.DataFrame(
            [
                {
                    "_parent": parent_name,
                    "_driver_mode": driver_mode,
                    "_strength": 1.0,
                }
                for parent_name in parent_names
                for driver_mode in driver_modes
            ]
        )

        # Pure SES state: one level trajectory per leaf × alpha.
        alpha_state = (
            ids.select(uid_col)
            .join(leaf_bounds, on=uid_col, how="left")
            .join(alpha_params, how="cross")
            .join(state0, on=uid_col, how="left")
            .with_columns(
                pl.lit(0.0).alias("_score_ae_b0_y"),
                pl.lit(0.0).alias("_score_den_b0_y"),
                pl.lit(0.0).alias("_score_se_b0_y"),
                pl.lit(0.0).alias("_score_ae_b1_y"),
                pl.lit(0.0).alias("_score_den_b1_y"),
                pl.lit(0.0).alias("_score_se_b1_y"),
                pl.lit(0.0).alias("_score_ae_b2_y"),
                pl.lit(0.0).alias("_score_den_b2_y"),
                pl.lit(0.0).alias("_score_se_b2_y"),
                pl.lit(0.0).alias("_score_ae_b3_y"),
                pl.lit(0.0).alias("_score_den_b3_y"),
                pl.lit(0.0).alias("_score_se_b3_y"),
                pl.lit(0.0).alias("_score_ae_b0_v"),
                pl.lit(0.0).alias("_score_den_b0_v"),
                pl.lit(0.0).alias("_score_se_b0_v"),
                pl.lit(0.0).alias("_score_ae_b1_v"),
                pl.lit(0.0).alias("_score_den_b1_v"),
                pl.lit(0.0).alias("_score_se_b1_v"),
                pl.lit(0.0).alias("_score_ae_b2_v"),
                pl.lit(0.0).alias("_score_den_b2_v"),
                pl.lit(0.0).alias("_score_se_b2_v"),
                pl.lit(0.0).alias("_score_ae_b3_v"),
                pl.lit(0.0).alias("_score_den_b3_v"),
                pl.lit(0.0).alias("_score_se_b3_v"),
            )
        )

        # Parent scores contain NO SES state and NO alpha. They evaluate only
        # the shape added on top of the alpha/level selected by pure SES.
        parent_state = (
            leaf_parent_availability
            .join(parent_params, how="cross")
            .filter(
                (
                    (pl.col("_parent") == "store")
                    & pl.col("_store_parent_available")
                )
                | (
                    (pl.col("_parent") == "section")
                    & pl.col("_section_parent_available")
                )
            )
            .select(uid_col, "_parent", "_driver_mode", "_strength")
            .with_columns(
                pl.lit(0.0).alias("_cae_parent_y"),
                pl.lit(0.0).alias("_cden_parent_y"),
                pl.lit(0.0).alias("_cae_parent_v"),
                pl.lit(0.0).alias("_cden_parent_v"),
            )
        )
        # Parent selection is the simple expanding prior wMAPE requested by
        # the client. No decay, no "none", no partial strengths.
        driver_score_decay = 1.0

        # Partition actual rows once. Missing calendar dates are zeros and are
        # handled analytically below; the full history is never densified.
        obs_by_block: dict[int, pl.DataFrame] = {}
        if actual_resid.height:
            for key, frame in actual_resid.partition_by(
                "_block", as_dict=True, maintain_order=True
            ).items():
                k = key[0] if isinstance(key, tuple) else key
                obs_by_block[int(k)] = frame
        if memory_safe:
            del actual_resid

        # Robust reference window is the same finite history used by regime
        # detection. No separate "stability" subsystem exists in v11.
        regime_history_blocks = int(
            getattr(settings, "LEAF_REGIME_HISTORY_BLOCKS", 8)
        )
        if regime_history_blocks < 1:
            raise ValueError("LEAF_REGIME_HISTORY_BLOCKS must be >= 1")
        stability_days = regime_history_blocks * block_days
        stability_eps = float(
            getattr(settings, "LEAF_REGIME_EPS", 1e-9)
        )
        stability_blocks = regime_history_blocks

        stability_block_sums = (
            actual_obs.with_columns(
                block_expr(),
                (
                    (pl.col("ds") - pl.lit(train_start)).dt.total_days()
                    % block_days
                )
                .cast(pl.Int32)
                .alias("_day_in_block"),
            )
            .group_by([uid_col, "_block"])
            .agg(
                pl.when(pl.col("y").is_finite())
                .then(pl.col("y").clip(lower_bound=0.0))
                .otherwise(0.0)
                .sum()
                .alias("_ref_sum_y"),
                pl.when(pl.col("value").is_finite())
                .then(pl.col("value").clip(lower_bound=0.0))
                .otherwise(0.0)
                .sum()
                .alias("_ref_sum_v"),
                (
                    (pl.col("y").fill_null(0.0) > 0)
                    & pl.col("y").is_finite()
                )
                .sum()
                .cast(pl.Int32)
                .alias("_ref_nz_y"),
                (
                    (pl.col("value").fill_null(0.0) > 0)
                    & pl.col("value").is_finite()
                )
                .sum()
                .cast(pl.Int32)
                .alias("_ref_nz_v"),
                pl.when(
                    (pl.col("_day_in_block") >= block_days - 14)
                    & pl.col("y").is_finite()
                )
                .then(pl.col("y").clip(lower_bound=0.0))
                .otherwise(0.0)
                .sum()
                .alias("_last14_sum_y"),
                pl.when(
                    (pl.col("_day_in_block") >= block_days - 14)
                    & pl.col("value").is_finite()
                )
                .then(pl.col("value").clip(lower_bound=0.0))
                .otherwise(0.0)
                .sum()
                .alias("_last14_sum_v"),
            )
        )
        regime_min_history_blocks = int(
            getattr(settings, "LEAF_REGIME_MIN_HISTORY_BLOCKS", 4)
        )

        if memory_safe:
            # 1d creates roughly 28x more origin rows than 28d. The historical
            # implementation materialized 5 shifted copies twice (structural
            # history + reference sums), which can exceed RAM before the model
            # reaches OOS. Keep one compact block-stat partition and assemble
            # the six exact prior blocks only for the current origin.
            stability_by_block: dict[int, pl.DataFrame] = {}
            if stability_block_sums.height:
                for key, frame in stability_block_sums.partition_by(
                    "_block", as_dict=True, maintain_order=True
                ).items():
                    k = key[0] if isinstance(key, tuple) else key
                    stability_by_block[int(k)] = frame.drop("_block")
            del stability_block_sums
            del actual_obs

            def _recent_ref_base_memory_safe(origin_block: int) -> pl.DataFrame:
                base = ids.select(uid_col).join(leaf_bounds, on=uid_col, how="left")
                # B0 = immediately closed block. B1..B5 are exact calendar
                # update blocks before B0; an absent block stays null/zero,
                # matching the previous shifted exact-block joins.
                b0 = stability_by_block.get(origin_block - 1)
                if b0 is not None:
                    base = base.join(
                        b0.select(
                            uid_col,
                            pl.col("_ref_sum_y").alias("_recent28_sum_y"),
                            pl.col("_ref_sum_v").alias("_recent28_sum_v"),
                            pl.col("_ref_nz_y").alias("_recent28_nz_y"),
                            pl.col("_ref_nz_v").alias("_recent28_nz_v"),
                            pl.col("_last14_sum_y").alias("_recent14_sum_y"),
                            pl.col("_last14_sum_v").alias("_recent14_sum_v"),
                        ),
                        on=uid_col,
                        how="left",
                    )
                else:
                    base = base.with_columns(
                        pl.lit(None, dtype=pl.Float64).alias("_recent28_sum_y"),
                        pl.lit(None, dtype=pl.Float64).alias("_recent28_sum_v"),
                        pl.lit(None, dtype=pl.Int32).alias("_recent28_nz_y"),
                        pl.lit(None, dtype=pl.Int32).alias("_recent28_nz_v"),
                        pl.lit(None, dtype=pl.Float64).alias("_recent14_sum_y"),
                        pl.lit(None, dtype=pl.Float64).alias("_recent14_sum_v"),
                    )

                for hist_i in range(1, regime_history_blocks + 1):
                    src = stability_by_block.get(origin_block - 1 - hist_i)
                    cy, cv = f"_hist_b{hist_i}_y", f"_hist_b{hist_i}_v"
                    if src is not None:
                        base = base.join(
                            src.select(
                                uid_col,
                                (pl.col("_ref_sum_y") / pl.lit(float(block_days))).alias(cy),
                                (pl.col("_ref_sum_v") / pl.lit(float(block_days))).alias(cv),
                            ),
                            on=uid_col,
                            how="left",
                        )
                    else:
                        base = base.with_columns(
                            pl.lit(None, dtype=pl.Float64).alias(cy),
                            pl.lit(None, dtype=pl.Float64).alias(cv),
                        )

                hist_y = [pl.col(f"_hist_b{i}_y") for i in range(1, regime_history_blocks + 1)]
                hist_v = [pl.col(f"_hist_b{i}_v") for i in range(1, regime_history_blocks + 1)]
                ref_hist_n = max(0, min(stability_blocks - 1, regime_history_blocks))
                ref_y_terms = [pl.col("_recent28_sum_y").fill_null(0.0)] + [
                    pl.col(f"_hist_b{i}_y").fill_null(0.0) * pl.lit(float(block_days))
                    for i in range(1, ref_hist_n + 1)
                ]
                ref_v_terms = [pl.col("_recent28_sum_v").fill_null(0.0)] + [
                    pl.col(f"_hist_b{i}_v").fill_null(0.0) * pl.lit(float(block_days))
                    for i in range(1, ref_hist_n + 1)
                ]
                base = base.with_columns(
                    (pl.col("_recent28_sum_y") / pl.lit(float(block_days))).alias("_b0_y"),
                    (pl.col("_recent28_sum_v") / pl.lit(float(block_days))).alias("_b0_v"),
                    *[
                        pl.col(f"_hist_b{i}_y").alias(f"_b{i}_y")
                        for i in range(1, min(3, regime_history_blocks) + 1)
                    ],
                    *[
                        pl.col(f"_hist_b{i}_v").alias(f"_b{i}_v")
                        for i in range(1, min(3, regime_history_blocks) + 1)
                    ],
                    pl.concat_list(hist_y).list.median().alias("_structural_y"),
                    pl.concat_list(hist_v).list.median().alias("_structural_v"),
                    pl.sum_horizontal(*[x.is_not_null().cast(pl.Int32) for x in hist_y])
                    .alias("_structural_history_blocks_y"),
                    pl.sum_horizontal(*[x.is_not_null().cast(pl.Int32) for x in hist_v])
                    .alias("_structural_history_blocks_v"),
                    pl.sum_horizontal(*ref_y_terms).alias("_ref_sum_y"),
                    pl.sum_horizontal(*ref_v_terms).alias("_ref_sum_v"),
                )
                # The downstream regime code expects B1/B2/B3 even when the
                # configured history is shorter (production is >=4 today).
                missing = []
                for i in (1, 2, 3):
                    for suffix in ("y", "v"):
                        c = f"_b{i}_{suffix}"
                        if c not in base.columns:
                            missing.append(pl.lit(None, dtype=pl.Float64).alias(c))
                if missing:
                    base = base.with_columns(*missing)
                return base

            # Global shifted frames are deliberately absent in memory-safe mode.
            recent_block_stats = pl.DataFrame()
            regime_block_stats = pl.DataFrame()
            structural_history_stats = pl.DataFrame()
            stability_ref_sums = pl.DataFrame()
        else:
            recent_block_stats = stability_block_sums.with_columns(
                (pl.col("_block") + 1).cast(pl.Int32).alias("_block")
            ).select(
                uid_col,
                "_block",
                pl.col("_ref_sum_y").alias("_recent28_sum_y"),
                pl.col("_ref_sum_v").alias("_recent28_sum_v"),
                pl.col("_ref_nz_y").alias("_recent28_nz_y"),
                pl.col("_ref_nz_v").alias("_recent28_nz_v"),
                pl.col("_last14_sum_y").alias("_recent14_sum_y"),
                pl.col("_last14_sum_v").alias("_recent14_sum_v"),
            )
            # Regime blocks at each forecast origin:
            # B0 = immediately closed block, B1/B2/B3 = previous blocks.
            regime_block_stats = recent_block_stats.with_columns(
                (pl.col("_recent28_sum_y") / pl.lit(float(block_days))).alias("_b0_y"),
                (pl.col("_recent28_sum_v") / pl.lit(float(block_days))).alias("_b0_v"),
            )
            for _lag, _name in ((2, "b1"), (3, "b2"), (4, "b3")):
                _lag_stats = stability_block_sums.with_columns(
                    (pl.col("_block") + _lag).cast(pl.Int32).alias("_block")
                ).select(
                    uid_col,
                    "_block",
                    (pl.col("_ref_sum_y") / pl.lit(float(block_days))).alias(f"_{_name}_y"),
                    (pl.col("_ref_sum_v") / pl.lit(float(block_days))).alias(f"_{_name}_v"),
                )
                regime_block_stats = regime_block_stats.join(
                    _lag_stats, on=[uid_col, "_block"], how="left"
                )
            structural_shifted: list[pl.DataFrame] = []
            # lag=2 is B1 at the next forecast origin; B0 (lag=1) is excluded.
            for _lag in range(2, 2 + regime_history_blocks):
                structural_shifted.append(
                    stability_block_sums.with_columns(
                        (pl.col("_block") + _lag).cast(pl.Int32).alias("_block")
                    ).select(
                        uid_col,
                        "_block",
                        (pl.col("_ref_sum_y") / pl.lit(float(block_days))).alias("_hist_y"),
                        (pl.col("_ref_sum_v") / pl.lit(float(block_days))).alias("_hist_v"),
                    )
                )
            structural_history_stats = (
                pl.concat(structural_shifted, how="vertical_relaxed")
                .group_by([uid_col, "_block"])
                .agg(
                    pl.col("_hist_y").median().alias("_structural_y"),
                    pl.col("_hist_v").median().alias("_structural_v"),
                    pl.col("_hist_y").count().cast(pl.Int32).alias("_structural_history_blocks_y"),
                    pl.col("_hist_v").count().cast(pl.Int32).alias("_structural_history_blocks_v"),
                )
                if structural_shifted else pl.DataFrame()
            )
            stability_shifted: list[pl.DataFrame] = []
            for lag_block in range(1, stability_blocks + 1):
                stability_shifted.append(
                    stability_block_sums.with_columns(
                        (pl.col("_block") + lag_block).cast(pl.Int32).alias("_block")
                    )
                )
            stability_ref_sums = (
                pl.concat(stability_shifted, how="vertical_relaxed")
                .group_by([uid_col, "_block"])
                .agg(
                    pl.col("_ref_sum_y").sum().alias("_ref_sum_y"),
                    pl.col("_ref_sum_v").sum().alias("_ref_sum_v"),
                )
                if stability_shifted else pl.DataFrame()
            )

        alpha_score_weights = tuple(
            float(x)
            for x in getattr(
                settings,
                "LEAF_ALPHA_SCORE_WEIGHTS",
                (0.60, 0.30, 0.10, 0.00),
            )
        )
        if (
            len(alpha_score_weights) != 4
            or any(x < 0 for x in alpha_score_weights)
            or sum(alpha_score_weights) <= 0
        ):
            raise ValueError(
                "LEAF_ALPHA_SCORE_WEIGHTS must contain four non-negative "
                f"weights with positive sum; got {alpha_score_weights}"
            )
        _w_sum = float(sum(alpha_score_weights))
        alpha_score_weights = tuple(x / _w_sum for x in alpha_score_weights)
        score_min_den = float(
            getattr(settings, "LEAF_ALPHA_SCORE_MIN_DEN", 1e-9)
        )
        alpha_bias_weight = float(
            getattr(settings, "LEAF_ALPHA_BIAS_WEIGHT", 0.20)
        )

        def _weighted_score_component(
            component: str,
            suffix: str,
        ) -> pl.Expr:
            expr = pl.lit(0.0)
            for _i, _weight in enumerate(alpha_score_weights):
                expr = expr + (
                    pl.col(f"_score_{component}_b{_i}_{suffix}")
                    * pl.lit(float(_weight))
                )
            return expr

        def _score_history_block_count(suffix: str) -> pl.Expr:
            expr = pl.lit(0).cast(pl.Int32)
            for _i, _weight in enumerate(alpha_score_weights):
                if _weight <= 0:
                    continue
                expr = expr + (
                    pl.col(f"_score_den_b{_i}_{suffix}") > score_min_den
                ).cast(pl.Int32)
            return expr
        regime_trend_up_ratio = float(
            getattr(settings, "LEAF_REGIME_TREND_UP_RATIO", 1.05)
        )
        regime_trend_down_ratio = float(
            getattr(settings, "LEAF_REGIME_TREND_DOWN_RATIO", 0.95)
        )
        regime_trend_max_step_ratio = float(
            getattr(settings, "LEAF_REGIME_TREND_MAX_STEP_RATIO", 2.0)
        )
        regime_transient_ratio = float(
            getattr(settings, "LEAF_REGIME_TRANSIENT_RATIO", 1.75)
        )
        reset_transient = bool(
            getattr(settings, "LEAF_RESET_TRANSIENT", True)
        )
        transient_down_alpha = float(
            getattr(settings, "LEAF_TRANSIENT_DOWN_ALPHA", 0.20)
        )
        if transient_down_alpha not in alpha_candidates:
            raise ValueError(
                "LEAF_TRANSIENT_DOWN_ALPHA must be present in "
                f"LEAF_SES_ALPHA_CANDIDATES; got {transient_down_alpha}"
            )

        # With 1d/7d/14d there can be hundreds of model origins. Keeping one
        # wide chosen-state frame per origin in RAM is the main source of the
        # multi-cadence memory explosion. Spill each state to parquet and join
        # it back lazily after the sequential state update loop.
        chosen_frames: list[pl.DataFrame] = []
        chosen_cols: list[str] | None = None
        default_parent = str(
            getattr(settings, "LEAF_PARENT_DEFAULT", "store")
        )
        if default_parent not in {"store", "section"}:
            raise ValueError(
                "LEAF_PARENT_DEFAULT must be 'store' or 'section'"
            )

        for block_i in range(1, max_block + 1):
            block_start = train_start + dt.timedelta(days=block_i * block_days)
            block_end = block_start + dt.timedelta(days=block_days - 1)

            # Number of DAILY SES updates available after this leaf's own
            # 28-day warm-up. This is independent of parent/RLS.
            alpha_state = (
                alpha_state.with_columns(
                    pl.max_horizontal(
                        pl.col("_leaf_warmup_end") + pl.duration(days=1),
                        pl.lit(block_start),
                    ).alias("_update_start")
                )
                .with_columns(
                    pl.when(pl.col("_update_start") <= pl.lit(block_end))
                    .then(
                        (pl.lit(block_end) - pl.col("_update_start"))
                        .dt.total_days()
                        + 1
                    )
                    .otherwise(0)
                    .cast(pl.Int32)
                    .alias("_n_calendar")
                )
            )

            # Causal regime references for THIS model origin.
            # If calendar time advances through a PARTIAL observed tail, no
            # 28-day model update occurs. In that case keep using the most
            # recent complete actual block instead of treating the incomplete
            # calendar block as 28 zeros.
            regime_ref_block = min(block_i, actual_last_block + 1)
            recent_ref_base = (
                _recent_ref_base_memory_safe(regime_ref_block)
                if memory_safe
                else (
                    ids.select(uid_col)
                    .join(leaf_bounds, on=uid_col, how="left")
                    .join(
                        stability_ref_sums.filter(
                            pl.col("_block") == regime_ref_block
                        ).select(uid_col, "_ref_sum_y", "_ref_sum_v"),
                        on=uid_col,
                        how="left",
                    )
                    .join(
                        regime_block_stats.filter(
                            pl.col("_block") == regime_ref_block
                        ).drop("_block"),
                        on=uid_col,
                        how="left",
                    )
                    .join(
                        structural_history_stats.filter(
                            pl.col("_block") == regime_ref_block
                        ).drop("_block"),
                        on=uid_col,
                        how="left",
                    )
                )
            )
            recent_ref = (
                recent_ref_base.with_columns(
                    pl.col("_b1_y").is_not_null().alias("_b1_available_y"),
                    pl.col("_b2_y").is_not_null().alias("_b2_available_y"),
                    pl.col("_b3_y").is_not_null().alias("_b3_available_y"),
                    pl.col("_b1_v").is_not_null().alias("_b1_available_v"),
                    pl.col("_b2_v").is_not_null().alias("_b2_available_v"),
                    pl.col("_b3_v").is_not_null().alias("_b3_available_v"),
                )
                .with_columns(
                    (
                        pl.lit(block_start) - pl.col("_leaf_start")
                    )
                    .dt.total_days()
                    .cast(pl.Int32)
                    .alias("_leaf_age_days"),
                    pl.col("_ref_sum_y").fill_null(0.0),
                    pl.col("_ref_sum_v").fill_null(0.0),
                    pl.col("_recent28_sum_y").fill_null(0.0),
                    pl.col("_recent28_sum_v").fill_null(0.0),
                    pl.col("_recent28_nz_y").fill_null(0),
                    pl.col("_recent28_nz_v").fill_null(0),
                    pl.col("_recent14_sum_y").fill_null(0.0),
                    pl.col("_recent14_sum_v").fill_null(0.0),
                    pl.col("_b0_y").fill_null(0.0),
                    pl.col("_b0_v").fill_null(0.0),
                    pl.col("_b1_y").fill_null(0.0),
                    pl.col("_b1_v").fill_null(0.0),
                    pl.col("_b2_y").fill_null(0.0),
                    pl.col("_b2_v").fill_null(0.0),
                    pl.col("_b3_y").fill_null(0.0),
                    pl.col("_b3_v").fill_null(0.0),
                )
                .with_columns(
                    pl.when(pl.col("_leaf_age_days") <= 0)
                    .then(1)
                    .when(pl.col("_leaf_age_days") < stability_days)
                    .then(pl.col("_leaf_age_days"))
                    .otherwise(stability_days)
                    .cast(pl.Int32)
                    .alias("_ref_days")
                )
                .with_columns(
                    (
                        pl.col("_ref_sum_y")
                        / pl.col("_ref_days").cast(pl.Float64)
                    ).alias("_reference_y"),
                    (
                        pl.col("_ref_sum_v")
                        / pl.col("_ref_days").cast(pl.Float64)
                    ).alias("_reference_v"),
                    (
                        pl.col("_recent28_sum_y") / pl.lit(float(block_days))
                    ).alias("_recent28_y"),
                    (
                        pl.col("_recent28_sum_v") / pl.lit(float(block_days))
                    ).alias("_recent28_v"),
                    (
                        pl.col("_recent14_sum_y") / pl.lit(14.0)
                    ).alias("_recent14_y"),
                    (
                        pl.col("_recent14_sum_v") / pl.lit(14.0)
                    ).alias("_recent14_v"),
                    (
                        pl.col("_recent28_nz_y").cast(pl.Float64)
                        / pl.lit(float(block_days))
                    ).alias("_coverage_y"),
                    (
                        pl.col("_recent28_nz_v").cast(pl.Float64)
                        / pl.lit(float(block_days))
                    ).alias("_coverage_v"),
                )
                .with_columns(
                    pl.col("_structural_y")
                    .fill_null(pl.col("_reference_y"))
                    .alias("_structural_y"),
                    pl.col("_structural_v")
                    .fill_null(pl.col("_reference_v"))
                    .alias("_structural_v"),
                    pl.col("_structural_history_blocks_y")
                    .fill_null(0)
                    .cast(pl.Int32),
                    pl.col("_structural_history_blocks_v")
                    .fill_null(0)
                    .cast(pl.Int32),
                )
                .with_columns(
                    (
                        pl.col("_b1_available_y")
                        & pl.col("_b2_available_y")
                        & pl.col("_b3_available_y")
                        & (pl.col("_b3_y") > stability_eps)
                        & (pl.col("_b2_y") > pl.col("_b3_y") * regime_trend_up_ratio)
                        & (pl.col("_b1_y") > pl.col("_b2_y") * regime_trend_up_ratio)
                        & (pl.col("_b0_y") > pl.col("_b1_y") * regime_trend_up_ratio)
                        & (
                            pl.col("_b2_y") / pl.col("_b3_y").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                        & (
                            pl.col("_b1_y") / pl.col("_b2_y").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                        & (
                            pl.col("_b0_y") / pl.col("_b1_y").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                    ).alias("_trend_up_y"),
                    (
                        pl.col("_b1_available_y")
                        & pl.col("_b2_available_y")
                        & pl.col("_b3_available_y")
                        & (pl.col("_b1_y") > stability_eps)
                        & (pl.col("_b2_y") < pl.col("_b3_y") * regime_trend_down_ratio)
                        & (pl.col("_b1_y") < pl.col("_b2_y") * regime_trend_down_ratio)
                        & (pl.col("_b0_y") < pl.col("_b1_y") * regime_trend_down_ratio)
                        & (
                            pl.col("_b3_y") / pl.col("_b2_y").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                        & (
                            pl.col("_b2_y") / pl.col("_b1_y").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                        & (
                            pl.col("_b1_y") / pl.col("_b0_y").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                    ).alias("_trend_down_y"),
                    (
                        pl.col("_b1_available_v")
                        & pl.col("_b2_available_v")
                        & pl.col("_b3_available_v")
                        & (pl.col("_b3_v") > stability_eps)
                        & (pl.col("_b2_v") > pl.col("_b3_v") * regime_trend_up_ratio)
                        & (pl.col("_b1_v") > pl.col("_b2_v") * regime_trend_up_ratio)
                        & (pl.col("_b0_v") > pl.col("_b1_v") * regime_trend_up_ratio)
                        & (
                            pl.col("_b2_v") / pl.col("_b3_v").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                        & (
                            pl.col("_b1_v") / pl.col("_b2_v").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                        & (
                            pl.col("_b0_v") / pl.col("_b1_v").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                    ).alias("_trend_up_v"),
                    (
                        pl.col("_b1_available_v")
                        & pl.col("_b2_available_v")
                        & pl.col("_b3_available_v")
                        & (pl.col("_b1_v") > stability_eps)
                        & (pl.col("_b2_v") < pl.col("_b3_v") * regime_trend_down_ratio)
                        & (pl.col("_b1_v") < pl.col("_b2_v") * regime_trend_down_ratio)
                        & (pl.col("_b0_v") < pl.col("_b1_v") * regime_trend_down_ratio)
                        & (
                            pl.col("_b3_v") / pl.col("_b2_v").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                        & (
                            pl.col("_b2_v") / pl.col("_b1_v").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                        & (
                            pl.col("_b1_v") / pl.col("_b0_v").clip(lower_bound=stability_eps)
                            <= regime_trend_max_step_ratio
                        )
                    ).alias("_trend_down_v"),
                )
                .with_columns(
                    (
                        (pl.col("_structural_history_blocks_y") >= regime_min_history_blocks)
                        & (pl.col("_structural_y") > stability_eps)
                        & (pl.col("_b0_y") > pl.col("_structural_y") * regime_transient_ratio)
                        & ~pl.col("_trend_up_y")
                    ).alias("_transient_up_y"),
                    (
                        (pl.col("_structural_history_blocks_y") >= regime_min_history_blocks)
                        & (pl.col("_structural_y") > stability_eps)
                        & (pl.col("_b0_y") < pl.col("_structural_y") / regime_transient_ratio)
                        & ~pl.col("_trend_down_y")
                    ).alias("_transient_down_y"),
                    (
                        (pl.col("_structural_history_blocks_v") >= regime_min_history_blocks)
                        & (pl.col("_structural_v") > stability_eps)
                        & (pl.col("_b0_v") > pl.col("_structural_v") * regime_transient_ratio)
                        & ~pl.col("_trend_up_v")
                    ).alias("_transient_up_v"),
                    (
                        (pl.col("_structural_history_blocks_v") >= regime_min_history_blocks)
                        & (pl.col("_structural_v") > stability_eps)
                        & (pl.col("_b0_v") < pl.col("_structural_v") / regime_transient_ratio)
                        & ~pl.col("_trend_down_v")
                    ).alias("_transient_down_v"),
                )
                .with_columns(
                    pl.when(pl.col("_transient_up_y"))
                    .then(pl.lit("transient_up"))
                    .when(pl.col("_transient_down_y"))
                    .then(pl.lit("transient_down"))
                    .when(pl.col("_trend_up_y"))
                    .then(pl.lit("trend_up"))
                    .when(pl.col("_trend_down_y"))
                    .then(pl.lit("trend_down"))
                    .otherwise(pl.lit("stable"))
                    .alias("_regime_class_y"),
                    pl.when(pl.col("_transient_up_v"))
                    .then(pl.lit("transient_up"))
                    .when(pl.col("_transient_down_v"))
                    .then(pl.lit("transient_down"))
                    .when(pl.col("_trend_up_v"))
                    .then(pl.lit("trend_up"))
                    .when(pl.col("_trend_down_v"))
                    .then(pl.lit("trend_down"))
                    .otherwise(pl.lit("stable"))
                    .alias("_regime_class_v"),
                )
                .with_columns(
                    pl.col("_structural_y").alias("_regime_anchor_y"),
                    pl.col("_structural_v").alias("_regime_anchor_v"),
                )
                .with_columns(
                    pl.when(pl.col("_regime_anchor_y") > stability_eps)
                    .then(pl.col("_regime_anchor_y"))
                    .otherwise(pl.col("_reference_y"))
                    .alias("_regime_anchor_y"),
                    pl.when(pl.col("_regime_anchor_v") > stability_eps)
                    .then(pl.col("_regime_anchor_v"))
                    .otherwise(pl.col("_reference_v"))
                    .alias("_regime_anchor_v"),
                )
                .with_columns(
                    pl.lit(regime_ref_block)
                    .cast(pl.Int32)
                    .alias("_regime_ref_block")
                )
                .select(
                    uid_col,
                    "_regime_ref_block",
                    "_reference_y",
                    "_reference_v",
                    "_recent28_y",
                    "_recent28_v",
                    "_recent14_y",
                    "_recent14_v",
                    "_coverage_y",
                    "_coverage_v",
                    "_b0_y", "_b0_v",
                    "_b1_y", "_b1_v",
                    "_b2_y", "_b2_v",
                    "_b3_y", "_b3_v",
                    "_structural_y", "_structural_v",
                    "_structural_history_blocks_y",
                    "_structural_history_blocks_v",
                    "_regime_class_y", "_regime_class_v",
                    "_regime_anchor_y",
                    "_regime_anchor_v",
                )
            )

            # ── STAGE 1: PURE SES level ────────────────────────────────
            # A transient is handled explicitly as a state reinitialization.
            # This is the only regime intervention on level. Trend/stable
            # origins leave the SES recurrence untouched.
            reset_ref = recent_ref.select(
                uid_col,
                "_regime_class_y",
                "_regime_class_v",
                "_structural_y",
                "_structural_v",
                "_structural_history_blocks_y",
                "_structural_history_blocks_v",
            )
            alpha_state = (
                alpha_state.join(reset_ref, on=uid_col, how="left")
                .with_columns(
                    (
                        pl.lit(reset_transient)
                        & (pl.col("_regime_class_y") == "transient_up")
                        & (
                            pl.col("_structural_history_blocks_y")
                            >= regime_min_history_blocks
                        )
                    ).alias("_reset_y"),
                    (
                        pl.lit(reset_transient)
                        & (pl.col("_regime_class_v") == "transient_up")
                        & (
                            pl.col("_structural_history_blocks_v")
                            >= regime_min_history_blocks
                        )
                    ).alias("_reset_v"),
                )
                .with_columns(
                    pl.when(pl.col("_reset_y"))
                    .then(pl.col("_structural_y").clip(lower_bound=0.0))
                    .otherwise(pl.col("_level_y"))
                    .alias("_level_y"),
                    pl.when(pl.col("_reset_v"))
                    .then(pl.col("_structural_v").clip(lower_bound=0.0))
                    .otherwise(pl.col("_level_v"))
                    .alias("_level_v"),
                )
                .drop(
                    "_regime_class_y",
                    "_regime_class_v",
                    "_structural_y",
                    "_structural_v",
                    "_structural_history_blocks_y",
                    "_structural_history_blocks_v",
                )
            )

            if block_i == 1:
                alpha_choices = (
                    ids.select(uid_col)
                    .join(recent_ref, on=uid_col, how="left")
                    .with_columns(
                        pl.lit(default_alpha).alias("_alpha_y"),
                        pl.lit(default_alpha).alias("_alpha_v"),
                        pl.lit(default_alpha).alias("_alpha_y_unconstrained"),
                        pl.lit(default_alpha).alias("_alpha_v_unconstrained"),
                        pl.lit(False).alias("_ses_guard_y"),
                        pl.lit(False).alias("_ses_guard_v"),
                        pl.lit("WARMUP_DEFAULT").alias("_ses_guard_status_y"),
                        pl.lit("WARMUP_DEFAULT").alias("_ses_guard_status_v"),
                        pl.lit(True).alias("_stable_pool_y"),
                        pl.lit(True).alias("_stable_pool_v"),
                        pl.lit(0).cast(pl.Int32).alias("_score_history_blocks_y"),
                        pl.lit(0).cast(pl.Int32).alias("_score_history_blocks_v"),
                    )
                )
                alpha_selection_source = (
                    alpha_state.join(alpha_choices, on=uid_col, how="inner")
                    .with_columns(
                        pl.lit(None).cast(pl.Float64).alias("_wmape_ses_y"),
                        pl.lit(None).cast(pl.Float64).alias("_bias_ses_y"),
                        pl.lit(None).cast(pl.Float64).alias("_score_ses_y"),
                        pl.lit(None).cast(pl.Float64).alias("_wmape_ses_v"),
                        pl.lit(None).cast(pl.Float64).alias("_bias_ses_v"),
                        pl.lit(None).cast(pl.Float64).alias("_score_ses_v"),
                    )
                )
            else:
                ranked_alpha = (
                    alpha_state.join(recent_ref, on=uid_col, how="left")
                    .with_columns(
                        _weighted_score_component("ae", "y")
                        .alias("_finite_ae_y"),
                        _weighted_score_component("den", "y")
                        .alias("_finite_den_y"),
                        _weighted_score_component("se", "y")
                        .alias("_finite_se_y"),
                        _weighted_score_component("ae", "v")
                        .alias("_finite_ae_v"),
                        _weighted_score_component("den", "v")
                        .alias("_finite_den_v"),
                        _weighted_score_component("se", "v")
                        .alias("_finite_se_v"),
                        _score_history_block_count("y")
                        .alias("_score_history_blocks_y"),
                        _score_history_block_count("v")
                        .alias("_score_history_blocks_v"),
                    )
                    .with_columns(
                        pl.when(pl.col("_finite_den_y") > score_min_den)
                        .then(pl.col("_finite_ae_y") / pl.col("_finite_den_y"))
                        .otherwise(float("inf"))
                        .alias("_wmape_ses_y"),
                        pl.when(pl.col("_finite_den_y") > score_min_den)
                        .then(pl.col("_finite_se_y") / pl.col("_finite_den_y"))
                        .otherwise(0.0)
                        .alias("_bias_ses_y"),
                        pl.when(pl.col("_finite_den_v") > score_min_den)
                        .then(pl.col("_finite_ae_v") / pl.col("_finite_den_v"))
                        .otherwise(float("inf"))
                        .alias("_wmape_ses_v"),
                        pl.when(pl.col("_finite_den_v") > score_min_den)
                        .then(pl.col("_finite_se_v") / pl.col("_finite_den_v"))
                        .otherwise(0.0)
                        .alias("_bias_ses_v"),
                    )
                    .with_columns(
                        (
                            pl.col("_wmape_ses_y")
                            + pl.lit(alpha_bias_weight)
                            * pl.col("_bias_ses_y").abs()
                        ).alias("_score_ses_y"),
                        (
                            pl.col("_wmape_ses_v")
                            + pl.lit(alpha_bias_weight)
                            * pl.col("_bias_ses_v").abs()
                        ).alias("_score_ses_v"),
                    )
                )

                alpha_choices = (
                    ranked_alpha.group_by(uid_col)
                    .agg(
                        pl.col("_alpha")
                        .sort_by("_score_ses_y", "_alpha")
                        .first()
                        .alias("_alpha_y"),
                        pl.col("_alpha")
                        .sort_by("_score_ses_v", "_alpha")
                        .first()
                        .alias("_alpha_v"),
                        pl.col("_score_history_blocks_y").max()
                        .alias("_score_history_blocks_y"),
                        pl.col("_score_history_blocks_v").max()
                        .alias("_score_history_blocks_v"),
                        pl.col("_regime_ref_block").first()
                        .alias("_regime_ref_block"),
                        pl.col("_reference_y").first().alias("_reference_y"),
                        pl.col("_reference_v").first().alias("_reference_v"),
                        pl.col("_recent28_y").first().alias("_recent28_y"),
                        pl.col("_recent28_v").first().alias("_recent28_v"),
                        pl.col("_recent14_y").first().alias("_recent14_y"),
                        pl.col("_recent14_v").first().alias("_recent14_v"),
                        pl.col("_coverage_y").first().alias("_coverage_y"),
                        pl.col("_coverage_v").first().alias("_coverage_v"),
                        pl.col("_regime_anchor_y").first().alias("_regime_anchor_y"),
                        pl.col("_regime_anchor_v").first().alias("_regime_anchor_v"),
                        pl.col("_regime_class_y").first().alias("_regime_class_y"),
                        pl.col("_regime_class_v").first().alias("_regime_class_v"),
                        pl.col("_structural_y").first().alias("_structural_y"),
                        pl.col("_structural_v").first().alias("_structural_v"),
                        pl.col("_structural_history_blocks_y").first()
                        .alias("_structural_history_blocks_y"),
                        pl.col("_structural_history_blocks_v").first()
                        .alias("_structural_history_blocks_v"),
                        pl.col("_b0_y").first().alias("_b0_y"),
                        pl.col("_b0_v").first().alias("_b0_v"),
                        pl.col("_b1_y").first().alias("_b1_y"),
                        pl.col("_b1_v").first().alias("_b1_v"),
                        pl.col("_b2_y").first().alias("_b2_y"),
                        pl.col("_b2_v").first().alias("_b2_v"),
                        pl.col("_b3_y").first().alias("_b3_y"),
                        pl.col("_b3_v").first().alias("_b3_v"),
                        pl.col("_reset_y").any().alias("_reset_y"),
                        pl.col("_reset_v").any().alias("_reset_v"),
                    )
                    .with_columns(
                        pl.col("_alpha_y").alias("_alpha_y_unconstrained"),
                        pl.col("_alpha_v").alias("_alpha_v_unconstrained"),
                    )
                    .with_columns(
                        pl.when(pl.col("_regime_class_y") == "transient_down")
                        .then(pl.lit(transient_down_alpha))
                        .otherwise(pl.col("_alpha_y"))
                        .alias("_alpha_y"),
                        pl.when(pl.col("_regime_class_v") == "transient_down")
                        .then(pl.lit(transient_down_alpha))
                        .otherwise(pl.col("_alpha_v"))
                        .alias("_alpha_v"),
                        pl.lit(True).alias("_stable_pool_y"),
                        pl.lit(True).alias("_stable_pool_v"),
                        pl.when(pl.col("_reset_y"))
                        .then(pl.lit("SES_RESET_TRANSIENT_UP"))
                        .when(pl.col("_regime_class_y") == "transient_down")
                        .then(pl.lit("SES_REACTIVE_TRANSIENT_DOWN"))
                        .when(pl.col("_score_history_blocks_y") == 0)
                        .then(pl.lit("NO_SCORE_HISTORY"))
                        .otherwise(pl.lit("OK"))
                        .alias("_ses_guard_status_y"),
                        pl.when(pl.col("_reset_v"))
                        .then(pl.lit("SES_RESET_TRANSIENT_UP"))
                        .when(pl.col("_regime_class_v") == "transient_down")
                        .then(pl.lit("SES_REACTIVE_TRANSIENT_DOWN"))
                        .when(pl.col("_score_history_blocks_v") == 0)
                        .then(pl.lit("NO_SCORE_HISTORY"))
                        .otherwise(pl.lit("OK"))
                        .alias("_ses_guard_status_v"),
                    )
                    .with_columns(
                        (pl.col("_ses_guard_status_y") != "OK")
                        .alias("_ses_guard_y"),
                        (pl.col("_ses_guard_status_v") != "OK")
                        .alias("_ses_guard_v"),
                    )
                )

                alpha_selection_source = ranked_alpha.join(
                    alpha_choices.select(
                        uid_col,
                        "_alpha_y",
                        "_alpha_v",
                        "_alpha_y_unconstrained",
                        "_alpha_v_unconstrained",
                        "_ses_guard_y",
                        "_ses_guard_v",
                        "_ses_guard_status_y",
                        "_ses_guard_status_v",
                        "_stable_pool_y",
                        "_stable_pool_v",
                        "_score_history_blocks_y",
                        "_score_history_blocks_v",
                    ),
                    on=uid_col,
                    how="inner",
                )

            selected_y = (
                alpha_selection_source
                .filter(pl.col("_alpha") == pl.col("_alpha_y"))
                .select(
                    uid_col,
                    "_alpha_y",
                    "_alpha_y_unconstrained",
                    "_ses_guard_y",
                    "_ses_guard_status_y",
                    "_stable_pool_y",
                    "_score_history_blocks_y",
                    "_regime_ref_block",
                    "_reference_y",
                    "_recent28_y",
                    "_recent14_y",
                    "_coverage_y",
                    "_regime_anchor_y",
                    "_regime_class_y",
                    "_structural_y",
                    "_structural_history_blocks_y",
                    "_b0_y", "_b1_y", "_b2_y", "_b3_y",
                    pl.col("_level_y"),
                    pl.col("_wmape_ses_y").alias("_pure_ses_wmape_y"),
                    pl.col("_bias_ses_y").alias("_pure_ses_bias_y"),
                    pl.col("_score_ses_y").alias("_pure_ses_score_y"),
                )
            )
            selected_v = (
                alpha_selection_source
                .filter(pl.col("_alpha") == pl.col("_alpha_v"))
                .select(
                    uid_col,
                    "_alpha_v",
                    "_alpha_v_unconstrained",
                    "_ses_guard_v",
                    "_ses_guard_status_v",
                    "_stable_pool_v",
                    "_score_history_blocks_v",
                    "_reference_v",
                    "_recent28_v",
                    "_recent14_v",
                    "_coverage_v",
                    "_regime_anchor_v",
                    "_regime_class_v",
                    "_structural_v",
                    "_structural_history_blocks_v",
                    "_b0_v", "_b1_v", "_b2_v", "_b3_v",
                    pl.col("_level_v"),
                    pl.col("_wmape_ses_v").alias("_pure_ses_wmape_v"),
                    pl.col("_bias_ses_v").alias("_pure_ses_bias_v"),
                    pl.col("_score_ses_v").alias("_pure_ses_score_v"),
                )
            )
            selected_level = selected_y.join(
                selected_v, on=uid_col, how="inner"
            )

            # ── STAGE 2: select parent + driver mode with SES fixed ─────
            # Candidates: store/section × shape_only/level_shape.  The first
            # scored block starts conservatively with shape_only; later blocks
            # choose only from cumulative PRIOR-block errors.
            if block_i == 1:
                parent_choices = (
                    leaf_parent_availability.with_columns(
                        pl.when(pl.col("_store_parent_available"))
                        .then(pl.lit("store"))
                        .otherwise(pl.lit("section"))
                        .alias("_parent_y"),
                        pl.when(pl.col("_store_parent_available"))
                        .then(pl.lit("store"))
                        .otherwise(pl.lit("section"))
                        .alias("_parent_v"),
                        pl.lit(default_driver_mode).alias("_driver_mode_y"),
                        pl.lit(default_driver_mode).alias("_driver_mode_v"),
                        pl.lit(1.0).alias("_strength_y"),
                        pl.lit(1.0).alias("_strength_v"),
                    )
                    .select(
                        uid_col,
                        "_parent_y", "_parent_v",
                        "_driver_mode_y", "_driver_mode_v",
                        "_strength_y", "_strength_v",
                    )
                )
            else:
                ranked_parent = (
                    parent_state.with_columns(
                        pl.when(pl.col("_cden_parent_y") > 0)
                        .then(pl.col("_cae_parent_y") / pl.col("_cden_parent_y"))
                        .otherwise(float("inf"))
                        .alias("_wmape_parent_y"),
                        pl.when(pl.col("_cden_parent_v") > 0)
                        .then(pl.col("_cae_parent_v") / pl.col("_cden_parent_v"))
                        .otherwise(float("inf"))
                        .alias("_wmape_parent_v"),
                        pl.when(pl.col("_parent") == default_parent)
                        .then(0).otherwise(1).cast(pl.Int8).alias("_parent_priority"),
                        pl.when(pl.col("_driver_mode") == default_driver_mode)
                        .then(0).otherwise(1).cast(pl.Int8).alias("_mode_priority"),
                    )
                )
                sort_y = ["_wmape_parent_y", "_mode_priority", "_parent_priority"]
                sort_v = ["_wmape_parent_v", "_mode_priority", "_parent_priority"]
                parent_choices = ranked_parent.group_by(uid_col).agg(
                    pl.col("_parent").sort_by(*sort_y).first().alias("_parent_y"),
                    pl.col("_driver_mode").sort_by(*sort_y).first().alias("_driver_mode_y"),
                    pl.col("_parent").sort_by(*sort_v).first().alias("_parent_v"),
                    pl.col("_driver_mode").sort_by(*sort_v).first().alias("_driver_mode_v"),
                ).with_columns(
                    pl.lit(1.0).alias("_strength_y"),
                    pl.lit(1.0).alias("_strength_v"),
                )

            parent_choices = parent_choices.select(
                uid_col,
                "_parent_y", "_parent_v",
                "_driver_mode_y", "_driver_mode_v",
                "_strength_y", "_strength_v",
            )
            chosen_frame = (
                selected_level.join(
                    parent_choices, on=uid_col, how="inner"
                ).with_columns(
                    pl.lit(block_i).cast(pl.Int32).alias("_block")
                )
            )

            # Defensive schema invariant for both RAM and disk-spill paths.
            if chosen_cols is None:
                chosen_cols = list(chosen_frame.columns)
            else:
                if set(chosen_frame.columns) != set(chosen_cols):
                    raise RuntimeError(
                        "Leaf chosen-frame schema mismatch before consolidation: "
                        f"block={block_i}, expected={chosen_cols}, "
                        f"got={chosen_frame.columns}"
                    )
                chosen_frame = chosen_frame.select(chosen_cols)

            if memory_safe:
                assert spill_path is not None
                chosen_frame.write_parquet(
                    spill_path / f"chosen_state_{block_i:05d}.parquet",
                    compression=str(getattr(settings, "MULTIBLOCK_SPILL_COMPRESSION", "zstd")),
                    statistics=True,
                )
            else:
                chosen_frames.append(chosen_frame)

            # Forecast-only has no actuals and is never allowed to affect
            # alpha/parent selection or SES state.
            if block_i > actual_last_block:
                continue

            obs_block = obs_by_block.get(block_i)

            # ── Score ALL alpha candidates using SES ONLY ──────────────────
            if obs_block is not None and obs_block.height:
                alpha_eval = (
                    obs_block.join(
                        alpha_state.select(
                            uid_col,
                            "_alpha",
                            "_level_y",
                            "_level_v",
                            "_n_calendar",
                        ),
                        on=uid_col,
                        how="inner",
                    )
                    .with_columns(
                        (
                            pl.lit(block_end) - pl.col("ds")
                        ).dt.total_days().cast(pl.Int32).alias("_days_to_end")
                    )
                    .with_columns(
                        (
                            pl.col("_alpha")
                            * (pl.lit(1.0) - pl.col("_alpha")).pow(
                                pl.col("_days_to_end")
                            )
                        ).alias("_w")
                    )
                    .group_by([uid_col, "_alpha"])
                    .agg(
                        # Official client objective: score ONLY dates with
                        # positive actual demand. Missing/zero sale dates still
                        # participate in the SES state decay below, but they do
                        # not enter wMAPE/BIAS model selection.
                        pl.when(pl.col("y") > 0)
                        .then((pl.col("y") - pl.col("_level_y")).abs())
                        .otherwise(0.0)
                        .sum().alias("_ses_ae_pos_y"),
                        pl.when(pl.col("y") > 0)
                        .then(pl.col("y").abs())
                        .otherwise(0.0)
                        .sum().alias("_ses_den_y"),
                        pl.when(pl.col("y") > 0)
                        .then(pl.col("_level_y") - pl.col("y"))
                        .otherwise(0.0)
                        .sum().alias("_ses_signed_pos_y"),
                        pl.col("y").sum().alias("_ses_sum_y"),
                        (pl.col("y") > 0).sum().alias("_ses_nz_y"),
                        pl.when(pl.col("value") > 0)
                        .then((pl.col("value") - pl.col("_level_v")).abs())
                        .otherwise(0.0)
                        .sum().alias("_ses_ae_pos_v"),
                        pl.when(pl.col("value") > 0)
                        .then(pl.col("value").abs())
                        .otherwise(0.0)
                        .sum().alias("_ses_den_v"),
                        pl.when(pl.col("value") > 0)
                        .then(pl.col("_level_v") - pl.col("value"))
                        .otherwise(0.0)
                        .sum().alias("_ses_signed_pos_v"),
                        pl.col("value").sum().alias("_ses_sum_v"),
                        (pl.col("value") > 0).sum().alias("_ses_nz_v"),
                        (pl.col("y").clip(lower_bound=0.0) * pl.col("_w"))
                        .sum().alias("_wzy"),
                        (pl.col("value").clip(lower_bound=0.0) * pl.col("_w"))
                        .sum().alias("_wzv"),
                    )
                )
                alpha_state = alpha_state.join(
                    alpha_eval, on=[uid_col, "_alpha"], how="left"
                )
            else:
                alpha_state = alpha_state.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("_ses_ae_pos_y"),
                    pl.lit(None).cast(pl.Float64).alias("_ses_den_y"),
                    pl.lit(None).cast(pl.Float64).alias("_ses_signed_pos_y"),
                    pl.lit(None).cast(pl.Float64).alias("_ses_sum_y"),
                    pl.lit(None).cast(pl.Int64).alias("_ses_nz_y"),
                    pl.lit(None).cast(pl.Float64).alias("_ses_ae_pos_v"),
                    pl.lit(None).cast(pl.Float64).alias("_ses_den_v"),
                    pl.lit(None).cast(pl.Float64).alias("_ses_signed_pos_v"),
                    pl.lit(None).cast(pl.Float64).alias("_ses_sum_v"),
                    pl.lit(None).cast(pl.Int64).alias("_ses_nz_v"),
                    pl.lit(None).cast(pl.Float64).alias("_wzy"),
                    pl.lit(None).cast(pl.Float64).alias("_wzv"),
                )

            # PURE SES recurrence over daily actuals. Missing calendar days
            # are zeros; finite scoring only changes alpha selection, never the
            # SES state equation itself.
            full_block = pl.col("_n_calendar") == block_days
            alpha_state = (
                alpha_state.with_columns(
                    pl.when(full_block)
                    .then(pl.col("_ses_ae_pos_y").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_ses_ae_y"),
                    pl.when(full_block)
                    .then(pl.col("_ses_den_y").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_ses_den_y"),
                    pl.when(full_block)
                    .then(pl.col("_ses_ae_pos_v").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_ses_ae_v"),
                    pl.when(full_block)
                    .then(pl.col("_ses_den_v").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_ses_den_v"),
                    pl.when(full_block)
                    .then(pl.col("_ses_signed_pos_y").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_ses_signed_y"),
                    pl.when(full_block)
                    .then(pl.col("_ses_signed_pos_v").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_ses_signed_v"),
                )
                .with_columns(
                    (
                        (pl.lit(1.0) - pl.col("_alpha")).pow(
                            pl.col("_n_calendar")
                        ) * pl.col("_level_y")
                        + pl.col("_wzy").fill_null(0.0)
                    ).alias("_level_y_next"),
                    (
                        (pl.lit(1.0) - pl.col("_alpha")).pow(
                            pl.col("_n_calendar")
                        ) * pl.col("_level_v")
                        + pl.col("_wzv").fill_null(0.0)
                    ).alias("_level_v_next"),
                    pl.when(full_block)
                    .then(pl.col("_score_ae_b2_y"))
                    .otherwise(pl.col("_score_ae_b3_y"))
                    .alias("_next_score_ae_b3_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_ae_b1_y"))
                    .otherwise(pl.col("_score_ae_b2_y"))
                    .alias("_next_score_ae_b2_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_ae_b0_y"))
                    .otherwise(pl.col("_score_ae_b1_y"))
                    .alias("_next_score_ae_b1_y"),
                    pl.when(full_block)
                    .then(pl.col("_block_ses_ae_y"))
                    .otherwise(pl.col("_score_ae_b0_y"))
                    .alias("_next_score_ae_b0_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_den_b2_y"))
                    .otherwise(pl.col("_score_den_b3_y"))
                    .alias("_next_score_den_b3_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_den_b1_y"))
                    .otherwise(pl.col("_score_den_b2_y"))
                    .alias("_next_score_den_b2_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_den_b0_y"))
                    .otherwise(pl.col("_score_den_b1_y"))
                    .alias("_next_score_den_b1_y"),
                    pl.when(full_block)
                    .then(pl.col("_block_ses_den_y"))
                    .otherwise(pl.col("_score_den_b0_y"))
                    .alias("_next_score_den_b0_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_se_b2_y"))
                    .otherwise(pl.col("_score_se_b3_y"))
                    .alias("_next_score_se_b3_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_se_b1_y"))
                    .otherwise(pl.col("_score_se_b2_y"))
                    .alias("_next_score_se_b2_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_se_b0_y"))
                    .otherwise(pl.col("_score_se_b1_y"))
                    .alias("_next_score_se_b1_y"),
                    pl.when(full_block)
                    .then(pl.col("_block_ses_signed_y"))
                    .otherwise(pl.col("_score_se_b0_y"))
                    .alias("_next_score_se_b0_y"),
                    pl.when(full_block)
                    .then(pl.col("_score_ae_b2_v"))
                    .otherwise(pl.col("_score_ae_b3_v"))
                    .alias("_next_score_ae_b3_v"),
                    pl.when(full_block)
                    .then(pl.col("_score_ae_b1_v"))
                    .otherwise(pl.col("_score_ae_b2_v"))
                    .alias("_next_score_ae_b2_v"),
                    pl.when(full_block)
                    .then(pl.col("_score_ae_b0_v"))
                    .otherwise(pl.col("_score_ae_b1_v"))
                    .alias("_next_score_ae_b1_v"),
                    pl.when(full_block)
                    .then(pl.col("_block_ses_ae_v"))
                    .otherwise(pl.col("_score_ae_b0_v"))
                    .alias("_next_score_ae_b0_v"),
                    pl.when(full_block)
                    .then(pl.col("_score_den_b2_v"))
                    .otherwise(pl.col("_score_den_b3_v"))
                    .alias("_next_score_den_b3_v"),
                    pl.when(full_block)
                    .then(pl.col("_score_den_b1_v"))
                    .otherwise(pl.col("_score_den_b2_v"))
                    .alias("_next_score_den_b2_v"),
                    pl.when(full_block)
                    .then(pl.col("_score_den_b0_v"))
                    .otherwise(pl.col("_score_den_b1_v"))
                    .alias("_next_score_den_b1_v"),
                    pl.when(full_block)
                    .then(pl.col("_block_ses_den_v"))
                    .otherwise(pl.col("_score_den_b0_v"))
                    .alias("_next_score_den_b0_v"),
                    pl.when(full_block)
                    .then(pl.col("_score_se_b2_v"))
                    .otherwise(pl.col("_score_se_b3_v"))
                    .alias("_next_score_se_b3_v"),
                    pl.when(full_block)
                    .then(pl.col("_score_se_b1_v"))
                    .otherwise(pl.col("_score_se_b2_v"))
                    .alias("_next_score_se_b2_v"),
                    pl.when(full_block)
                    .then(pl.col("_score_se_b0_v"))
                    .otherwise(pl.col("_score_se_b1_v"))
                    .alias("_next_score_se_b1_v"),
                    pl.when(full_block)
                    .then(pl.col("_block_ses_signed_v"))
                    .otherwise(pl.col("_score_se_b0_v"))
                    .alias("_next_score_se_b0_v"),
                )
                .select(
                    uid_col,
                    "_leaf_start",
                    "_leaf_warmup_end",
                    "_alpha",
                    pl.col("_level_y_next").alias("_level_y"),
                    pl.col("_level_v_next").alias("_level_v"),
                pl.col("_next_score_ae_b0_y").alias("_score_ae_b0_y"),
                pl.col("_next_score_den_b0_y").alias("_score_den_b0_y"),
                pl.col("_next_score_se_b0_y").alias("_score_se_b0_y"),
                pl.col("_next_score_ae_b1_y").alias("_score_ae_b1_y"),
                pl.col("_next_score_den_b1_y").alias("_score_den_b1_y"),
                pl.col("_next_score_se_b1_y").alias("_score_se_b1_y"),
                pl.col("_next_score_ae_b2_y").alias("_score_ae_b2_y"),
                pl.col("_next_score_den_b2_y").alias("_score_den_b2_y"),
                pl.col("_next_score_se_b2_y").alias("_score_se_b2_y"),
                pl.col("_next_score_ae_b3_y").alias("_score_ae_b3_y"),
                pl.col("_next_score_den_b3_y").alias("_score_den_b3_y"),
                pl.col("_next_score_se_b3_y").alias("_score_se_b3_y"),
                pl.col("_next_score_ae_b0_v").alias("_score_ae_b0_v"),
                pl.col("_next_score_den_b0_v").alias("_score_den_b0_v"),
                pl.col("_next_score_se_b0_v").alias("_score_se_b0_v"),
                pl.col("_next_score_ae_b1_v").alias("_score_ae_b1_v"),
                pl.col("_next_score_den_b1_v").alias("_score_den_b1_v"),
                pl.col("_next_score_se_b1_v").alias("_score_se_b1_v"),
                pl.col("_next_score_ae_b2_v").alias("_score_ae_b2_v"),
                pl.col("_next_score_den_b2_v").alias("_score_den_b2_v"),
                pl.col("_next_score_se_b2_v").alias("_score_se_b2_v"),
                pl.col("_next_score_ae_b3_v").alias("_score_ae_b3_v"),
                pl.col("_next_score_den_b3_v").alias("_score_den_b3_v"),
                pl.col("_next_score_se_b3_v").alias("_score_se_b3_v"),
                )
            )

            # ── Score parent shapes using ONLY the already-selected SES level ──
            calendar_by_uid = (
                alpha_state.select(uid_col, "_leaf_warmup_end")
                .unique(subset=[uid_col])
                .with_columns(
                    pl.max_horizontal(
                        pl.col("_leaf_warmup_end") + pl.duration(days=1),
                        pl.lit(block_start),
                    ).alias("_update_start")
                )
                .with_columns(
                    pl.when(pl.col("_update_start") <= pl.lit(block_end))
                    .then(
                        (pl.lit(block_end) - pl.col("_update_start"))
                        .dt.total_days()
                        + 1
                    )
                    .otherwise(0)
                    .cast(pl.Int32)
                    .alias("_n_calendar")
                )
                .select(uid_col, "_n_calendar")
            )
            parent_eval_state = (
                parent_state.join(
                    selected_level, on=uid_col, how="left"
                )
                .join(calendar_by_uid, on=uid_col, how="left")
            )

            if obs_block is not None and obs_block.height:
                parent_eval = (
                    obs_block.join(
                        parent_eval_state.select(
                            uid_col,
                            "_parent", "_driver_mode", "_strength",
                            "_level_y", "_level_v", "_n_calendar",
                        ),
                        on=uid_col,
                        how="inner",
                    )
                    .with_columns(
                        pl.when((pl.col("_parent") == "store") & (pl.col("_driver_mode") == "level_shape"))
                        .then(pl.col("_store_ey_level").exp())
                        .when(pl.col("_parent") == "store")
                        .then(pl.col("_store_ey_shape").exp())
                        .when((pl.col("_parent") == "section") & (pl.col("_driver_mode") == "level_shape"))
                        .then(pl.col("_sec_ey_level").exp())
                        .when(pl.col("_parent") == "section")
                        .then(pl.col("_sec_ey_shape").exp())
                        .otherwise(1.0)
                        .alias("_raw_factor_y"),
                        pl.when((pl.col("_parent") == "store") & (pl.col("_driver_mode") == "level_shape"))
                        .then(pl.col("_store_ev_level").exp())
                        .when(pl.col("_parent") == "store")
                        .then(pl.col("_store_ev_shape").exp())
                        .when((pl.col("_parent") == "section") & (pl.col("_driver_mode") == "level_shape"))
                        .then(pl.col("_sec_ev_level").exp())
                        .when(pl.col("_parent") == "section")
                        .then(pl.col("_sec_ev_shape").exp())
                        .otherwise(1.0)
                        .alias("_raw_factor_v"),
                    )
                    .with_columns(
                        (1.0 + pl.col("_strength") * (pl.col("_raw_factor_y") - 1.0)).alias("_factor_candidate_y"),
                        (1.0 + pl.col("_strength") * (pl.col("_raw_factor_v") - 1.0)).alias("_factor_candidate_v"),
                    )
                    .with_columns(
                        (pl.col("_level_y") * pl.col("_factor_candidate_y")).clip(lower_bound=0.0).alias("_pred_y"),
                        (pl.col("_level_v") * pl.col("_factor_candidate_v")).clip(lower_bound=0.0).alias("_pred_v"),
                    )
                    .group_by([uid_col, "_parent", "_driver_mode", "_strength"])
                    .agg(
                        pl.when(pl.col("y") > 0).then((pl.col("y") - pl.col("_pred_y")).abs()).otherwise(0.0).sum().alias("_parent_ae_pos_y"),
                        pl.when(pl.col("y") > 0).then(pl.col("y").abs()).otherwise(0.0).sum().alias("_parent_den_y"),
                        pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("_pred_v")).abs()).otherwise(0.0).sum().alias("_parent_ae_pos_v"),
                        pl.when(pl.col("value") > 0).then(pl.col("value").abs()).otherwise(0.0).sum().alias("_parent_den_v"),
                    )
                )
                parent_state = parent_state.join(
                    parent_eval,
                    on=[uid_col, "_parent", "_driver_mode", "_strength"],
                    how="left",
                )
            else:
                parent_state = parent_state.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("_parent_ae_pos_y"),
                    pl.lit(None).cast(pl.Float64).alias("_parent_den_y"),
                    pl.lit(None).cast(pl.Float64).alias("_parent_ae_pos_v"),
                    pl.lit(None).cast(pl.Float64).alias("_parent_den_v"),
                )

            parent_state = (
                parent_state.join(
                    selected_level.select(
                        uid_col, "_level_y", "_level_v"
                    ),
                    on=uid_col,
                    how="left",
                )
                .join(calendar_by_uid, on=uid_col, how="left")
                .with_columns(
                    pl.when(pl.col("_n_calendar") == block_days)
                    .then(pl.col("_parent_ae_pos_y").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_parent_ae_y"),
                    pl.when(pl.col("_n_calendar") == block_days)
                    .then(pl.col("_parent_den_y").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_parent_den_y"),
                    pl.when(pl.col("_n_calendar") == block_days)
                    .then(pl.col("_parent_ae_pos_v").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_parent_ae_v"),
                    pl.when(pl.col("_n_calendar") == block_days)
                    .then(pl.col("_parent_den_v").fill_null(0.0))
                    .otherwise(0.0)
                    .alias("_block_parent_den_v"),
                )
                .with_columns(
                    (
                        pl.lit(driver_score_decay)
                        * pl.col("_cae_parent_y")
                        + pl.col("_block_parent_ae_y")
                    ).alias("_cae_parent_y"),
                    (
                        pl.lit(driver_score_decay)
                        * pl.col("_cden_parent_y")
                        + pl.col("_block_parent_den_y")
                    ).alias("_cden_parent_y"),
                    (
                        pl.lit(driver_score_decay)
                        * pl.col("_cae_parent_v")
                        + pl.col("_block_parent_ae_v")
                    ).alias("_cae_parent_v"),
                    (
                        pl.lit(driver_score_decay)
                        * pl.col("_cden_parent_v")
                        + pl.col("_block_parent_den_v")
                    ).alias("_cden_parent_v"),
                )
                .select(
                    uid_col,
                    "_parent", "_driver_mode", "_strength",
                    "_cae_parent_y", "_cden_parent_y",
                    "_cae_parent_v", "_cden_parent_v",
                )
            )

        chosen_states = (
            pl.concat(chosen_frames, how="vertical_relaxed")
            if not memory_safe
            else None
        )

        logger.info(
            "⏱ Sección %s: selección leaf 2-etapas (SES puro %d alphas + %d parent-modes): %.1fs",
            section_id,
            len(alpha_candidates),
            parent_params.height,
            time.perf_counter() - t_candidates,
        )

        # Reference level for audit: reuse the block statistics already built
        # above. v11.0.1 rescanned actual_obs once per block; that duplicated a
        # costly group_by/filter loop without changing the result.
        recent_refs = (
            recent_block_stats.select(
                uid_col,
                "_block",
                (pl.col("_recent28_sum_y") / pl.lit(float(block_days))).alias("_recent28_mean_y"),
                (pl.col("_recent28_sum_v") / pl.lit(float(block_days))).alias("_recent28_mean_v"),
            )
            if not memory_safe else pl.DataFrame()
        )

        if memory_safe:
            assert spill_path is not None
            chosen_lf = pl.scan_parquet(str(spill_path / "chosen_state_*.parquet"))
            # Streaming join avoids materializing the whole chosen-state table
            # beside the already-large daily leaf frame. The final daily frame
            # still exists (it must be saved), but the extra N-origins copy does not.
            rows = (
                rows.lazy()
                .join(chosen_lf, on=[uid_col, "_block"], how="left")
                .collect(engine="streaming")
                .with_columns(
                    pl.col("_recent28_y").alias("_recent28_mean_y"),
                    pl.col("_recent28_v").alias("_recent28_mean_v"),
                )
            )
            # Windows may retain file handles until collection completes; only
            # clean after the streaming join has fully materialized.
            if spill_tmp is not None:
                spill_tmp.cleanup()
                spill_tmp = None
        else:
            assert chosen_states is not None
            rows = (
                rows.join(chosen_states, on=[uid_col, "_block"], how="left")
                .join(recent_refs, on=[uid_col, "_block"], how="left")
            )

        rows = (
            rows.with_columns(
                (
                    (pl.col("ds") - pl.lit(train_start)).dt.total_days()
                    % block_days
                )
                .cast(pl.Int32)
                .alias("_day_in_block"),
                pl.when((pl.col("_parent_y") == "store") & (pl.col("_driver_mode_y") == "level_shape"))
                .then(pl.col("_store_ey_level").exp())
                .when(pl.col("_parent_y") == "store")
                .then(pl.col("_store_ey_shape").exp())
                .when((pl.col("_parent_y") == "section") & (pl.col("_driver_mode_y") == "level_shape"))
                .then(pl.col("_sec_ey_level").exp())
                .when(pl.col("_parent_y") == "section")
                .then(pl.col("_sec_ey_shape").exp())
                .otherwise(1.0)
                .alias("_raw_factor_y"),
                pl.when((pl.col("_parent_v") == "store") & (pl.col("_driver_mode_v") == "level_shape"))
                .then(pl.col("_store_ev_level").exp())
                .when(pl.col("_parent_v") == "store")
                .then(pl.col("_store_ev_shape").exp())
                .when((pl.col("_parent_v") == "section") & (pl.col("_driver_mode_v") == "level_shape"))
                .then(pl.col("_sec_ev_level").exp())
                .when(pl.col("_parent_v") == "section")
                .then(pl.col("_sec_ev_shape").exp())
                .otherwise(1.0)
                .alias("_raw_factor_v"),
                pl.when((pl.col("_parent_y") == "store") & (pl.col("_driver_mode_y") == "level_shape"))
                .then(pl.col("_store_level_factor_y"))
                .when((pl.col("_parent_y") == "section") & (pl.col("_driver_mode_y") == "level_shape"))
                .then(pl.col("_sec_level_factor_y"))
                .otherwise(1.0)
                .alias("_selected_level_factor_y"),
                pl.when((pl.col("_parent_v") == "store") & (pl.col("_driver_mode_v") == "level_shape"))
                .then(pl.col("_store_level_factor_v"))
                .when((pl.col("_parent_v") == "section") & (pl.col("_driver_mode_v") == "level_shape"))
                .then(pl.col("_sec_level_factor_v"))
                .otherwise(1.0)
                .alias("_selected_level_factor_v"),
            )
        )

        # Parent RLS is mandatory and applied at 100% strength.  shape_only
        # keeps the v11.4 mean-one contract; level_shape also carries the
        # causal parent block uplift selected from prior historical errors.
        rows = (
            rows.with_columns(
                pl.lit(1.0).alias("_effective_strength_y"),
                pl.lit(1.0).alias("_effective_strength_v"),
            )
            .with_columns(
                (
                    1.0
                    + pl.col("_effective_strength_y")
                    * (pl.col("_raw_factor_y") - 1.0)
                ).alias("_factor_y_selected"),
                (
                    1.0
                    + pl.col("_effective_strength_v")
                    * (pl.col("_raw_factor_v") - 1.0)
                ).alias("_factor_v_selected"),
            )
            .with_columns(
                pl.col("_factor_y_selected").log().alias("_ey"),
                pl.col("_factor_v_selected").log().alias("_ev"),
            )
            .with_columns(
                (
                    pl.col("_level_y")
                    * pl.col("_ey").exp()
                )
                .clip(lower_bound=0.0)
                .alias("_yhat_raw"),
                (
                    pl.col("_level_v")
                    * pl.col("_ev").exp()
                )
                .clip(lower_bound=0.0)
                .alias("_valuehat_raw"),
            )
            .with_columns(
                pl.col("_yhat_raw").round(0).alias("yhat"),
                pl.col("_valuehat_raw").round(2).alias("valuehat"),
                pl.col("_yhat_raw").alias("yhat_raw"),
                pl.col("_valuehat_raw").alias("valuehat_raw"),
                pl.concat_str(
                    [
                        pl.lit("leaf_ses_level_driver:"),
                        pl.col("_parent_y"), pl.lit(":"), pl.col("_driver_mode_y"),
                        pl.lit("@"), pl.col("_effective_strength_y").round(2).cast(pl.Utf8),
                        pl.lit("/"),
                        pl.col("_parent_v"), pl.lit(":"), pl.col("_driver_mode_v"),
                        pl.lit("@"), pl.col("_effective_strength_v").round(2).cast(pl.Utf8),
                    ]
                ).alias("modelo_seleccionado"),
                pl.col("_alpha_y").alias("ses_alpha_y"),
                pl.col("_alpha_v").alias("ses_alpha_value"),
                pl.col("_alpha_y_unconstrained").alias("ses_alpha_unconstrained_y"),
                pl.col("_alpha_v_unconstrained").alias("ses_alpha_unconstrained_value"),
                pl.col("_parent_y").alias("parent_model_y"),
                pl.col("_parent_v").alias("parent_model_value"),
                pl.col("_driver_mode_y").alias("parent_driver_mode_y"),
                pl.col("_driver_mode_v").alias("parent_driver_mode_value"),
                pl.col("_selected_level_factor_y").alias("driver_level_factor_y"),
                pl.col("_selected_level_factor_v").alias("driver_level_factor_value"),
                pl.col("_effective_strength_y").alias("driver_strength_y"),
                pl.col("_effective_strength_v").alias("driver_strength_value"),
                pl.col("_level_y").alias("ses_level_y"),
                pl.col("_level_v").alias("ses_level_value"),
                pl.col("_pure_ses_wmape_y").alias("pure_ses_wmape_y"),
                pl.col("_pure_ses_wmape_v").alias("pure_ses_wmape_value"),
                pl.col("_pure_ses_bias_y").alias("pure_ses_bias_y"),
                pl.col("_pure_ses_bias_v").alias("pure_ses_bias_value"),
                pl.col("_pure_ses_score_y").alias("pure_ses_score_y"),
                pl.col("_pure_ses_score_v").alias("pure_ses_score_value"),
                pl.col("_reference_y").alias("ses_stability_reference_y"),
                pl.col("_reference_v").alias("ses_stability_reference_value"),
                pl.col("_recent28_y").alias("ses_recent28_y"),
                pl.col("_recent28_v").alias("ses_recent28_value"),
                pl.col("_recent14_y").alias("ses_recent14_y"),
                pl.col("_recent14_v").alias("ses_recent14_value"),
                pl.col("_coverage_y").alias("ses_recent28_coverage_y"),
                pl.col("_coverage_v").alias("ses_recent28_coverage_value"),
                pl.col("_regime_anchor_y").alias("ses_regime_anchor_y"),
                pl.col("_regime_anchor_v").alias("ses_regime_anchor_value"),
                pl.col("_regime_class_y").alias("ses_regime_class_y"),
                pl.col("_regime_class_v").alias("ses_regime_class_value"),
                pl.col("_regime_ref_block").alias("ses_model_reference_block"),
                pl.col("_structural_y").alias("ses_structural_level_y"),
                pl.col("_structural_v").alias("ses_structural_level_value"),
                pl.col("_structural_history_blocks_y")
                .alias("ses_structural_history_blocks_y"),
                pl.col("_structural_history_blocks_v")
                .alias("ses_structural_history_blocks_value"),
                pl.col("_b0_y").alias("ses_block_b0_y"),
                pl.col("_b0_v").alias("ses_block_b0_value"),
                pl.col("_b1_y").alias("ses_block_b1_y"),
                pl.col("_b1_v").alias("ses_block_b1_value"),
                pl.col("_b2_y").alias("ses_block_b2_y"),
                pl.col("_b2_v").alias("ses_block_b2_value"),
                pl.col("_b3_y").alias("ses_block_b3_y"),
                pl.col("_b3_v").alias("ses_block_b3_value"),
                pl.col("_ses_guard_y").alias("ses_stability_guard_y"),
                pl.col("_ses_guard_v").alias("ses_stability_guard_value"),
                pl.col("_ses_guard_status_y").alias("ses_guard_status_y"),
                pl.col("_ses_guard_status_v").alias("ses_guard_status_value"),
                pl.col("_score_history_blocks_y").alias("ses_score_history_blocks_y"),
                pl.col("_score_history_blocks_v").alias("ses_score_history_blocks_value"),
                pl.lit("B0+B1+B2").alias("ses_score_window_y"),
                pl.lit("B0+B1+B2").alias("ses_score_window_value"),
                pl.col("_stable_pool_y").alias("ses_stable_pool_found_y"),
                pl.col("_stable_pool_v").alias("ses_stable_pool_found_value"),
                pl.col("_recent28_mean_y")
                .fill_null(0.0)
                .alias("recent28_mean_y"),
                pl.col("_recent28_mean_v")
                .fill_null(0.0)
                .alias("recent28_mean_value"),
                pl.when(pl.col("_recent28_mean_y") > 1e-12)
                .then(pl.col("_level_y") / pl.col("_recent28_mean_y"))
                .otherwise(None)
                .alias("ses_vs_recent28_ratio_y"),
                pl.when(pl.col("_recent28_mean_v") > 1e-12)
                .then(pl.col("_level_v") / pl.col("_recent28_mean_v"))
                .otherwise(None)
                .alias("ses_vs_recent28_ratio_value"),
                pl.col("_ey").alias("driver_effect"),
                pl.col("_ev").alias("driver_effect_value"),
                pl.col("_ey").exp().alias("driver_factor_y"),
                pl.col("_ev").exp().alias("driver_factor_value"),
                pl.lit(True).alias("rls_metric_eligible"),
                pl.col("_block").alias("rls_block"),
                (pl.col("_block") * block_days)
                .cast(pl.Int32)
                .alias("rls_train_days"),
            )
        )

        # v11.2: only high-risk leaves are reconsidered. Selection for OOS is
        # causal and uses the immediately preceding 28-day validation block.
        # The RLS factor itself is never changed; only the level source may
        # switch to deseasonalized SES or a robust deseasonalized mean.
        _t_fallback = time.perf_counter()
        rows = apply_robust_leaf_fallbacks(rows, horizons)
        logger.info(
            "⏱ Sección %s: fallbacks leaf robustos: %.1fs",
            section_id,
            time.perf_counter() - _t_fallback,
        )

        # v12 challenger: independent SKU-total + occurrence/store-share path.
        # The incumbent v11 forecast is retained. v12.3 uses shared true-hurdle
        # allocation plus robust closed-block selection with a recent-win guard.
        _t_v12 = time.perf_counter()
        rows = apply_v12_occurrence_share_challenger(
            rows=rows,
            train_obs=train_obs,
            oos_obs=oos_obs,
            horizons=horizons,
            section_id=str(section_id),
            n_jobs=self._workers(),
        )
        logger.info(
            "⏱ Sección %s: challenger v12.6 LGBM-shape+occurrence+share: %.1fs",
            section_id,
            time.perf_counter() - _t_v12,
        )

        # These temporary leaf columns were always excluded from the final
        # artifact. Drop them before audits so their buffers do not overlap
        # with the audit frames at peak RAM. Output schema is unchanged.
        drop_tmp = [
            c for c in (
                "_block", "_store_uid", "_v12_sku", "_v12_store_uid",
                "_store_ey_shape", "_store_ev_shape", "_store_ey_level", "_store_ev_level",
                "_sec_ey_shape", "_sec_ev_shape", "_sec_ey_level", "_sec_ev_level",
                "_store_level_factor_y", "_store_level_factor_v",
                "_sec_level_factor_y", "_sec_level_factor_v",
                "_candidate_y", "_candidate_v", "_level_y", "_level_v",
                "_parent_y", "_parent_v", "_driver_mode_y", "_driver_mode_v",
                "_selected_level_factor_y", "_selected_level_factor_v",
                "_alpha_y", "_alpha_v", "_ey", "_ev",
                "_recent28_mean_y", "_recent28_mean_v",
                "_yhat_raw", "_valuehat_raw",
                "_reference_y", "_reference_v", "_ses_guard_y", "_ses_guard_v",
                "_stable_pool_y", "_stable_pool_v",
            ) if c in rows.columns
        ]
        rows = rows.drop(drop_tmp)
        # Production invariant: every leaf forecast must use a real RLS
        # parent and full normalized driver strength. There is no "none" model.
        #
        # MEMORY-SAFE AUDIT (v12.9.12): never filter the full wide post-v12
        # frame. For section 1, 23,445 leaves × 56 target days = 1,312,920
        # rows, so one Float64 column alone requires exactly 10,503,360 bytes
        # (the failed allocation observed after the challenger). Projection
        # pushdown keeps each audit frame intentionally narrow.
        def _target_audit_rows(*cols: str) -> pl.DataFrame:
            keep = [c for c in cols if c in rows.columns]
            lf = (
                rows.lazy()
                .filter(pl.col("period_type").is_in(["out_sample", "forecast_only"]))
                .select(keep)
            )
            try:
                return lf.collect(engine="streaming")
            except TypeError:  # Polars compatibility
                return lf.collect(streaming=True)

        parent_audit_rows = _target_audit_rows(
            uid_col, "period_type",
            "parent_model_y", "parent_model_value",
            "parent_driver_mode_y", "parent_driver_mode_value",
            "driver_strength_y", "driver_strength_value",
        )
        bad_parent = parent_audit_rows.filter(
            ~pl.col("parent_model_y").is_in(["store", "section"])
            | ~pl.col("parent_model_value").is_in(["store", "section"])
            | ~pl.col("parent_driver_mode_y").is_in(["shape_only", "level_shape"])
            | ~pl.col("parent_driver_mode_value").is_in(["shape_only", "level_shape"])
            | ((pl.col("driver_strength_y") - 1.0).abs() > 1e-12)
            | ((pl.col("driver_strength_value") - 1.0).abs() > 1e-12)
        )
        if bad_parent.height:
            raise RuntimeError(
                "Leaf parent invariant violated: store/section RLS at 100% "
                f"is mandatory; examples={bad_parent.head(5).select(
                    uid_col,
                    'period_type',
                    'parent_model_y',
                    'parent_model_value',
                    'parent_driver_mode_y',
                    'parent_driver_mode_value',
                    'driver_strength_y',
                    'driver_strength_value',
                ).to_dicts()}"
            )
        del bad_parent, parent_audit_rows

        # Lightweight production invariants. Detailed descriptive diagnostics
        # were removed from the hot path in v11.2; diagnose_leaf/validate_v11
        # provide them on demand.
        block_audit_rows = _target_audit_rows(
            uid_col, "period_type", "ds",
            "driver_factor_y", "driver_factor_value",
            "driver_level_factor_y", "driver_level_factor_value",
            "sku_seasonal_multiplier_y", "sku_seasonal_multiplier_value",
            "parent_driver_mode_y", "parent_driver_mode_value",
            "leaf_model_family_y", "leaf_model_family_value",
            "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
            "v12_sku_forecast_y", "v12_sku_forecast_value",
            "v12_store_share_y", "v12_store_share_value",
            "ses_level_y", "ses_level_value", "yhat_raw", "valuehat_raw",
        )
        audit_blocks = (
            block_audit_rows.group_by([uid_col, "period_type"])
            .agg(
                pl.col("ds").min().alias("_block_start"),
                pl.col("ds").max().alias("_block_end"),
                pl.col("ds").n_unique().alias("_n_dates"),
                pl.col("driver_factor_y").mean().alias("_mean_factor_y"),
                pl.col("driver_factor_value").mean().alias("_mean_factor_v"),
                pl.col("driver_level_factor_y").first().alias("_expected_factor_y"),
                pl.col("driver_level_factor_value").first().alias("_expected_factor_v"),
                pl.col("sku_seasonal_multiplier_y").first().fill_null(1.0).alias("_seasonal_multiplier_y"),
                pl.col("sku_seasonal_multiplier_value").first().fill_null(1.0).alias("_seasonal_multiplier_v"),
                pl.col("sku_seasonal_multiplier_y").n_unique().alias("_n_seasonal_multiplier_y"),
                pl.col("sku_seasonal_multiplier_value").n_unique().alias("_n_seasonal_multiplier_v"),
                pl.col("parent_driver_mode_y").first().alias("_driver_mode_y"),
                pl.col("parent_driver_mode_value").first().alias("_driver_mode_v"),
                pl.col("leaf_model_family_y").first().fill_null("v11_ses_rls").alias("_family_y"),
                pl.col("leaf_model_family_value").first().fill_null("v11_ses_rls").alias("_family_v"),
                pl.col("v12_candidate_yhat_raw").mean().alias("_v12_mean_yhat"),
                pl.col("v12_candidate_valuehat_raw").mean().alias("_v12_mean_vhat"),
                pl.col("v12_sku_forecast_y").mean().alias("_v12_mean_sku_y"),
                pl.col("v12_sku_forecast_value").mean().alias("_v12_mean_sku_v"),
                pl.col("v12_store_share_y").mean().alias("_v12_mean_share_y"),
                pl.col("v12_store_share_value").mean().alias("_v12_mean_share_v"),
                pl.col("ses_level_y").first().alias("_ses_y"),
                pl.col("ses_level_value").first().alias("_ses_v"),
                pl.col("yhat_raw").mean().alias("_mean_yhat_raw"),
                pl.col("valuehat_raw").mean().alias("_mean_vhat_raw"),
            )
            .with_columns(
                (pl.col("_expected_factor_y") * pl.col("_seasonal_multiplier_y")).alias("_expected_forecast_factor_y"),
                (pl.col("_expected_factor_v") * pl.col("_seasonal_multiplier_v")).alias("_expected_forecast_factor_v"),
                pl.when(pl.col("_ses_y") > 1e-12)
                .then(pl.col("_mean_yhat_raw") / pl.col("_ses_y"))
                .otherwise(None)
                .alias("_forecast_ses_ratio_y"),
                pl.when(pl.col("_ses_v") > 1e-12)
                .then(pl.col("_mean_vhat_raw") / pl.col("_ses_v"))
                .otherwise(None)
                .alias("_forecast_ses_ratio_v"),
            )
        )
        metric_horizon_days = int(getattr(settings, "METRIC_HORIZON_DAYS", 28))
        bad_horizon_days = audit_blocks.filter(pl.col("_n_dates") != metric_horizon_days)
        if bad_horizon_days.height:
            examples = bad_horizon_days.head(5).select(
                uid_col, "period_type", "_block_start", "_block_end", "_n_dates"
            ).to_dicts()
            raise RuntimeError(
                "Leaf horizon invariant violated: OOS/forecast_only must contain "
                f"exactly {metric_horizon_days} dates. Examples={examples}"
            )

        factor_tol = float(getattr(settings, "LEAF_DRIVER_MEAN_TOLERANCE", 1e-6))
        bad_factor = audit_blocks.filter(
            ((pl.col("_mean_factor_y") - pl.col("_expected_factor_y")).abs() > factor_tol)
            | ((pl.col("_mean_factor_v") - pl.col("_expected_factor_v")).abs() > factor_tol)
        )
        if bad_factor.height:
            examples = bad_factor.head(5).select(
                uid_col, "period_type", "_driver_mode_y", "_mean_factor_y",
                "_expected_factor_y", "_driver_mode_v", "_mean_factor_v",
                "_expected_factor_v",
            ).to_dicts()
            raise RuntimeError(
                "Leaf driver invariant violated: horizon mean factor must match "
                f"the selected causal parent level factor. Examples={examples}"
            )

        bad_seasonal_multiplier = audit_blocks.filter(
            (pl.col("_n_seasonal_multiplier_y") != 1)
            | (pl.col("_n_seasonal_multiplier_v") != 1)
            | ~pl.col("_seasonal_multiplier_y").is_finite()
            | ~pl.col("_seasonal_multiplier_v").is_finite()
            | (pl.col("_seasonal_multiplier_y") <= 0.0)
            | (pl.col("_seasonal_multiplier_v") <= 0.0)
        )
        if bad_seasonal_multiplier.height:
            examples = bad_seasonal_multiplier.head(5).select(
                uid_col, "period_type", "_seasonal_multiplier_y",
                "_seasonal_multiplier_v", "_n_seasonal_multiplier_y",
                "_n_seasonal_multiplier_v",
            ).to_dicts()
            raise RuntimeError(
                "Leaf seasonal invariant violated: SKU seasonal multiplier must "
                f"be one finite positive constant per horizon. Examples={examples}"
            )

        # Multi-cadence-safe v11 identity.  With update cadences shorter than
        # the fixed 28-day OOS horizon, the effective SES state (and therefore
        # the forecast level) can legitimately vary inside that horizon.  The
        # old 28d-only audit compared a 28-day mean forecast against the first
        # SES state and produced false failures for 1d/7d/14d scenarios.
        # Validate the actual production equation row by row instead:
        #   yhat_raw = ses_level * driver_factor * seasonal_multiplier
        # This is exact for every supported update cadence and still detects
        # real corruption of the v11 leaf forecast path.
        level_identity_tol = float(
            getattr(settings, "LEAF_FORECAST_LEVEL_IDENTITY_TOLERANCE", 1e-7)
        )
        # v12.9.10 may apply a productive sparse Value uplift *after* the
        # selected v11/v12 baseline forecast has been built.  Therefore the
        # legacy family identity must audit the pre-rescue baseline, while a
        # separate invariant below audits the sparse write-back itself.
        value_identity_col = (
            "v12910_valuehat_raw_before_sparse"
            if "v12910_valuehat_raw_before_sparse" in rows.columns
            else "valuehat_raw"
        )
        v11_identity_rows = _target_audit_rows(
            uid_col, "period_type",
            "leaf_model_family_y", "leaf_model_family_value",
            "ses_level_y", "ses_level_value",
            "driver_factor_y", "driver_factor_value",
            "sku_seasonal_multiplier_y", "sku_seasonal_multiplier_value",
            "yhat_raw", "valuehat_raw", "v12910_valuehat_raw_before_sparse",
        ).filter(
            (pl.col("leaf_model_family_y") == "v11_ses_rls")
            | (pl.col("leaf_model_family_value") == "v11_ses_rls")
        ).with_columns(
            (
                pl.col("ses_level_y")
                * pl.col("driver_factor_y")
                * pl.col("sku_seasonal_multiplier_y").fill_null(1.0)
            ).clip(lower_bound=0.0).alias("_expected_v11_yhat_raw"),
            (
                pl.col("ses_level_value")
                * pl.col("driver_factor_value")
                * pl.col("sku_seasonal_multiplier_value").fill_null(1.0)
            ).clip(lower_bound=0.0).alias("_expected_v11_valuehat_raw"),
        )
        bad_forecast_level = v11_identity_rows.filter(
            (
                (pl.col("leaf_model_family_y") == "v11_ses_rls")
                & (
                    (pl.col("yhat_raw") - pl.col("_expected_v11_yhat_raw")).abs()
                    > level_identity_tol
                    * pl.max_horizontal(pl.col("_expected_v11_yhat_raw").abs(), pl.lit(1.0))
                )
            )
            | (
                (pl.col("leaf_model_family_value") == "v11_ses_rls")
                & (
                    (pl.col(value_identity_col) - pl.col("_expected_v11_valuehat_raw")).abs()
                    > level_identity_tol
                    * pl.max_horizontal(pl.col("_expected_v11_valuehat_raw").abs(), pl.lit(1.0))
                )
            )
        )
        if bad_forecast_level.height:
            examples = bad_forecast_level.head(5).select(
                uid_col, "period_type", "leaf_model_family_y",
                "ses_level_y", "driver_factor_y", "sku_seasonal_multiplier_y",
                "yhat_raw", "_expected_v11_yhat_raw",
                "leaf_model_family_value", "ses_level_value",
                "driver_factor_value", "sku_seasonal_multiplier_value",
                value_identity_col, "valuehat_raw", "_expected_v11_valuehat_raw",
            ).to_dicts()
            raise RuntimeError(
                "Leaf level invariant violated: v11 raw forecast must equal "
                "effective SES level × driver factor × selected SKU seasonal "
                f"multiplier row by row. Examples={examples}"
            )

        del bad_forecast_level, v11_identity_rows

        # v12.9.10 sparse-rescue identity.  The final Value raw forecast is
        # allowed to differ from its v11/v12 family baseline only through the
        # explicitly gated causal multiplier.  This keeps both invariants
        # simultaneously true: family baseline identity and final write-back.
        sparse10_required = {
            "v12910_valuehat_raw_before_sparse",
            "v12910_value_sparse_promoted",
            "v12910_value_sparse_applied",
            "v1299_value_sparse_candidate",
            "v1299_value_sparse_factor",
        }
        if sparse10_required.issubset(set(rows.columns)):
            expected_sparse_apply = (
                pl.col("v12910_value_sparse_promoted").fill_null(False)
                & pl.col("v1299_value_sparse_candidate").fill_null(False)
                & (pl.col("v1299_value_sparse_factor").fill_null(1.0) > 1.0)
            )
            expected_sparse_raw = pl.when(expected_sparse_apply).then(
                pl.col("v12910_valuehat_raw_before_sparse")
                * pl.col("v1299_value_sparse_factor").fill_null(1.0)
            ).otherwise(pl.col("v12910_valuehat_raw_before_sparse"))
            sparse_audit_rows = _target_audit_rows(
                uid_col, "period_type", "leaf_model_family_value",
                "v12910_value_sparse_promoted", "v1299_value_sparse_candidate",
                "v1299_value_sparse_factor", "v12910_value_sparse_applied",
                "v12910_valuehat_raw_before_sparse", "valuehat_raw",
            )
            bad_sparse_writeback = sparse_audit_rows.filter(
                (pl.col("v12910_value_sparse_applied").fill_null(False) != expected_sparse_apply)
                | (
                    (pl.col("valuehat_raw") - expected_sparse_raw).abs()
                    > level_identity_tol
                    * pl.max_horizontal(expected_sparse_raw.abs(), pl.lit(1.0))
                )
            )
            if bad_sparse_writeback.height:
                examples = bad_sparse_writeback.head(5).select(
                    uid_col, "period_type", "leaf_model_family_value",
                    "v12910_value_sparse_promoted", "v1299_value_sparse_candidate",
                    "v1299_value_sparse_factor", "v12910_value_sparse_applied",
                    "v12910_valuehat_raw_before_sparse", "valuehat_raw",
                ).to_dicts()
                raise RuntimeError(
                    "v12.9.10 sparse Value invariant violated: final raw forecast "
                    f"must equal pre-rescue baseline × gated factor. Examples={examples}"
                )
            del bad_sparse_writeback, sparse_audit_rows

        # Historical audit wording: parent level factor × selected SKU seasonal multiplier.
        # In multi-cadence mode the production check above is deliberately row-wise.

        # v12 candidate identity is independent from the v11 SES identity.
        # For every target row candidate = SKU total × normalized store share;
        # if v12 wins the causal A/B, the final raw forecast must equal it.
        v12_rows = _target_audit_rows(
            uid_col, "period_type", "ds",
            "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
            "v12_sku_forecast_y", "v12_sku_forecast_value",
            "v12_store_share_y", "v12_store_share_value",
            "v12_selected_y", "v12_selected_value",
            "yhat_raw", "valuehat_raw", "v12910_valuehat_raw_before_sparse",
            "v12_occurrence_gate_open_y", "v12_occurrence_gate_open_value",
            "v12_occurrence_gate_open_joint",
        ).filter(
            pl.col("v12_candidate_yhat_raw").is_not_null()
            | pl.col("v12_candidate_valuehat_raw").is_not_null()
        )
        if v12_rows.height:
            v12_tol = 1e-7
            bad_v12_identity = v12_rows.filter(
                (
                    (pl.col("v12_candidate_yhat_raw") - pl.col("v12_sku_forecast_y") * pl.col("v12_store_share_y")).abs()
                    > v12_tol * pl.max_horizontal(pl.col("v12_candidate_yhat_raw").abs(), pl.lit(1.0))
                )
                | (
                    (pl.col("v12_candidate_valuehat_raw") - pl.col("v12_sku_forecast_value") * pl.col("v12_store_share_value")).abs()
                    > v12_tol * pl.max_horizontal(pl.col("v12_candidate_valuehat_raw").abs(), pl.lit(1.0))
                )
                | (
                    pl.col("v12_selected_y")
                    & ((pl.col("yhat_raw") - pl.col("v12_candidate_yhat_raw")).abs()
                       > v12_tol * pl.max_horizontal(pl.col("yhat_raw").abs(), pl.lit(1.0)))
                )
                | (
                    pl.col("v12_selected_value")
                    & ((pl.col(value_identity_col) - pl.col("v12_candidate_valuehat_raw")).abs()
                       > v12_tol * pl.max_horizontal(pl.col(value_identity_col).abs(), pl.lit(1.0)))
                )
            )
            if bad_v12_identity.height:
                raise RuntimeError(
                    "v12 occurrence/share invariant violated; "
                    f"examples={bad_v12_identity.head(5).select(uid_col, 'period_type', 'v12_selected_y', 'v12_selected_value').to_dicts()}"
                )

            # v12.3 production invariants: one occurrence event and, by
            # default, one structural family decision for quantity + value.
            bad_shared_gate = v12_rows.filter(
                (pl.col("v12_occurrence_gate_open_y").fill_null(False)
                 != pl.col("v12_occurrence_gate_open_value").fill_null(False))
                | (pl.col("v12_occurrence_gate_open_y").fill_null(False)
                   != pl.col("v12_occurrence_gate_open_joint").fill_null(False))
            )
            if bad_shared_gate.height:
                raise RuntimeError(
                    "v12 shared occurrence gate invariant violated; "
                    f"examples={bad_shared_gate.head(5).select(uid_col, 'period_type').to_dicts()}"
                )
            if bool(getattr(settings, "V12_REQUIRE_JOINT_TARGET_WIN", True)):
                bad_joint_selection = v12_rows.filter(
                    pl.col("v12_selected_y").fill_null(False)
                    != pl.col("v12_selected_value").fill_null(False)
                )
                if bad_joint_selection.height:
                    raise RuntimeError(
                        "v12 joint target selection invariant violated; "
                        f"examples={bad_joint_selection.head(5).select(uid_col, 'period_type', 'v12_selected_y', 'v12_selected_value').to_dicts()}"
                    )

            share_sums = (
                v12_rows.with_columns(
                    pl.col(uid_col).str.extract(r"\|\|S:(.+)$", 1).alias("_v12_sku_audit")
                )
                .group_by(["_v12_sku_audit", "ds", "period_type"])
                .agg(
                    pl.col("v12_store_share_y").sum().alias("_sum_share_y"),
                    pl.col("v12_store_share_value").sum().alias("_sum_share_v"),
                )
            )
            bad_shares = share_sums.filter(
                ((pl.col("_sum_share_y") - 1.0).abs() > 1e-6)
                | ((pl.col("_sum_share_v") - 1.0).abs() > 1e-6)
            )
            if bad_shares.height:
                raise RuntimeError(
                    "v12 store-share invariant violated: shares must sum to 1 "
                    f"per SKU/date; examples={bad_shares.head(5).to_dicts()}"
                )

        del v12_rows, block_audit_rows, audit_blocks
        gc.collect()

        logger.info(
            "⏱ Sección %s: leaf 2-etapas SES puro (%d alphas) + parent-modes (%d): %.1fs",
            section_id,
            len(alpha_candidates),
            parent_params.height,
            time.perf_counter() - t_candidates,
        )

        # Warm-up is visual continuity only; never official metrics.
        warm = (
            train_obs.join(leaf_bounds, on=uid_col, how="left")
            .filter(pl.col("ds") <= pl.col("_leaf_warmup_end"))
            .join(state0, on=uid_col, how="left")
            .with_columns(
                pl.col("_level_y").clip(lower_bound=0.0).round(0).alias("yhat"),
                pl.col("_level_v").clip(lower_bound=0.0).round(2).alias("valuehat"),
                pl.lit("in_sample").alias("period_type"),
                pl.lit("leaf_warmup_calendar_mean").alias("modelo_seleccionado"),
                pl.lit(default_alpha).alias("ses_alpha_y"),
                pl.lit(default_alpha).alias("ses_alpha_value"),
                pl.lit("warmup").alias("parent_model_y"),
                pl.lit("warmup").alias("parent_model_value"),
                pl.lit("warmup").alias("parent_driver_mode_y"),
                pl.lit("warmup").alias("parent_driver_mode_value"),
                pl.lit(1.0).alias("driver_level_factor_y"),
                pl.lit(1.0).alias("driver_level_factor_value"),
                pl.col("_level_y").alias("ses_level_y"),
                pl.col("_level_v").alias("ses_level_value"),
                pl.lit(0.0).alias("driver_effect"),
                pl.lit(0.0).alias("driver_effect_value"),
                pl.lit(1.0).alias("driver_factor_y"),
                pl.lit(1.0).alias("driver_factor_value"),
                pl.lit(False).alias("rls_metric_eligible"),
                pl.lit(0).cast(pl.Int32).alias("rls_block"),
                pl.lit(warmup_days).cast(pl.Int32).alias("rls_train_days"),
            )
        )

        warm = warm.drop(
            [c for c in ("_level_y", "_level_v", "_leaf_start", "_leaf_warmup_end")
             if c in warm.columns]
        )

        if meta:
            warm = warm.with_columns([pl.lit(v).alias(k) for k, v in meta.items()])
            rows = rows.with_columns([pl.lit(v).alias(k) for k, v in meta.items()])

        logger.info(
            "Sección %s: leaf SES + RLS parent | %d hojas | alphas=%s | "
            "candidatos tienda/sección × modo(s) habilitados por wMAPE previo",
            section_id, ids.height, alpha_candidates,
        )
        logger.info(
            "⏱ Sección %s: FAST SKU+tienda total: %.1fs",
            section_id,
            time.perf_counter() - t_leaf_total,
        )
        out = pl.concat([warm, rows], how="diagonal_relaxed")
        if bool(getattr(settings, "LEAF_SORT_OUTPUT", False)):
            out = out.sort([uid_col, "ds"])
        return out
