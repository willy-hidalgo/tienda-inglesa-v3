"""RLS forecasting runner and leaf-level derivation."""
from __future__ import annotations
import logging
import numpy as np
import polars as pl
import settings
from rls_opt import RecursiveLeastSquaresRegression, RLSConstantPrior, RLSPrior
from app.forecasting.metrics import compute_wmape
from app.forecasting.robust_baseline import robust_baseline_forecast, validation_score as robust_validation_score, wmape_np

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

        train = res_df.filter(pl.col("period_type") == "in_sample")
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

        is_corr = pl.col("period_type").is_in(["out_sample", "forecast_only"])
        if "modelo_seleccionado" in out.columns:
            is_corr = is_corr & ~pl.col("modelo_seleccionado").cast(pl.Utf8).str.starts_with("baseline:")
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

    # ── SKU+tienda: efecto RLS seleccionado + SES causal ──────────────────
    @staticmethod
    def _apply_ses(
        df: pl.DataFrame, src_col: str, out_col: str, alpha: float
    ) -> pl.DataFrame:
        """Causal SES for train and frozen residual state for OOS/forecast.

        In-sample predictions are one-step-ahead: the state used at t contains
        residuals only through t-1.  For out_sample/forecast_only, the state is
        the *final raw SES state from train*, including the last train residual,
        and remains frozen for the entire horizon.  This avoids both leakage
        from OOS actuals and the former off-by-one state bug.
        """
        df = df.sort(["unique_id", "ds"])
        has_period = "period_type" in df.columns
        is_train = (
            (pl.col("period_type") == "in_sample") if has_period else pl.lit(True)
        )

        train = (
            df.filter(is_train)
            .with_columns(
                pl.col(src_col)
                .ewm_mean(alpha=alpha, adjust=False)
                .over("unique_id")
                .alias("_s_raw")
            )
            .with_columns(
                pl.col("_s_raw")
                .shift(1)
                .over("unique_id")
                .alias("_s_prev")
            )
            .with_columns(
                pl.coalesce(pl.col("_s_prev"), pl.col(src_col))
                .fill_null(0.0)
                .alias(out_col)
            )
        )

        if not has_period:
            return train.drop(["_s_raw", "_s_prev"]).sort(["unique_id", "ds"])

        # IMPORTANT: use raw final state, not the shifted one-step state.
        last_state = train.group_by("unique_id").agg(
            pl.col("_s_raw").last().alias("_last_s")
        )
        train = train.drop(["_s_raw", "_s_prev"])

        future = df.filter(~is_train)
        if future.height:
            future = (
                future.join(last_state, on="unique_id", how="left")
                .with_columns(pl.col("_last_s").fill_null(0.0).alias(out_col))
                .drop("_last_s")
            )
            return pl.concat([train, future], how="diagonal_relaxed").sort(
                ["unique_id", "ds"]
            )
        return train.sort(["unique_id", "ds"])

    @staticmethod
    def _apply_ses_tuned(
        df: pl.DataFrame,
        src_col: str,
        out_col: str,
        *,
        actual_col: str,
        base_log_cols: tuple[str, ...],
        default_alpha: float,
    ) -> pl.DataFrame:
        """Tune SES alpha per unique_id using only the tail of train.

        Candidate alphas are scored on one-step-ahead forecasts reconstructed in
        the original target space, using client WMAPE (y==0 excluded). OOS and
        forecast-only keep the final train state frozen, so there is no leakage.
        """
        if df.height == 0:
            return df
        out = df.sort(["unique_id", "ds"])
        uids = np.asarray(out["unique_id"].to_list(), dtype=object)
        periods = (
            np.asarray(out["period_type"].to_list(), dtype=object)
            if "period_type" in out.columns
            else np.full(out.height, "in_sample", dtype=object)
        )
        residual = out[src_col].to_numpy().astype(np.float64, copy=False)
        actual = out[actual_col].to_numpy().astype(np.float64, copy=False)
        base_log = np.zeros(out.height, dtype=np.float64)
        for col in base_log_cols:
            base_log += out[col].to_numpy().astype(np.float64, copy=False)

        candidates = tuple(
            float(a) for a in getattr(
                settings, "SES_ALPHA_CANDIDATES",
                (0.05, 0.10, 0.20, 0.35, 0.50, 0.70),
            )
            if 0.0 < float(a) <= 1.0
        )
        if not candidates:
            candidates = (float(default_alpha),)
        val_days = int(getattr(settings, "SES_TUNE_VALIDATION_DAYS", 28))
        min_points = int(getattr(settings, "SES_TUNE_MIN_VALID_POINTS", 7))

        pred_state = np.zeros(out.height, dtype=np.float64)
        alpha_used = np.full(out.height, float(default_alpha), dtype=np.float64)

        change = np.empty(len(uids), dtype=bool)
        change[0] = True
        change[1:] = uids[1:] != uids[:-1]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(uids)]

        tuned = 0
        for a, b in zip(starts, ends):
            p = periods[a:b]
            train_idx = np.flatnonzero(p == "in_sample")
            future_idx = np.flatnonzero(p != "in_sample")
            if train_idx.size == 0:
                continue

            r = residual[a:b]
            y = actual[a:b]
            base = base_log[a:b]
            best_alpha = float(default_alpha)
            best_score = float("inf")

            # Score each alpha on recent train only.
            for alpha in candidates:
                states = np.empty(train_idx.size, dtype=np.float64)
                state = None
                for j, local_i in enumerate(train_idx):
                    rv = r[local_i]
                    if not np.isfinite(rv):
                        rv = 0.0
                    states[j] = rv if state is None else state
                    state = rv if state is None else alpha * rv + (1.0 - alpha) * state

                n_val = min(val_days, max(0, train_idx.size - 1))
                if n_val < 1:
                    continue
                val_local = train_idx[-n_val:]
                val_states = states[-n_val:]
                y_val = y[val_local]
                pred_val = np.maximum(
                    0.0,
                    np.expm1(base[val_local] + val_states),
                )
                mask = np.isfinite(y_val) & np.isfinite(pred_val) & (y_val != 0.0)
                if int(mask.sum()) < min_points:
                    continue
                denom = float(np.abs(y_val[mask]).sum())
                if denom <= 0:
                    continue
                score = float(np.abs(y_val[mask] - pred_val[mask]).sum() / denom)
                if score < best_score:
                    best_score = score
                    best_alpha = float(alpha)

            # Generate train one-step states and freeze the final train state.
            state = None
            for local_i in train_idx:
                rv = r[local_i]
                if not np.isfinite(rv):
                    rv = 0.0
                pred_state[a + local_i] = rv if state is None else state
                state = rv if state is None else best_alpha * rv + (1.0 - best_alpha) * state
            frozen = 0.0 if state is None or not np.isfinite(state) else float(state)
            for local_i in future_idx:
                pred_state[a + local_i] = frozen
            alpha_used[a:b] = best_alpha
            if abs(best_alpha - float(default_alpha)) > 1e-12:
                tuned += 1

        result = out.with_columns(
            pl.Series(out_col, pred_state),
            pl.Series(f"{out_col}_alpha", alpha_used),
        )
        if tuned:
            logger.info(
                "SES adaptativo %s: %d series con alpha distinto del default %.2f",
                actual_col, tuned, float(default_alpha),
            )
        return result


    @staticmethod
    def _select_model_wmape(
        y: np.ndarray,
        yhat_sec: np.ndarray,
        yhat_sto: np.ndarray,
        uid: np.ndarray,
        is_train: np.ndarray,
    ) -> dict[str, str]:
        """WMAPE/BIAS solo in-sample → {unique_id: 'seccion'|'tienda'}.

        Vectorizado con reduceat sobre grupos contiguos de uid (sin loop
        Python por punto; solo un paso por serie).
        """
        selection: dict[str, str] = {}
        mask = is_train & np.isfinite(y) & (y != 0)
        if not np.any(mask):
            for u in np.unique(uid):
                selection[str(u)] = "seccion"
            return selection

        # Solo filas train con venta; ordenar por uid
        idx = np.flatnonzero(mask)
        uid_m = np.asarray(uid)[idx]
        order = np.argsort(uid_m, kind="mergesort")
        uid_s = uid_m[order]
        y_s = np.asarray(y, dtype=np.float64)[idx][order]
        ys_s = np.asarray(yhat_sec, dtype=np.float64)[idx][order]
        yt_s = np.asarray(yhat_sto, dtype=np.float64)[idx][order]

        # Bordes de grupo
        change = np.empty(len(uid_s), dtype=bool)
        change[0] = True
        change[1:] = uid_s[1:] != uid_s[:-1]
        starts = np.flatnonzero(change)
        # reduceat acumula por grupo
        sum_abs_y = np.add.reduceat(np.abs(y_s), starts)
        err_sec = np.add.reduceat(np.abs(y_s - ys_s), starts)
        err_sto = np.add.reduceat(np.abs(y_s - yt_s), starts)
        sum_y = np.add.reduceat(y_s, starts)
        bias_sec = np.add.reduceat(ys_s - y_s, starts)
        bias_sto = np.add.reduceat(yt_s - y_s, starts)

        uids_grp = uid_s[starts]
        for k, u in enumerate(uids_grp):
            denom = float(sum_abs_y[k])
            if denom == 0.0:
                selection[str(u)] = "seccion"
                continue
            es, et = float(err_sec[k]), float(err_sto[k])
            if et < es:
                selection[str(u)] = "tienda"
            elif et > es:
                selection[str(u)] = "seccion"
            else:
                sy = float(sum_y[k])
                if sy == 0.0:
                    selection[str(u)] = "seccion"
                else:
                    selection[str(u)] = (
                        "tienda"
                        if abs(float(bias_sto[k]) / sy) <= abs(float(bias_sec[k]) / sy)
                        else "seccion"
                    )

        # Series sin puntos train con venta → sección por defecto
        for u in np.unique(uid):
            su = str(u)
            if su not in selection:
                selection[su] = "seccion"
        return selection

    def _apply_leaf_guardrail(
        self, block: pl.DataFrame, section_id: str
    ) -> pl.DataFrame:
        """Replace/clip weak leaf OOS forecasts using a leakage-free baseline.

        A baseline is allowed to replace the RLS+SES candidate only when it
        beats the model by a configurable margin on the recent train tail.
        Regardless of replacement, future model forecasts are clipped to a
        broad multiple of the robust baseline to prevent numerical explosions.
        """
        if not bool(getattr(settings, "LEAF_BASELINE_GUARDRAIL", True)):
            return block
        required = {"unique_id", "ds", "period_type", "y", "yhat"}
        if block.height == 0 or not required.issubset(block.columns):
            return block

        method_map = getattr(settings, "LEAF_BASELINE_METHOD_BY_SECTION", {})
        fallback_method = str(method_map.get(str(section_id), "median_pos56"))
        methods = tuple(
            str(m)
            for m in getattr(
                settings,
                "LEAF_BASELINE_CANDIDATES",
                (fallback_method,),
            )
        ) or (fallback_method,)
        val_days = int(getattr(settings, "LEAF_BASELINE_VALIDATION_DAYS", 28))
        lookback = int(getattr(settings, "LEAF_BASELINE_LOOKBACK_DAYS", 56))
        min_points = int(getattr(settings, "LEAF_BASELINE_MIN_VALID_POINTS", 7))
        min_improvement = float(
            getattr(settings, "LEAF_BASELINE_MIN_IMPROVEMENT", 0.03)
        )
        clip_lo, clip_hi = tuple(
            getattr(settings, "LEAF_BASELINE_CLIP_RATIO", (0.35, 2.50))
        )

        out = block.sort(["unique_id", "ds"])
        uids = np.asarray(out["unique_id"].to_list(), dtype=object)
        periods = np.asarray(out["period_type"].to_list(), dtype=object)
        dates = out["ds"].to_list()
        y = out["y"].to_numpy().astype(np.float64, copy=False)
        yhat = out["yhat"].to_numpy().astype(np.float64, copy=True)
        model_labels = (
            np.asarray(out["modelo_seleccionado"].to_list(), dtype=object)
            if "modelo_seleccionado" in out.columns
            else np.full(len(out), "rls_ses", dtype=object)
        )

        if len(uids) == 0:
            return out
        change = np.empty(len(uids), dtype=bool)
        change[0] = True
        change[1:] = uids[1:] != uids[:-1]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(uids)]

        replaced = 0
        clipped = 0
        for a, b in zip(starts, ends):
            p = periods[a:b]
            train_local = np.flatnonzero(p == "in_sample")
            future_local = np.flatnonzero(
                (p == "out_sample") | (p == "forecast_only")
            )
            if train_local.size < max(val_days + 1, min_points + 1) or future_local.size == 0:
                continue

            yy = y[a:b]
            yh = yhat[a:b]
            dd = dates[a:b]
            y_train = yy[train_local]
            d_train = [dd[i] for i in train_local]
            n_val = min(val_days, len(y_train) - 1)
            if np.count_nonzero(y_train[-n_val:] > 0) < min_points:
                continue

            # Challenger adaptativo: elegir el mejor baseline por serie usando
            # exclusivamente la cola de train. No se fija un método por sección.
            scored_methods: list[tuple[float, str]] = []
            for candidate in methods:
                try:
                    score = robust_validation_score(
                        y_train,
                        d_train,
                        method=candidate,
                        validation_days=n_val,
                        lookback_days=lookback,
                    )
                except ValueError:
                    continue
                if np.isfinite(score):
                    scored_methods.append((float(score), candidate))
            if not scored_methods:
                continue
            base_score, method = min(scored_methods, key=lambda x: x[0])

            model_score = wmape_np(
                y_train[-n_val:], yh[train_local][-n_val:]
            )
            future_dates = [dd[i] for i in future_local]
            base_future = robust_baseline_forecast(
                y_train,
                d_train,
                future_dates,
                method=method,
                lookback_days=lookback,
            )

            target_idx = a + future_local
            baseline_wins = (
                np.isfinite(base_score)
                and np.isfinite(model_score)
                and base_score <= model_score * (1.0 - min_improvement)
            )

            # IMPORTANTE: nunca recortar un RLS que ganó la validación. La
            # versión anterior hacía clipping incondicional y podía aumentar
            # el WMAPE OOS. Solo el challenger ganador puede reemplazar.
            if baseline_wins:
                before = yhat[target_idx].copy()
                replacement = np.maximum(0.0, np.round(base_future, 0))
                yhat[target_idx] = replacement
                clipped += int(np.count_nonzero(before != replacement))
                model_labels[target_idx] = f"baseline:{method}"
                replaced += 1

        out = out.with_columns(
            pl.Series("yhat", yhat).clip(lower_bound=0.0).round(0),
            pl.Series("modelo_seleccionado", model_labels),
        )
        if replaced or clipped:
            logger.info(
                "Sección %s: guardrail adaptativo → %d series reemplazadas, %d puntos cambiados",
                section_id,
                replaced,
                clipped,
            )
        return out

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
              yhat = expm1(intercept + efecto + SES_causal(log-residuo))

        donde `residuo_log = log1p(y) − (intercept + efecto)`, e `intercept` +
        `efecto` provienen del modelo (sección o tienda) elegido por
        `_select_model_wmape`. `_apply_ses` calcula el SES **causal** (shift 1)
        sobre TODO el período con actuals (in_sample + out_sample) como una
        única serie continua: s(t) solo usa residuales hasta t-1. Así el
        pronóstico queda alineado en fecha con y(t) (sin adelanto de 1 día).

        `_yhat_rls`/`_valuehat_rls` se conservan solo como entrada de
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
        # ~150k × 76 × float32 ≈ 45 MiB por matriz; configurable
        max_rows = int(getattr(settings, "DERIVE_MAX_ROWS_PER_CHUNK", 150_000))

        def _process_sub(sub: pl.DataFrame, coef_store) -> pl.DataFrame | None:
            """Procesa un bloque (tienda completa o lote de SKUs)."""
            if sub.height == 0:
                return None

            X_y = np.ascontiguousarray(
                sub.select(driver_cols).to_numpy(), dtype=np.float32
            )
            X_p = np.ascontiguousarray(
                sub.select(driver_cols_price).to_numpy(), dtype=np.float32
            )
            y = sub["y"].to_numpy().astype(np.float64, copy=False)
            value = (
                sub["value"].to_numpy().astype(np.float64, copy=False)
                if "value" in sub.columns
                else np.zeros(len(y), dtype=np.float64)
            )
            uids = sub["unique_id"].to_numpy()
            if has_period:
                periods = sub["period_type"].to_numpy()
                is_train = periods == "in_sample"
            else:
                is_train = np.ones(len(y), dtype=bool)

            coef_y_sto = np.ascontiguousarray(coef_store[0], dtype=np.float32).ravel()
            coef_p_sto = np.ascontiguousarray(coef_store[1], dtype=np.float32).ravel()

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
            del y, value, use_sto, log_y, log_v, log_resid_y, log_resid_v

            block = self._apply_ses_tuned(
                block,
                "_y_neto",
                "_y_neto_hat",
                actual_col="y",
                base_log_cols=("_intercept_y", "driver_effect"),
                default_alpha=alpha,
            )
            block = self._apply_ses_tuned(
                block,
                "_v_neto",
                "_v_neto_hat",
                actual_col="value",
                base_log_cols=("_intercept_v", "driver_effect_value"),
                default_alpha=alpha,
            )

            if has_period:
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
            out_b = block.drop(drop_tmp)
            out_b = self._apply_leaf_guardrail(out_b, section_id)
            del block
            return out_b

        for si, store_uid in enumerate(store_uids):
            coef_store = store_coefs.get(store_uid)
            if coef_store is None:
                continue

            sub = panel.filter(pl.col("_store_uid") == store_uid)
            if sub.height == 0:
                continue

            if sub.height <= max_rows:
                batches = [sub]
            else:
                uid_list = sub.get_column("unique_id").unique().to_list()
                n_uid = len(uid_list)
                avg = max(1, sub.height // max(1, n_uid))
                batch_uids = max(1, max_rows // avg)
                batches = [
                    sub.filter(pl.col("unique_id").is_in(uid_list[i : i + batch_uids]))
                    for i in range(0, n_uid, batch_uids)
                ]
                logger.info(
                    "  tienda %s: %d filas / %d SKU → %d lotes (≤%d filas)",
                    store_uid,
                    sub.height,
                    n_uid,
                    len(batches),
                    max_rows,
                )

            for bi, chunk in enumerate(batches):
                out_b = None
                try:
                    out_b = _process_sub(chunk, coef_store)
                except Exception as e:
                    msg = str(e)
                    is_mem = isinstance(e, MemoryError) or (
                        "Unable to allocate" in msg
                        or "ArrayMemoryError" in type(e).__name__
                    )
                    if not is_mem:
                        raise
                    logger.warning(
                        "MemoryError %s lote %d (%d filas); fallback SKU a SKU: %s",
                        store_uid,
                        bi,
                        chunk.height,
                        msg[:120],
                    )
                    for one_uid in chunk.get_column("unique_id").unique().to_list():
                        one = chunk.filter(pl.col("unique_id") == one_uid)
                        try:
                            part = _process_sub(one, coef_store)
                        except Exception as e2:
                            logger.error("SKU %s omitido: %s", one_uid, e2)
                            part = None
                        if part is not None and part.height:
                            frames.append(part)
                        del one, part
                        gc.collect()
                if out_b is not None and out_b.height:
                    frames.append(out_b)
                del chunk, out_b
                if (bi + 1) % 3 == 0:
                    gc.collect()

            del sub, batches
            if (si + 1) % 2 == 0:
                gc.collect()
            if len(frames) >= 20:
                frames = [pl.concat(frames, how="vertical")]
                gc.collect()

        if not frames:
            return pl.DataFrame()

        out = pl.concat(frames, how="vertical")
        del frames
        gc.collect()
        return out
