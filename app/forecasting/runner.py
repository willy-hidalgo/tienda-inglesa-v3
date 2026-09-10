"""RLS forecasting runner and leaf-level derivation."""
from __future__ import annotations
import logging
import time
import threading
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
from app.forecasting.leaf_ses_rls import build_leaf_forecasts

logger = logging.getLogger(__name__)

class RLSForecastRunner:
    """
    RLS a nivel sección/tienda + modelo leaf SES+RLS coherente.

      - `fit_and_predict_sections`: RLS expanding para sección y tienda.
      - `predict_leaf_series`: SES a nivel SKU+tienda sobre magnitud positiva ajustada,
        aplicando exactamente un parent RLS (tienda o sección) elegido causalmente.
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
        n_jobs: int | None = None,
        optimization_diagnostics: bool = False,
        optimization_phase2: bool = False,
    ):
        self._driver_cols = driver_cols
        _price_exclude = {"asp", "edp", "discount"}
        self._driver_cols_value_base = [c for c in driver_cols if c not in _price_exclude]
        self._rmse_error = rmse_error
        self._forgetting_factor = forgetting_factor
        self._min_y_to_update = min_y_to_update
        # n_jobs ahora es Nº de THREADS (ver docstring de clase), no procesos.
        self._n_jobs = n_jobs
        self._optimization_diagnostics = bool(optimization_diagnostics)
        self._optimization_phase2 = bool(optimization_phase2)
        self._optimization_lock = threading.Lock()
        self._optimization_candidate_records: list[dict] = []
        self._optimization_driver_records: list[dict] = []
        self._optimization_driver_refit_records: list[dict] = []

    def optimization_candidate_frame(self) -> pl.DataFrame:
        """Small RLS candidate audit; never participates in production selection."""
        with self._optimization_lock:
            rows = list(self._optimization_candidate_records)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def optimization_driver_frame(self) -> pl.DataFrame:
        """Fast driver-contribution screening for the selected RLS path."""
        with self._optimization_lock:
            rows = list(self._optimization_driver_records)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def optimization_driver_refit_frame(self) -> pl.DataFrame:
        """Exact same-family RLS driver refit tests (Phase 2 only)."""
        with self._optimization_lock:
            rows = list(self._optimization_driver_refit_records)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    @staticmethod
    def _driver_groups(columns: list[str]) -> dict[str, list[int]]:
        """Group only EXISTING RLS drivers; this function never creates drivers."""
        weekdays = {"Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
        months = {"Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
        price = {"asp", "edp", "discount"}
        groups: dict[str, list[int]] = {}
        for idx, col in enumerate(columns):
            if col == "intercept":
                continue
            if col in price:
                group = "price"
            elif col in weekdays:
                group = "weekday"
            elif col in months:
                group = "month"
            else:
                group = "holiday_other"
            groups.setdefault(group, []).append(idx)
        return groups

    def _driver_columns_for_node(
        self,
        uid: str,
        target: str,
        *,
        force_value_price: bool | None = None,
        ignore_exclusion_groups: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Return the productive EXISTING-driver design for one RLS node/target.

        v13.2.x may manually exclude a driver *group* only after closed-history
        evidence plus holdout validation.  This helper never creates a driver;
        it only selects columns already present in ``self._driver_cols``.

        ``ignore_exclusion_groups`` is diagnostic-only: Phase 2 can add a
        promoted group back and verify whether the earlier removal still holds.
        """
        uid = str(uid)
        target = str(target)
        if target == "Valor ($)":
            if force_value_price is None:
                price_nodes = {
                    str(x) for x in getattr(settings, "RLS_VALUE_PRICE_NODE_IDS", ())
                }
                use_price = uid in price_nodes
            else:
                use_price = bool(force_value_price)
            columns = list(self._driver_cols if use_price else self._driver_cols_value_base)
        else:
            columns = list(self._driver_cols)

        exclusions_cfg = getattr(settings, "RLS_DRIVER_GROUP_EXCLUSIONS", {}) or {}
        target_cfg = exclusions_cfg.get(uid, {}) if isinstance(exclusions_cfg, dict) else {}
        excluded = {str(x) for x in target_cfg.get(target, ())}
        excluded.difference_update(str(x) for x in ignore_exclusion_groups)
        if not excluded:
            return columns

        grouped = self._driver_groups(columns)
        remove_idx = {idx for group in excluded for idx in grouped.get(group, ())}
        return [col for idx, col in enumerate(columns) if idx not in remove_idx]

    def _append_optimization_records(
        self,
        candidate_rows: list[dict],
        driver_rows: list[dict],
        driver_refit_rows: list[dict] | None = None,
    ) -> None:
        if not self._optimization_diagnostics:
            return
        with self._optimization_lock:
            self._optimization_candidate_records.extend(candidate_rows)
            self._optimization_driver_records.extend(driver_rows)
            if driver_refit_rows:
                self._optimization_driver_refit_records.extend(driver_refit_rows)

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

    @staticmethod
    def _normalize_partition_dict(d: dict) -> dict:
        return {(k[0] if isinstance(k, tuple) else k): v for k, v in d.items()}

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
        source_period = np.asarray(actual["_source_period"].to_list(), dtype=object)
        n = actual.height
        if n == 0:
            return [], None

        seed_n = min(max(block_days, int(getattr(settings, "RLS_INITIAL_SEED_DAYS", 28))), n)
        base = actual.drop("_source_period")
        level_name = "seccion" if "||" not in uid else "tienda"
        section_name = str(uid).split("||", 1)[0]
        # Same RLS family, node-specific EXISTING-driver design.  Promotions
        # may add/remove only validated groups; the model class never changes.
        unit_driver_cols = self._driver_columns_for_node(uid, "Unidades")
        value_driver_cols = self._driver_columns_for_node(uid, "Valor ($)")
        y = base["y"].to_numpy().astype(np.float64, copy=False)
        v = base["value"].to_numpy().astype(np.float64, copy=False)
        log_y = np.log1p(np.clip(y, 0.0, None))
        log_v = np.log1p(np.clip(v, 0.0, None))
        Xy_base = self._finite_matrix(base, unit_driver_cols, uid=uid, stage="RLS rolling fit y")
        Xv_base = self._finite_matrix(base, value_driver_cols, uid=uid, stage="RLS rolling fit value")

        ar_enabled = bool(getattr(settings, "RLS_AUTOREGRESSIVE_DRIVERS", True))
        mode_candidates = tuple(getattr(settings, "RLS_DYNAMICS_CANDIDATES", ("base", "ar" if ar_enabled else "base")))
        mode_candidates = tuple(dict.fromkeys(m for m in mode_candidates if m in {"base", "ar"} and (m != "ar" or ar_enabled))) or ("base",)
        productive_lambdas = tuple(sorted({float(x) for x in getattr(settings, "RLS_FORGETTING_FACTOR_CANDIDATES", (self._forgetting_factor,)) if 0.0 < float(x) <= 1.0} | {float(self._forgetting_factor)}))
        extra_lambdas = (
            tuple(
                float(x)
                for x in getattr(settings, "STAT_OPT_RLS_EXTRA_LAMBDAS", ())
                if 0.0 < float(x) <= 1.0
            )
            if self._optimization_phase2 and self._optimization_diagnostics
            else ()
        )
        diagnostic_lambdas = tuple(sorted(set(productive_lambdas) | set(extra_lambdas)))
        default_lambda = float(self._forgetting_factor)
        default_mode = str(getattr(settings, "RLS_DEFAULT_DYNAMICS", "base"))
        if default_mode not in mode_candidates:
            default_mode = mode_candidates[0]
        candidates = tuple((mode, lam) for mode in mode_candidates for lam in productive_lambdas)
        diagnostic_candidates = tuple((mode, lam) for mode in mode_candidates for lam in diagnostic_lambdas)
        default_candidate = (default_mode, default_lambda)

        paths_y: dict[tuple[str, float], np.ndarray] = {}
        paths_v: dict[tuple[str, float], np.ndarray] = {}
        for mode, lam in diagnostic_candidates:
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
        # Driver contribution on the log1p scale.  The leaf SES supplies the
        # baseline/intercept; these effects contain every RLS term except the
        # intercept, so the same parent model can be applied at SKU+store
        # level without importing the parent's absolute sales level.
        driver_effect_y = np.zeros(n, dtype=np.float64)
        driver_effect_v = np.zeros(n, dtype=np.float64)
        intercept_y_idx = unit_driver_cols.index("intercept") if "intercept" in unit_driver_cols else None
        intercept_v_idx = value_driver_cols.index("intercept") if "intercept" in value_driver_cols else None
        cum_ae_y = {c: 0.0 for c in candidates}; cum_den_y = {c: 0.0 for c in candidates}
        cum_ae_v = {c: 0.0 for c in candidates}; cum_den_v = {c: 0.0 for c in candidates}

        def choose(cae, cden):
            scored = [(cae[c] / cden[c], c) for c in candidates if cden[c] > 0]
            return min(scored, key=lambda z: (z[0], z[1][0], z[1][1]))[1] if scored else default_candidate

        opt_candidate_rows: list[dict] = []
        opt_driver_rows: list[dict] = []
        opt_driver_refit_rows: list[dict] = []

        def _score(mask, actual_values, pred_values):
            if not mask.any():
                return None
            yy = actual_values[mask]
            pp = pred_values[mask]
            den = float(np.abs(yy).sum())
            if den <= 0.0:
                return None
            err = pp - yy
            return (
                float(np.abs(err).sum()),
                den,
                float(err.sum()),
                int(mask.sum()),
            )

        def _record_candidate_block(start, end, block_no, origin_date, chosen_y, chosen_v, cand_y, cand_v):
            if not self._optimization_diagnostics:
                return
            for period_name in ("in_sample", "out_sample"):
                period_mask = source_period[start:end] == period_name
                valid_y = period_mask & np.isfinite(y[start:end]) & (y[start:end] > 0.0)
                # Contractual support is y>0 for BOTH targets. Value itself
                # may be zero and must still contribute forecast error.
                valid_v = valid_y & np.isfinite(v[start:end])
                for c in diagnostic_candidates:
                    for target, actual_values, predictions, mask, chosen in (
                        ("Unidades", y[start:end], cand_y[c], valid_y, chosen_y),
                        ("Valor ($)", v[start:end], cand_v[c], valid_v, chosen_v),
                    ):
                        scored = _score(mask, actual_values, predictions)
                        if scored is None:
                            continue
                        ae, den, se, npts = scored
                        opt_candidate_rows.append({
                            "record_type": "rls_candidate",
                            "section": section_name,
                            "node_level": level_name,
                            "unique_id": str(uid),
                            "period_type": period_name,
                            "block": int(block_no),
                            "forecast_origin": origin_date,
                            "target": target,
                            "dynamics": str(c[0]),
                            "lambda": float(c[1]),
                            "productive_candidate": bool(c in candidates),
                            "selected_by_prior": bool(c == chosen),
                            "sum_abs_error": ae,
                            "sum_abs_y": den,
                            "sum_signed_error": se,
                            "n_points": npts,
                            "wmape": ae / den,
                            "bias": se / den,
                        })

        def _record_driver_screen(
            start, end, block_no, origin_date, chosen, pred, actual_values,
            Xbase, coef, columns, target, base_no_ar_pred=None, support_values=None
        ):
            if not self._optimization_diagnostics:
                return
            for period_name in ("in_sample", "out_sample"):
                period_mask = source_period[start:end] == period_name
                support = actual_values if support_values is None else support_values
                valid = (
                    period_mask
                    & np.isfinite(support[start:end])
                    & (support[start:end] > 0.0)
                    & np.isfinite(actual_values[start:end])
                )
                full = _score(valid, actual_values[start:end], pred)
                if full is None:
                    continue
                full_ae, den, full_se, npts = full
                log_full = np.log1p(np.clip(pred, 0.0, None))
                for group, idxs in self._driver_groups(columns).items():
                    if not idxs:
                        continue
                    contribution = Xbase[start:end][:, idxs] @ coef[idxs]
                    zeroed = np.expm1(np.clip(log_full - contribution, -20.0, 20.0))
                    zeroed = np.clip(zeroed, 0.0, None)
                    zscore = _score(valid, actual_values[start:end], zeroed)
                    if zscore is None:
                        continue
                    z_ae, _, z_se, _ = zscore
                    opt_driver_rows.append({
                        "record_type": "driver_screen",
                        "audit_method": "zero_contribution_no_refit",
                        "section": section_name,
                        "node_level": level_name,
                        "unique_id": str(uid),
                        "period_type": period_name,
                        "block": int(block_no),
                        "forecast_origin": origin_date,
                        "target": target,
                        "dynamics": str(chosen[0]),
                        "lambda": float(chosen[1]),
                        "driver_group": group,
                        "sum_abs_y": den,
                        "n_points": npts,
                        "wmape_full": full_ae / den,
                        "bias_full": full_se / den,
                        "wmape_zeroed": z_ae / den,
                        "bias_zeroed": z_se / den,
                        "wmape_delta_if_removed": (z_ae - full_ae) / den,
                        "mean_abs_log_contribution": float(np.mean(np.abs(contribution))) if len(contribution) else 0.0,
                    })

                # AR is the only group for which an exact same-family refit is
                # already available at zero extra fitting cost: the runner has
                # fitted the base candidate with the same lambda.
                if chosen[0] == "ar" and base_no_ar_pred is not None:
                    zscore = _score(valid, actual_values[start:end], base_no_ar_pred)
                    if zscore is not None:
                        z_ae, _, z_se, _ = zscore
                        log_base = np.log1p(np.clip(base_no_ar_pred, 0.0, None))
                        opt_driver_rows.append({
                            "record_type": "driver_screen",
                            "audit_method": "refit_same_rls_without_ar",
                            "section": section_name,
                            "node_level": level_name,
                            "unique_id": str(uid),
                            "period_type": period_name,
                            "block": int(block_no),
                            "forecast_origin": origin_date,
                            "target": target,
                            "dynamics": str(chosen[0]),
                            "lambda": float(chosen[1]),
                            "driver_group": "autoregressive",
                            "sum_abs_y": den,
                            "n_points": npts,
                            "wmape_full": full_ae / den,
                            "bias_full": full_se / den,
                            "wmape_zeroed": z_ae / den,
                            "bias_zeroed": z_se / den,
                            "wmape_delta_if_removed": (z_ae - full_ae) / den,
                            "mean_abs_log_contribution": float(np.mean(np.abs(log_full - log_base))),
                        })

        bno = 1
        frozen_choice_y = None
        frozen_choice_v = None
        s = seed_n
        while s < n:
            # v13.2.16: a block may never straddle in-sample/OOS. The OOS
            # boundary is a true fixed forecast origin for every cadence.
            current_period = source_period[s]
            period_end = s + 1
            while period_end < n and source_period[period_end] == current_period:
                period_end += 1
            e = min(s + block_days, period_end)
            boundary = s - 1
            candidate_y = choose(cum_ae_y, cum_den_y)
            candidate_v = choose(cum_ae_v, cum_den_v)
            # v13.2.10: dynamics/lambda are snapshotted at the first OOS
            # origin. OOS actuals can update the recursive coefficient state,
            # but they can never re-select the RLS configuration.
            block_has_history = bool(np.any(source_period[s:e] == "in_sample"))
            if block_has_history:
                chosen_y, chosen_v = candidate_y, candidate_v
            else:
                if frozen_choice_y is None:
                    frozen_choice_y, frozen_choice_v = candidate_y, candidate_v
                chosen_y, chosen_v = frozen_choice_y, frozen_choice_v
            cand_y = {}; cand_v = {}
            for c in diagnostic_candidates:
                mode, lam = c
                # Contrato v13: el forecast de cada bloque usa exclusivamente
                # el estado disponible en SU origen. Un bloque OOS cerrado puede
                # avanzar el estado para el siguiente origen, pero nunca cambia
                # lambda/dynamics seleccionados ni su propio forecast.
                cand_y[c] = predict_block(Xy_base, paths_y[c][boundary], log_y, s, e, mode)
                cand_v[c] = predict_block(Xv_base, paths_v[c][boundary], log_v, s, e, mode)
            yh[s:e] = np.round(cand_y[chosen_y], 0); vh[s:e] = np.round(cand_v[chosen_v], 2)
            # Exact RLS non-intercept contribution.  With a log1p link:
            #   log(1 + forecast) = intercept + driver_effect
            # The leaf model later combines its SES baseline with this same
            # driver_effect, which keeps section/store and leaf semantics
            # identical for every update cadence (including 1-day blocks).
            cy_coef = paths_y[chosen_y][boundary]
            cv_coef = paths_v[chosen_v][boundary]
            lp_y = np.log1p(np.clip(cand_y[chosen_y], 0.0, None))
            lp_v = np.log1p(np.clip(cand_v[chosen_v], 0.0, None))
            if intercept_y_idx is not None:
                lp_y = lp_y - Xy_base[s:e, intercept_y_idx] * cy_coef[intercept_y_idx]
            if intercept_v_idx is not None:
                lp_v = lp_v - Xv_base[s:e, intercept_v_idx] * cv_coef[intercept_v_idx]
            driver_effect_y[s:e] = lp_y
            driver_effect_v[s:e] = lp_v
            eligible[s:e] = True; block[s:e] = bno; train_days[s:e] = s
            mode_y[s:e], lambda_y[s:e] = chosen_y; mode_v[s:e], lambda_v[s:e] = chosen_v
            origin_date = base["ds"][boundary]
            origin[s:e] = [origin_date] * (e - s)
            _record_candidate_block(
                s, e, bno, origin_date, chosen_y, chosen_v, cand_y, cand_v
            )
            _record_driver_screen(
                s, e, bno, origin_date, chosen_y, cand_y[chosen_y], y,
                Xy_base, cy_coef, unit_driver_cols, "Unidades",
                cand_y.get(("base", chosen_y[1])), support_values=y,
            )
            _record_driver_screen(
                s, e, bno, origin_date, chosen_v, cand_v[chosen_v], v,
                Xv_base, cv_coef, value_driver_cols, "Valor ($)",
                cand_v.get(("base", chosen_v[1])), support_values=y,
            )
            # Productive dynamics/lambda selection is HISTORY-ONLY. OOS is a
            # holdout: its actuals may advance the recursive RLS state for a
            # later operational origin, but its errors never enter the
            # cumulative scores that choose dynamics/lambda. This keeps the
            # selection contract identical across 1d/7d/14d/28d.
            history_mask = source_period[s:e] == "in_sample"
            # Official selection support: y>0. Value uses the same sale-event
            # support so both targets follow the client's contractual metric.
            valid_y = history_mask & np.isfinite(y[s:e]) & (y[s:e] > 0.0)
            valid_v = valid_y & np.isfinite(v[s:e])
            for c in candidates:
                if valid_y.any():
                    cum_ae_y[c] += float(np.abs(y[s:e][valid_y] - cand_y[c][valid_y]).sum())
                    cum_den_y[c] += float(np.abs(y[s:e][valid_y]).sum())
                if valid_v.any():
                    cum_ae_v[c] += float(np.abs(v[s:e][valid_v] - cand_v[c][valid_v]).sum())
                    cum_den_v[c] += float(np.abs(v[s:e][valid_v]).sum())
            bno += 1
            s = e

        # ── Phase 2: exact same-family refit of existing driver groups ─────
        # This runs only on explicit optimization diagnostics.  It does NOT
        # alter the productive candidate set or any forecast emitted above.
        if self._optimization_phase2 and self._optimization_diagnostics:
            t_refit = time.perf_counter()

            def _best_historical_candidate(target_name: str):
                scores: dict[tuple[str, float], tuple[float, float]] = {}
                for row in opt_candidate_rows:
                    if row.get("period_type") != "in_sample" or row.get("target") != target_name:
                        continue
                    key = (str(row["dynamics"]), float(row["lambda"]))
                    ae0, den0 = scores.get(key, (0.0, 0.0))
                    scores[key] = (ae0 + float(row["sum_abs_error"]), den0 + float(row["sum_abs_y"]))
                valid = [
                    (ae / den, key)
                    for key, (ae, den) in scores.items()
                    if den > 0.0
                ]
                return min(valid, key=lambda z: (z[0], z[1][0], z[1][1]))[1] if valid else default_candidate

            def _fit_refit_path(
                Xbase: np.ndarray,
                log_target: np.ndarray,
                *,
                columns: list[str],
                candidate: tuple[str, float],
                min_y: float,
                remove_group: str | None = None,
            ):
                groups = self._driver_groups(columns)
                removed = set(groups.get(remove_group, [])) if remove_group else set()
                keep = np.asarray(
                    [idx for idx in range(len(columns)) if idx not in removed],
                    dtype=np.int64,
                )
                if keep.size == 0:
                    return None
                Xred = np.ascontiguousarray(Xbase[:, keep], dtype=np.float64)
                mode, lam = candidate
                Xfit = (
                    np.column_stack([Xred, self._ar_training_matrix(log_target)])
                    if mode == "ar"
                    else Xred
                )
                model = self._new_rls(
                    min_y, return_all_coefs=True, forgetting_factor=float(lam)
                )
                model.fit(
                    x=Xfit,
                    y=log_target,
                    priors=self._default_priors(Xfit.shape[1]),
                    seed_n_obs=seed_n,
                )
                return Xred, np.asarray(model.all_coef_[0], dtype=np.float64)

            def _record_refit_blocks(
                *,
                target_name: str,
                actual_values: np.ndarray,
                history_log: np.ndarray,
                full_X: np.ndarray,
                full_paths: dict,
                candidate: tuple[str, float],
                challenger_X: np.ndarray,
                challenger_path: np.ndarray,
                driver_group: str,
                test_action: str,
                challenger_mode: str | None = None,
            ) -> None:
                mode, lam = candidate
                challenger_mode = mode if challenger_mode is None else str(challenger_mode)
                block_no = 1
                for start2 in range(seed_n, n, block_days):
                    end2 = min(start2 + block_days, n)
                    boundary2 = start2 - 1
                    full_pred = predict_block(
                        full_X, full_paths[candidate][boundary2], history_log,
                        start2, end2, mode,
                    )
                    challenger_pred = predict_block(
                        challenger_X, challenger_path[boundary2], history_log,
                        start2, end2, challenger_mode,
                    )
                    for period_name in ("in_sample", "out_sample"):
                        period_mask = source_period[start2:end2] == period_name
                        if target_name == "Unidades":
                            valid = (
                                period_mask
                                & np.isfinite(actual_values[start2:end2])
                                & (actual_values[start2:end2] > 0.0)
                            )
                        else:
                            valid = (
                                period_mask
                                & np.isfinite(y[start2:end2])
                                & (y[start2:end2] > 0.0)
                                & np.isfinite(actual_values[start2:end2])
                            )
                        full_sc = _score(valid, actual_values[start2:end2], full_pred)
                        ch_sc = _score(valid, actual_values[start2:end2], challenger_pred)
                        if full_sc is None or ch_sc is None:
                            continue
                        full_ae, den, full_se, npts = full_sc
                        ch_ae, _, ch_se, _ = ch_sc
                        opt_driver_refit_rows.append({
                            "record_type": "driver_refit",
                            "audit_method": "refit_same_rls",
                            "section": section_name,
                            "node_level": level_name,
                            "unique_id": str(uid),
                            "period_type": period_name,
                            "block": int(block_no),
                            "forecast_origin": base["ds"][boundary2],
                            "target": target_name,
                            "dynamics": str(mode),
                            "lambda": float(lam),
                            "driver_group": driver_group,
                            "test_action": test_action,
                            "sum_abs_y": den,
                            "n_points": npts,
                            "wmape_current": full_ae / den,
                            "bias_current": full_se / den,
                            "wmape_challenger": ch_ae / den,
                            "bias_challenger": ch_se / den,
                            # Positive = challenger improves current same-family RLS.
                            "wmape_improvement": (full_ae - ch_ae) / den,
                            "abs_bias_change": abs(ch_se / den) - abs(full_se / den),
                        })
                    block_no += 1

            best_y_hist = _best_historical_candidate("Unidades")
            best_v_hist = _best_historical_candidate("Valor ($)")
            allowed_groups = set(
                str(x) for x in getattr(
                    settings, "STAT_OPT_REFIT_DRIVER_GROUPS",
                    ("weekday", "month", "holiday_other", "price"),
                )
            )

            for target_name, actual_values, history_log, Xbase, columns, full_paths, best_candidate, min_y in (
                ("Unidades", y, log_y, Xy_base, unit_driver_cols, paths_y, best_y_hist, self._min_y_to_update),
                ("Valor ($)", v, log_v, Xv_base, value_driver_cols, paths_v, best_v_hist, 1e-8),
            ):
                groups = self._driver_groups(columns)
                for group in sorted(set(groups) & allowed_groups):
                    fitted = _fit_refit_path(
                        Xbase, history_log, columns=columns, candidate=best_candidate,
                        min_y=min_y, remove_group=group,
                    )
                    if fitted is None:
                        continue
                    Xred, path_red = fitted
                    _record_refit_blocks(
                        target_name=target_name, actual_values=actual_values,
                        history_log=history_log, full_X=Xbase, full_paths=full_paths,
                        candidate=best_candidate, challenger_X=Xred,
                        challenger_path=path_red, driver_group=group,
                        test_action="remove",
                    )

                # Exact AR removal is already represented by the fitted base
                # candidate with the same lambda; no extra fit is needed.
                if best_candidate[0] == "ar" and ("base", best_candidate[1]) in full_paths:
                    _record_refit_blocks(
                        target_name=target_name, actual_values=actual_values,
                        history_log=history_log, full_X=Xbase, full_paths=full_paths,
                        candidate=best_candidate, challenger_X=Xbase,
                        challenger_path=full_paths[("base", best_candidate[1])],
                        driver_group="autoregressive", test_action="remove",
                        challenger_mode="base",
                    )

            # A manually promoted exclusion remains auditable. Phase 2 adds
            # the SAME existing group back, fits the SAME RLS and can therefore
            # veto the promotion in a later cycle if the evidence reverses.
            exclusions_cfg = getattr(settings, "RLS_DRIVER_GROUP_EXCLUSIONS", {}) or {}
            node_exclusions = exclusions_cfg.get(str(uid), {}) if isinstance(exclusions_cfg, dict) else {}
            for target_name, actual_values, history_log, current_X, current_cols, full_paths, best_candidate, min_y in (
                ("Unidades", y, log_y, Xy_base, unit_driver_cols, paths_y, best_y_hist, self._min_y_to_update),
                ("Valor ($)", v, log_v, Xv_base, value_driver_cols, paths_v, best_v_hist, 1e-8),
            ):
                for group in sorted(set(node_exclusions.get(target_name, ())) & allowed_groups):
                    challenger_cols = self._driver_columns_for_node(
                        uid, target_name,
                        ignore_exclusion_groups=frozenset({str(group)}),
                    )
                    if challenger_cols == current_cols:
                        continue
                    challenger_X = self._finite_matrix(
                        base, challenger_cols, uid=uid,
                        stage=f"RLS phase2 {target_name} add-back {group}",
                    )
                    mode_hist, lam_hist = best_candidate
                    Xfit = (
                        np.column_stack([challenger_X, self._ar_training_matrix(history_log)])
                        if mode_hist == "ar" else challenger_X
                    )
                    model = self._new_rls(
                        min_y, return_all_coefs=True, forgetting_factor=float(lam_hist)
                    )
                    model.fit(
                        x=Xfit, y=history_log,
                        priors=self._default_priors(Xfit.shape[1]),
                        seed_n_obs=seed_n,
                    )
                    _record_refit_blocks(
                        target_name=target_name, actual_values=actual_values,
                        history_log=history_log, full_X=current_X, full_paths=full_paths,
                        candidate=best_candidate, challenger_X=challenger_X,
                        challenger_path=np.asarray(model.all_coef_[0], dtype=np.float64),
                        driver_group=str(group), test_action="add",
                    )

            # For Value nodes that still use the no-price baseline, Phase 2
            # tests causal inclusion of the EXISTING price group. Nodes where
            # price was manually promoted (currently Section 1) instead test
            # it naturally through the remove-refit loop above.
            price_idxs = self._driver_groups(self._driver_cols).get("price", [])
            value_has_price = bool(self._driver_groups(value_driver_cols).get("price", []))
            if price_idxs and "price" in allowed_groups and not value_has_price:
                value_with_price_cols = self._driver_columns_for_node(
                    uid, "Valor ($)", force_value_price=True
                )
                Xv_with_price = self._finite_matrix(
                    base, value_with_price_cols, uid=uid,
                    stage="RLS phase2 value + existing price drivers",
                )
                mode_v_hist, lam_v_hist = best_v_hist
                Xfit_vp = (
                    np.column_stack([Xv_with_price, self._ar_training_matrix(log_v)])
                    if mode_v_hist == "ar"
                    else Xv_with_price
                )
                mv_price = self._new_rls(
                    1e-8, return_all_coefs=True, forgetting_factor=float(lam_v_hist)
                )
                mv_price.fit(
                    x=Xfit_vp, y=log_v,
                    priors=self._default_priors(Xfit_vp.shape[1]),
                    seed_n_obs=seed_n,
                )
                _record_refit_blocks(
                    target_name="Valor ($)", actual_values=v, history_log=log_v,
                    full_X=Xv_base, full_paths=paths_v,
                    candidate=best_v_hist, challenger_X=Xv_with_price,
                    challenger_path=np.asarray(mv_price.all_coef_[0], dtype=np.float64),
                    driver_group="price", test_action="add",
                )

            logger.info(
                "⏱ %s %s: Phase-2 refit drivers (%d registros): %.2fs",
                desc, uid, len(opt_driver_refit_rows), time.perf_counter() - t_refit,
            )

        out = pl.DataFrame({
            "unique_id": [uid] * n, "ds": actual["ds"], "value": actual["value"], "valuehat": vh,
            "y": actual["y"], "yhat": yh, "period_type": actual["_source_period"],
            "rls_metric_eligible": eligible, "rls_block": block, "rls_train_days": train_days,
            "rls_lambda_y": lambda_y, "rls_lambda_value": lambda_v,
            "rls_dynamics_y": mode_y.tolist(), "rls_dynamics_value": mode_v.tolist(),
            "rls_forecast_origin": origin.tolist(),
            "driver_effect": driver_effect_y, "driver_effect_value": driver_effect_v,
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
            Xyf_base = self._finite_matrix(fcst_g, unit_driver_cols, uid=uid, stage="RLS forecast y")
            Xvf_base = self._finite_matrix(fcst_g, value_driver_cols, uid=uid, stage="RLS forecast value")
            cy = choose(cum_ae_y, cum_den_y); cv = choose(cum_ae_v, cum_den_v)
            def predict_future(Xbase, coef, hist_actual, mode):
                xrows = np.ascontiguousarray(Xbase, dtype=np.float64)
                c = np.ascontiguousarray(coef, dtype=np.float64)
                if mode == "ar":
                    tail = np.ascontiguousarray(hist_actual[-28:], dtype=np.float64)
                    return _predict_loglink_ar(xrows, c, tail)
                return _predict_loglink_base(xrows, c)
            fy_raw = predict_future(Xyf_base, paths_y[cy][-1], log_y, cy[0])
            fv_raw = predict_future(Xvf_base, paths_v[cv][-1], log_v, cv[0])
            fy = np.round(fy_raw, 0)
            fv = np.round(fv_raw, 2)
            f_effect_y = np.log1p(np.clip(fy_raw, 0.0, None))
            f_effect_v = np.log1p(np.clip(fv_raw, 0.0, None))
            if intercept_y_idx is not None:
                f_effect_y = f_effect_y - Xyf_base[:, intercept_y_idx] * paths_y[cy][-1][intercept_y_idx]
            if intercept_v_idx is not None:
                f_effect_v = f_effect_v - Xvf_base[:, intercept_v_idx] * paths_v[cv][-1][intercept_v_idx]
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
                "driver_effect": f_effect_y, "driver_effect_value": f_effect_v,
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
            desc, uid, block_days, mode_candidates, productive_lambdas, final_cy[0], final_cy[1], final_cv[0], final_cv[1],
        )
        logger.info(
            "⏱ %s %s: RLS candidatos + bloques: %.1fs",
            desc,
            uid,
            time.perf_counter() - t_rls_total,
        )
        self._append_optimization_records(
            opt_candidate_rows, opt_driver_rows, opt_driver_refit_rows
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

        The implementation avoids store-by-store orchestration, which repeatedly
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
        block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))

        def _one(uid: str):
            g = train_parts.get(uid)
            if g is None or g.height == 0:
                return uid, [], None
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
    # ── SKU+tienda: misma familia SES+RLS en in-sample/OOS/forecast-only ──
    def predict_leaf_series(
        self,
        train_leaves: pl.DataFrame,
        oos_leaves: pl.DataFrame,
        section_id: str,
        horizons: dict,
        meta: dict | None = None,
        parent_forecasts: pl.DataFrame | None = None,
        oos_observable_store_days: pl.DataFrame | None = None,
        diagnostics_out: list[pl.DataFrame] | None = None,
        parent_diagnostics_out: list[pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Pronostica hojas con el único contrato productivo SES+RLS de v13."""
        if train_leaves.height == 0:
            return pl.DataFrame()
        if parent_forecasts is None or parent_forecasts.height == 0:
            raise RuntimeError(
                f"Sección {section_id}: no hay forecasts RLS parent para derivar SKU+tienda"
            )
        return build_leaf_forecasts(
            train_leaves=train_leaves,
            oos_leaves=oos_leaves,
            parent_forecasts=parent_forecasts,
            oos_observable_store_days=oos_observable_store_days,
            section_id=str(section_id),
            horizons=horizons,
            meta=meta,
            diagnostics_out=diagnostics_out,
            parent_diagnostics_out=parent_diagnostics_out,
            optimization_phase2=self._optimization_phase2,
        )

