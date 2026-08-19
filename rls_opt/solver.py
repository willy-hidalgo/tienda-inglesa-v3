"""Public API for recursive least-squares regression."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numba.core.registry import CPUDispatcher

from rls_opt.kernels import (
    RLSConvergenceError,
    log_weighting,
    numba_outer,
    rls_in_sample_forecasts,
    rls_kernel,
    rls_out_of_sample_forecasts,
    rls_predict_kernel,
)
from rls_opt.priors import RLSConstantPrior, RLSPrior, RLSPriorBase, RLSRelativePrior

# Backwards-compatible private aliases for existing internal consumers.
_rls = rls_kernel
_log_weighting = log_weighting
_rls_predict = rls_predict_kernel
_numba_outer = numba_outer
_rls_out_of_sample_forecasts = rls_out_of_sample_forecasts
_rls_in_sample_forecasts = rls_in_sample_forecasts


@dataclass
class RecursiveLeastSquaresRegression:
    forgetting_factor: float
    min_y_to_update: float
    weighting_function: Callable[[float], float] | CPUDispatcher | None = field(
        default=None
    )
    return_all_coefs: bool = False
    check_inputs: bool = False
    final_coef_: list[np.ndarray] = field(default_factory=list, repr=False)
    all_coef_: list[np.ndarray] = field(default_factory=list, repr=False)
    errors: list[float] = field(default_factory=list, repr=False)

    @property
    def all_coef(self) -> list[np.ndarray]:
        if not self.all_coef_:
            raise ValueError(
                f"{self} does not have final_coeffs_indexors. Must set return_all_coeffs = True."
            )
        return self.all_coef_

    def _check_priors(self, priors: Sequence[RLSPriorBase]) -> None:
        num_constant_priors = sum(
            isinstance(prior, RLSConstantPrior) for prior in priors
        )
        if num_constant_priors > 1:
            raise ValueError("At most one RLSConstantPrior is supported")
        if num_constant_priors == 1 and not isinstance(priors[0], RLSConstantPrior):
            raise ValueError("RLSConstantPrior must be the first prior")

    def _check_inputs(
        self, x: np.ndarray, y: np.ndarray, priors: np.ndarray, xxt_inv: np.ndarray
    ) -> None:
        if x.ndim != 2:
            raise ValueError("x must be a 2-dimensional array")
        num_rows, num_cols = x.shape
        if y.ndim != 1 or len(y) != num_rows:
            raise ValueError("y must be 1-dimensional with one value per x row")
        if len(priors) != num_cols:
            raise ValueError("priors must contain one value per x column")
        if xxt_inv.shape != (num_cols, num_cols):
            raise ValueError("xxt_inv must be square with one row/column per feature")

        for name, values in (
            ("x", x),
            ("y", y),
            ("priors", priors),
            ("xxt_inv", xxt_inv),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} is not finite, includes nan or inf")

    @staticmethod
    def _validate_solver_parameters(
        forgetting_factor: float, min_y_to_update: float
    ) -> None:
        if not np.isfinite(forgetting_factor) or not 0.0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be finite and satisfy 0 < value <= 1")
        if not np.isfinite(min_y_to_update) or min_y_to_update <= 0.0:
            raise ValueError("min_y_to_update must be finite and greater than 0")

    @staticmethod
    def _validate_metrics_flags(
        in_sample_metrics: bool, out_sample_metrics: bool
    ) -> None:
        if in_sample_metrics or out_sample_metrics:
            raise NotImplementedError(
                "in_sample_metrics and out_sample_metrics are not implemented yet"
            )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        priors: Sequence[RLSPriorBase],
        in_sample_metrics: bool = False,
        out_sample_metrics: bool = False,
    ) -> None:
        self._validate_solver_parameters(self.forgetting_factor, self.min_y_to_update)
        self._validate_metrics_flags(in_sample_metrics, out_sample_metrics)
        self._check_priors(priors)

        coeffs_0 = self._coefficient_seeds(x=x, y=y, priors=priors)
        xxt_inv_0 = self._xxt_inv_seeds(y=y, priors=priors, coefficient_seeds=coeffs_0)

        if self.check_inputs:
            self._check_inputs(x=x, y=y, priors=coeffs_0, xxt_inv=xxt_inv_0)

        coeffs, all_coeffs, errors = rls_kernel(
            x=x,
            y=y,
            priors=coeffs_0,
            xxt_inv_seed=xxt_inv_0,
            forgetting_factor=self.forgetting_factor,
            weighting_function=self.weighting_function,
            return_all_coeffs=self.return_all_coefs,
            min_y_to_update=self.min_y_to_update,
        )
        if self.return_all_coefs:
            self.all_coef_.append(all_coeffs)
        self.final_coef_.append(coeffs)
        self.errors = errors

    def fit_stacked(
        self,
        x: np.ndarray,
        y: np.ndarray,
        priors: Sequence[RLSPriorBase],
        indexors: Sequence[slice] | Sequence[np.ndarray] | None = None,
        in_sample_metrics: bool = False,
        out_sample_metrics: bool = False,
    ) -> None:
        self._validate_metrics_flags(in_sample_metrics, out_sample_metrics)
        if x.ndim != 2:
            raise ValueError("x must be a 2-dimensional array")
        num_rows, num_cols = x.shape
        if y.ndim != 1 or len(y) != num_rows:
            raise ValueError("y must be 1-dimensional with one value per x row")
        if len(priors) != num_cols:
            raise ValueError("priors must contain one value per x column")

        if indexors is None:
            indexors = [slice(None, None, None)]

        for indexor in indexors:
            x_i = np.ascontiguousarray(x[indexor])
            y_i = np.ascontiguousarray(y[indexor])
            self.fit(x=x_i, y=y_i, priors=priors)

    def predict(
        self, x, indexors: Sequence[slice] | Sequence[np.ndarray] | None = None
    ):
        if indexors is None:
            indexors = [slice(None, None, None)]

        if len(self.final_coef_) < len(indexors):
            raise ValueError("Not enough fitted coefficient sets for the requested indexors")

        predictions = []
        for i, indexor in enumerate(indexors):
            predictions.append(rls_predict_kernel(coeffs=self.final_coef_[i], x=x[indexor]))

        return predictions

    def _coefficient_seeds(self, x, y, priors: Sequence[RLSPriorBase]) -> np.ndarray:
        coefficients = []
        for index, prior in reversed(list(enumerate(priors))):
            if isinstance(prior, RLSPrior):
                coefficients.append(prior.coefficient_value())
            elif isinstance(prior, RLSRelativePrior):
                coefficients.append(prior.coefficient_value(x[:, index]))
            elif isinstance(prior, RLSConstantPrior):
                x_no_constants = x[:, 1:]
                priors_no_constant = np.array(coefficients)[::-1]
                coefficients.append(
                    prior.coefficient_value(
                        x=x_no_constants, y=y, priors_coeffs=priors_no_constant
                    )
                )

        return np.ascontiguousarray(np.array(coefficients)[::-1])

    def _xxt_inv_seeds(
        self,
        y: np.ndarray,
        priors: Sequence[RLSPriorBase],
        coefficient_seeds: np.ndarray,
    ) -> np.ndarray:
        xxt_inv_seed_diag = []
        for prior, coeff in zip(priors, coefficient_seeds):
            if isinstance(prior, RLSPrior):
                xxt_inv_seed_diag.append(prior.standard_error_value())
            elif isinstance(prior, RLSRelativePrior):
                xxt_inv_seed_diag.append(prior.standard_error_value(coeff))
            elif isinstance(prior, RLSConstantPrior):
                xxt_inv_seed_diag.append(prior.standard_error_value(coeff))

        return np.diag(xxt_inv_seed_diag)
