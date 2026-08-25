from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numba.core.registry import CPUDispatcher

from rls_opt.priors import RLSConstantPrior, RLSPrior, RLSPriorBase, RLSRelativePrior
from rls_opt.kernels import (
    RLSConvergenceError,
    _log_weighting,
    _numba_outer,
    _rls,
    _rls_in_sample_forecasts,
    _rls_out_of_sample_forecasts,
    _rls_predict,
)


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

    # TODO: Add xxt to return values. This value supplies the confidence of each coef value

    @property
    def all_coef(self) -> list[np.ndarray]:
        if not self.all_coef_:
            raise ValueError(
                f"{self} does not have final_coeffs_indexors. Must set return_all_coeffs = True."
            )
        return self.all_coef_

    def _check_priors(self, priors: Sequence[RLSPriorBase]) -> None:
        num_constant_priors = sum(isinstance(prior, RLSConstantPrior) for prior in priors)
        if num_constant_priors > 1:
            raise ValueError("At most one RLSConstantPrior is allowed")
        if num_constant_priors == 1 and not isinstance(priors[0], RLSConstantPrior):
            raise ValueError("RLSConstantPrior must be the first prior")

    def _check_inputs(
        self, x: np.ndarray, y: np.ndarray, priors: np.ndarray, xxt_inv: np.ndarray
    ) -> None:
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if priors.ndim != 1:
            raise ValueError("priors must be a 1D array")
        if xxt_inv.ndim != 2:
            raise ValueError("xxt_inv must be a 2D array")
        num_rows, num_cols = x.shape
        if len(y) != num_rows:
            raise ValueError("x and y must contain the same number of observations")
        if len(priors) != num_cols:
            raise ValueError("number of priors must equal number of x columns")
        if xxt_inv.shape != (num_cols, num_cols):
            raise ValueError("xxt_inv must be square with one row/column per feature")
        if not (0.0 < float(self.forgetting_factor) <= 1.0):
            raise ValueError("forgetting_factor must be in (0, 1]")
        if float(self.min_y_to_update) <= 0.0:
            raise ValueError("min_y_to_update must be > 0")
        for name, value in (("x", x), ("y", y), ("priors", priors), ("xxt_inv", xxt_inv)):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} includes nan or inf")

    def _check_fit_inputs(self, x: np.ndarray, y: np.ndarray, priors: Sequence[RLSPriorBase]) -> None:
        if x.ndim != 2 or y.ndim != 1:
            raise ValueError("x must be 2D and y must be 1D")
        rows, cols = x.shape
        if len(y) != rows:
            raise ValueError("x and y must contain the same number of observations")
        if len(priors) != cols:
            raise ValueError("number of priors must equal number of x columns")
        if not np.isfinite(x).all():
            raise ValueError("x includes nan or inf")
        if not np.isfinite(y).all():
            raise ValueError("y includes nan or inf")
        if not (0.0 < float(self.forgetting_factor) <= 1.0):
            raise ValueError("forgetting_factor must be in (0, 1]")
        if float(self.min_y_to_update) <= 0.0:
            raise ValueError("min_y_to_update must be > 0")

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        priors: Sequence[RLSPriorBase],
        in_sample_metrics: bool = False,
        out_sample_metrics: bool = False,
        seed_n_obs: int | None = None,
    ) -> None:
        if in_sample_metrics or out_sample_metrics:
            raise NotImplementedError("RLS metric generation is not implemented; compute metrics in the forecasting layer")

        self._check_fit_inputs(x, y, priors)
        self._check_priors(priors)

        if seed_n_obs is not None:
            seed_n = max(1, min(int(seed_n_obs), len(y)))
            x_seed, y_seed = x[:seed_n], y[:seed_n]
        else:
            x_seed, y_seed = x, y
        coeffs_0 = self._coefficient_seeds(x=x_seed, y=y_seed, priors=priors)
        xxt_inv_0 = self._xxt_inv_seeds(
            y=y_seed, priors=priors, coefficient_seeds=coeffs_0
        )

        # Public API validation is always on; Numba kernels keep internal asserts.
        self._check_inputs(x=x, y=y, priors=coeffs_0, xxt_inv=xxt_inv_0)

        coeffs, all_coeffs, errors = _rls(
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
        if in_sample_metrics or out_sample_metrics:
            raise NotImplementedError("RLS metric generation is not implemented; compute metrics in the forecasting layer")
        if x.ndim != 2 or y.ndim != 1:
            raise ValueError("x must be 2D and y must be 1D")
        num_rows, num_cols = x.shape
        if len(y) != num_rows:
            raise ValueError("x and y must contain the same number of observations")
        if len(priors) != num_cols:
            raise ValueError("number of priors must equal number of x columns")

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

        if len(indexors) > len(self.final_coef_):
            raise ValueError("predict received more indexors than fitted models")
        predictions = []
        for i, indexor in enumerate(indexors):
            predictions.append(_rls_predict(coeffs=self.final_coef_[i], x=x[indexor]))

        return predictions

    def _in_sample_metrics(self):
        pass

    def _out_sample_metrics(self):
        pass

    def _coefficient_seeds(self, x, y, priors: Sequence[RLSPriorBase]) -> np.ndarray:
        coefficients = []
        for index, prior in reversed(list(enumerate(priors))):
            if isinstance(prior, RLSPrior):
                coefficients.append(prior.coefficient_value())
            elif isinstance(prior, RLSRelativePrior):
                coefficients.append(prior.coefficient_value(x[:, index]))
            elif isinstance(prior, RLSConstantPrior):
                # Always will be the last prior
                x_no_constants = x[:, 1:]
                priors_no_constant = np.array(coefficients)[::-1]
                # if (priors_no_constant!=0).sum() == 0: # si es todo cero
                #     coefficients.append(1)
                # else:
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
        """

        Returns
        -------
        xxt_inv_seed: np.ndarray of shape (num_cols, num_cols)

        """
        xxt_inv_seed_diag = []
        for prior, coeff in zip(priors, coefficient_seeds):
            if isinstance(prior, RLSPrior):
                xxt_inv_seed_diag.append(prior.standard_error_value())
            elif isinstance(prior, RLSRelativePrior):
                xxt_inv_seed_diag.append(prior.standard_error_value(coeff))
            elif isinstance(prior, RLSConstantPrior):
                # Always will be the last prior
                xxt_inv_seed_diag.append(prior.standard_error_value(coeff))

        return np.diag(xxt_inv_seed_diag)
