import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numba import jit
from numba.core.registry import CPUDispatcher
from numpy import exp, sqrt

from rls_opt.priors import RLSConstantPrior, RLSPrior, RLSPriorBase, RLSRelativePrior


@dataclass
class RecursiveLeastSquaresRegression:
    forgetting_factor: float
    min_y_to_update: float
    weighting_function: Optional[Union[Callable[[float], float], CPUDispatcher]] = field(default=None)
    return_all_coefs: bool = False
    check_inputs: bool = False
    final_coef_: List[np.ndarray] = field(default_factory=list, repr=False)
    all_coef_: List[np.ndarray] = field(default_factory=list, repr=False)
    errors: List[float] = field(default_factory=list, repr=False)
    
    # TODO: Add xxt to return values. This value supplies the confidence of each coef value

    @property
    def all_coef(self) -> List[np.ndarray]:
        if not self.all_coef_:
            raise ValueError(f"{self} does not have final_coeffs_indexors. Must set return_all_coeffs = True.")
        return self.all_coef_

    def _check_priors(self, priors: Sequence[RLSPriorBase]):
        num_constant_priors = sum(isinstance(prior, RLSConstantPrior) for prior in priors)
        assert num_constant_priors <= 1
        if num_constant_priors == 1:
            assert isinstance(priors[0], RLSConstantPrior)

    def _check_inputs(self, x: np.ndarray, y: np.ndarray, priors: np.ndarray, xxt_inv: np.ndarray) -> None:
        assert x.ndim == 2
        num_rows, num_cols = x.shape
        assert len(y) == num_rows
        assert len(priors) == num_cols

        if not np.isfinite(x).any():
            raise ValueError("x is not finite, includes nan or inf")
        if not np.isfinite(y).any():
            raise ValueError("y is not finite, includes nan or inf")
        if not np.isfinite(priors).any():
            raise ValueError("priors is not finite, includes nan or inf")
        if not np.isfinite(xxt_inv).any():
            raise ValueError("xxt_inv is not finite, includes nan or inf")

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            priors: Sequence[RLSPriorBase],
            in_sample_metrics: bool = False,
            out_sample_metrics: bool = False) -> None:
        # TODO: Add support for `in_sample_metrics` and `out_sample_metrics`

        self._check_priors(priors)

        coeffs_0 = self._coefficient_seeds(x=x, y=y, priors=priors)
        xxt_inv_0 = self._xxt_inv_seeds(y=y, priors=priors, coefficient_seeds=coeffs_0)

        if self.check_inputs:
            self._check_inputs(x=x, y=y, priors=coeffs_0, xxt_inv=xxt_inv_0)

        coeffs, all_coeffs, errors = _rls(x=x,
                                  y=y,
                                  priors=coeffs_0,
                                  xxt_inv_seed=xxt_inv_0,
                                  forgetting_factor=self.forgetting_factor,
                                  weighting_function=self.weighting_function,
                                  return_all_coeffs=self.return_all_coefs,
                                  min_y_to_update=self.min_y_to_update)
        if self.return_all_coefs:
            self.all_coef_.append(all_coeffs)
        self.final_coef_.append(coeffs)
        self.errors = errors

    def fit_stacked(self,
                    x: np.ndarray,
                    y: np.ndarray,
                    priors: Sequence[RLSPriorBase],
                    indexors: Optional[Union[Sequence[slice], Sequence[np.ndarray]]] = None,
                    in_sample_metrics: bool = False,
                    out_sample_metrics: bool = False) -> None:
        # TODO: Add support for `in_sample_metrics` and `out_sample_metrics`

        assert x.ndim == 2
        num_rows, num_cols = x.shape
        assert len(y) == num_rows
        assert len(priors) == num_cols

        if indexors is None:
            indexors = [slice(None, None, None)]

        for indexor in indexors:
            x_i = np.ascontiguousarray(x[indexor])
            y_i = np.ascontiguousarray(y[indexor])
            self.fit(x=x_i, y=y_i, priors=priors)

    def predict(self, x, indexors: Optional[Union[Sequence[slice], Sequence[np.ndarray]]] = None):
        if indexors is None:
            indexors = [slice(None, None, None)]

        predictions = []
        for i, indexors in enumerate(indexors):
            predictions.append(_rls_predict(coeffs=self.final_coef_[i], x=x[indexors]))

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
                coefficients.append(prior.coefficient_value(x=x_no_constants, y=y, priors_coeffs=priors_no_constant))

        return np.ascontiguousarray(np.array(coefficients)[::-1])

    def _xxt_inv_seeds(self, y: np.ndarray, priors: Sequence[RLSPriorBase], coefficient_seeds: np.ndarray) -> np.ndarray:
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


class RLSConvergenceError(Exception):
    def __init__(self, message="RLS Solver found an convergence error. Try increasing Forgetting Factor."):
        super(RLSConvergenceError, self).__init__(message)
        self.message = message


@jit(nopython=True, cache=True)
def _rls(x: np.ndarray, y: np.ndarray, priors: np.ndarray, xxt_inv_seed: np.ndarray, forgetting_factor: float = 1,
         weighting_function: Optional[Callable[[float], float]] = None, return_all_coeffs: bool = False,
         min_y_to_update: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """ Weighted Recursive Least Squares

    Parameters
    ----------
    x : numpy array of shape (num_obs, num_vars)
        Rows are observations (x[i, :] is observation i)

        The ordering the rows of x is important as the influence of y[i], x[i] will be based on information already
        learned from y[:i-1], x[:i-1, :]
        For Example, with timeseries data, it is best to sort x and y by the time dimension in an ascending fashion.

    y : numpy array of shape (num_obs, )
        Values are realizations y[i] is realization i

    priors : numpy array of shape (num_vars, )
        TODO: Add doc

    xxt_inv_seed : numpy array of shape (num_vars, num_vars)
        TODO: Add doc

    weighting_function: numba no python jit function that takes a float and returns a float
        numba function  signature double(double) or f8(f8)
        Commonly used functions:
            sqrt(exp(x) / x)

        TODO: Check weighting_function is proper function type:
            Use weighting_function.nopython_signatures

    return_all_coeffs: bool, optional. Default is False
        Flag whether to return all learned

    Returns
    -------
    priors : numpy array of shape (num_vars, )
        Learned Coefficients after

    """
    assert x.ndim == 2
    assert y.ndim == 1
    assert priors.ndim == 1
    assert xxt_inv_seed.ndim == 2

    num_obs, num_vars = x.shape
    assert len(y) == num_obs
    assert len(priors) == num_vars
    assert xxt_inv_seed.shape == (num_vars, num_vars)

    assert 0. < forgetting_factor <= 1.
    assert min_y_to_update > 0.

    B = np.copy(xxt_inv_seed)  # We will refer to xxt_inv_seed as B in later steps to make code more readable:
    coeffs = np.copy(priors)

    all_coeffs = np.zeros_like(x, dtype=np.float64)

    BxxtwB = np.empty((num_vars, num_vars))
    oos_error = []
    for i in range(num_obs):
        y_i = y[i]

        if y_i > min_y_to_update:
            x_i = x[i, :]
            z_i = _rls_predict(coeffs=coeffs, x=x_i)

            if weighting_function is None:
                xtwB = x_i @ B
            else:
                xtwB = (weighting_function(y_i) * x_i) @ B

            Bx = B @ x_i
            _numba_outer(Bx, xtwB, BxxtwB)  # Re-implement numpy.outer as inplace operation.
            xtwBx: float = xtwB @ x_i
            alpha = 1 / (forgetting_factor + xtwBx)

            if math.isnan(alpha):
                _N = 10000
                print("forgetting_factor of ~"
                      + str(int(_N*forgetting_factor)) + "/" + str(_N)
                      + " is to small. Please increase forgetting_factor.")
                # We need to print here instead of including the error message in the RLSConvergenceError error as
                # Numba requires all error message to be compile time constants. See
                # http://numba.pydata.org/numba-doc/dev/reference/pysupported.html
                raise RLSConvergenceError()

            B -= (alpha * BxxtwB)
            B /= forgetting_factor
            oos_error.append(y_i - z_i)
            coeffs += (alpha * (y_i - z_i)) * xtwB

        if return_all_coeffs:
            # The following for loop is equivalent to
            # #all_coeffs[i, :] = coeffs
            for j in range(num_vars): all_coeffs[i, j] = coeffs[j]

    return coeffs, all_coeffs, oos_error


@jit(nopython=True, cache=True)
def _log_weighting(x):
    return sqrt(exp(x) / x)


@jit(nopython=True, cache=True)
def _rls_predict(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    x
    coeffs

    Returns
    -------

    """
    assert x.ndim in [1, 2]
    assert coeffs.ndim == 1

    if x.ndim == 1:
        num_obs, num_vars = 1, len(x)
    else:
        num_obs, num_vars = x.shape

    assert num_vars == len(coeffs)

    return x @ coeffs


@jit(nopython=True, cache=True)
def _numba_outer(x1: np.ndarray, x2: np.ndarray, out: np.ndarray) -> None:
    """ Re-implements np.outer as an inplace operation at ~10x the speed of np.outer.

    See np.outer for more details.
    """
    assert x1.ndim == 1
    assert x2.ndim == 1
    len_x1 = len(x1)
    assert len_x1 == len(x2)
    assert out.shape == (len_x1, len_x1)

    for i in range(len_x1):
        for j in range(len_x1):
            out[i, j] = x1[i] * x2[j]


@jit(nopython=True, cache=True)
def _rls_out_of_sample_forecasts(x: np.ndarray, all_coeffs: np.ndarray, ignore_first_n: Optional[int] = None,
                                 forecast_horizon: int = 12) -> List[np.ndarray]:
    """
    Parameters
    ----------
    x : numpy array of shape (num_obs, num_vars)
        Rows are observations (x[i, :] is observation i)

        The ordering the rows of x is important as the influence of y[i], x[i] will be based on information already
        learned from y[:i-1], x[:i-1, :]
        For Example, with timeseries data, it is best to sort x and y by the time dimension in an ascending fashion.

    all_coeffs : numpy array of shape (num_obs, num_vars)
        Rows are the coefficients learned on data  (y[:i-1], x[:i-1])

    ignore_first_n : int
        Defaults to min(num_obs, num_vars)
        Number of observations to skip before forecasts are calculated

    forecast_horizon : int
        Defaults to 12
        Number of observations b

    Returns
    -------
    out_of_sample_forecasts : List of num_obs numpy arrays.
        out_of_sample_forecasts[j] will be forecasts j.

            Therefore for:
                i) k < ignore_first_n,
                    the lists will be empty arrays as there are no forecasts.
                ii) ignore_first_n <= j <= (num_obs-forecast_horizon),
                    will be numpy arrays of shape (forecast_horizon,)
                iii) j > num_obs-forecast_horizon
                    will be numpy arrays of shape (num_obs - j + 1)

        This allows you to access forecast of week j from week i (assuming they are valid according to the above logic)
            at out_of_sample_forecasts[i][j-i].
                In other words, based on our data from the first i weeks, our forecast for week j is
                out_of_sample_forecasts[i][j-i]

    """
    assert x.ndim == 2
    assert all_coeffs.ndim == 2

    num_obs, num_vars = x.shape
    assert all_coeffs.shape == (num_obs, num_vars)

    if ignore_first_n is None:
        ignore_first_n = min(num_vars, num_obs)
    else:
        assert ignore_first_n < num_obs

    assert forecast_horizon < num_obs

    forecasts = []
    for coeff_index in range(num_obs):
        index_forecasts = []

        if coeff_index >= ignore_first_n:
            index_coeffs = all_coeffs[coeff_index, :]
            for horizon_index in range(coeff_index, min(num_obs, coeff_index + forecast_horizon)):
                x_i = x[horizon_index, :]
                index_forecasts.append(_rls_predict(coeffs=index_coeffs, x=x_i))

        forecasts.append(np.array(index_forecasts))

    return forecasts


def _rls_in_sample_forecasts(x: np.ndarray, all_coeffs: np.ndarray,
                             ignore_first_n: Optional[int] = None) -> List[np.ndarray]:
    """

    Parameters
    ----------
    x
    all_coeffs
    ignore_first_n

    Returns
    -------

    """
    assert x.ndim == 2
    assert all_coeffs.ndim == 2

    num_obs, num_vars = x.shape
    assert all_coeffs.shape == (num_obs, num_vars)

    if ignore_first_n is None:
        ignore_first_n = min(num_vars, num_obs)
    else:
        assert ignore_first_n < num_obs

    forecasts = []
    for coeff_index in range(num_obs):
        if coeff_index >= ignore_first_n:
            forecasts.append(_rls_predict(all_coeffs[coeff_index, :], x[:coeff_index, :]))
        else:
            forecasts.append(np.array([]))
    return forecasts
