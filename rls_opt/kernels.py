"""Numba kernels for Recursive Least Squares.

Kept separate from the public API so validation/orchestration stays testable.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numba import jit
from numpy import exp, sqrt

class RLSConvergenceError(Exception):
    def __init__(
        self,
        message="RLS Solver found an convergence error. Try increasing Forgetting Factor.",
    ):
        super().__init__(message)
        self.message = message

def _rls(
    x: np.ndarray,
    y: np.ndarray,
    priors: np.ndarray,
    xxt_inv_seed: np.ndarray,
    forgetting_factor: float = 1,
    weighting_function: Callable[[float], float] | None = None,
    return_all_coeffs: bool = False,
    min_y_to_update: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Weighted Recursive Least Squares

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

    assert 0.0 < forgetting_factor <= 1.0
    assert min_y_to_update > 0.0

    B = np.copy(
        xxt_inv_seed
    )  # We will refer to xxt_inv_seed as B in later steps to make code more readable:
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
            _numba_outer(
                Bx, xtwB, BxxtwB
            )  # Re-implement numpy.outer as inplace operation.
            xtwBx: float = xtwB @ x_i
            alpha = 1 / (forgetting_factor + xtwBx)

            if math.isnan(alpha):
                _N = 10000
                print(
                    "forgetting_factor of ~"
                    + str(int(_N * forgetting_factor))
                    + "/"
                    + str(_N)
                    + " is to small. Please increase forgetting_factor."
                )
                # We need to print here instead of including the error message in the RLSConvergenceError error as
                # Numba requires all error message to be compile time constants. See
                # http://numba.pydata.org/numba-doc/dev/reference/pysupported.html
                raise RLSConvergenceError()

            B -= alpha * BxxtwB
            B /= forgetting_factor
            oos_error.append(y_i - z_i)
            coeffs += (alpha * (y_i - z_i)) * xtwB

        if return_all_coeffs:
            # The following for loop is equivalent to
            # #all_coeffs[i, :] = coeffs
            for j in range(num_vars):
                all_coeffs[i, j] = coeffs[j]

    return coeffs, all_coeffs, oos_error

def _log_weighting(x):
    return sqrt(exp(x) / x)

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

def _numba_outer(x1: np.ndarray, x2: np.ndarray, out: np.ndarray) -> None:
    """Re-implements np.outer as an inplace operation at ~10x the speed of np.outer.

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

def _rls_out_of_sample_forecasts(
    x: np.ndarray,
    all_coeffs: np.ndarray,
    ignore_first_n: int | None = None,
    forecast_horizon: int = 12,
) -> list[np.ndarray]:
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
            for horizon_index in range(
                coeff_index, min(num_obs, coeff_index + forecast_horizon)
            ):
                x_i = x[horizon_index, :]
                index_forecasts.append(_rls_predict(coeffs=index_coeffs, x=x_i))

        forecasts.append(np.array(index_forecasts))

    return forecasts

def _rls_in_sample_forecasts(
    x: np.ndarray, all_coeffs: np.ndarray, ignore_first_n: int | None = None
) -> list[np.ndarray]:
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
            forecasts.append(
                _rls_predict(all_coeffs[coeff_index, :], x[:coeff_index, :])
            )
        else:
            forecasts.append(np.array([]))
    return forecasts
