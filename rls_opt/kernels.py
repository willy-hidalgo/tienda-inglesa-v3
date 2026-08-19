"""Numba-compiled numerical kernels for the RLS solver.

This module intentionally contains no public model orchestration. Keeping the
hot numerical loops isolated makes the solver API easier to test and lets us
benchmark the kernels independently from validation and prior construction.
"""

import math
from collections.abc import Callable

import numpy as np
from numba import jit
from numpy import exp, sqrt


class RLSConvergenceError(Exception):
    """Raised when the recursive update becomes numerically unstable."""

    def __init__(
        self,
        message="RLS Solver found an convergence error. Try increasing Forgetting Factor.",
    ):
        super().__init__(message)
        self.message = message


@jit(nopython=True, cache=True)
def rls_kernel(
    x: np.ndarray,
    y: np.ndarray,
    priors: np.ndarray,
    xxt_inv_seed: np.ndarray,
    forgetting_factor: float = 1,
    weighting_function: Callable[[float], float] | None = None,
    return_all_coeffs: bool = False,
    min_y_to_update: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Weighted Recursive Least Squares numerical kernel."""
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

    B = np.copy(xxt_inv_seed)
    coeffs = np.copy(priors)
    all_coeffs = np.zeros_like(x, dtype=np.float64)

    BxxtwB = np.empty((num_vars, num_vars))
    oos_error = []
    for i in range(num_obs):
        y_i = y[i]

        if y_i > min_y_to_update:
            x_i = x[i, :]
            z_i = rls_predict_kernel(coeffs=coeffs, x=x_i)

            if weighting_function is None:
                xtwB = x_i @ B
            else:
                xtwB = (weighting_function(y_i) * x_i) @ B

            Bx = B @ x_i
            numba_outer(Bx, xtwB, BxxtwB)
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
                raise RLSConvergenceError()

            B -= alpha * BxxtwB
            B /= forgetting_factor
            oos_error.append(y_i - z_i)
            coeffs += (alpha * (y_i - z_i)) * xtwB

        if return_all_coeffs:
            for j in range(num_vars):
                all_coeffs[i, j] = coeffs[j]

    return coeffs, all_coeffs, oos_error


@jit(nopython=True, cache=True)
def log_weighting(x):
    return sqrt(exp(x) / x)


@jit(nopython=True, cache=True)
def rls_predict_kernel(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Predict using learned coefficients."""
    assert x.ndim in [1, 2]
    assert coeffs.ndim == 1

    if x.ndim == 1:
        num_obs, num_vars = 1, len(x)
    else:
        num_obs, num_vars = x.shape

    assert num_vars == len(coeffs)
    return x @ coeffs


@jit(nopython=True, cache=True)
def numba_outer(x1: np.ndarray, x2: np.ndarray, out: np.ndarray) -> None:
    """Compute an outer product in-place."""
    assert x1.ndim == 1
    assert x2.ndim == 1
    len_x1 = len(x1)
    assert len_x1 == len(x2)
    assert out.shape == (len_x1, len_x1)

    for i in range(len_x1):
        for j in range(len_x1):
            out[i, j] = x1[i] * x2[j]


@jit(nopython=True, cache=True)
def rls_out_of_sample_forecasts(
    x: np.ndarray,
    all_coeffs: np.ndarray,
    ignore_first_n: int | None = None,
    forecast_horizon: int = 12,
) -> list[np.ndarray]:
    """Generate recursive out-of-sample forecasts from coefficient history."""
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
                index_forecasts.append(
                    rls_predict_kernel(coeffs=index_coeffs, x=x[horizon_index, :])
                )

        forecasts.append(np.array(index_forecasts))

    return forecasts


def rls_in_sample_forecasts(
    x: np.ndarray, all_coeffs: np.ndarray, ignore_first_n: int | None = None
) -> list[np.ndarray]:
    """Generate in-sample forecasts from coefficient history."""
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
                rls_predict_kernel(all_coeffs[coeff_index, :], x[:coeff_index, :])
            )
        else:
            forecasts.append(np.array([]))
    return forecasts
