"""RLS update semantics and 28-day validation contract."""
from __future__ import annotations

import numpy as np

import settings
from rls_opt import RecursiveLeastSquaresRegression, RLSConstantPrior, RLSPrior


def test_validation_block_is_28_days():
    assert settings.RLS_BLOCK_DAYS == 28


def test_rls_updates_recursively_per_observation_not_only_every_28_days():
    # RLS is recursive: return_all_coefs exposes one coefficient state per row.
    n = 35
    x = np.column_stack([np.ones(n), np.linspace(0.0, 1.0, n)])
    y = 1.0 + 2.0 * x[:, 1]
    model = RecursiveLeastSquaresRegression(
        forgetting_factor=0.995,
        min_y_to_update=1e-8,
        return_all_coefs=True,
    )
    priors = [
        RLSConstantPrior(standard_error=0.5, rmse_error=0.2),
        RLSPrior(coefficient=0.0, standard_error=0.5, rmse_error=0.2),
    ]
    model.fit(x=x, y=y, priors=priors)
    coef_path = model.all_coef_[0]
    assert coef_path.shape == x.shape
    # Coefficients already change before day 28: no block-only refit behavior.
    assert np.any(np.abs(np.diff(coef_path[:10], axis=0)) > 1e-12)
