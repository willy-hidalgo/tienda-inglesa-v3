from __future__ import annotations

import numpy as np
import pytest

from rls_opt import RLSConstantPrior, RLSPrior, RecursiveLeastSquaresRegression


def priors(n=2):
    return [RLSConstantPrior(standard_error=0.5, rmse_error=0.2)] + [
        RLSPrior(coefficient=0.0, standard_error=0.5, rmse_error=0.2)
        for _ in range(n - 1)
    ]


def model(**kw):
    return RecursiveLeastSquaresRegression(forgetting_factor=kw.get("forgetting_factor", 0.99), min_y_to_update=kw.get("min_y_to_update", 1e-8))


def test_fit_predict_finite():
    x=np.c_[np.ones(20),np.arange(20,dtype=float)]
    y=1+2*x[:,1]
    m=model(); m.fit(x,y,priors()); p=m.predict(x)[0]
    assert np.isfinite(p).all()

@pytest.mark.parametrize("bad", [np.nan,np.inf,-np.inf])
def test_reject_nonfinite_x(bad):
    x=np.c_[np.ones(10),np.arange(10,dtype=float)];x[3,1]=bad;y=np.arange(10,dtype=float)+1
    with pytest.raises(ValueError): model().fit(x,y,priors())


def test_reject_invalid_forgetting_factor():
    x=np.c_[np.ones(5),np.arange(5,dtype=float)];y=np.arange(5,dtype=float)+1
    with pytest.raises(ValueError): model(forgetting_factor=1.1).fit(x,y,priors())


def test_reject_multiple_constant_priors():
    x=np.c_[np.ones(5),np.arange(5,dtype=float)];y=np.arange(5,dtype=float)+1
    ps=[RLSConstantPrior(.5,.2),RLSConstantPrior(.5,.2)]
    with pytest.raises(ValueError): model().fit(x,y,ps)


def test_unimplemented_metrics_fail_explicitly():
    x=np.c_[np.ones(5),np.arange(5,dtype=float)];y=np.arange(5,dtype=float)+1
    with pytest.raises(NotImplementedError): model().fit(x,y,priors(),in_sample_metrics=True)


def test_seed_28_boundary_is_independent_of_future_actuals():
    """El modelo usado para días 29-56 no puede depender de actuals posteriores."""
    x = np.c_[np.ones(84), np.linspace(0.0, 1.0, 84)]
    y1 = 5.0 + 2.0 * x[:, 1]
    y2 = y1.copy()
    y2[28:] = 500.0 + 100.0 * x[28:, 1]

    m1 = RecursiveLeastSquaresRegression(
        forgetting_factor=0.99, min_y_to_update=1e-8, return_all_coefs=True
    )
    m2 = RecursiveLeastSquaresRegression(
        forgetting_factor=0.99, min_y_to_update=1e-8, return_all_coefs=True
    )
    m1.fit(x, y1, priors(), seed_n_obs=28)
    m2.fit(x, y2, priors(), seed_n_obs=28)

    # Snapshot after day 28 is identical: no future leakage.
    assert np.allclose(m1.all_coef_[0][27], m2.all_coef_[0][27], rtol=1e-12, atol=1e-12)


def test_expanding_state_changes_after_next_28_actuals():
    """El modelo para 57-84 incorpora los actuals 29-56."""
    x = np.c_[np.ones(84), np.linspace(0.0, 1.0, 84)]
    y = np.r_[5.0 + 2.0 * x[:28, 1], 20.0 + 9.0 * x[28:, 1]]
    m = RecursiveLeastSquaresRegression(
        forgetting_factor=0.99, min_y_to_update=1e-8, return_all_coefs=True
    )
    m.fit(x, y, priors(), seed_n_obs=28)
    c28 = m.all_coef_[0][27]
    c56 = m.all_coef_[0][55]
    assert not np.allclose(c28, c56)
