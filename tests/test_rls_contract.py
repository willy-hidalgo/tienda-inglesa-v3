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
