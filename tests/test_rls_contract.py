import numpy as np
import pytest

from rls_opt import RLSPrior, RLSConstantPrior, RecursiveLeastSquaresRegression


def _model(**kwargs):
    defaults = dict(forgetting_factor=0.98, min_y_to_update=0.01, check_inputs=True)
    defaults.update(kwargs)
    return RecursiveLeastSquaresRegression(**defaults)


def _inputs():
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=float)
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    priors = [RLSConstantPrior(0.5, 0.2), RLSPrior(0.5, 0.5, 1.0)]
    return x, y, priors


def test_fit_produces_finite_coefficients_and_predictions():
    x, y, priors = _inputs()
    model = _model()
    model.fit(x, y, priors)

    assert len(model.final_coef_) == 1
    assert np.isfinite(model.final_coef_[0]).all()
    prediction = model.predict(x)[0]
    np.testing.assert_allclose(prediction, x @ model.final_coef_[0])


def test_check_inputs_rejects_any_non_finite_value():
    x, y, priors = _inputs()
    x[1, 1] = np.nan
    model = _model()

    with pytest.raises(ValueError, match="x is not finite"):
        model.fit(x, y, priors)


def test_check_inputs_rejects_non_finite_y():
    x, y, priors = _inputs()
    y[1] = np.inf
    model = _model()

    with pytest.raises(ValueError, match="y is not finite"):
        model.fit(x, y, priors)


def test_invalid_solver_parameters_are_rejected():
    x, y, priors = _inputs()

    with pytest.raises(ValueError, match="forgetting_factor"):
        _model(forgetting_factor=0).fit(x, y, priors)

    with pytest.raises(ValueError, match="min_y_to_update"):
        _model(min_y_to_update=0).fit(x, y, priors)


def test_multiple_constant_priors_are_rejected():
    x, y, _ = _inputs()
    priors = [RLSConstantPrior(0.5, 0.2), RLSConstantPrior(0.5, 0.2)]

    with pytest.raises(ValueError, match="At most one"):
        _model().fit(x, y, priors)


def test_metrics_flags_fail_explicitly_until_implemented():
    x, y, priors = _inputs()

    with pytest.raises(NotImplementedError, match="metrics"):
        _model().fit(x, y, priors, in_sample_metrics=True)

    with pytest.raises(NotImplementedError, match="metrics"):
        _model().fit(x, y, priors, out_sample_metrics=True)


def test_predict_requires_enough_fitted_models():
    x, _, _ = _inputs()
    model = _model()

    with pytest.raises(ValueError, match="Not enough fitted"):
        model.predict(x)
