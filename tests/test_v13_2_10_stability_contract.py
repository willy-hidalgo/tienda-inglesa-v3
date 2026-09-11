"""v13.3.3 contracts: frozen OOS selection + robust causal leaf stability."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def _settings_assignments():
    tree = ast.parse((ROOT / "settings.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


def _kernel_without_numba():
    import numpy as np

    source = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ses_walkforward_kernel"
    )
    fn.decorator_list = []
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, "<ses-kernel-v13210>", "exec"), namespace)
    return namespace["_ses_walkforward_kernel"]


def test_v13_2_10_keeps_statistical_baseline_and_adds_only_stability_guards():
    s = _settings_assignments()
    assert s["APP_VERSION"] == "13.3.3"
    assert tuple(s["LEAF_SES_ALPHA_CANDIDATES"]) == (
        0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80
    )
    assert tuple(s["RLS_FORGETTING_FACTOR_CANDIDATES"]) == (0.970, 0.985, 0.995)
    assert tuple(s["RLS_VALUE_PRICE_NODE_IDS"]) == ()
    assert s["RLS_DRIVER_GROUP_EXCLUSIONS"] == {}
    assert s["LEAF_SES_UPDATE_FACTOR_MIN"] == 0.50
    assert s["LEAF_SES_UPDATE_FACTOR_MAX"] == 2.00
    assert s["LEAF_FORECAST_HISTORY_MAX_MULTIPLIER"] == 2.00
    assert s["LEAF_FORECAST_GUARD_MIN_POSITIVE_POINTS"] == 7


def test_oos_model_selection_is_explicitly_snapshotted():
    leaf = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    runner = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    assert "frozen_best_y" in leaf and "frozen_best_v" in leaf
    assert "_parent_frozen_y" in leaf and "_parent_frozen_v" in leaf
    assert "frozen_choice_y" in runner and "frozen_choice_v" in runner
    assert "block_has_history" in runner


def test_robust_ses_update_prevents_one_peak_from_resetting_level():
    import numpy as np

    kernel = _kernel_without_numba()
    # Eight stable history points, one extreme history peak, then OOS. With
    # alpha=.8 the unguarded SES would jump near 800; v13.3.3 constrains the
    # deseasonalized innovation to at most 2x the pre-event state.
    y = np.array([10.] * 8 + [1000., 10., 10.])
    n = len(y)
    periods = np.array([0] * 9 + [1, 1], dtype=np.int8)
    result = kernel(
        np.zeros(n, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.uint8),
        periods,
        np.arange(1, n + 1, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        y,
        y * 10.0,
        np.ones(n),
        np.ones(n),
        np.full(n, 10.0),
        np.full(n, 100.0),
        np.array([0.80]),
        np.array([1], dtype=np.uint8),
        0,
        0,
        0.50,
        2.00,
        2.00,
        7,
        1,
        1,
        28,
    )
    # Forecast immediately after the 1000-unit peak remains near the prior
    # operating scale rather than inheriting the peak.
    assert float(result[0][9]) <= 20.0 + 1e-12
    assert float(result[2][9]) <= 20.0 + 1e-12


def test_causal_history_cap_is_block_origin_only_and_traced():
    import numpy as np

    kernel = _kernel_without_numba()
    # Very slow alpha keeps the latent state far above actuals. After seven
    # closed positive observations with max=10, the next block must be capped
    # at 2*10=20 using only prior history.
    y = np.array([10.] * 8 + [10., 10.])
    n = len(y)
    result = kernel(
        np.zeros(n, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.uint8),
        np.array([0] * 8 + [1, 1], dtype=np.int8),
        np.arange(1, n + 1, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        y,
        y * 10.0,
        np.ones(n),
        np.ones(n),
        np.full(n, 100.0),
        np.full(n, 1000.0),
        np.array([0.005]),
        np.array([1], dtype=np.uint8),
        0,
        0,
        0.50,
        2.00,
        2.00,
        7,
        1,
        1,
        28,
    )
    yhat, vhat = result[0], result[1]
    cap_y, cap_v = result[6], result[7]
    assert cap_y[8] == 20.0
    assert cap_v[8] == 200.0
    assert yhat[8] <= cap_y[8] + 1e-12
    assert vhat[8] <= cap_v[8] + 1e-12


def test_validator_and_dashboard_audit_know_about_v13210_guards():
    validator = (ROOT / "app/forecasting/validate_v13.py").read_text(encoding="utf-8")
    consistency = (ROOT / "app/dashboard_consistency.py").read_text(encoding="utf-8")
    assert "leaf_forecast_cap_y" in validator
    assert "leaf_forecast_cap_value" in validator
    assert "yhat_raw excede leaf_forecast_cap_y" in validator
    artifacts = (ROOT / "app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert 'ARTIFACT_VERSION = 33' in artifacts
    assert '"rls_metric_eligible"' in artifacts
    assert '_WMAPE_COLS' in artifacts
    assert 'sort("unique_id")' in consistency
    assert "scale-aware tolerance" in consistency
