"""Regression contracts for v13.2.17 parent-RLS→leaf transfer and release gate."""
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


def test_leaf_driver_transfer_uses_identifiable_parent_forecast_relative_level():
    s = _settings_assignments()
    assert s["APP_VERSION"] == "13.2.17"
    assert s["LEAF_DRIVER_REFERENCE_DAYS"] == 28
    assert s["LEAF_DRIVER_REFERENCE_MIN_POINTS"] == 7
    assert s["LEAF_DRIVER_REFERENCE_SHIFT_FACTOR"] == 1.50
    assert s["LEAF_DRIVER_FACTOR_MIN"] == 0.50
    assert s["LEAF_DRIVER_FACTOR_MAX"] == 2.00

    text = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "def _relative_parent_forecast_effect_path" in text
    assert "parent_rls_forecast_y" in text
    assert "parent_reference_level_y" in text
    assert "rls_parent_forecast_relative_centered_causal" in text
    assert "raw - ref_log" in text
    assert "parent_effect_coef_raw_y" in text  # audit only


def test_validator_rejects_driver_explosion_and_wrong_transfer_source():
    text = (ROOT / "app/forecasting/validate_v13.py").read_text(encoding="utf-8")
    assert "driver_factor_y fuera del guard" in text
    assert "driver_factor_value fuera del guard" in text
    assert "driver_effect_raw != log1p(forecast RLS parent)" in text
    assert "driver_effect_reference != log1p(nivel causal parent)" in text


def test_release_gate_is_mandatory_in_dashboard_ready_all():
    ready = (ROOT / "app/dashboard_ready.py").read_text(encoding="utf-8")
    gate = (ROOT / "app/forecasting/regression_gate.py").read_text(encoding="utf-8")
    main = (ROOT / "app/main.py").read_text(encoding="utf-8")
    assert "app.forecasting.regression_gate" in ready
    assert "--all-update-blocks" in ready
    assert "RELEASE_GATE_SENTINELS" in gate
    assert "RELEASE_GATE_MAX_CROSS_CADENCE_MEDIAN_RATIO" in gate
    assert "regression-gate-all" in main


def test_v13_1_1_productive_parameter_baseline_stays_frozen():
    s = _settings_assignments()
    assert tuple(s["LEAF_SES_ALPHA_CANDIDATES"]) == (
        0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80
    )
    assert tuple(s["RLS_FORGETTING_FACTOR_CANDIDATES"]) == (0.970, 0.985, 0.995)
    assert tuple(s["RLS_VALUE_PRICE_NODE_IDS"]) == ()
    assert s["RLS_DRIVER_GROUP_EXCLUSIONS"] == {}


def test_relative_parent_helper_caps_spike_and_is_block_causal_without_polars():
    import math
    import numpy as np

    source = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_relative_parent_forecast_effect_path"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, "<relative-parent-helper>", "exec"), namespace)
    helper = namespace["_relative_parent_forecast_effect_path"]

    actual = np.full(60, 100.0)
    forecast = np.full(60, 100.0)
    forecast[35] = 100000.0
    blocks = np.arange(60, dtype=np.int64)
    support = np.ones(60, dtype=bool)
    effect, ref, _, _, center = helper(
        actual, forecast, blocks, support,
        reference_days=28, reference_min_points=7,
        reference_shift_factor=1.50,
        factor_min=0.50, factor_max=2.00,
    )
    assert math.exp(float(effect[35])) == 2.0
    assert float(ref[35]) == 100.0

    # 28d: actuals inside the target block cannot alter that block reference.
    blocks28 = np.repeat(np.arange(3), 28)[:60]
    actual28 = np.r_[np.full(28, 100.0), np.full(28, 1000.0), np.full(4, 1000.0)]
    effect28, ref28, _, _, _ = helper(
        actual28, np.full(60, 100.0), blocks28, np.ones(60, dtype=bool),
        reference_days=28, reference_min_points=7,
        reference_shift_factor=1.50,
        factor_min=0.50, factor_max=2.00,
    )
    assert np.all(ref28[28:56] == 100.0)
    assert np.all(ref28[56:] == 1000.0)
    assert np.isfinite(effect28).all()

    # Persistent identifiable block-wide uplift is level/calibration, not a
    # leaf driver. A 14-day 2x shift is centered to ~1x while within-block
    # deviations would remain available.
    blocks14 = np.repeat(np.arange(5), 14)
    actual14 = np.full(70, 100.0)
    forecast14 = np.full(70, 100.0)
    forecast14[56:70] = 220.0
    effect14, _, _, _, center14 = helper(
        actual14, forecast14, blocks14, np.ones(70, dtype=bool),
        reference_days=28, reference_min_points=7,
        reference_shift_factor=1.50,
        factor_min=0.50, factor_max=2.00,
    )
    assert np.max(np.abs(effect14[56:70])) < 1e-12
    assert np.min(center14[56:70]) > 0.0
