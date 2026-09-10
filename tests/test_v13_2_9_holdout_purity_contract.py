"""v13.2.17 contracts: OOS is validation/recurrence, never tuning."""
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


def test_v13_2_9_keeps_statistical_baseline_frozen():
    s = _settings_assignments()
    assert s["APP_VERSION"] == "13.2.17"
    assert tuple(s["LEAF_SES_ALPHA_CANDIDATES"]) == (
        0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80
    )
    assert tuple(s["RLS_FORGETTING_FACTOR_CANDIDATES"]) == (0.970, 0.985, 0.995)
    assert tuple(s["RLS_VALUE_PRICE_NODE_IDS"]) == ()
    assert s["RLS_DRIVER_GROUP_EXCLUSIONS"] == {}


def test_oos_errors_never_select_rls_dynamics_or_lambda():
    text = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    assert 'history_mask = source_period[s:e] == "in_sample"' in text
    assert "valid_y = history_mask & np.isfinite" in text
    assert "cum_ae_y[c] +=" in text and "cum_ae_v[c] +=" in text


def test_oos_errors_never_select_leaf_alpha():
    text = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "if pc == 0:  # only closed in-sample history selects alpha" in text
    assert "if pc == 0:  # same holdout purity for Value" in text
    # OOS still exists in diagnostics and can update the operational state.
    assert "collect_diagnostics == 1 and pc <= 1" in text
    assert "period_codes[r] == 2" in text


def test_oos_errors_never_select_store_vs_section_parent():
    text = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    # Four metric terms (qty ae/den, value ae/den) are history-only.
    assert text.count('(pl.col("period_type") == "in_sample")') >= 4
    assert "parent_prior_wmape_y" in text
    assert "parent_prior_wmape_value" in text


def test_identifiable_transfer_has_explicit_causal_center_trace():
    s = _settings_assignments()
    assert s["LEAF_DRIVER_REFERENCE_SHIFT_FACTOR"] == 1.50
    text = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "calibration_center" in text
    assert "driver_effect_center" in text
    assert "driver_effect_value_center" in text
    validator = (ROOT / "app/forecasting/validate_v13.py").read_text(encoding="utf-8")
    assert "raw-reference-center + guard" in validator


def test_ses_alpha_is_frozen_by_history_when_oos_errors_reverse_preference_without_polars():
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
    exec(compile(module, "<ses-kernel>", "exec"), namespace)
    kernel = namespace["_ses_walkforward_kernel"]

    # The exact history-selected alpha may change under the v13.2.17 robust
    # recurrence, but once OOS starts the selection must remain frozen.
    y = np.array([10., 20., 40., 40., 5., 5., 200., 200., 5., 5., 200., 50.])
    value = y * 10.0
    n = len(y)
    result = kernel(
        np.zeros(n, dtype=np.int64),          # one leaf
        np.arange(n, dtype=np.int64),        # 1d blocks
        np.zeros(n, dtype=np.uint8),          # post-warmup fixture
        np.r_[np.zeros(6, dtype=np.int8), np.ones(6, dtype=np.int8)],
        np.arange(1, n + 1, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        y,
        value,
        np.ones(n),
        np.ones(n),
        np.full(n, 10.0),
        np.full(n, 100.0),
        np.array([0.05, 0.80]),
        np.array([1, 1], dtype=np.uint8),
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
    alpha_y = result[4]
    assert alpha_y[6] in (0.05, 0.80)
    assert np.all(alpha_y[6:] == alpha_y[6])
