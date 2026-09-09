from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "app" / "forecasting" / "leaf_model.py"

spec = importlib.util.spec_from_file_location("leaf_model_v11", MODULE_PATH)
leaf_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = leaf_model
spec.loader.exec_module(leaf_model)


def test_299994_is_transient_not_trend_and_resets_near_structural_level():
    # Reconstructed pre-OOS 28-day value means for SKU 299994 @ store 00001.
    b0 = 6899.128929
    b1 = 5090.930714
    b2 = 582.746071
    b3 = 794.917857
    prior = [
        b1,
        b2,
        b3,
        601.285714,
        800.935714,
        1013.436071,
        1890.148214,
        1826.832500,
    ]
    d = leaf_model.classify_regime(
        b0, b1, b2, b3, prior_block_means=prior
    )
    assert d.kind == "transient_up"
    assert d.reset_level is True
    assert 750.0 < d.structural_level < 850.0

    level, status = leaf_model.reset_ses_level(6538.334881, d)
    assert status == "SES_RESET_TRANSIENT_UP"
    assert level == d.structural_level
    assert level < 0.20 * 6538.334881


def test_real_persistent_trend_requires_three_coherent_transitions():
    d = leaf_model.classify_regime(
        160.0,
        140.0,
        120.0,
        100.0,
        prior_block_means=[140.0, 120.0, 100.0, 95.0, 90.0],
    )
    assert d.kind == "trend_up"
    assert d.reset_level is False


def test_single_large_jump_is_transient():
    d = leaf_model.classify_regime(
        500.0,
        110.0,
        105.0,
        100.0,
        prior_block_means=[110.0, 105.0, 100.0, 98.0, 102.0],
    )
    assert d.kind == "transient_up"
    assert d.reset_level is True


def test_two_large_recent_blocks_do_not_become_trend():
    d = leaf_model.classify_regime(
        500.0,
        400.0,
        100.0,
        95.0,
        prior_block_means=[400.0, 100.0, 95.0, 105.0, 90.0, 100.0],
    )
    assert d.kind == "transient_up"
    assert d.reset_level is True




def test_261298_transient_down_uses_reactive_ses_not_old_structural_reset():
    # Reconstructed pre-OOS value-block means for 261298 @ 00001.
    d = leaf_model.classify_regime(
        477.4775,
        1886.396786,
        3509.134286,
        2585.200357,
        prior_block_means=[
            1886.396786,
            3509.134286,
            2585.200357,
            1650.0,
            1400.0,
            1500.0,
            1200.0,
            1300.0,
        ],
    )
    assert d.kind == "transient_down"
    assert d.reset_level is False

    alpha, status = leaf_model.alpha_for_regime(0.005, d)
    assert alpha == 0.20
    assert status == "SES_REACTIVE_TRANSIENT_DOWN"

    # The old failure used an ultra-slow alpha and stayed around 1.5k.
    # Pure SES alpha=.20 on the actual pre-OOS history ends near 51,
    # which is on the same scale as the observed OOS mean (~46).
    assert alpha > 0.005


def test_594880_transient_up_uses_recent_robust_structural_level():
    d = leaf_model.classify_regime(
        5768.89,
        2378.55,
        1963.76,
        2213.75,
        prior_block_means=[
            2378.55, 1963.76, 2213.75, 2815.10, 2367.95
        ],
    )
    assert d.kind == "transient_up"
    assert d.reset_level is True
    assert 2300.0 < d.structural_level < 2450.0


def test_587833_transient_up_does_not_follow_latest_spike():
    d = leaf_model.classify_regime(
        1430.25,
        139.54,
        103.46,
        374.88,
        prior_block_means=[
            139.54, 103.46, 374.88, 929.10, 454.80
        ],
    )
    assert d.kind == "transient_up"
    assert d.reset_level is True
    assert 350.0 < d.structural_level < 400.0

def test_parent_choice_is_always_store_or_section():
    assert leaf_model.choose_parent(0.20, 0.30) == "store"
    assert leaf_model.choose_parent(0.40, 0.25) == "section"
    assert leaf_model.choose_parent(None, 0.25) == "section"
    assert leaf_model.choose_parent(0.25, None) == "store"
    assert leaf_model.choose_parent(None, None) == "store"
    assert leaf_model.choose_parent(None, None, default="section") == "section"


def test_source_contract_has_no_leaf_none_model_or_partial_strengths():
    runner = (ROOT / "app" / "forecasting" / "runner.py").read_text(
        encoding="utf-8"
    )
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert '{"_parent": "none"' not in runner
    assert 'pl.lit("none")' not in runner
    assert "LEAF_DRIVER_STRENGTH_CANDIDATES" not in settings
    assert 'LEAF_PARENT_DRIVER_STRENGTH: float = 1.00' in settings
    assert "Leaf parent invariant violated" in runner


def test_source_contract_oos_price_drivers_are_causal():
    pipeline = (ROOT / "app" / "forecasting" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert 'drivers OOS causales' in pipeline
    assert "self._calculate_edp(df_oos)" not in pipeline
    assert "self._carry_forward_prices(df_train, df_oos)" in pipeline


def test_source_contract_forecast_only_does_not_overlap_actuals():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'oos_end = last_actual' in settings
    assert 'forecast_start = oos_end + dt.timedelta(days=1)' in settings
    assert '"observed_tail_start": None' in settings
    assert '"actual_extension_end": None' in settings


def test_runtime_driver_invariant_audits_complete_horizon_not_rls_fragments():
    runner = (ROOT / "app" / "forecasting" / "runner.py").read_text(
        encoding="utf-8"
    )
    # forecast_only is normalized over its complete 28-day horizon. It can cross
    # an internal expanding-28 block, so the invariant must not fragment it.
    assert '.group_by([uid_col, "period_type"])' in runner
    assert '.group_by([uid_col, "period_type", "rls_block"])' not in runner
    assert 'Leaf horizon invariant violated' in runner
