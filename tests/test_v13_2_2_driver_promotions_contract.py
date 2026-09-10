"""Promociones v13.2.2: solo grupos existentes del mismo RLS."""
from pathlib import Path

import settings


ROOT = Path(__file__).resolve().parent.parent


def test_v13_2_2_driver_exclusions_are_frozen_out_of_production_in_v13_2_6():
    assert settings.RLS_DRIVER_GROUP_EXCLUSIONS == {}


def test_runner_applies_node_target_driver_design_to_history_and_future():
    text = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    assert 'unit_driver_cols = self._driver_columns_for_node(uid, "Unidades")' in text
    assert 'value_driver_cols = self._driver_columns_for_node(uid, "Valor ($)")' in text
    assert 'Xy_base = self._finite_matrix(base, unit_driver_cols' in text
    assert 'Xyf_base = self._finite_matrix(fcst_g, unit_driver_cols' in text
    assert 'ignore_exclusion_groups=frozenset({str(group)})' in text
    assert 'test_action="add"' in text


def test_promotions_do_not_create_new_model_family_or_pandas():
    text = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8").lower()
    assert "lightgbm" not in text
    assert "occurrence" not in text
    assert "pandas" not in text
