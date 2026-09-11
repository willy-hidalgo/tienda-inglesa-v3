from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_gap_staleness_recomputed_each_block_origin_without_state_decay():
    text = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "frozen_gap_factor_y" not in text
    assert "frozen_gap_factor_v" not in text
    assert "selected_decay_y = candidate_decay_y" in text
    assert "selected_decay_v = candidate_decay_v" in text
    assert "block_level_y = state_y[best_y] * selected_decay_y" in text
    assert "block_level_v = state_v[best_v] * selected_decay_v" in text
