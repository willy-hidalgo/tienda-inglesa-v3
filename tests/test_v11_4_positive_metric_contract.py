from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_leaf_fallback_uses_positive_sale_history():
    src = (ROOT / 'app' / 'forecasting' / 'leaf_fallback.py').read_text(encoding='utf-8')
    assert 'Positive-sale deseasonalized history' in src
    assert '& (pl.col(actual_col) > 0.0)' in src
    assert 'pl.col("_clipped").mean().alias("_level")' in src
    assert 'LEAF_FALLBACK_MIN_POSITIVE_HISTORY' in src


def test_leaf_selection_scores_positive_actual_only():
    src = (ROOT / 'app' / 'forecasting' / 'runner.py').read_text(encoding='utf-8')
    assert 'Official client objective: score ONLY dates with' in src
    assert 'alias("_ses_ae_pos_y")' in src
    assert 'alias("_ses_signed_pos_y")' in src
    assert 'alias("_parent_ae_pos_y")' in src
    assert '_ses_ae_adjust_y' not in src
    assert '_parent_ae_adjust_y' not in src


def test_version_and_min_positive_history():
    import ast
    settings_src = (ROOT / 'settings.py').read_text(encoding='utf-8')
    tree = ast.parse(settings_src)
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                vals[node.target.id] = ast.literal_eval(node.value)
            except Exception:
                pass
    assert vals['APP_VERSION'] == '12.9.12'
    assert vals['LEAF_FALLBACK_MIN_POSITIVE_HISTORY'] >= 7
