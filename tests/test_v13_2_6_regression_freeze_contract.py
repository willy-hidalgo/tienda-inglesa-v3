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


def test_v13_2_6_restores_v13_1_1_productive_statistical_baseline():
    s = _settings_assignments()
    assert s["APP_VERSION"] == "13.2.6"
    assert tuple(s["LEAF_SES_ALPHA_CANDIDATES"]) == (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80)
    assert tuple(s["RLS_FORGETTING_FACTOR_CANDIDATES"]) == (0.970, 0.985, 0.995)
    assert tuple(s["RLS_VALUE_PRICE_NODE_IDS"]) == ()
    assert s["RLS_DRIVER_GROUP_EXCLUSIONS"] == {}


def test_promoted_v13_2_candidates_are_diagnostic_only_again():
    s = _settings_assignments()
    assert 0.30 in tuple(s["STAT_OPT_SES_EXTRA_ALPHAS"])
    assert 0.990 in tuple(s["STAT_OPT_RLS_EXTRA_LAMBDAS"])
    assert 0.9975 in tuple(s["STAT_OPT_RLS_EXTRA_LAMBDAS"])
    assert 1.0 in tuple(s["STAT_OPT_RLS_EXTRA_LAMBDAS"])
