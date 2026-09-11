"""v13.3.3 contracts: robust history cap + common OOS-origin state envelope."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _kernel_without_numba():
    import numpy as np
    source = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ses_walkforward_kernel")
    fn.decorator_list = []
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<v13214-kernel>", "exec"), ns)
    return ns["_ses_walkforward_kernel"]


def _call(y, blocks, periods, *, init=10.0, alpha=0.8, obs_seq=None):
    import numpy as np
    y = np.asarray(y, dtype=float)
    n = len(y)
    if obs_seq is None:
        obs_seq = np.arange(1, n + 1, dtype=np.int64)
    else:
        obs_seq = np.asarray(obs_seq, dtype=np.int64)
    k = _kernel_without_numba()
    return k(
        np.zeros(n, dtype=np.int64),
        np.asarray(blocks, dtype=np.int64),
        np.zeros(n, dtype=np.uint8),
        np.asarray(periods, dtype=np.int8),
        obs_seq,
        np.maximum(obs_seq - 1, 0),
        np.zeros(n, dtype=np.int64),
        y,
        y * 10.0,
        np.ones(n),
        np.ones(n),
        np.full(n, init),
        np.full(n, init * 10.0),
        np.array([alpha]),
        np.array([1], dtype=np.uint8),
        0,
        0,
        0.50,
        2.00,
        2.00,
        7,
        0,
        1,
        28,
        0.025,
        56,
        0.25,
        0.20,
        2.00,
        28,
        1e-12,
        4.00,
        2.00,
        0.75,
        1.25,
        28,
        7,
        1.50,
    )


def test_version_and_artifact_27():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    artifacts = (ROOT / "app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "13.3.3"' in settings
    assert 'LEAF_FORECAST_HISTORY_MEAN_MULTIPLIER: float = 4.00' in settings
    assert 'LEAF_LONG_GAP_FORECAST_MEAN_MULTIPLIER: float = 2.00' in settings
    assert 'LEAF_OOS_STATE_ANCHOR_MIN_FACTOR: float = 0.75' in settings
    assert 'LEAF_OOS_STATE_ANCHOR_MAX_FACTOR: float = 1.25' in settings
    assert 'ARTIFACT_VERSION = 33' in artifacts


def test_history_mean_cap_prevents_old_peak_from_authorizing_high_forecast():
    # Closed history is mostly 1 with one old peak 20. 2*max would allow 40,
    # but 4*positive-mean should be much tighter at the OOS origin.
    y = [1.0] * 9 + [20.0, 0.0]
    periods = [0] * 10 + [1]
    out = _call(y, list(range(len(y))), periods, init=30.0, alpha=0.01)
    mean_hist = (9.0 + 20.0) / 10.0
    assert float(out[6][-1]) <= 4.0 * mean_hist + 1e-12
    assert float(out[0][-1]) <= float(out[6][-1]) + 1e-12


def test_long_gap_tightens_cap_without_decaying_state():
    # Same stale level, but a long observable gap. State remains 10 while the
    # forecast cap tightens to 2x closed positive-history mean.
    out = _call([2.0, 2.0, 2.0, 0.0], [0, 1, 2, 3], [0, 0, 0, 1], init=10.0, alpha=0.1, obs_seq=[1, 2, 3, 100])
    assert abs(float(out[10][-1]) - 1.0) < 1e-12
    assert float(out[6][-1]) <= 4.0 + 1e-12
    assert float(out[2][-1]) > 0.0


def test_oos_state_is_clamped_to_common_origin_envelope():
    # OOS starts from level 10. Repeated large positives would compound a fast
    # cadence upward; after each closed OOS block the latent state cannot exceed
    # 1.25x the common OOS-origin state.
    y = [10.0] * 8 + [100.0, 100.0, 0.0]
    periods = [0] * 8 + [1, 1, 1]
    out = _call(y, list(range(len(y))), periods, init=10.0, alpha=0.8)
    # Level used by final OOS block is the state after previous closure.
    assert float(out[2][-1]) <= 12.5 + 1e-12
