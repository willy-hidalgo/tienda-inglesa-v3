"""v13.3.3 regression contracts: diagnostic gaps, causal reactivation restart."""
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
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<v13213-kernel>", "exec"), ns)
    return ns["_ses_walkforward_kernel"]


def _call_kernel(observable_seq, blocks, periods, y, value, *, init_y=10.0, init_v=100.0, alpha=0.10):
    import numpy as np
    n=len(y)
    k=_kernel_without_numba()
    return k(
        np.zeros(n,dtype=np.int64), np.asarray(blocks,dtype=np.int64),
        np.zeros(n,dtype=np.uint8), np.asarray(periods,dtype=np.int8),
        np.asarray(observable_seq,dtype=np.int64),
        np.asarray([max(int(x)-1,0) for x in observable_seq],dtype=np.int64),
        np.zeros(n,dtype=np.int64), np.asarray(y,dtype=float), np.asarray(value,dtype=float),
        np.ones(n), np.ones(n), np.full(n,init_y), np.full(n,init_v),
        np.array([alpha]), np.array([1],dtype=np.uint8),
        0,0,0.50,2.00,2.00,7,0,1,28,0.025,56,0.25,0.20,2.00,
    )


def test_version_and_artifact_contract():
    settings=(ROOT/"settings.py").read_text(encoding="utf-8")
    artifacts=(ROOT/"app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "13.3.3"' in settings
    assert 'LEAF_GAP_AWARE_ENABLED: bool = True' in settings
    assert 'LEAF_GAP_REACTIVATION_STATE_FACTOR: float = 2.00' in settings
    assert 'RELEASE_GATE_MIN_CROSS_CADENCE_SCALED_GAP: float = 2.0' in settings
    assert 'ARTIFACT_VERSION = 33' in artifacts


def test_long_gap_is_trace_only_and_does_not_decay_forecast_state():
    out=_call_kernel([1,701],[0,1],[0,1],[10.0,10.0],[100.0,100.0])
    assert abs(float(out[2][1])-10.0)<1e-12
    assert abs(float(out[3][1])-100.0)<1e-12
    assert float(out[10][1])==1.0
    assert float(out[11][1])==1.0
    assert int(out[8][1])>=699


def test_long_gap_reactivation_reanchors_only_next_block():
    # stale state 100, first real reactivation 10 after a long gap.
    # Forecast for the reactivation block stays 100 (no leakage); after close
    # the stale state is clamped to <=2*10 and normal SES update applies.
    out=_call_kernel([1,62,63],[0,1,2],[0,1,1],[100.0,10.0,10.0],[1000.0,100.0,100.0],init_y=100.0,init_v=1000.0,alpha=0.10)
    assert float(out[0][1]) >= 99.0
    assert float(out[2][2]) <= 20.0 + 1e-12
    assert float(out[3][2]) <= 200.0 + 1e-12


def test_no_productive_gap_decay_code_remains():
    text=(ROOT/"app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert 'block_level_y = state_y[best_y] * selected_decay_y' in text
    assert 'frozen_gap_factor_y' not in text and 'frozen_gap_factor_v' not in text
    assert 'sy *= event_decay' not in text
    assert 'sv *= event_decay' not in text
