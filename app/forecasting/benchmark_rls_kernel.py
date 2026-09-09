"""Micro-benchmark for the production RLS numerical kernel.

Run with:
    python -m app.forecasting.benchmark_rls_kernel

The first call includes Numba compilation; the second is the steady-state time
that matters when many store/candidate fits share the same process.
"""
from __future__ import annotations

import time
import numpy as np

from rls_opt.kernels import _rls


def main() -> int:
    rng = np.random.default_rng(42)
    n_obs, n_features = 700, 100
    x = np.ascontiguousarray(rng.normal(size=(n_obs, n_features)), dtype=np.float64)
    y = np.ascontiguousarray(np.abs(rng.normal(size=n_obs)) + 1.0, dtype=np.float64)
    priors = np.zeros(n_features, dtype=np.float64)
    inv = np.eye(n_features, dtype=np.float64)

    t0 = time.perf_counter()
    _rls(x[:32], y[:32], priors, inv, 0.995, None, True, 1e-2)
    compile_s = time.perf_counter() - t0

    repeats = 5
    t0 = time.perf_counter()
    for _ in range(repeats):
        _rls(x, y, priors, inv, 0.995, None, True, 1e-2)
    steady_s = (time.perf_counter() - t0) / repeats

    print("RLS NUMBA KERNEL BENCHMARK")
    print(f"shape              : {n_obs} x {n_features}")
    print(f"first/compile call : {compile_s:.3f}s")
    print(f"steady-state fit   : {steady_s:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
