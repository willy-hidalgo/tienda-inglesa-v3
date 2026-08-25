# Speed + OOS accuracy refactor — 2026-08-19

## Why wMAPE could reach millions of percent

The metric itself was mathematically correct: `sum(abs(y-yhat))/sum(abs(y))`, with
`y == 0` excluded.  A leaf with very small positive actual demand and an exploded
log-space forecast can therefore produce arbitrarily large percentages.  The real
bug was allowing unstable leaf `valuehat`/`yhat` forecasts to reach the metric.
Those leaf errors then propagated bottom-up to store and section.

## Production leaf model

`FAST_LEAF_MODE=True` is now the default.  RLS remains on section/store, but
SKU+store no longer receives a dense full-history EDP/features/RLS/SES pipeline.
Future leaf forecasts use robust profiles learned only from train:

- section 1 units: positive median, recent 28 days
- section 1 value: positive median, recent 56 days
- section 23 units/value: same-weekday positive median, recent 56 days

The full leaf history stays sparse (observed days); only the short OOS+forecast
horizon is made dense.  This removes the `n_leaves * full_history_days` explosion.

## Benchmark on bundled raw sales files

Client wMAPE definition, same configured OOS windows:

| Section | Target | Fast method | Section bottom-up OOS wMAPE | Leaf median | Leaf P75 |
|---|---|---|---:|---:|---:|
| 1 | Units | median28 | 51.70% | 35.71% | 54.55% |
| 1 | Value | median56 | 48.78% | 33.33% | 50.00% |
| 23 | Units | weekday56 | 46.42% | 44.44% | 62.50% |
| 23 | Value | weekday56 | 45.77% | 44.17% | 62.01% |

Store medians were approximately 51.9%, 46.7%, 48.4% and 44.6% respectively.
These are independent raw-data benchmarks, not a claim that the full application
run will be bit-for-bit identical before it is executed with Polars in the target
environment.

## RLS and the 28-day interval

The prior code did **not** perform a full RLS refit every 28 days;
`COMPUTE_ROLLING_28=False`.  RLS is recursive: inside one `fit()` the coefficients
are updated observation by observation in chronological order.  This is more
frequent than a 28-day block refit and is much cheaper.

`RLS_VALIDATION_BLOCK_DAYS=28` now documents the reporting/backtest interval.
A costly full refit every 28 days should be an optional backtest operation, not
the production default, because it adds runtime without improving the recursive
update semantics.

## Complexity change

Before: all leaf series were densified across the full train horizon and then EDP
and calendar/price features were computed for that huge panel.

After: only section/store nodes are densified across full history.  Leaves remain
sparse in train and are densified only across the short OOS+forecast horizon.
