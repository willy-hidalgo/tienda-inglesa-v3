# v12.9.8 — Sparse-demand bias rescue diagnostic + safe finalization

## Statistical scope

v12.9.8 **freezes the productive v12.9.7 policy**. No SES/RLS, LightGBM shape,
SKU-total, occurrence/share, Sec23 L4 stress gate, high-impact guard or 15% budget
threshold is changed.

The new sparse-demand layer is diagnostic only. It targets leaves whose latest
**closed** 28-day block has 7–13 positive-sales days (`07-09`, `10-13`). For each
target (Qty/Value), it estimates a section/bucket uplift from up to 12 closed
historical blocks:

1. reconstruct the historical forecast total using the family selected for the
   target period (v11 or v12);
2. aggregate actual/forecast by closed block and sparse bucket;
3. calculate robust ratio statistics (median, p25, worst);
4. require at least 8 folds, median ratio >= 1.05, p25 >= 1.00 and worst >= 0.90;
5. shrink the median correction 50% toward 1.0 and clip to `[1.00, 1.15]`.

The factor is emitted as `v1298_sparse_*` metadata. It **does not modify**
`yhat`, `valuehat`, SKU totals, occurrence gates or store shares. Section 31 of
the acceptance report scores the challenger on current OOS only after the
causal factor has been fixed.

Forecast-only is also causal: the just-observed OOS becomes the newest closed
block, followed by older historical blocks. Forecast-only actuals are never
used.

## Operational fix

A successful `forecast.parquet` is now registered with `mark_success()` before
building dashboard artifacts. The large in-memory forecast frames are released
first. Therefore, an external Rust/Polars memory abort while creating dashboard
artifacts cannot leave a fully-written forecast stuck in `status="running"`.

Dashboard artifacts keep the v12.9.7 memory-safe section-by-section writer.

## Acceptance

After a 28d run:

```bash
uv run python -m app.forecasting.validate_v12 --update-block-days 28
uv run python -m app.dashboard_consistency --update-block-days 28
uv run python -m app.forecasting_acceptance_report --update-block-days 28
```

Required invariants:

- model audit OK;
- dashboard audit OK;
- v12.9.7 Section 30 temporal replay remains PASS;
- Section 31 reports the sparse challenger by section/target;
- promotion to production is **not automatic** in v12.9.8.
