---
name: changes_summary
description: Summary of changes made to forecasting logic for store-level RLS
metadata:
  type: project
---

## Changes Made

### File: app/forecasts.py

#### Problem
Store-level forecasts were not being generated because the code was filtering store data from section-only data (`train_section`) instead of the full training data (`df_train`) which contains all unique_ids.

#### Fix
1. Changed line ~1701 from:
   ```python
   train_store = train_section.filter(pl.col("unique_id") == store_unique_id)
   ```
   to:
   ```python
   train_store = df_train.filter(pl.col("unique_id") == store_unique_id)
   ```

2. Added logging to show number of store-level forecast rows obtained:
   ```python
   logger.info(f"Sección {seccion}: obtenidas {res_store.height} filas de pronósticos a nivel tienda")
   ```

#### Effect
- Section level ("1"): RLS fitting (unchanged)
- Store level ("1||T:xxxxxx"): Now fitted with RLS (same logic as section)
- SKU level ("1||S:xxxxxx"): No estimates (as requested)
- Store+SKU level ("1||T:xxxxxx||S:xxxxxx"): No estimates (as requested)
- Rolling 28d: Still computed only for section level (unchanged)

#### Expected Outcome
- Output forecasts.parquet will include rows for stores (unique_id like "1||T:00122")
- WMAPE calculations will include store-level errors
- Ranking table and graphs should now display store-level forecasts

#### Verification
User should run the pipeline and check:
- Logs for store-level forecast row counts
- forecast.parquet for presence of store-level unique_ids
- Ranking table and graphs showing store data