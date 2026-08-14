---
name: final_implementation_summary
description: Summary of the final implementation for store-level RLS with SKU-level model selection
metadata:
  type: project
---

## Implementation Summary

### Requirements Fulfilled
1. ��� � � ✅ **Section-level logic unchanged**: RLS fitting for section level ("1") remains exactly as before
2. ��� � � ✅ **Store-level RLS fitting**: Stores ("1||T:xxxxxx") now use pure RLS fitting (same logic as section) instead of derivation
3. ��� � � ✅ **SKU-level model selection**: For in-sample period, store+SKU level ("1||T:xxxxxx||S:SKUxxx") calculates forecasts using both section and store model coefficients, selecting the better one based on wMAPE (then BIAS closer to zero) - **WITHOUT SES**
4. ��� � � ✅ **Forecast-only period**: No estimates calculated for SKU and store+SKU levels (as requested initially)
5. ��� � � ✅ **Rolling 28d unchanged**: Still computed only for section level as before

### Key Changes in `app/forecasts.py`

#### Section Fitting (lines 1657-1666)
- Changed variable name from `coefs` to `section_coefs_raw` for clarity
- Added extraction: `section_coefs = {seccion: section_coefs_raw[seccion]} if section_coefs_raw and seccion in section_coefs_raw else {}`

#### Store Fitting (lines 1710-1722)
- Added coefficient storage: 
  ```python
  # Almacenar coeficientes de esta tienda
  if store_coefs_raw and store_unique_id in store_coefs_raw:
      store_coefs[store_unique_id] = store_coefs_raw[store_unique_id]
  ```

#### SKU-Level Model Selection (lines 1748-1930)
- **In-sample only**: Uses train + out_sample data, excludes forecast_only
- **Model comparison**: For each store+SKU unique_id:
  - Section model forecast: `yhat_section = X_y @ coef_section_y` (direct driver effect, NO SES)
  - Store model forecast: `yhat_store = X_y @ coef_store_y` (direct driver effect, NO SES)
  - Selection criteria: 
    - Lower wMAPE wins
    - If tied, BIAS closer to zero wins
- **Forecast calculation**: Uses **direct driver effect only** (no SES application) for in-sample as requested
- **Result combination**: Combines section, store, and SKU-tienda results

### Expected Behavior
- **Section level ("1")**: RLS fitting (unchanged)
- **Store level ("1||T:xxxxxx"): RLS fitting (now active instead of derivation)
- **Store+SKU level** ("1||T:xxxxxx||S:SKUxxx"):
  - In-sample: Selection between section model and store model (direct driver effects only, NO SES) → pronósticos
  - Forecast-only: No estimates (as se solicitó inicialmente)
- **Rolling 28d**: Solo nivel sección (sin cambios)

### Important Clarification on SES
As per user feedback: "en in-sample no vamos a usar SES para nada en absoluto"
- For **in-sample** forecasts of store+SKU level: We use ONLY the direct driver effect (`X_y @ coef_y`) with rounding
- NO causal SES (`ewm_mean`) is applied to in-sample forecasts
- SES is still used for:
  - Section level forecast_only predictions (existing logic)
  - Store level forecast_only predictions (existing logic)

### Verification Points
Users should verify:
1. Logs show store-level forecast counts: `Sección 1: obtenidas X filas de pronósticos a nivel tienda`
2. forecast.parquet contains store-level unique_ids ("1||T:xxxxxx")
3. forecast.parquet contains store+SKU level unique_ids ("1||T:xxxxxx||S:SKUxxx") with values reflecting:
   - In-sample: Direct driver effect from selected model (no SES)
   - Forecast-only: Zeros or empty (as requested)
4. Ranking table and gráficos ahora muestren datos a nivel tienda
5. No forecast_only estimates for SKU/store+SKU levels (zeros or empty as appropriate)