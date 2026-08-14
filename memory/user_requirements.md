---
name: user_requirements
description: User requirements for modifying the forecasting logic to apply RLS to stores
metadata:
  type: user
---

User wants to modify the forecasting pipeline to:
1. Keep the section-level RLS logic unchanged (do not modify)
2. Apply the same pure RLS logic to stores (instead of deriving from section coefficients)
3. For now, do not calculate estimates at the store/sku level (skip derivation for SKU and store+SKU levels)
4. UPDATE: For store+SKU level (section+store+SKU), in the in-sample period, calculate two forecast columns using section and store model coefficients respectively, and select the better model based on wMAPE (then BIAS closer to zero)

This means:
- Section level ("1"): RLS fitting (unchanged)
- Store level ("1||T:xxxxxx"): RLS fitting (changed from derive to fit)
- Store+SKU level ("1||T:xxxxxx||S:SKUxxx"): 
   * In-sample: Calculate forecasts using both section and store models, select better based on wMAPE then BIAS
   * Forecast_only: No estimates (as per initial request)
- SKU level ("1||S:SKUxxx"): Not explicitly addressed in user request (would require additional specification)
- Rolling 28d: Still computed only for section level (as before)