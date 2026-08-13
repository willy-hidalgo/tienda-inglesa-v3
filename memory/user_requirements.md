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

This means:
- Section level ("1"): RLS fitting (unchanged)
- Store level ("1||T:xxxxxx"): RLS fitting (changed from derive to fit)
- SKU level ("1||S:xxxxxx"): No estimates (changed from derive to none)
- Store+SKU level ("1||T:xxxxxx||S:xxxxxx"): No estimates (changed from derive to none)
- Rolling 28d: Still computed only for section level (as before)