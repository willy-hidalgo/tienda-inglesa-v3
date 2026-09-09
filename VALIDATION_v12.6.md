# Validación v12.6.0 — pooled LightGBM SHAPE SKU-total

## Contrato causal

- Un modelo pooled por sección/target; nunca SKU×tienda.
- Cada target se ajusta con 16 bloques de 28 días estrictamente anteriores.
- OOS no entra al ajuste OOS; para forecast-only el OOS ya cerrado puede entrar.
- Eventos/calendario son conocidos ex ante; price-state usa solo 28/84 días previos.
- Categoría es metadata estática, no actual futuro.
- Gamma productivo fijo = 1.0, elegido en rolling PRE-OOS.

## Invariante principal

Para cada SKU y horizonte:

`sum_28d(v12_sku_forecast_after_lgbm) == sum_28d(v12_sku_forecast_base)`

con tolerancia relativa `V12_SKU_SHAPE_LGBM_TOTAL_TOLERANCE=1e-9`. La capa cambia solo
forma diaria. Occurrence, share 28×84 y el selector leaf anidado permanecen
después de la corrección.

## Validación requerida tras una corrida

```bash
uv run python -m app.forecasting.validate_v12
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
```

La sección 18 del acceptance report debe mostrar `total28 bad=0` para ambos
targets/secciones y reportar base→LGBM a nivel SKU agregado. La decisión final
se toma con la sección 1 Active bottom-up y los A/B v11/v12/finales.
