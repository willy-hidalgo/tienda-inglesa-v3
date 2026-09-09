# VALIDATION v12.4.0

## Contrato estadístico

- v11.6.1 permanece como incumbent leaf.
- v12 construye un candidato independiente `SKU total diario × store-share`.
- Toda estadística del candidato para un bloque usa exclusivamente `ds < origin`.
- OOS usa **4 bloques cerrados** de 28 días: bloques 2–4 para discovery y bloque 1 (más reciente) como confirmation temporal fuera del pooled discovery.
- Forecast-only usa OOS ya cerrado como confirmation + los 3 bloques históricos anteriores como discovery.
- Discovery usa solo días con actual positivo y wMAPE como objetivo primario.
- `V12_MIN_IMPROVEMENT=0.02`: mejora pooled discovery >=2 pp.
- Se exigen >=2 bloques discovery válidos y >=2 victorias discovery.
- El bloque confirmation debe mejorar >=2 pp y pasar el guard de BIAS.
- Ningún bloque discovery puede empeorar más de `V12_MAX_BLOCK_DEGRADATION=0.08`.
- BIAS solo veta cambios riesgosos; nunca compensa peor wMAPE.
- `V12_REQUIRE_JOINT_TARGET_WIN=True`: cantidad y valor deben pasar simultáneamente discovery+confirmation.
- Los portfolio guards de cantidad y valor se calculan sobre el mismo conjunto conjunto y ambos deben confirmar.
- `V12_MIN_VALIDATION_SALES=7`: no se cambia de familia con evidencia insuficiente.

## Invariantes

Para cada SKU/fecha del candidato v12:

```text
occurrence_gate_y = occurrence_gate_value = occurrence_gate_joint
sum(store_share_y)     = 1
sum(store_share_value) = 1
si existe alguna tienda con gate abierto:
    gate cerrado => store_share = 0
candidate_yhat         = sku_forecast_y     × store_share_y
candidate_valuehat     = sku_forecast_value × store_share_value
```

Con selección conjunta habilitada:

```text
v12_selected_y == v12_selected_value
v12_selected => v12_section_portfolio_gate_joint == True
```

Las hojas que permanecen en `v11_ses_rls` siguen sujetas a la identidad v11:

```text
forecast = SES × parent_driver × sku_seasonal_multiplier
```

## Validaciones incluidas

```bash
uv run python -m app.forecasting.validate_v12
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
```

El acceptance report incluye:

- sección 11: wMAPE/BIAS final por familia;
- sección 12: A/B contrafactual OOS `v11_incumbent`, `v12_all`, `final_selected`;
- sección 13: estabilidad discovery + confirmation solo sobre hojas seleccionadas;
- sección 14: v11 vs v12 sobre exactamente las mismas hojas seleccionadas;
- sección 15: consistencia conjunta quantity/value y shared occurrence gate.

## Baseline congelado v11.6.1 Active OOS

- Unidades sec 1: 59.14%
- Unidades sec 23: 60.13%
- Valor sec 1: 58.40%
- Valor sec 23: 61.14%

v12 no se considera aceptada hasta que `final_selected` sea no peor que el incumbent en los cuatro portfolios y muestre mejora material sin deterioro severo de BIAS o zero-demand.

## Motivo de v12.3

v12.2 mejoró tres de cuatro portfolios, pero `Valor ($)` sección 1 quedó 0.20 pp peor que v11.6.1. El contrafactual sobre las mismas hojas mostró que cantidad y valor podían discrepar: el mismo enfoque de asignación ayudaba a quantity pero dañaba value. Además, los gains históricos del portfolio eran optimistas porque el bloque más reciente participaba simultáneamente en discovery y confirmation. v12.3 elimina ambas fuentes de inestabilidad.


## v12.4 — contrato adicional SKU-total

1. Cada forecast de total SKU usa únicamente información disponible antes del
   origen de su bloque de 28 días.
2. Los lags mínimos de actuals son 28/35/42/49 días; la trayectoria anual usa
   364 días.
3. `v11_sum` es un forecast causal ya producido por el incumbent y puede actuar
   como anchor del ensamble.
4. La elección del submodelo usa tres bloques cerrados previos, mínimo dos con
   al menos siete días positivos.
5. Se validan modelos permitidos, no negatividad y la identidad final
   `v12_candidate = v12_sku_forecast × v12_store_share`.
6. El reporte de aceptación añade la sección 16 con wMAPE/BIAS OOS del total SKU
   por submodelo seleccionado.
