# Validación v12.8.6 — Multi-block memory-safe

## Contratos de ingeniería

- `APP_VERSION=12.8.5`.
- `ARTIFACT_VERSION=19` (sin cambio de esquema).
- La lógica estadística v12.8.4/12.8.3 queda congelada.
- Para 1d/7d/14d, estados leaf por origen se persisten temporalmente y se reinyectan con `scan_parquet` + `collect(engine="streaming")`.
- Secciones completas se derraman a `_section_spill` antes de procesar la siguiente.
- Dashboard artifacts se construyen después de liberar `res_df`/`wmapes_df`.
- Caps de workers: 1d=1, 7d=2, 14d=4, 28d=8.
- OOS y forecast-only siguen fijos en 28 días.

Validación recomendada:

```bash
uv run pytest
uv run python app/forecasts.py --update-block-days 1 --n-jobs 8
uv run python app/forecasts.py --update-block-days 7 --n-jobs 8
uv run python app/forecasts.py --all-update-blocks --n-jobs 8 --skip-existing
```

---

# Validación histórica v12.8.4 — Demo multi-cadencia

## Contrato demo 1/7/14/28d

- `APP_VERSION=12.8.4`.
- `ARTIFACT_VERSION=19`.
- Cadencias soportadas: 1, 7, 14 y 28 días.
- OOS y forecast-only permanecen fijos en 28 días en todos los escenarios.
- Cada escenario se precalcula en `data/output/update_blocks/block_XXd/`.
- El dashboard solo lee artefactos precalculados; no entrena en runtime.
- El escenario 28d conserva la semántica baseline de v12.8.3.
- La optimización estadística queda congelada en esta versión.

Validación local recomendada:

```bash
uv run pytest
uv run python -m app.dashboard_consistency --all-update-blocks
uv run python -m app.forecasting.validate_v12 --update-block-days 28
```


---

# Validación v12.8.3

## Objetivo

Validar el tuning causal del meta-selector sin modificar los generadores de forecast de v12.8.1.

## Contratos estructurales

- `APP_VERSION=12.8.3`.
- `ARTIFACT_VERSION=18`.
- `V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER=False` por defecto.
- LightGBM sigue siendo shape-only y preserva exactamente el total SKU 28d (`total28 bad=0`).
- Store-share suma 1 por SKU/día y occurrence gate sigue compartido entre Qty/Valor.
- `v12_selected_y` y `v12_selected_value` pueden diferir.
- Modos permitidos por sección × target: `v11_all`, `v12_all`, `meta_leaf`.
- Threshold adaptativo permitido: conjunto configurado en `V12_META_SELECTOR_THRESHOLDS`.
- El legacy `v12_section_portfolio_gate_*` es solo diagnóstico; no bloquea selección.
- Probabilidades meta están en `[0,1]` y los thresholds en `(0,1)`.

## Temporalidad estricta

Para calibrar la policy que se aplicará al target:

1. un meta-modelo antiguo aprende la etiqueta del bloque 2 usando features de bloques 3/4;
2. ese modelo puntúa el bloque cerrado 1 usando features 2/3/4;
3. el bloque 1 elige causalmente `v11_all`, `v12_all` o `meta_leaf` y, para `meta_leaf`, el threshold;
4. el modelo productivo aprende la etiqueta del bloque 1 con features 2/3/4;
5. el target se puntúa con features 1/2/3.

Nunca se utiliza actual del bloque objetivo para seleccionar su forecast.

## Guard de BIAS

- Leaf meta: `|BIAS_v12_hist| <= |BIAS_v11_hist| + 3 pp`.
- Portfolio policy: el modo candidato no puede empeorar `|BIAS|` histórico en más de 3 pp.
- La policy solo abandona `v11_all` si mejora la utilidad histórica al menos el mínimo configurado.

## Validación después de corrida

```bash
uv run python -m app.forecasting.validate_v12
uv run python -m app.dashboard_artifacts
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
```

Revisar especialmente las secciones 1, 12, 14, 18, 20 y 21. La sección 21 debe mostrar por sección/target el modo elegido, threshold, disponibilidad de policy, gain de utilidad, BIAS guard y cobertura meta.

## Dashboard

Regenerar artifacts 17 y abrir:

```bash
uv run streamlit run app/dashboard.py
```

El panel diagnóstico debe mostrar final/v11/v12/oracle, modo causal portfolio, threshold aprendido, P(v12 gana), gain de policy y cobertura del meta-selector.

## Compatibilidad v12.8.1 conservada

Se mantienen los fixes de v12.8.1: exclusión de `y=0` en métricas, ranking legacy, horizontes explícitos del dashboard y facade EDP monkeypatchable.


## Contrato v12.8.3 Value Portfolio Safety

- Qty debe conservar la policy v12.8.2.
- `Value v12_all` solo es válido con `dominance_pass`, al menos 2 confirmaciones recientes y `bias_coverage_pass`.
- `Value meta_leaf` requiere `meta_margin_pass` frente al mejor all-mode seguro.
- Sin evidencia suficiente, Value vuelve a `v11_all`.
- Dashboard: KPI OOS desde `metrics.parquet`; diagnóstico avanzado lazy.
- `ARTIFACT_VERSION=18`.


## v12.9.0

- Qty congelado con policy v12.8.6 (4 bloques).
- Value Portfolio Selector walk-forward con 5 bloques / 3 folds pseudo-OOS.
- Guardias: win-rate, mediana gain, worst-fold, |BIAS| y utilidad bottom-up ponderada por volumen.
- `ARTIFACT_VERSION=20`; acceptance añade sección 23.
