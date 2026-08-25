# VALIDATION — Tienda Inglesa v9.1

## Release

- APP_VERSION: 9.1
- pyproject version: 9.1.0
- dashboard ARTIFACT_VERSION: 6
- Python target: >=3.13,<3.14

## Objetivo de esta revisión

v9.1 corrige específicamente hojas intermitentes/shock-heavy donde el estado SES
al origen OOS seguía alineándose con picos recientes, mientras conserva la lógica
de hojas densas que ya presentan wMAPE OOS alrededor de 20–30%.

Casos de referencia reportados por el usuario:

- SKU 587833 @ tienda 00001: nivel OOS demasiado alto.
- SKU 261298 @ tienda 00001: nivel OOS demasiado alto.
- SKU 567332, 483046 y 58905: comportamiento aceptable; no se modifica la rama densa.

## Cambios estadísticos

1. Warm-up provisional sin cambios:
   `sum(actuals primeros 28 días) / 28 días calendario`.

2. Hojas densas:
   - se conserva selector causal 14/28/112 de v9.0.1.
   - se conserva competencia de drivers 0/25/50/75/100%.

3. Hojas intermitentes:
   - se calculan las medias diarias de los cuatro bloques de 28 días cerrados
     previos al origen.
   - se usa la mediana de esas cuatro medias como referencia robusta.
   - un bloque reciente no puede dominar la referencia por un único pico.
   - la referencia solo selecciona entre estados SES puros existentes.
   - no existe clipping ni sustitución del nivel SES.

4. Guard sparse:
   - `LEAF_REGIME_SPARSE_SHOCK_RATIO = 1.35`.
   - `LEAF_REGIME_SPARSE_STABILITY_RATIO = 1.25`.

5. Drivers:
   - continúan con media 1.
   - no modifican el nivel medio SES.
   - guard de dirección permanece activo.

6. Bias correction:
   - continúa prohibido estructuralmente en SKU+tienda.

## Diagnósticos nuevos

`forecast.parquet` exporta:

- `ses_sparse_robust_y`
- `ses_sparse_robust_value`
- `ses_sparse_shock_y`
- `ses_sparse_shock_value`
- `ses_robust_block_median_y`
- `ses_robust_block_median_value`

## Quality gate ejecutado en este entorno

- Errores de sintaxis Python: 0
- Patrones `with_columns(...filter...)` de riesgo: 0
- Tests source/regression ejecutables sin Polars: 139 passed

### Tests funcionales diferidos

Este runtime no incluye Polars. No se contabilizan como aprobados los 9 módulos
que importan Polars directamente:

- tests/test_aggregator.py
- tests/test_backend.py
- tests/test_densify.py
- tests/test_edp_batch.py
- tests/test_forecasting_components.py
- tests/test_forecasts_logspace.py
- tests/test_forecasts_runner.py
- tests/test_rolling28.py
- tests/test_wmape.py

## Regresiones específicas añadidas

1. Un bloque spike-heavy no debe dominar la referencia sparse de 112 días.
2. El selector robusto solo puede devolver niveles que correspondan a estados
   SES puros existentes.
3. Sobre el fixture real de `587833 @ 00001`, reducir el nivel desde 598.6 hacia
   el nivel estructural reduce simultáneamente wMAPE y |BIAS|.
4. Un patrón sparse en caída tipo `261298` debe poder descartar estados SES que
   siguen alineados al régimen antiguo.
5. La rama densa permanece explícitamente separada y sin cambios conceptuales.

## Validación requerida en el entorno productivo

```powershell
pytest -q
python app\forecasts.py --n-jobs 8
python -m app.dashboard_artifacts
python -m app.dashboard_consistency
streamlit run app\dashboard.py
```

Prioridad de inspección:

```powershell
python -m app.forecasting.diagnose_leaf --uid "1||T:00001||S:587833"
```

Revisar especialmente:

- `ses_level_value`
- `ses_sparse_robust_value`
- `ses_robust_block_median_value`
- `ses_recent28_coverage_value`
- `ses_alpha_value`
- `driver_strength_value`
- `driver_factor_value`
- `valuehat_raw`

Para 587833 y 261298, el nivel SES OOS debe quedar en el orden del régimen
estructural reciente y no alineado con los picos individuales.
