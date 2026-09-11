# Validación v13


## Interfaz operativa

Las validaciones también pueden lanzarse desde `uv run python main.py`: el menú expone `pytest`, `validate_v13`, `dashboard_consistency` y el reporte de aceptación, tanto por bloque cuando aplica como para todos los escenarios. Esto no reemplaza los comandos documentados abajo; centraliza su ejecución.

## Principio

v13 debe validarse por **contrato estructural** además de precisión. Un forecast no es aceptable si in-sample/OOS/forecast-only provienen de familias distintas, aunque el OOS aislado parezca bueno.

## Suite

```bash
uv run pytest -q
```

El usuario ejecutará la suite completa en su entorno `uv`, donde están Polars y Streamlit. Los errores runtime de Polars se corrigen en iteraciones posteriores si aparecen.

## Auditoría del modelo

Baseline:

```bash
uv run python -m app.forecasting.validate_v13
```

Todos los escenarios existentes:

```bash
uv run python -m app.forecasting.validate_v13 --all-update-blocks
```

El validator comprueba:

- columnas v13 requeridas;
- ausencia de columnas legacy incompatibles;
- presencia de in-sample/OOS/forecast-only;
- 28 días exactos de OOS y forecast-only por hoja;
- continuidad OOS → forecast-only;
- una única familia `SES+RLS(...)`;
- parent Sección/Tienda válido post-warm-up;
- warm-up con efecto RLS neutro;
- forecast-only no elegible para métricas;
- identidad algebraica SES + efecto RLS;
- factor RLS = `exp(driver_effect)`;
- `driver_effect = clip(log1p(forecast_RLS_parent) - log1p(nivel_parent_causal), log(0.50), log(2.00))` post-warm-up;
- `driver_factor_y/value` dentro de `[0.50, 2.00]` post-warm-up.

## Auditoría del dashboard

```bash
uv run python -m app.dashboard_consistency --all-update-blocks
```

Comprueba:

- `ARTIFACT_VERSION = 33`;
- fingerprint exacto del forecast;
- filas/keys consistentes;
- wMAPE/BIAS bottom-up;
- métricas all-points;
- horizonte OOS.

## Acceptance report

Para una corrida concreta:

```bash
uv run python -m app.forecasting_acceptance_report \
  --forecast data/output/update_blocks/block_28d/forecast.parquet
```

El reporte es deliberadamente compacto. Muestra:

- contrato estructural;
- métricas oficiales + prueba ácida;
- distribución de parent RLS;
- alphas SES;
- continuidad de familia entre fases;
- higiene de esquema.

No contiene challengers ni tuning experimental.

## Tests contractuales v13

La suite incluye contratos para:

- mediana positiva inicial;
- robustez frente a ceros/picos;
- SES recursivo más allá del warm-up;
- selección causal de parent por wMAPE oficial;
- misma familia en todas las fases;
- métricas all-points sin impacto en ranking;
- dashboard multi-bloque;
- exportación Excel de auditoría;
- ausencia de pandas en código/dependencias productivas;
- ausencia física de módulos de modelos legacy retirados;
- versionado v13/artefacto 22.

## Restricción de performance

Validar siempre con una corrida real además de tests unitarios. Registrar tiempos de:

- RLS sección;
- RLS tiendas;
- SES leaf;
- escritura forecast;
- construcción de artefactos.

Una corrección estadística no debe convertir el pipeline o dashboard en inviables operativamente.

## Warning Numba

Si aparece `NumbaPendingDeprecationWarning` por reflected lists en `rls_opt/edp.py`, se considera deuda técnica no bloqueante mientras los resultados/test del kernel sean correctos. No se introduce una dependencia alternativa para ocultarlo.
## Validación de la optimización estadística

Ejecutar primero el escenario 28d:

```bash
uv run python app/forecasts.py --update-block-days 28 --n-jobs 8 --optimization-diagnostics
```

Revisar `statistical_optimization/optimization_summary.txt`. La historia `in_sample` sirve para comparar parámetros; `out_sample` se presenta como holdout y no debe utilizarse para escoger el candidato. Ningún diagnóstico cambia automáticamente `settings.py`.


## Validación de trazabilidad del dashboard

La auditoría de una hoja debe permitir reconstruir `forecast_raw = max(exp(log(1+nivel_SES) + efecto_RLS) - 1, 0)`, comparar contra el forecast almacenado, auditar el actual desestacionalizado y reproducir numeradores/denominadores de wMAPE y BIAS. La bandera de discrepancia es exclusivamente visual/diagnóstica.

## Validación Phase 2

Ejecutar en 28d antes de cualquier promoción:

```bash
uv run python app/forecasts.py \
  --update-block-days 28 \
  --n-jobs 8 \
  --optimization-phase2
```

Revisar además de los artefactos Phase 1:

- `driver_refit.parquet`;
- `driver_refit_summary.parquet`;
- `optimization_summary.txt`, sección 6.

Condiciones de gobierno:

- candidatos marcados `diag-only` no pueden modificar el forecast;
- OOS actual se usa solo como holdout;
- un driver no se elimina/agrega por un único fold;
- cualquier cambio productivo requiere una promoción explícita posterior;
- el dashboard debe seguir abriendo con `ARTIFACT_VERSION=22`.


## Validación v13.2 después de promociones manuales

Ejecutar primero 28d:

```bash
uv run pytest -q
uv run python app/forecasts.py --update-block-days 28 --n-jobs 8 --optimization-phase2
uv run python -m app.forecasting.validate_v13 --update-block-days 28
uv run python -m app.dashboard_consistency --update-block-days 28
```

Verificar específicamente:

- `APP_VERSION=13.2.0`;
- `alpha=0.30` puede participar en selección SES; `0.0025/0.0075/0.50` siguen `diag-only`;
- RLS productivo puede elegir `lambda=0.990/0.9975`; `1.0` sigue `diag-only`;
- el nodo `unique_id="1"` de Valor ($) incorpora `asp/edp/discount`;
- los refits de drivers en `optimization_summary.txt` muestran `unique_id` y no mezclan tiendas;
- no existe Pandas;
- el dashboard continúa leyendo artefactos v22.


## Validación de candidatos v13.2.1

Para cada cambio de driver propuesto históricamente se reportan `history_wmape_improvement`, `history_positive_fold_share`, `holdout_wmape_improvement` y `holdout_abs_bias_change`. El OOS no interviene en la generación del candidato. `validated_candidate` requiere no empeorar wMAPE OOS y un deterioro de |BIAS| no mayor a 0.5 pp; un empeoramiento OOS o >1 pp de |BIAS| produce `holdout_veto`; los casos intermedios quedan `mixed_holdout`.


## Validación de promociones v13.2.2

Además de los contratos generales, la suite verifica que las únicas exclusiones node-specific promovidas sean `month` para `1||T:00211` (Unidades/Valor) y `price` para `23||T:00006` (Unidades), y que el mismo diseño de columnas se use tanto en historia como en forecast-only.

## Diagnóstico residual v13.2.3

La Phase 3 de optimización valida información no explicada sobre el forecast ya emitido. No participa en tuning ni selección. La regla de extremo es simétrica Actual/Forecast y utiliza como escala causal `max(nivel SES, nivel inicial, 1)`, de modo que el OOS actual no define sus propios umbrales.

`required_driver_effect = log1p(actual) - log1p(nivel_SES)` y `unexplained_driver_effect = required_driver_effect - driver_effect_actual`. Valores absolutos grandes y repetidos, especialmente si concentran una parte material del error oficial, se marcan para revisión de drivers. La persistencia por fecha, weekday, mes, tienda o parent se reporta separadamente. Ningún resultado de esta fase altera forecasts.


### Reauditoría de promociones

`promotion_reaudit.parquet` compara el RLS productivo actual con el add-back del grupo previamente excluido usando la nueva selección de parámetros. Un cambio de conclusión se reporta como watch/reversal candidate; nunca se revierte automáticamente y el OOS sigue siendo solo validación.

## Diagnóstico Phase 4 v13.2.4

`--optimization-phase2` genera también:

- `driver_strength_candidates.parquet`;
- `driver_strength_summary.parquet`;
- `driver_strength_recommendation.parquet`;
- `calendar_day_yearly.parquet`;
- `calendar_day_recurrence.parquet`.

Validar que `gamma=1.0` esté siempre presente y reproduzca la identidad productiva. Un `history_candidate` de gamma requiere al menos 0.5 pp de mejora histórica y 60% de folds no peores frente a gamma 1; solo después se observa OOS. `validated_calibration` no implica promoción automática.

En `leaf_residual_summary.parquet` revisar conjuntamente `wmape_official`, `wmape_all_points`, `zero_sale_error_share`, `zero_sale_extreme_share` y `driver_help_metric_extreme_share`. Si gran parte del error proviene de `y=0`, no atribuirlo automáticamente a un driver de magnitud. Si el driver empeora la mayoría de extremos positivos, revisar su intensidad antes de proponer variables nuevas.

`calendar_day_recurrence.parquet` debe usar exclusivamente años in-sample para la mediana histórica del mismo `MM-DD`; el OOS solo se compara contra esa referencia. El diagnóstico no debe crear columnas de producción ni modificar `settings.HOLIDAYS`.


## Diagnóstico Phase 5 v13.2.5 — parent Section/Store por hoja

`--optimization-phase2` genera además:

- `leaf_parent_candidates.parquet`;
- `leaf_parent_summary.parquet`;
- `leaf_parent_section_summary.parquet`;
- `leaf_parent_recommendation.parquet`;
- `leaf_zero_rate_summary.parquet`;
- `leaf_zero_rate_patterns.parquet`.

Validar que `leaf_parent_candidates.parquet` contenga únicamente `current_policy`, `store`, `section` y `ses_only`; `ses_only` no debe aparecer como `candidate_parent` en recomendaciones. Los forecasts candidatos deben usar el mismo `ses_level` productivo y el mismo redondeo oficial, variando únicamente el efecto del parent existente.

Un candidato histórico Section/Store requiere >=1 pp de mejora, >=4 folds, >=60% de folds no peores y deterioro de |BIAS| <=1 pp. El OOS no puede crear candidatos. `validated_parent_switch` exige no empeorar wMAPE OOS y deterioro de |BIAS| <=0.5 pp; empeorar wMAPE o deteriorar |BIAS| >1 pp produce `holdout_veto`.

La auditoría `leaf_zero_rate_*` es descriptiva y nunca debe alimentar wMAPE/BIAS oficial, selección de alpha/lambda, selección de parent o forecast-only.

## Gate bloqueante v13.2.11 — estabilidad, gaps y no-regresión end-to-end

`validate_v13` es bloqueante para los cuatro escenarios. Además de la identidad SES+RLS, rechaza cualquier forecast post-warm-up si:

- `driver_factor_y` o `driver_factor_value` sale de `[0.50, 2.00]`;
- el efecto transferido no coincide con `clip(log1p(forecast_RLS_parent)-log1p(nivel_parent_causal)-centro_causal)`;
- el forecast raw no puede reconstruirse exactamente desde `SES level + driver_effect + leaf_forecast_cap`;
- `yhat_raw/valuehat_raw` excede el tope causal persistido cuando ese guard está activo.


`dashboard_consistency` debe auditar la misma población que `metrics.parquet`: el panel slim de métricas conserva `rls_metric_eligible`, por lo que warm-ups que alcanzan OOS quedan excluidos de ambos lados de la identidad. `ARTIFACT_VERSION=24` identifica este contrato.

El objetivo de este gate es impedir la reaparición de los defectos observados en el dashboard: colapso/salto sostenido del nivel OOS por importar un offset de nivel del parent, picos aislados y contaminación del holdout que haga divergir las cadencias. La preparación `dashboard-ready-all` no debe declarar `DASHBOARD READY` si cualquiera de los cuatro bloques falla esta auditoría.


Además, `python -m app.forecasting.regression_gate --all-update-blocks` es parte obligatoria de `dashboard-ready-all`. Bloquea la release si detecta spikes leaf fuera de escala frente a la historia positiva, incoherencias de órdenes de magnitud entre cadencias o regresiones patológicas en los casos centinela `455436@00154` y `478160@00003`. Este gate no modifica forecasts.


El gate v13.2.11 verifica además que las cuatro cadencias compartan la misma ventana in-sample al origen OOS y bloquea reactivaciones positivas tras >=28 días observables sin venta cuando el error forecast/actual supera el factor configurado.
