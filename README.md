# Tienda Inglesa Forecasting — v13.2.6

> **v13.2.6 stability baseline.** The productive SES/RLS statistical configuration has been rolled back to the v13.1.1 baseline after regressions were observed in later v13.2 promotions. The current dashboard/main/all-block operational improvements remain available.

Herramienta de forecasting jerárquico para **Sección → Tienda → SKU+Tienda**, diseñada para ser causal, explicable, rápida y auditable.

## Contrato productivo v13

La arquitectura productiva se simplificó deliberadamente:

1. **Sección:** RLS expansivo usando la librería `rls_opt` incluida en el proyecto.
2. **Tienda:** RLS expansivo con la misma familia y los mismos drivers.
3. **SKU+Tienda:** un único modelo **SES + RLS parent**:
   - nivel inicial = mediana robusta de observaciones positivas en el warm-up inicial;
   - SES recursivo sobre toda la historia disponible, actualizando magnitud solo cuando `y > 0`;
   - el efecto RLS de **Tienda o Sección** se aplica sobre el nivel SES;
   - el parent se elige causalmente por el menor wMAPE oficial acumulado en bloques cerrados previos.
4. **In-sample, OOS y forecast-only usan exactamente la misma composición.** Solo cambia la información disponible en cada origen temporal.

No hay occurrence/share, LightGBM, SKU-total ensembles, sparse rescue ni meta-selectores en el forecast productivo v13.

## Métricas

Las métricas oficiales siguen siendo:

```text
wMAPE = Σ|y - ŷ| / Σ|y|
BIAS  = Σ(ŷ - y) / Σ|y|
```

con soporte **`y != 0`**. Son la única base para:

- selección de parent Sección/Tienda;
- rankings;
- KPI oficiales del dashboard.

El dashboard también precalcula `wMAPE incl. y=0` y `BIAS incl. y=0` como **prueba ácida diagnóstica**. Estas métricas no cambian ranking ni selección.

## Horizontes y cadencias

- cadencias de actualización: `1d / 7d / 14d / 28d`;
- OOS comparable: `28 días`;
- forecast-only: `28 días`;
- warm-up inicial SES: `28 días calendario`, usando solo magnitudes positivas para calcular la mediana inicial.

El in-sample es walk-forward/expanding: cada bloque se pronostica con el estado disponible al inicio del bloque, igual que OOS y forecast-only.

## Instalación

```bash
uv sync
```

Python soportado: `>=3.13,<3.14`.

El proyecto usa **Polars** para procesamiento tabular. No existe dependencia productiva de pandas.

## Ejecución

### Interfaz operativa única

Todos los procesos necesarios pueden lanzarse desde:

```bash
uv run python main.py
```

El menú incluye ingestión, selección de secciones, forecasts por bloque o para 1/7/14/28d, optimización estadística SES+RLS (refit de drivers + residuos + calibración + selección diagnóstica Section/Store por hoja), reconstrucción de artefactos, validación del modelo, consistencia del dashboard, reporte de aceptación, `pytest` y lanzamiento de Streamlit. Las opciones de un solo escenario solicitan el bloque de actualización y las de forecast permiten configurar `n_jobs`.

También puede usarse sin menú, por ejemplo:

```bash
uv run python main.py --run forecast --update-block-days 28 --n-jobs 8
uv run python main.py --run optimization-phase2 --update-block-days 28 --n-jobs 8
uv run python main.py --run validate-all
uv run python main.py --run dashboard-check-all
uv run python main.py --run dashboard
```

### Escenario principal 28d

```bash
uv run python -m app.forecasts
```

### Todos los bloques para dejar el dashboard multi-bloque listo

```bash
uv run python app/forecasts.py --all-update-blocks --n-jobs 8
```

El pipeline genera automáticamente los artefactos rápidos del dashboard al finalizar cada escenario.

Si los forecasts ya existen y solo deben reconstruirse artefactos:

```bash
uv run python -m app.dashboard_artifacts --all-update-blocks
```

## Dashboard

```bash
uv run streamlit run app/dashboard.py
```

El dashboard:

- permite cambiar entre escenarios 1/7/14/28d ya materializados;
- usa artefactos precalculados para no recalcular modelos al navegar;
- mantiene ranking por wMAPE oficial;
- permite mostrar/ocultar métricas all-points;
- permite descargar una auditoría Excel detallada para una hoja SKU+Tienda.

## Validación

```bash
uv run pytest -q
uv run python -m app.forecasting.validate_v13 --all-update-blocks
uv run python -m app.dashboard_consistency --all-update-blocks
```

Para un único 28d baseline:

```bash
uv run python -m app.forecasting.validate_v13
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
```

## Salidas

Baseline:

- `data/output/forecast.parquet`
- `data/output/wmape.parquet`
- `data/output/dashboard/`

Multi-bloque:

- `data/output/update_blocks/block_01d/`
- `data/output/update_blocks/block_07d/`
- `data/output/update_blocks/block_14d/`
- `data/output/update_blocks/block_28d/`

Cada escenario contiene su `forecast.parquet` y sus artefactos `dashboard/`.

## Gobierno actual

La optimización estadística está **activa nuevamente**, pero exclusivamente dentro de la arquitectura v13 vigente: RLS en Sección/Tienda + SES en SKU+Tienda. No se incorporan familias de modelos distintas ni capas que desconecten in-sample, OOS y forecast-only.

La primera corrida de tuning debe hacerse sobre 28d con:

```bash
uv run python app/forecasts.py --update-block-days 28 --n-jobs 8 --optimization-diagnostics
```

Además del forecast y los artefactos normales del dashboard, se crea `data/output/update_blocks/block_28d/statistical_optimization/` con:

- `ses_candidates.parquet` / `ses_summary.parquet` / `ses_recommendation.parquet`;
- `rls_candidates.parquet` / `rls_summary.parquet` / `rls_node_summary.parquet` / `rls_recommendation.parquet`;
- `driver_screen.parquet` / `driver_summary.parquet`;
- `driver_refit.parquet` / `driver_refit_summary.parquet` / `driver_refit_validation.parquet` / `promotion_reaudit.parquet`;
- `leaf_residual_extremes.parquet` / `leaf_residual_summary.parquet` / `residual_signal_summary.parquet`;
- `driver_strength_candidates.parquet` / `driver_strength_summary.parquet` / `driver_strength_recommendation.parquet`;
- `calendar_day_yearly.parquet` / `calendar_day_recurrence.parquet`;
- `leaf_parent_candidates.parquet` / `leaf_parent_summary.parquet` / `leaf_parent_section_summary.parquet` / `leaf_parent_recommendation.parquet`;
- `leaf_zero_rate_summary.parquet` / `leaf_zero_rate_patterns.parquet`;
- `optimization_summary.txt`.

El reporte **no promueve automáticamente** parámetros ni drivers. La elección se realiza con bloques históricos cerrados; el OOS actual queda como holdout final.

## Documentación

- [Arquitectura](docs/ARCHITECTURE.md)
- [Modelo y métricas](docs/MODEL.md)
- [Dashboard](docs/DASHBOARD.md)
- [Validación](docs/VALIDATION.md)
- [Experimentos cerrados](docs/EXPERIMENTS.md)
- [Changelog](CHANGELOG.md)

### Diagnóstico visual v13.1

- **Discrepancias Actual vs Forecast**: se marcan cuando la diferencia es material y extrema en términos multiplicativos/absolutos; no cambian métricas ni forecast.
- **EDP SKU+Tienda**: segundo eje con EDP observado de la hoja. En forecast-only se prolonga el último EDP observado.
- **Auditoría Excel SKU+Tienda**: exporta el proceso completo de cálculo (actual, EDP/ASP, nivel SES, desestacionalización, parent RLS, efecto/factor de drivers, forecast reconstruido, error, wMAPE y BIAS).

Para agregaciones, el mecanismo recomendado es: EDP ponderado por unidades para SKU entre tiendas; para Tienda/Sección, índice EDP de mix fijo (base 100) para no confundir cambio de precio con cambio de mix.

## Optimización estadística v13.2 — promociones manuales dentro de SES+RLS

Tras revisar Phase 2 se promovieron **solo** cambios dentro de la misma familia:

- SES: `alpha=0.30` entra en la grilla productiva; `0.0025`, `0.0075` y `0.50` siguen siendo diagnósticos.
- RLS: `lambda=0.990` y `0.9975` entran en la grilla productiva; `1.0` sigue siendo diagnóstico.
- Valor ($), nodo de Sección 1 (`unique_id="1"`): se habilitan los drivers de precio ya existentes (`asp`, `edp`, `discount`). Los demás nodos Valor mantienen el diseño sin precio hasta nueva evidencia.

No se introdujo ningún modelo nuevo ni selección basada en el holdout. In-sample, OOS y forecast-only siguen usando exactamente SES + RLS.

Para continuar el diagnóstico con el nuevo baseline v13.2 y obtener el refit **por nodo**:

```bash
uv run python app/forecasts.py --update-block-days 28 --n-jobs 8 --optimization-phase2
```

El reporte ahora identifica `unique_id` en cada resultado de refit para evitar agregaciones ambiguas entre tiendas. El dashboard continúa operativo con `ARTIFACT_VERSION=22`.


## Optimización v13.2.1 — validación holdout sin tuning

v13.2.1 no cambia la familia ni los parámetros productivos de v13.2.0. Amplía el diagnóstico `--optimization-phase2` para separar estrictamente dos pasos:

1. **candidato histórico**: nace únicamente de bloques cerrados in-sample;
2. **validación OOS**: el holdout actual solo puede validar o vetar ese candidato.

Se escribe `driver_refit_validation.parquet` y el resumen añade una sección 7 con todos los candidatos históricos y su comportamiento OOS. El reporte de holdout SES/RLS también se balancea por sección/target para evitar que un grupo desaparezca por truncamiento global.


## Optimización v13.2.2 — promociones validadas por historia + holdout

La sección 7 del diagnóstico v13.2.1 validó tres cambios de drivers existentes. Se promueven manualmente, sin cambiar de familia de modelo:

- `1||T:00211`, **Unidades**: remover grupo `month`;
- `1||T:00211`, **Valor ($)**: remover grupo `month`;
- `23||T:00006`, **Unidades**: remover grupo `price`.

Los candidatos `1||T:00411 Valor + price`, `23||T:00005 Valor + price`, `23||T:00155 Unidades - month` y `23||T:00200 Unidades - holiday_other` fueron vetados por el holdout y no se incorporan. La configuración se centraliza en `RLS_DRIVER_GROUP_EXCLUSIONS`; Phase 2 puede volver a añadir temporalmente un grupo excluido y refitear el **mismo RLS** para auditar si la promoción sigue siendo válida.

`APP_VERSION=13.2.2`; `ARTIFACT_VERSION=22` permanece.

## Optimización v13.2.4 — atribución residual y calibración del efecto RLS

El baseline productivo v13.2.2/v13.2.3 se conserva: no hay nuevas promociones automáticas ni nuevos drivers. `--optimization-phase2` añade una Phase 4 diagnóstica sobre el forecast causal ya emitido.

La Phase 4 incorpora tres controles adicionales dentro de la misma familia SES+RLS:

1. **Taxonomía del error extremo**: separa error de días con `y=0`, error oficial con venta positiva y cuánto ayudó/empeoró el efecto RLS frente a un forecast SES-only.
2. **Calibración diagnóstica gamma**: evalúa `log1p(forecast)=log1p(nivel_SES)+gamma*efecto_RLS`. `gamma=1` reproduce producción; otros valores solo se puntúan con historia cerrada y OOS se usa exclusivamente como holdout.
3. **Recurrencia calendario**: compara cada `MM-DD` del OOS contra el mismo día de años históricos cerrados para distinguir un patrón recurrente de un cambio de régimen o un problema de cobertura/ceros. No crea variables productivas.

Nuevos artefactos compactos:

- `driver_strength_candidates.parquet`;
- `driver_strength_summary.parquet`;
- `driver_strength_recommendation.parquet`;
- `calendar_day_yearly.parquet`;
- `calendar_day_recurrence.parquet`.

Los artefactos residuales existentes incorporan además `wmape_all_points`, participación del error en días con `y=0` y una medida de si el driver mejora o empeora el error extremo respecto al nivel SES sin drivers.

`APP_VERSION=13.2.4`; `ARTIFACT_VERSION=22`.


## Optimización v13.2.5 — selección Section vs Store en contexto SKU+Tienda

La Phase 5 no crea un modelo leaf alternativo. Para cada hoja y target toma el **mismo nivel SES y alpha productivos** y compara cuatro trazas diagnósticas: `current_policy`, `store`, `section` y `ses_only`. Solo `store`/`section` pueden ser candidatos; `ses_only` sirve para auditar si el efecto RLS realmente agrega valor sobre el mismo nivel.

El candidato nace únicamente de bloques históricos cerrados y exige: mejora leaf >=1 pp de wMAPE frente a `current_policy`, al menos 4 folds, >=60% de folds no peores y deterioro de |BIAS| histórico <=1 pp. El OOS actual solo clasifica el candidato como `validated_parent_switch`, `holdout_veto`, `mixed_holdout` o `already_current_parent`; nunca crea el switch.

El diagnóstico persiste únicamente agregados leaf×período×candidato, no series diarias alternativas, para conservar memoria. La misma corrida audita `zero_rate` por SKU+Tienda y patrones por weekday/mes/bloque sin afectar la métrica oficial (`y!=0`) ni la selección productiva.

`Phase 5` permanece disponible solo como diagnóstico. La versión productiva actual es `APP_VERSION=13.2.6`; `ARTIFACT_VERSION=22`.

## Preparación urgente del dashboard 1d/7d/14d/28d

Para dejar los cuatro escenarios disponibles, sincronizados y auditados sin ejecutar optimización estadística:

```bash
uv run python main.py --run dashboard-ready-all --n-jobs 8
```

El comando es reanudable: reutiliza únicamente bloques de la misma `APP_VERSION` cuya corrida esté registrada como exitosa; recalcula los demás, repara artefactos y ejecuta `validate_v13` y `dashboard_consistency` para todos los bloques. Al terminar debe imprimir `DASHBOARD READY` y `Escenarios disponibles y validados: 1d, 7d, 14d, 28d`. Después:

```bash
uv run python main.py --run dashboard
```

La preparación estable **no** ejecuta `--optimization-phase2` ni modifica SES/RLS/drivers.
