# Forecast RLS – Secciones 1 & 23

Pipeline de forecasting jerárquico con **Recursive Least Squares (RLS)** para retail (Tienda Inglesa).  
Niveles: **sección → tienda → SKU**.

## Arquitectura

```
data/input/          → archivos crudos (xlsx, dat, Valida_Profi*)
data/output/         → master.parquet, sales.parquet, selected.parquet,
                       forecast.parquet, wmape.parquet,
                       forecast_seccion_<n>_partial.parquet (checkpoints)
app/
  ingestor.py
  categories_selector.py   → filtro secciones + locales de SECCIONES
  forecasts.py             → agregación 3 niveles + ventanas por sección + RLS
  dashboard.py             → Streamlit Explorer (paginación, sync, línea punteada)
settings.py
tests/
```

**Principio de diseño (offline vs online):** todo el cómputo pesado (agregación,
features, EDP, fits RLS, rolling 28d) vive en `forecasts.py` y corre **offline**,
una vez, escribiendo `forecast.parquet` / `wmape.parquet`. El dashboard
(`dashboard.py` + `backend.py` + `dashboard_data.py`) **nunca reajusta un
modelo ni recalcula EDP/features en runtime** — solo lee esos Parquet
precalculados y aplica selección de UI (unidad, frecuencia, nivel) sobre
datos ya resueltos. Esto es intencional y no cambió con esta optimización;
lo que cambió es cuánto tarda la etapa offline en producir esos Parquet.

## Filtros independientes y esquema de `unique_id`

**Sección** es obligatoria (un solo select). **Tienda** y **SKU** son dos
filtros **independientes** al mismo nivel — reemplazan la navegación
jerárquica anterior (sección → tienda → sku). Cualquiera de los dos puede
estar vacío, uno solo, o ambos a la vez:

| Filtro elegido | `unique_id` (`settings.make_unique_id`) | Significado |
|---|---|---|
| Sección | `1` | Total de la sección |
| Sección + Tienda | `1\|\|T:00122` | Tienda, todos los SKU |
| Sección + SKU | `1\|\|S:SKU123` | SKU, agregado sobre **todas** las tiendas |
| Sección + Tienda + SKU | `1\|\|T:00122\|\|S:SKU123` | Combinación específica |

El orden interno del id siempre es `T` antes de `S` (canónico), sin importar
en qué orden el usuario haya elegido los filtros en la UI. `settings.
split_unique_id` hace el parseo inverso; `settings.display_label` /
`ranking_code` / `ranking_description` generan las etiquetas para cada uno
de los 4 casos.

**Orden de selección en el dashboard:** el filtro tocado más recientemente
"ancla" y acota las opciones del otro (`st.session_state["_last_touched"]`
en `dashboard.py`, actualizado vía `on_change` de cada `selectbox`):
- Elegís Tienda → el select de SKU se acota a los SKU que existen en esa
  tienda (`backend.skus_for_store`).
- Elegís SKU → el select de Tienda se acota a las tiendas donde existe ese
  SKU (`backend.stores_for_sku`).
- Cambiar de Sección invalida cualquier Tienda/SKU elegido previamente.

## Ventanas por sección (`SECCIONES`)

Cada sección tiene su propio `test_start`, `test_end` y lista de locales.

| Concepto | Regla |
|----------|--------|
| **train_end** | `SECCIONES[s].test_start` |
| **train_start** | `max(S0, train_end − n·28d)` con `S0` = domingo ≥ primer dato |
| **In-sample** | `[train_start, train_end]` — modelo ajustado con toda esta data |
| **Out-of-sample** | lunes ≥ `test_start` → `test_end` |
| **Solo forecast** | `test_end+1` → `test_end+28` (sin actuals → línea punteada) |

Ejemplo sección 23: `test_start=2025-11-30`, `test_end=2025-12-07`.  
Ejemplo sección 1: `test_start=2026-03-29`, `test_end=2026-04-26`.

## Modelo jerárquico: RLS solo a nivel sección; tienda/sku derivado sin RLS

Cambio de arquitectura (no solo de performance): **RLS se ajusta
ÚNICAMENTE a nivel sección** (2 modelos por sección: variable `y`, variable
`value`, exactamente igual que antes pero restringido a esos 1-2 ids). Los
nodos de tienda, sku y tienda+sku **ya no se ajustan con RLS individual**
— se derivan de los coeficientes de la sección con este procedimiento (por
cada unidad, `y` o `value`, indistintamente):

1. **Efecto de drivers**: `efecto(t) = drivers_propios_del_nodo(t) ·
   coef_sección` (producto punto, **sin el intercepto**, con los drivers
   propios del nodo — no los de la sección — multiplicados por los
   coeficientes YA ajustados a nivel sección).
2. **y_neto**: `y_neto(t) = y(t) − efecto(t)`.
3. **SES causal**: suavización exponencial simple sobre `y_neto`, con
   `alpha` fijo (`settings.SES_ALPHA`, default `0.1`). La predicción de
   cada fecha usa el estado suavizado **hasta el día anterior**
   (`s(t-1)`, nunca `s(t)`) — mismo principio de causalidad que el
   rolling28: no hay look-ahead. Fuera de la muestra (OOS/forecast-only),
   el estado se propaga **constante** en el último valor conocido
   (propiedad estándar de una SES a cualquier horizonte).
4. **yhat**: `yhat(t) = y_neto_hat(t) + efecto(t)`.

Implementación: `RLSForecastRunner.compute_derived_forecasts` +
`_apply_causal_ses`, con `pl.Expr.ewm_mean(alpha=..., adjust=False)`
(nativo de Polars) para el paso 3 — **vectorizado sobre todas las series a
la vez, sin loop en Python y sin numba** (ver § Rendimiento).

**Se usó `pl.ewm_mean`, no `statsmodels`.** `statsmodels.
SimpleExpSmoothing` ajusta una serie a la vez desde Python — con miles de
series eso es el mismo patrón de loop-con-overhead-por-serie que ya se
eliminó para RLS. `ewm_mean` con `alpha` fijo es una operación columnar
pura en Rust sobre todas las series simultáneamente.

**Persistencia**: `driver_effect` y `driver_effect_value` (el término
"efecto" en cada unidad) se persisten en el parquet — útil para debug y
auditoría. Para nodos de sección estos campos van `null` (no aplica, el
`yhat` de sección es la predicción RLS directa, sin descomposición).
`y_neto` no se persiste (se puede recalcular al vuelo: `y − driver_effect`).

**Asunción explícita**: la SES se alimenta con `y_neto` de **todos los
días del panel denso**, incluyendo días sin venta (`y=0`) — a diferencia
del RLS, que excluye actualizaciones vía `min_y_to_update`. Un día sin
venta es información real para la línea de base local del nodo.

## Rendimiento del modelo jerárquico

Este cambio de arquitectura es, en sí mismo, la optimización de
performance más grande del proyecto — no un ajuste incremental:

- **Antes**: 1 fit RLS por serie (miles: sección × tienda × sku × 2
  variables).
- **Ahora**: **2-4 fits RLS en total** (uno por sección × variable). Todo
  lo demás (tienda/sku/tienda+sku) es una **única pasada vectorizada**
  (matmul de drivers + `ewm_mean`), sin loop por serie y sin threads — no
  hay nada que paralelizar por serie en ese camino.
- Benchmark sintético (8 tiendas × 15 SKU parcial, 111 series, 561 días de
  historia): pipeline completo de la sección en **~4s** (1.5s EDP + 0.1s
  RLS sección + 1.2s derivado vectorizado + agregación/features), de los
  cuales el cómputo de tienda/sku es <1.5s para 109 series simultáneas.

## Dashboard

- Unidad por defecto: **Valor ($)**.
- Filtros independientes Sección + Tienda + SKU (ver § Filtros
  independientes) — reemplazan la navegación jerárquica anterior.
- **Dos tablas de ranking** (Tiendas / SKU), cada una con filtro cruzado
  según el otro eje — ver § Ranking.
- Click en fila de cualquiera de las dos tablas → aplica ese filtro y lo
  marca como "ancla" (mismo comportamiento que elegirlo manualmente en el
  select correspondiente).
- Gráfico: área con actuals (incluyendo períodos con `y=0`); **línea
  punteada** en zona solo-forecast.
- El dashboard es **puramente de lectura**: consume `forecast.parquet` /
  `wmape.parquet` ya calculados (ver § Arquitectura).
- El contexto de tienda (`📍 Tienda: {código} — {nombre}`) solo se muestra
  cuando **ambos** filtros (Tienda y SKU) están activos a la vez — un SKU
  sin tienda es, por diseño, un agregado cross-tienda y no tiene "una"
  tienda que mostrar. Va **arriba** del encabezado del nodo. El
  encabezado dice `SECCIÓN:` / `TIENDA:` / `SKU:` según qué combinación de
  filtros esté activa (`view.node_kind`).
- El control "Series de forecast visibles" (yhat vs yhat28) y la sección
  "Métricas Rolling 28d" **solo se muestran si el nodo actualmente
  seleccionado tiene rolling28 calculado** — desde el modelo jerárquico,
  eso significa **solo el nodo de sección** (ver § Rolling 28d).
- `prepare_dashboard_state` está envuelto en `st.cache_data` (ver
  `_cached_dashboard_state` en `dashboard.py`): Streamlit re-ejecuta todo
  el script en cada interacción, así que sin cache se recalculaban
  agregación temporal y rankings en cada rerun aunque los datos de origen
  no cambiaran. La clave de cache es la selección de filtros (unidad,
  freq, sección, tienda, sku) — volver a un nodo ya visitado es
  instantáneo.

## Ranking

**Dos tablas** (Tiendas / SKU), cada una con las mismas columnas:

| Código | Descripción | wMAPE (%) | Rotación | N puntos | % ≠0 |
|---|---|---|---|---|---|

Filtro cruzado (`backend.ranking_table`, parámetro `axis`):
- **Tabla de Tiendas**: si no hay SKU elegido, compara tiendas puras de la
  sección; si hay un SKU elegido, compara **tienda+sku** entre tiendas
  para ese SKU fijo (ranking cross-tienda para ese producto puntual).
- **Tabla de SKU**: si no hay tienda elegida, compara SKU puros
  (agregados sobre todas las tiendas); si hay una tienda elegida, compara
  **tienda+sku** entre SKU para esa tienda fija (ranking cross-sku dentro
  de esa tienda).
- Ambas tablas excluyen el nodo actualmente seleccionado en su propio eje,
  y las series sin ventas (`wmape == 0`, sin ningún punto scoreable).
- **Rotación** = suma de `y` (o `value` según unidad).
- **N puntos** = cantidad de puntos con venta (`y ≠ 0`), no la longitud
  del spine (constante para todas las filas de la sección).
- **% ≠0** = N puntos / longitud del spine — cobertura de venta.
- La longitud del spine se muestra como una **etiqueta única** sobre las
  tablas: `N puntos totales (spine): NNN`.
- El gráfico **no cambió** — siempre incluyó los períodos con `y=0` (ver
  `tests/test_densify.py::test_chart_series_includes_zeros`).

## Ejecución

## CLI (`app/main.py`)

```bash
# Menú interactivo
python -m app.main
python -m app.main --n-jobs 4

# Etapa directa (sin menú)
python -m app.main --run forecast --n-jobs 8
python -m app.main --run ingest
python -m app.main --run select
python -m app.main --run dashboard

# Variable de entorno
FORECAST_N_JOBS=4 python -m app.main --run forecast
```

`--n-jobs` / `FORECAST_N_JOBS` se propagan a `forecasts.py`. Desde esta
optimización, **`n_jobs` es Nº de threads** (antes: procesos) — ver § Rendimiento.

```bash
python -m app.ingestor
python -m app.categories_selector
python -m app.forecasts --n-jobs 8
python -m app.forecasts --n-jobs 8 --limit-series 200   # debug/benchmark rápido
streamlit run app/dashboard.py
pytest tests/ -q
```

## Tests

Cubren helpers de fecha (domingo, lunes, bloques 28 días), horizontes por
sección, las 4 combinaciones del esquema de `unique_id` (sección / tienda /
sku / tienda+sku), agregación con descripciones, filtrado cruzado
tienda↔sku, y equivalencia numérica del método derivado (efecto + SES
causal) contra referencias calculadas a mano en Python puro — ver §
Modelo jerárquico y § Rendimiento para el detalle de cada test.

## Dependencias

`polars`, `numpy`, `plotly`, `streamlit`, `python-dateutil`, `tqdm`, `fastexcel`, y el módulo interno `rls_opt`.


## Métricas

### WMAPE (Weighted MAPE)

$$
\mathrm{WMAPE} = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_j y_j}
$$

Equivalente a $\sum_i \frac{|e_i|}{y_i}\cdot\frac{y_i}{\sum y}$.

- Se **excluyen** observaciones con **ventas $y = 0$**.
- En el dashboard se muestra como porcentaje.

### BIAS

$$
\mathrm{BIAS} = \frac{\sum_i (\hat{y}_i - y_i)}{\sum_j y_j}
$$

(mismo filtro $y \neq 0$). Positivo → sobre-predicción.

### Export Excel por sección

Tras el pipeline, en `data/output/`:

- `forecast_seccion_1.xlsx`
- `forecast_seccion_23.xlsx`

Columnas: **SKU**, **Local**, **Forecast sumarizado** (suma de $\hat{y}$ en `[forecast_start, forecast_end]`).

## Tests (uv)

```bash
# Instalar pytest en el entorno del proyecto
uv add --dev pytest

# Ejecutar
uv run pytest
uv run pytest tests/ -q
```

Si aparece `program not found`, falta el paquete en el entorno:
`uv add --dev pytest` y volver a intentar.

## Rolling 28d (`yhat28` / `valuehat28`)

**Opcional** vía `settings.COMPUTE_ROLLING_28` (default: `False`). Cuando está
desactivado, `forecasts.py` no calcula nada de esto y **no agrega las
columnas** `yhat28`/`valuehat28` al parquet (ni siquiera nulas) — el
dashboard detecta si el rolling28 fue calculado por la sola presencia de
estas columnas con datos, tanto a nivel de dataset completo (para decidir si
mostrar el control "Series de forecast visibles") como a nivel del nodo
puntual seleccionado (para decidir si dibujar la línea y mostrar las
métricas).

**Desde el modelo jerárquico (RLS solo a nivel sección), el rolling28
también queda limitado al nivel sección** — es el único nivel con un RLS
real que tenga sentido reajustar por bloques; tienda/sku/tienda+sku usan el
método derivado (efecto + SES), que no tiene una noción de "reajuste por
bloque" análoga. `_run_section` restringe el panel de entrada de
`run_rolling_28` a `unique_id == seccion` antes de llamarlo.

Walk-forward por bloques de 28 días sobre **toda la historia** (desde el primer lunes ≥ primera fecha de la serie):

1. Se ajusta el modelo RLS **una sola vez** con todos los actuals disponibles, con `return_all_coefs=True` (opción B: el seed de coeficientes usa todo el historial de actuals, igual que antes).
2. Para el bloque que empieza en la posición `pos`, se predice usando el vector de coeficientes tal como quedó tras el **último actual estrictamente anterior a `pos`** (leído de la trayectoria de coeficientes ya calculada en el fit único — sin reajustar nada).
3. No modifica `yhat` / `valuehat`. Métricas `WMAPE₂₈` / `BIAS₂₈` en sección aparte del dashboard.

⚠️ Esto reemplaza el walk-forward anterior, que reajustaba el modelo completo en cada bloque de 28 días. Los valores numéricos de `yhat28`/`wmape_28` **cambian levemente** respecto de versiones previas del pipeline por este motivo — es un cambio intencional y documentado, no un bug.

## Panel denso

Por sección, el spine de fechas sale de `section_horizons` (settings):
- train: `[train_start, train_end]`
- OOS: `[test_start, test_end]`
- forecast-only: `[forecast_start, forecast_end]`

Cada `unique_id` tiene una fila por fecha del spine. Sin venta → `y=0`, `value=0`.
El filtro `y=0` aplica **solo a métricas**, no elimina filas del parquet ni del gráfico.

---

## Rendimiento

`forecasts.py` tardaba **~90 minutos sin concluir** antes de esta optimización.
Se identificaron 4 causas raíz y se corrigieron con impacto medido en un
smoke test sintético (22 series, ~560 días de historia, 3 targets +
rolling28): **pipeline completo de una sección en ~1.2s** (antes, extrapolando
las causas de abajo, esto escalaba de forma cuadrática y jamás terminaba con
miles de series reales).

### Causas raíz y fix

| # | Causa | Antes | Ahora |
|---|-------|-------|-------|
| 1 | **Triple fit redundante.** `_run_section` llamaba `runner.run(train, target)` una vez por cada `target` (in_sample / out_sample / forecast_only) sobre el **mismo** `train` → 3 fits idénticos (6 contando la variable precio) por serie. | 6 fits/serie | **2 fits/serie** (`RLSForecastRunner.fit_and_predict_multi`: 1 fit por variable, N predicciones) |
| 2 | **Rolling 28d O(n²).** `_rolling_28_one` reajustaba el modelo completo en cada bloque de 28 días sobre un prefijo creciente de la serie. | O(n²/28) por serie, `min_obs=2` (procesaba de más) | **O(n) por serie**: 1 fit con `return_all_coefs=True`, walk-forward leyendo la trayectoria de coeficientes ya calculada (`np.searchsorted` sobre los índices de actuals). `min_obs=28` alineado con el resto del pipeline. |
| 3 | **EDP no vectorizado.** `decompose_price` (numba) ya soportaba `indexors` para batchear múltiples series en una sola llamada, pero `_calculate_edp` hacía `partition_by` + 1 llamada numba por serie desde Python. | 1 llamada numba por serie | **1 sola llamada numba batched** (`indexors=[slice, ...]`) para todas las series del panel. Requirió un fix de tipado en `rls_opt/edp.py` (ver nota abajo). |
| 4 | **Paralelismo con procesos.** `ProcessPoolExecutor` serializa DataFrames Polars entre procesos; `run_rolling_28` ni siquiera paralelizaba. Los kernels numba (`_rls`, `decompose_price`) son `nopython=True` y **liberan el GIL**. | Procesos (pickling) o secuencial | **`ThreadPoolExecutor`** en `fit_and_predict_multi` y `run_rolling_28`. `--n-jobs` ahora es Nº de threads. |

### Fix necesario en `rls_opt/edp.py`

Al implementar el batching de EDP (causa #3), la primera versión fallaba en
runtime: numba no podía **unificar estáticamente** el tipo de `indexors`
entre la rama interna `if indexors is None: indexors = [slice(None,None,None)]`
y el tipo de la lista de slices real pasada desde Python
(`Cannot unify reflected list(slice<a:b>) and list(slice<a:b:c>)`). La
solución fue mover la resolución del default `None` **fuera** del kernel
`@jit(nopython=True)`, a un wrapper Python delgado (`decompose_price`), que
llama a un kernel interno (`_decompose_price_kernel`) que siempre recibe un
tipo de `indexors` homogéneo. El fallback automático a loop-por-serie sigue
existiendo como red de seguridad si el batching fallara por cualquier otro
motivo (`try/except` con log de warning) — se verificó en desarrollo que
efectivamente activa el fallback si se rompe el batching, y que ambos
caminos producen resultados idénticos (`tests/test_edp_batch.py`).

Nota: numba emite un `NumbaPendingDeprecationWarning` sobre "reflected list"
para el parámetro `indexors` de `_decompose_price_kernel` (ver salida de
`pytest`). Es un warning de una API de numba que eventualmente será removida
en una versión futura del compilador, no afecta el resultado actual, y no
bloquea la ejecución. Si en una migración futura de numba esto se rompe, la
solución es tipar `indexors` con `numba.typed.List` explícito.

### Otros cambios (resiliencia y config)

- **`min_obs` alineado a 28** en `run_rolling_28` (antes: 2), consistente con `_build_tasks` del resto del pipeline.
- **Checkpoint por sección**: `forecast_seccion_<n>_partial.parquet` se escribe apenas termina cada sección (antes: solo al final del pipeline completo). Si el proceso se corta, no se pierde el trabajo de las secciones ya completadas.
- **Try/except por serie**: una serie degenerada (fit que lanza excepción) se loguea con `logger.warning` y se salta, en vez de tumbar todo el run.
- **Instrumentación de tiempos** (`_stage_timer`): cada etapa (agregación, densify, EDP, features, fit+predict, rolling28) loguea su duración con `⏱`, por sección y total de pipeline. Esto fue clave para confirmar empíricamente dónde se iba el tiempo antes de optimizar.
- **`--limit-series N`**: muestrea N SKUs por sección (conservando los niveles sección/tienda) para iterar rápido en desarrollo sin correr el dataset completo. **Solo para debug/benchmark, no usar en producción** (loguea un warning si se activa).

### Cambio de comportamiento numérico (Fase 3, decisión del cliente)

El nuevo rolling 28d es **matemáticamente más correcto** (RLS genuinamente
recursivo, sin resetear la covarianza cada bloque) pero **no es bit-a-bit
idéntico** al anterior: `yhat28` y `wmape_28` van a diferir levemente de
corridas previas del pipeline. Esto fue una decisión explícita, confirmada
antes de implementar (ver historial de diseño), priorizando corrección
algorítmica sobre compatibilidad numérica exacta con corridas anteriores.
`yhat` / `valuehat` (no-rolling) **no cambian**.

### Cómo verificar / benchmarkear

```bash
# Correr con logging de tiempos por etapa (nivel INFO por defecto)
python -m app.forecasts --n-jobs 8

# Iterar rápido en desarrollo con una muestra de SKUs
python -m app.forecasts --n-jobs 8 --limit-series 200

# Tests de equivalencia de los refactors de performance
pytest tests/test_forecasts_runner.py tests/test_edp_batch.py tests/test_rolling28.py -v
```

### Tests de performance añadidos

- `tests/test_forecasts_runner.py`: `_apply_causal_ses` coincide con una referencia de SES causal calculada a mano en Python puro (serie única, dos series independientes sin mezclar estado, continuación constante en forecast-only); `compute_derived_forecasts` coincide con la referencia manual completa del procedimiento (efecto + y_neto + SES + yhat) para un nodo tienda+sku con coeficientes de sección inyectados; ignora ids de sección; `fit_and_predict_sections` solo ajusta los ids pedidos, nunca otros presentes en el DataFrame.
- `tests/test_edp_batch.py`: el EDP batched produce `asp`/`edp`/`discount` **idénticos** al loop por serie; fallback automático verificado; modo rápido aproximado para >5000 series sin llamar a `decompose_price`.
- `tests/test_rolling28.py`: `run_rolling_28` respeta `min_obs=28`; `_rolling_28_one` hace exactamente 2 llamadas a `.fit()` (y + precio) sin importar la longitud de la serie ni el número de bloques de 28 días; `COMPUTE_ROLLING_28=False` (default) no agrega columnas `yhat28`/`valuehat28` para ningún nodo, `=True` las agrega solo para el nodo de sección.
- `tests/test_aggregator.py`: las 4 combinaciones del esquema de id se generan correctamente; el nodo sección+sku suma sobre todas las tiendas; descripciones (sku_desc/store_name) atadas correctamente por tipo de nodo.
- `tests/test_backend.py`: filtrado cruzado tienda↔sku (`all_stores_in_section`, `stores_for_sku`, `skus_for_store`, etc.); `ranking_table` con y sin `fixed_peer`/`exclude` en ambos ejes; `prepare_dashboard_state` con las 4 combinaciones de filtros (`node_kind`, `store_context` solo cuando ambos filtros están activos, `has_rolling28` por nodo).

## Backend.py y dashboard_data.py — ¿por qué siguen separados de forecasts.py?

Se evaluó explícitamente (a pedido) si tenía sentido consolidar estos
módulos dado que todo el cómputo pesado ya vive en `forecasts.py`. La
conclusión fue **mantener la separación**, porque son responsabilidades
distintas:

- `forecasts.py` = **fit del modelo, offline**, corre una vez y escribe Parquet.
- `backend.py` / `dashboard_data.py` = **ensamblado de vista** a partir de esos Parquet ya calculados (agregación temporal para la UI, formateo de tablas, split hist/forecast para el gráfico) — nunca reajustan un modelo ni recalculan EDP/features.

El síntoma real reportado ("tiempo de carga demasiado lento, lo cual no
tiene sentido para datos precalculados") **no era un problema de
arquitectura de módulos** sino de falta de cache: Streamlit re-ejecuta
*todo* el script en cada interacción de UI, así que `prepare_dashboard_state` (agregación
temporal + ranking) se recalculaba sobre el DataFrame completo en cada
cambio de selección, aunque el archivo cargado no hubiera cambiado. Se
resolvió con `st.cache_data` sobre `prepare_dashboard_state` (ver §
Dashboard) — la clave de cache es la selección de UI, no el contenido del
DataFrame, así que seleccionar un nodo ya visitado es instantáneo sin haber
tocado la separación de módulos.

Se descartó (por ahora) la alternativa más invasiva de precalcular las
tablas de ranking por nodo dentro de `forecasts.py` y guardarlas en el
parquet — agrega complejidad de storage sin necesidad, ya que el cache en
el dashboard resuelve el síntoma real (recomputación innecesaria en cada
rerun) sin tocar el pipeline offline.
