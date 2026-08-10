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
  backend.py               → cálculos de visualización (contexto pre-agregado)
settings.py
tests/
```

**Principio de diseño (offline vs online):** todo el cómputo pesado (agregación,
features, EDP, fits RLS, rolling 28d — este último **opt-in**) vive en
`forecasts.py` y corre **offline**, una vez, escribiendo `forecast.parquet` /
`wmape.parquet`. El dashboard (`dashboard.py` + `backend.py`) **nunca
reajusta un modelo ni recalcula EDP/features en runtime**: solo lee esos
Parquet precalculados y aplica selección de UI (unidad, frecuencia, nivel)
sobre datos ya resueltos.

`backend.py` es la **única capa de cálculo del dashboard** (`dashboard_data.py`
se consolidó dentro de él). `build_dashboard_context` pre-agrega una vez por
(archivo, unidad) los mapas de etiquetas y el wMAPE/rotación por `unique_id`:
cada interacción del dashboard solo recorre **subconjuntos** (rankings = tabla
de ~1 fila por serie; gráfico/métricas = la serie elegida) → respuesta
sub-segunda, coherente con lo precalculado.

## Niveles y etiquetas

Jerarquía: **sección → tienda → SKU**.

| unique_id interno    | Código ranking | Descripción     |
| -------------------- | --------------- | ---------------- |
| `1`                | `1`           | Sección 1       |
| `1\|\|00122`         | `00122`       | Nombre de tienda |
| `1\|\|00122\|\|SKU123` | `SKU123`      | DESCRIPCION SKU  |

Ranking: columnas **Código | Descripción | métrica | N puntos** (paginado de 5).

## Modelo a nivel sección (descomposición causal + top-down)

> **Modo por defecto:** `settings.SECTION_LEVEL_MODEL = True`. El modelo RLS se
> ajusta **una sola vez por sección** (series `"1"` y `"23"`) y los pronósticos
> de tienda y SKU se obtienen **aplicando ese modelo** con una desagregación
> jerárquica *top-down*. Con `SECTION_LEVEL_MODEL = False` el pipeline vuelve al
> modo anterior (1 fit RLS por cada serie individual de sección/tienda/SKU).
>
> **Rendimiento:** de miles de fits RLS por ejecución → **2 fits por sección**
> (1 fit `model_y` + 1 fit `model_p`), es decir 4 fits totales.

### Descomposición causal de la sección

Con `settings.DECOMPOSE_EFFECTS = True` (default) se emite
`data/output/decomposition.parquet`: la demanda `log1p(y)` de la sección se
descompone como suma de contribuciones β·X por factor causal:

| Factor            | Features                                        |
| ----------------- | ----------------------------------------------- |
| `level`           | `intercept`                                      |
| `trend`           | `trend` (lineal, en años)                        |
| `seasonality`     | `weekday_*`, `month_*`                           |
| `edp`             | `edp`                                            |
| `discount`        | `discount`                                       |
| `feature_display` | festivos/promo (p. ej. `navidad_0`, ramps)       |
| `volume`          | `conteo_sku` (variedad/volumen de la sección)    |
| `price`           | `asp` (precio medio)                             |

Cada fila tiene el coeficiente y la contribución media en **train** y en la
ventana **solo forecast** (además del % sobre el total).

#### Series temporales de efectos por día (`ef_*`)

Además del parquet agregado (`decomposition.parquet`), las filas de **la
sección** en `forecast.parquet` llevan las contribuciones diarias β·X de cada
factor en espacio `ln(1+ŷ)` (demanda original transformada), con las columnas
siguientes (las filas hijas — tienda/SKU — dejan estas columnas nulas, pues la
descomposición se calcula a nivel sección):

| Columna                 | Factor                  |
| ----------------------- | ----------------------- |
| `ef_level`              | level (intercept)       |
| `ef_trend`              | trend                   |
| `ef_seasonality`        | seasonality             |
| `ef_edp`                | edp                     |
| `ef_discount`           | discount                |
| `ef_feature_display`    | feature_display         |
| `ef_volume`             | volume                  |
| `ef_price`              | price                   |
| `ef_total`              | ln(1+ŷ) total           |
| `ef_resid`              | residuo (corrección)    |

`ef_total = Σ ef_*` y `ef_resid = ef_total − Σ ef_*`; juntos coinciden con el
`ln(1+ŷ)` predicho a nivel sección (el residuo pequeño corrige el redondeo a
entero de `yhat`).

### Nivel y tendencia que corresponde a cada serie

Para cada tienda y SKU se calcula **una sola vez** (sobre el train):

- **share** = Σy_hijo / Σy_padre (nivel dentro de su padre)
- **pendiente** log-lineal semanal (en años) por serie

El pronóstico del padre se reparte a sus hijos aplicando el modelo de sección:

```
yhat_hijo(t) = yhat_padre(t) · share_hijo · exp((slope_hijo − slope_padre)·t)
```

normalizado por (padre, día) → **coherencia aditiva exacta**: Σ tiendas =
sección, y Σ SKUs de cada tienda = tienda (*top-down* en cascada
sección → tienda → SKU). El `valuehat` de los hijos se fija al último precio
observado (>0) en train. El rolling 28d en esta modalidad se calcula **solo a
nivel sección** (las columnas `yhat28`/`valuehat28` de tiendas/SKUs quedan
nulas).

## Ventanas por sección (`SECCIONES`)

Cada sección tiene su propio `test_start`, `test_end` y lista de locales.

| Concepto                | Regla                                                                |
| ----------------------- | -------------------------------------------------------------------- |
| **train_end**     | `SECCIONES[s].test_start`                                          |
| **train_start**   | `max(S0, train_end − n·28d)` con `S0` = domingo ≥ primer dato |
| **In-sample**     | `[train_start, train_end]` — modelo ajustado con toda esta data   |
| **Out-of-sample** | lunes ≥`test_start` → `test_end`                               |
| **Solo forecast** | `test_end+1` → `test_end+28` (sin actuals → línea punteada)   |

Ejemplo sección 23: `test_start=2025-11-30`, `test_end=2025-12-07`.
Ejemplo sección 1: `test_start=2026-03-29`, `test_end=2026-04-26`.

## Dashboard

- Unidad por defecto: **Valor ($)**.
- **Ranking único wMAPE + Rotación** (reemplaza los tabs "Mejor wMAPE" /
  "Mayor rotación"): todas las filas, **5 por página**, ordenado por wMAPE asc.
  Columnas `Código | Descripción | Rotación | wMAPE (%) | N puntos | % con venta`.
  - `N puntos` = períodos con venta (**y≠0**).
  - `% con venta` = proporción de esos puntos sobre el spine del nivel.
  - La etiqueta **"Total de puntos"** indica el spine total del nivel (igual
    para todas las series; alineado al panel real si es más largo que settings).
- A nivel **SKU** se indica en qué tienda estamos
  (`🏬 Tienda: 00122 — CENTRAL`). Cambiar de SKU mantiene la tienda (la
  jerarquía del sidebar la fija).
- Series de forecast visibles y **Métricas Rolling 28d** solo se muestran si el
  parquet trae `yhat28`/`valuehat28` (es decir, si el rolling se calculó — ver
  § Rolling 28d). Si no, el dashboard muestra solo `yhat`.
- El gráfico muestra **todos** los períodos, incluidos los de `y=0` (los ceros
  solo se excluyen de las métricas); línea punteada en zona solo-forecast.
- A nivel **sección**, un selector lateral
  "Descomposición de efectos (sección)" dibuja las contribuciones causales
  diarias (`ef_*`) como **áreas apiladas traslúcidas en el eje derecho**
  (`ln(1+ŷ)`), mientras la **predicción `yhat` aparece como línea negra** en el
  eje izquierdo. `ef_total` = Σ factores; `ef_resid` la corrección por
  redondeo.
- Click en fila → sincroniza selectores de la sidebar.
- El dashboard es **puramente de lectura**: consume `forecast.parquet` ya
  calculado y cada interacción es **O(subconjunto)** sobre el contexto
  pre-agregado (cacheado por archivo+unidad en `st.session_state`).

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
sección, etiquetas sin prefijo de sección, agregación con descripciones, y
(desde esta optimización) equivalencia numérica de los refactors de
rendimiento: fit único por serie, EDP batched, y conteo de fits del rolling
28d. Además:

- **Rolling 28d opt-in**: `settings.COMPUTE_ROLLING28` off por defecto,
  leído por `ForecastConfig`; `_attach_rolling28` no invoca `run_rolling_28`
  ni agrega columnas cuando está off.
- **Backend/dashboard**: tabla única wMAPE+Rotación (`N puntos` y `% con venta`, total del spine), contexto pre-agregado estable, `has_rolling`
  según columnas del parquet y contexto de tienda a nivel SKU.

Ver § Rendimiento para el detalle de cada test.

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

> **Opt-in (default OFF).** El rolling 28d es el cómputo más costoso del
> pipeline (walk-forward sobre toda la historia de todas las series). Por
> defecto está desactivado: `settings.COMPUTE_ROLLING28 = False`. Para
> habilitarlo, poner `True` en `settings.py` o ejecutar
> `python -m app.forecasts --rolling28` (`--no-rolling28` para forzar off).
>
> Con el flag **off**, el parquet de salida **no incluye** las columnas
> `yhat28`/`valuehat28`; el dashboard oculta el selector de series de
> forecast, la línea verde del rolling y las Métricas Rolling 28d. Con el
> flag **on**, esos controles aparecen automáticamente (la detección es por
> contenido del parquet, no por re-render: también funciona con archivos
> subidos).

Walk-forward por bloques de 28 días sobre **toda la historia** (desde el primer lunes ≥ primera fecha de la serie):

1. Se ajusta el modelo RLS **una sola vez** con todos los actuals disponibles, con `return_all_coefs=True` (opción B: el seed de coeficientes usa todo el historial de actuals, igual que antes).
2. Para el bloque que empieza en la posición `pos`, se predice usando el vector de coeficientes tal como quedó tras el **último actual estrictamente anterior a `pos`** (leído de la trayectoria de coeficientes ya calculada en el fit único — sin reajustar nada).
3. No modifica `yhat` / `valuehat`. Métricas `WMAPE₂₈` / `BIAS₂₈` en sección aparte del dashboard.

⚠️ Esto reemplaza el walk-forward anterior, que reajustaba el modelo completo en cada bloque de 28 días (ver § Rendimiento, Fase 3). Los valores numéricos de `yhat28`/`wmape_28` **cambian levemente** respecto de versiones previas del pipeline por este motivo — es un cambio intencional y documentado, no un bug.

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

> Con `SECTION_LEVEL_MODEL = True` (default) los `2 fits/serie` de abajo se
> reducen aún más: **2 fits por sección** (uno por variable `y`/precio), es
> decir **4 fits totales** para las secciones 1 y 23 — el resto de niveles
> tienda/SKU ya no ajusta modelo, aplica el de la sección (ver
> [Modelo a nivel sección](#modelo-a-nivel-sección-descomposición-causal--top-down)).

| # | Causa                                                                                                                                                                                                                                                           | Antes                                                 | Ahora                                                                                                                                                                                                                                   |
| - | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **Triple fit redundante.** `_run_section` llamaba `runner.run(train, target)` una vez por cada `target` (in_sample / out_sample / forecast_only) sobre el **mismo** `train` → 3 fits idénticos (6 contando la variable precio) por serie. | 6 fits/serie                                          | **2 fits/serie** (`RLSForecastRunner.fit_and_predict_multi`: 1 fit por variable, N predicciones)                                                                                                                                |
| 2 | **Rolling 28d O(n²).** `_rolling_28_one` reajustaba el modelo completo en cada bloque de 28 días sobre un prefijo creciente de la serie.                                                                                                              | O(n²/28) por serie,`min_obs=2` (procesaba de más) | **O(n) por serie**: 1 fit con `return_all_coefs=True`, walk-forward leyendo la trayectoria de coeficientes ya calculada (`np.searchsorted` sobre los índices de actuals). `min_obs=28` alineado con el resto del pipeline. |
| 3 | **EDP no vectorizado.** `decompose_price` (numba) ya soportaba `indexors` para batchear múltiples series en una sola llamada, pero `_calculate_edp` hacía `partition_by` + 1 llamada numba por serie desde Python.                              | 1 llamada numba por serie                             | **1 sola llamada numba batched** (`indexors=[slice, ...]`) para todas las series del panel. Requirió un fix de tipado en `rls_opt/edp.py` (ver nota abajo).                                                                  |
| 4 | **Paralelismo con procesos.** `ProcessPoolExecutor` serializa DataFrames Polars entre procesos; `run_rolling_28` ni siquiera paralelizaba. Los kernels numba (`_rls`, `decompose_price`) son `nopython=True` y **liberan el GIL**.        | Procesos (pickling) o secuencial                      | **`ThreadPoolExecutor`** en `fit_and_predict_multi` y `run_rolling_28`. `--n-jobs` ahora es Nº de threads.                                                                                                               |

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

- `tests/test_forecasts_runner.py`: el fit único (Fase 1) produce predicciones **idénticas** a fitear por separado para cada target (referencia manual), y el número de llamadas a `.fit()` no escala con el número de targets.
- `tests/test_edp_batch.py`: el EDP batched (Fase 2) produce `asp`/`edp`/`discount` **idénticos** al loop por serie; fallback automático verificado; modo rápido aproximado para >5000 series sin llamar a `decompose_price`.
- `tests/test_rolling28.py`: `run_rolling_28` respeta `min_obs=28` (Fase 5); `_rolling_28_one` hace exactamente 2 llamadas a `.fit()` (y + precio) sin importar la longitud de la serie ni el número de bloques de 28 días (Fase 3); la trayectoria de coeficientes tiene tantos estados como actuals usados en el fit (precondición del walk-forward por bloques). Además: `COMPUTE_ROLLING28` es opt-in (off por defecto) y `_attach_rolling28` salta `run_rolling_28` y deja `res_df` sin columnas rolling cuando está desactivado.
- `tests/test_backend.py`: tabla única de ranking (columnas `N puntos`/`% con venta`, total del spine, exclusión de series sin ventas y del id seleccionado), contexto pre-agregado `build_dashboard_context` estable, `has_rolling` según columnas del parquet, y contexto de tienda en nivel SKU.