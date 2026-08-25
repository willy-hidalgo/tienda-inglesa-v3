# Tienda Inglesa — Forecasting RLS v9.2

Aplicación de forecasting jerárquico para Tienda Inglesa con RLS, SES, Polars y Streamlit.

```python
APP_VERSION = "9.2"
```

Python recomendado:

```text
Python >=3.13,<3.14
```

## Objetivos productivos

1. Minimizar wMAPE OOS, especialmente en SKU+tienda.
2. Mantener BIAS próximo a cero sin correcciones post-hoc sobre hojas.
3. Preservar causalidad: un bloque nunca usa actuals futuros.
4. Mantener tiempos compatibles con la historia completa.
5. Mantener forecast, ranking, KPI y gráfico sincronizados.

## Contrato temporal

El modelo trabaja en bloques expanding de 28 días:

```text
bloque 1: train 28 días  -> forecast siguientes 28
bloque 2: train 56 días  -> forecast siguientes 28
...
OOS: 28 días
forecast-only: 28 días inmediatamente posteriores al OOS
```

El OOS empieza en lunes según la configuración de cada sección.

## SKU+tienda: arquitectura estadística

### 1. Warm-up

Se mantiene provisionalmente la definición acordada para evaluación:

```text
nivel inicial = Σ actuals de los primeros 28 días / 28 días calendario
```

La alternativa `Σ actuals / días con venta` queda pendiente de una comparación separada.

### 2. SES puro controla el nivel

Después del warm-up, cada hoja mantiene trayectorias SES independientes:

```python
LEAF_SES_ALPHA_CANDIDATES = (
    0.0002, 0.001,
    0.005, 0.01, 0.02, 0.05,
    0.10, 0.20, 0.40, 0.60, 0.70, 0.80,
)
```

La ecuación sigue siendo SES puro:

```text
L_t = alpha * y_t + (1-alpha) * L_(t-1)
```

Los drivers RLS nunca participan en la actualización del nivel.

### 3. Selector causal de régimen

La selección de alpha utiliza únicamente información cerrada antes del bloque a pronosticar.

Se calculan:

- últimos 14 días;
- último bloque de 28 días;
- referencia de 112 días;
- cobertura de días con venta;
- mediana robusta de bloques cerrados;
- **mediana estructural v9.2 de bloques anteriores, excluyendo el bloque recién cerrado**.

### 4. Corrección v9.2: shock alcista estructural

El error observado en v9.1 era que la referencia robusta todavía incluía el bloque inmediatamente anterior. Si ese bloque contenía un pico, el pico participaba en la definición de su propio techo.

v9.2 calcula una referencia estructural con los bloques anteriores al bloque recién cerrado:

```text
forecast origin
      │
      ├─ bloque -1 : bloque recién cerrado → se prueba como posible shock
      │
      └─ bloques -2 ... -7 → mediana estructural
```

Configuración:

```python
LEAF_REGIME_STRUCTURAL_BLOCKS = 6
LEAF_REGIME_UPWARD_SHOCK_MIN_BLOCKS = 3
LEAF_REGIME_UPWARD_SHOCK_RATIO = 1.75
LEAF_REGIME_UPWARD_SHOCK_STABILITY_RATIO = 1.30
```

Se declara shock alcista cuando:

```text
media del último bloque > 1.75 × mediana estructural previa
```

En ese caso se elige el estado **SES puro** más próximo al régimen estructural. La referencia nunca reemplaza ni recorta el nivel.

Los alphas `0.0002` y `0.001` se incorporan para disponer de trayectorias de memoria muy larga cuando un shock reciente ha contaminado incluso al antiguo mínimo `0.005`.

### Casos reales usados como regresión

Con los archivos de ventas incluidos en el proyecto:

```text
587833 @ 00001
último bloque ≈ 1,430
mediana estructural previa ≈ 260
SES alpha=0.0002 ≈ 324
```

```text
299993 @ 00001
último bloque ≈ 3,338
mediana estructural previa ≈ 291
SES alpha=0.0002 ≈ 280
```

La rama no se activa para series estables como los ejemplos `483046` y `58905`.

## Drivers RLS: solo forma

Una vez fijado el nivel SES compiten:

```text
SES puro                   0%
RLS tienda                25 / 50 / 75 / 100%
RLS sección               25 / 50 / 75 / 100%
```

La intensidad se aplica como:

```text
factor_lambda = 1 + lambda * (factor_RLS - 1)
```

Cada factor conserva media aritmética 1 por bloque:

```text
mean(driver_factor) = 1
```

Por construcción:

```text
forecast = SES_level × driver_factor
```

El driver puede modificar la forma diaria, pero no el promedio del nivel.

## Guard de dirección

En hojas densas se compara la dirección del driver con la tendencia reciente del SKU. Si son claramente contradictorias, la intensidad efectiva del driver se limita.

Esto evita imponer una pendiente descendente del padre cuando el SKU permanece plano o crece.

## Bias correction

SKU+tienda está excluido estructuralmente de cualquier corrección posterior de bias.

La detección depende de la estructura del `unique_id` (`T:` + `S:`), no del nombre del modelo.

## Métricas oficiales

```text
wMAPE = Σ |y-yhat| / Σ |y|
BIAS  = Σ (yhat-y) / Σ |y|
```

Tienda, SKU y sección se calculan bottom-up desde las hojas SKU+tienda.

## Diagnóstico

`forecast.parquet` conserva, entre otras:

```text
ses_alpha_y
ses_alpha_value
ses_level_y
ses_level_value
pure_ses_wmape_y
pure_ses_wmape_value
ses_recent14_value
ses_recent28_value
ses_robust_block_median_value
ses_structural_block_median_value
ses_upward_shock_value
ses_stability_guard_value
parent_model_value
driver_strength_value
driver_factor_value
valuehat_raw
valuehat
```

Diagnóstico puntual:

```powershell
python -m app.forecasting.diagnose_leaf --uid "1||T:00001||S:587833"
```

## Integridad del dashboard

Los artefactos usan:

```python
ARTIFACT_VERSION = 7
```

El fingerprint exige coincidencia de `mtime_ns`, tamaño, `APP_VERSION` y número de filas de `forecast.parquet`.

Si ranking, KPI y serie no corresponden a la misma corrida, el dashboard se detiene.

## Ejecución

```powershell
pytest -q
python app\forecasts.py --n-jobs 8
python -m app.dashboard_artifacts
python -m app.dashboard_consistency
streamlit run app\dashboard.py
```

El menú principal muestra:

```text
=== Pipeline Tienda Inglesa v9.2 ===
```

## Quality gates

Antes de aceptar una corrida:

1. `pytest -q` sin errores.
2. OOS de exactamente 28 días.
3. `mean(driver_factor)=1` por bloque.
4. `mean(forecast_raw)=ses_level` por bloque.
5. ninguna corrección de bias sobre SKU+tienda.
6. `dashboard_consistency` debe terminar en `OK`.
7. revisar distribución de wMAPE OOS SKU+tienda, no solo sección.
8. revisar especialmente `587833` y `299993` para confirmar que `ses_upward_shock_value=True` cuando corresponda.

## Notebooks

Los Jupyter notebooks se conservan como material de análisis y validación.
