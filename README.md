# Tienda Inglesa — Forecasting RLS v9.1

Aplicación de forecasting jerárquico para Tienda Inglesa con RLS, SES puro,
Polars y Streamlit.

```python
APP_VERSION = "9.1"
```

Python recomendado:

```text
Python >=3.13,<3.14
```

## Objetivos productivos

1. Minimizar wMAPE OOS, especialmente en SKU+tienda.
2. Mantener BIAS cercano a cero sin correcciones post-hoc en hojas.
3. Mantener causalidad expanding-28.
4. Conservar velocidad suficiente para la historia completa.
5. Garantizar coherencia forecast ↔ ranking ↔ KPI ↔ gráfico.

## Contrato temporal

Los modelos RLS de sección y tienda se recalibran expanding-28:

```text
primeros 28 días -> modelo -> forecast próximos 28
primeros 56 días -> modelo -> forecast próximos 28
...
```

OOS contiene 28 días y comienza en lunes. Forecast-only cubre idealmente los 28
días inmediatamente posteriores al OOS cuando no existen actuals.

## SKU+tienda: nivel SES puro

El warm-up provisional de esta versión se mantiene como:

```text
nivel inicial = Σ actuals de los primeros 28 días / 28 días calendario
```

Después, el nivel se actualiza exclusivamente mediante SES en escala original:

```text
L_t = alpha*y_t + (1-alpha)*L_(t-1)
```

Los días sin venta cuentan como `y_t = 0`. El RLS nunca entra en esta ecuación.

Alphas candidatos:

```python
LEAF_SES_ALPHA_CANDIDATES = (
    0.005, 0.01, 0.02, 0.05, 0.10,
    0.20, 0.40, 0.60, 0.70, 0.80,
)
```

## Selección de régimen v9.1

v9.1 mantiene intacta la lógica que ya funciona para SKU densos y refuerza solo
las hojas intermitentes/shock-heavy.

### Hojas densas

Cuando la cobertura de ventas de los últimos 28 días es alta, se mantiene la
selección causal basada en:

- últimos 14 días;
- últimos 28 días;
- referencia de 112 días;
- wMAPE histórico de SES puro.

Esto protege casos que ya presentan buen OOS, como SKU densos de venta diaria.

### Hojas intermitentes

La media de 112 días puede quedar contaminada por un solo bloque de picos. Para
estas hojas, v9.1 calcula adicionalmente la **mediana de las medias diarias de
los cuatro bloques de 28 días cerrados anteriores**:

```text
bloque -4 -> media diaria
bloque -3 -> media diaria
bloque -2 -> media diaria
bloque -1 -> media diaria
               ↓
           mediana robusta
```

Esta referencia solo sirve para **elegir entre estados SES puros existentes**.
Nunca reemplaza el nivel SES ni hace clipping del forecast.

Parámetros:

```python
LEAF_REGIME_SPARSE_SHOCK_RATIO = 1.35
LEAF_REGIME_SPARSE_STABILITY_RATIO = 1.25
```

Si una hoja es intermitente y el último bloque está muy por encima de la mediana
robusta, se interpreta como shock reciente y se evita seleccionar un estado SES
alineado artificialmente con los picos.

## Drivers RLS: forma, no nivel

Compiten causalmente:

```text
SES puro                    0%
RLS tienda                 25%
RLS tienda                 50%
RLS tienda                 75%
RLS tienda                100%
RLS sección                25%
RLS sección                50%
RLS sección                75%
RLS sección               100%
```

La intensidad se aplica como:

```text
factor_lambda = 1 + lambda*(factor_RLS - 1)
```

Cada bloque mantiene:

```text
mean(driver_factor) = 1
```

por lo que el driver modifica la forma diaria, no el nivel medio SES.

Un guard de dirección limita la intensidad cuando la forma RLS contradice la
tendencia reciente del SKU.

## Bias correction

SKU+tienda está excluido estructuralmente de cualquier bias correction posterior.
La identificación depende del `unique_id` con `T:` y `S:`, no del nombre del
modelo.

## Métricas oficiales

```text
wMAPE = Σ|y-yhat| / Σ|y|
BIAS  = Σ(yhat-y) / Σ|y|
```

Las métricas de SKU, tienda y sección se calculan bottom-up desde las hojas
SKU+tienda.

## Diagnóstico v9.1

`forecast.parquet` conserva, entre otros:

```text
ses_alpha_value
ses_level_value
ses_recent14_value
ses_recent28_value
ses_stability_reference_value
ses_regime_anchor_value
ses_sparse_robust_value
ses_sparse_shock_value
ses_robust_block_median_value
pure_ses_wmape_value

driver_strength_selected_value
driver_strength_value
driver_factor_value
driver_direction_guard_value

valuehat_raw
valuehat
```

Diagnóstico puntual:

```powershell
python -m app.forecasting.diagnose_leaf --uid "1||T:00001||S:587833"
```

## Dashboard y artefactos

```python
ARTIFACT_VERSION = 6
```

El fingerprint exige coincidencia de `mtime_ns`, tamaño, versión de aplicación y
número de filas. `dashboard_consistency` recalcula una muestra de métricas desde
`forecast.parquet`. Ante inconsistencia, el dashboard se detiene.

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
=== Pipeline Tienda Inglesa v9.1 ===
```

## Quality gates

Antes de aceptar una corrida:

1. `pytest -q` debe pasar.
2. OOS debe tener exactamente 28 días.
3. `mean(driver_factor)=1` por bloque.
4. `mean(forecast_raw)` debe coincidir con el nivel SES.
5. No puede existir bias correction en SKU+tienda.
6. `dashboard_consistency` debe terminar en `OK`.
7. Revisar especialmente distribución wMAPE OOS de hojas intermitentes, además
   de los SKU densos ya estabilizados.

## Notebooks

Los Jupyter notebooks se conservan como material de análisis y validación.
