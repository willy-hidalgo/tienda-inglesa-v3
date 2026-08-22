# Tienda Inglesa — Forecasting RLS v8.1

Aplicación de forecasting jerárquico para retail con dos objetivos de producción:

1. minimizar wMAPE/BIAS OOS, especialmente en SKU+tienda;
2. mantener tiempos y memoria compatibles con el volumen real.

## Contrato temporal

- El día 1 de cada sección es su primer lunes disponible.
- Los bloques de entrenamiento/pronóstico son de 28 días (4 semanas).
- OOS dura exactamente 28 días, lunes a domingo.
- `forecast_only` son los 28 días inmediatamente posteriores al OOS.
- Si existen actuals dentro del período `forecast_only`, no se utilizan para
  actualizar RLS/SES antes de generar ese forecast. Esto evita leakage.
- Los días 1–28 son warm-up y quedan fuera de las métricas oficiales.

## RLS expanding-28 v8.1

Contrato obligatorio:

```text
actuals 1–28   -> forecast 29–56
actuals 1–56   -> forecast 57–84
actuals 1–84   -> forecast 85–112
...
```

No se usan actuals del bloque que se está pronosticando.

### Drivers causales y adaptación

El RLS de sección/tienda incorpora calendario, feriados y drivers comerciales,
más cuatro señales autoregresivas causales en log-space:

```text
lag 7
lag 28
rolling mean 7
rolling mean 28
```

Dentro de un bloque objetivo los lags/rolling son recursivos: después del origen
del bloque se alimentan con forecasts, nunca con actuals todavía desconocidos.

Se evalúan pocos forgetting factors, solo en sección/tienda:

```python
RLS_FORGETTING_FACTOR_CANDIDATES = (0.970, 0.985, 0.995)
```

El lambda del bloque siguiente se elige por wMAPE acumulado de bloques anteriores.


### Selección de dinámica RLS v8.1

El componente autoregresivo deja de ser obligatorio. Para cada bloque se comparan:

```text
base = calendario + feriados + drivers comerciales
ar   = base + lag7 + lag28 + rolling7 + rolling28
```

La combinación `(dinámica, lambda)` del bloque siguiente se elige únicamente con
wMAPE acumulado de bloques anteriores. Así un AR recursivo que empieza a derivar
no se impone en OOS, mientras que sigue disponible cuando realmente mejora el ajuste.
El artefacto conserva `rls_dynamics_*` y `rls_forecast_origin` para auditar fecha a fecha.

## SKU+tienda v8: residual SES en log-space

Se elimina la escala multiplicativa `leaf_scale` de v7.x.

Primeros 28 días:

```text
nivel inicial = log1p(media de actuals disponibles)
```

Desde el día 29, cada padre RLS aporta únicamente su efecto temporal, centrado
por bloque para retirar el intercepto/nivel:

```text
efecto_RLS = log1p(parent_forecast) - media_bloque(log1p(parent_forecast))
```

Para actualizar SES después de cerrar un bloque:

```text
residuo = log1p(actual) - efecto_RLS
nivel_siguiente = SES(residuo)
```

Forecast:

```text
forecast = exp(nivel_SES + efecto_RLS) - 1
```

Se evalúan conjuntamente:

```text
5 alphas SES × 3 candidatos de forma (tienda/sección/ninguna)
```

por SKU+tienda. El candidato de un bloque se selecciona exclusivamente con
wMAPE acumulado de bloques anteriores. No hay tuning con el bloque actual.

Esto separa responsabilidades:

```text
SES -> nivel estructural SKU
RLS -> forma temporal, calendario, drivers y picos cuando mejora históricamente
SES directo -> protección para hojas donde el padre desalinearía la serie
```

## Métricas oficiales

```python
METRICS_MODE = "rolling_28"
METRICS_START_DAY = 29
```

Las métricas oficiales son siempre bottom-up desde las hojas SKU+tienda:

```text
SKU+tienda
    ↓
Tienda
    ↓
Sección
```

\[
wMAPE = \frac{\sum |y-\hat y|}{\sum |y|}
\]

\[
BIAS = \frac{\sum(\hat y-y)}{\sum y}
\]

Solo participan filas elegibles a partir del día 29, con actual distinto de cero
y valores finitos.

## Ranking SKU

```python
RANKING_SKU_MIN_NONZERO_POINTS = 15
```

Solo se muestran SKU con al menos 15 días con actual distinto de cero.

## Dashboard

Unidad, frecuencia, sección, tienda y SKU forman un único estado. Cambiar
`Valor ($)` ↔ `Unidades` actualiza conjuntamente:

- ranking de tiendas;
- ranking de SKU;
- wMAPE;
- BIAS;
- gráfico;
- detalle.

Las métricas de tienda y sección son bottom-up, nunca el wMAPE directo del nodo RLS.

## Diagnóstico disponible

`forecast.parquet` conserva para las hojas:

```text
ses_alpha_y
ses_alpha_value
parent_model_y
parent_model_value
driver_effect
driver_effect_value
modelo_seleccionado
```

Esto permite auditar por SKU qué `alpha`, escala y padre fueron utilizados.

## Rendimiento

La aplicación evita:

```text
n_SPU × n_tiendas × historia_completa × RLS
```

RLS se ejecuta en sección/tienda. La lógica leaf utiliza agregaciones Polars,
estados SES por bloque y operaciones vectorizadas.

## Ejecución

Python objetivo: 3.13.

```bash
python app/forecasts.py --n-jobs 8
python -m app.dashboard_artifacts
streamlit run app/dashboard.py
```

Tests:

```bash
pytest tests/ -q
```

## Criterios de aceptación

Una nueva versión no debe reemplazar la vigente si:

- empeora materialmente wMAPE OOS bottom-up;
- elimina picos explicados por drivers;
- introduce leakage;
- crea huecos entre OOS y forecast-only;
- deteriora de forma material tiempo o memoria;
- rompe la sincronización del dashboard.

El objetivo es recuperar y superar el desempeño histórico cercano al 16% wMAPE
sin volver a una arquitectura SKU+tienda que tarde horas o días.

## Optimización de rendimiento v8.2

La lógica estadística de v8.1 se conserva: SKU+tienda sigue comparando los
mismos 15 candidatos (`3 padres × 5 alphas`) y la selección continúa usando
únicamente wMAPE acumulado de bloques anteriores.

El cambio es computacional. v8.1 recorría candidatos y bloques con operaciones
Polars repetidas:

```text
padre × alpha × bloque
    -> filter
    -> join
    -> group_by
```

v8.2 usa `LEAF_CANDIDATE_ENGINE="vectorized_block"`:

```text
por cada bloque de 28 días
    -> expandir una vez a los 15 candidatos
    -> score + actualización SES en un group_by vectorizado
    -> conservar solo los estados elegidos
```

Además, las observaciones se particionan por bloque una sola vez. Esto reduce
drásticamente el número de scans, joins y group_by sin reducir el espacio de
modelos ni cambiar expanding-28, causalidad, SES residual o bottom-up.

El log incluye tiempos separados:

```text
⏱ ... RLS candidatos + bloques
⏱ Sección ... selección leaf vectorizada
⏱ Sección ... FAST SKU+tienda total
```

Estos tiempos deben utilizarse para localizar cualquier cuello de botella
restante antes de simplificar la lógica estadística.
