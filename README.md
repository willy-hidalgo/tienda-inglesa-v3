# Tienda Inglesa — Forecasting jerárquico RLS

Aplicación de forecasting retail para las secciones **1 y 23**, con Python,
Polars, NumPy/Numba, Streamlit y un RLS optimizado. Los niveles reportados son
**sección → tienda → SKU-tienda** y el wMAPE de tienda/sección se calcula
bottom-up desde las hojas.

## Runtime

El proyecto está normalizado a **Python 3.13** (`.python-version`). El lockfile
anterior correspondía a Python 3.14 y fue retirado por estar desfasado; debe
regenerarse en un entorno con acceso a PyPI:

```bash
uv venv --python 3.13
uv lock
uv sync --dev
```

## Arquitectura

```text
app/
├── main.py                     CLI/orquestación
├── ingestor.py                 ingestión master + ventas
├── categories_selector.py      selección de secciones/tiendas/SKU
├── forecasts.py                fachada compatible + CLI forecast
├── forecasting/
│   ├── config.py               contrato de configuración
│   ├── calendar.py             calendario retail/feriados
│   ├── features.py             drivers
│   ├── aggregation.py          agregación Polars
│   ├── panel.py                densificación diaria
│   ├── metrics.py              wMAPE bottom-up
│   ├── robust_baseline.py      guardrail leakage-free de hojas
│   ├── runner.py               RLS + derivación SKU-tienda
│   ├── pipeline.py             pipeline por sección
│   └── utils.py                utilidades compartidas
├── backend.py                  métricas/rankings/chart
├── dashboard_data.py           preparación de vista
├── dashboard_artifacts.py      artefactos precomputados
└── dashboard.py                Streamlit

rls_opt/
├── solver.py                   API/validación RLS
├── kernels.py                  kernels Numba
├── priors.py
└── edp.py
```

Los notebooks en `notebook/` son parte del proyecto y se conservan para auditoría
y experimentación.

## Modelo OOS de producción

### Sección y tienda

RLS se ajusta sobre `log1p(y)` y `log1p(value)` con drivers de calendario/precio.
La reconstrucción usa `expm1` y los coeficientes aprendidos exclusivamente en
train.

### SKU-tienda

1. Se calculan candidatos con coeficientes de sección y tienda.
2. La corrección residual usa SES en log-space.
3. El SES in-sample es one-step-ahead; para **todo el OOS/forecast** se usa el
   estado final del train y se congela. No se incorporan actuals OOS.
4. `SES_ALPHA` default es **0.20**. El anterior 0.74 resultó excesivamente
   reactivo en benchmark temporal de las hojas.
5. Se aplica un **guardrail robusto** leakage-free:
   - sección 1: mediana de ventas positivas recientes (`median_pos56`);
   - sección 23: mediana positiva por día de semana (`weekday_pos8`).
6. El baseline solo reemplaza RLS+SES si demuestra una mejora mínima en la cola
   de validación del train; en caso contrario RLS se conserva.
7. Para evitar explosiones, los forecasts futuros RLS se limitan a un rango
   configurable respecto del baseline robusto.
8. La corrección de bias no modifica filas que ya fueron reemplazadas por un
   baseline.

Parámetros centrales en `settings.py`:

```python
SES_ALPHA = 0.20
LEAF_BASELINE_GUARDRAIL = True
LEAF_BASELINE_VALIDATION_DAYS = 28
LEAF_BASELINE_LOOKBACK_DAYS = 56
LEAF_BASELINE_MIN_VALID_POINTS = 7
LEAF_BASELINE_MIN_IMPROVEMENT = 0.03
LEAF_BASELINE_CLIP_RATIO = (0.35, 2.50)
```

El benchmark independiente que justificó estos guardrails está documentado en
[`docs/OOS_BENCHMARK.md`](docs/OOS_BENCHMARK.md). Es un benchmark de baselines,
no una afirmación del wMAPE final del RLS completo.

## wMAPE

La definición usada por cliente y aplicación es:

```text
wMAPE = Σ |y - yhat| / Σ |y|
```

Se excluyen `y == 0`, `forecast_only` y valores no finitos. Para tienda y sección
se suman numeradores/denominadores de las hojas SKU-tienda; no se usa el `yhat`
del nodo agregado para el ranking.

Implementaciones principales:

- `app.forecasting.metrics.compute_wmape`
- `app.backend.wmape_bottom_up`
- `app.backend.wmape_por_id`

## Ventanas

`settings.section_horizons()` define por sección:

- train: hasta el domingo inmediatamente anterior/al inicio del holdout según la
  configuración vigente;
- OOS: lunes posterior/igual a `test_start` hasta `test_end`;
- forecast-only: desde el día siguiente a `test_end` hasta `forecast_end`.

## Ejecución

```bash
python -m app.main
python -m app.forecasts --n-jobs 8
python -m app.dashboard_artifacts
streamlit run app/dashboard.py
```

Pipeline por etapas:

```bash
python -m app.main --run ingest
python -m app.main --run select
python -m app.main --run forecast --n-jobs 8
python -m app.main --run dashboard
```

## Tests

```bash
pytest -q
```

Cobertura relevante:

- `test_settings.py`: horizontes e IDs.
- `test_wmape.py`: fórmula cliente y bottom-up.
- `test_backend.py`: rankings/filtros.
- `test_aggregator.py`: agregaciones.
- `test_densify.py`: panel diario.
- `test_forecasts_runner.py`: SES causal, OOS congelado y derivación.
- `test_forecasts_logspace.py`: consistencia `log1p/expm1`.
- `test_rls_contract.py`: contrato, finitud y parámetros del solver.
- `test_robust_baseline.py`: guardrail temporal sin dependencia de Polars.

## Dependencias directas

Declaradas en `pyproject.toml`: `polars`, `numpy`, `numba`, `streamlit`, `plotly`,
`python-dateutil`, `tqdm`, `fastexcel`, `xlsxwriter`, `matplotlib` e `ipykernel`.

`psutil` es opcional: si está instalado, `dashboard_artifacts.py` informa RSS;
si no, continúa sin esa métrica.

## Dashboard

El dashboard no reajusta modelos. Lee `forecast.parquet` y prefiere los artefactos
precomputados de `data/output/dashboard/`. Si no existen, usa el camino legacy.
La construcción de artefactos se ejecuta al guardar forecasts o manualmente con:

```bash
python -m app.dashboard_artifacts
```

## Troubleshooting

**`ModuleNotFoundError: polars`**: crear/sincronizar el entorno con `uv sync --dev`.

**Lockfile ausente**: es intencional en esta entrega; el anterior era de Python
3.14. Ejecutar `uv lock` con Python 3.13 y acceso a PyPI.

**Forecasts extremos en hojas**: revisar `modelo_seleccionado`, parámetros
`LEAF_BASELINE_*` y el benchmark OOS antes de ampliar los límites del guardrail.

**Cambios de wMAPE**: validar siempre por `period_type == "out_sample"` y por los
tres niveles. No optimizar contra in-sample como criterio principal.
