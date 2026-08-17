# Forecast RLS – Secciones 1 & 23

Pipeline de forecasting jerárquico con **Recursive Least Squares (RLS)** para retail (Tienda Inglesa).  
Niveles: **sección → tienda → SKU+tienda**.

## Arquitectura

```
data/input/          → archivos crudos
data/output/         → master.parquet, sales.parquet, selected.parquet,
                       forecast.parquet, wmape.parquet,
                       forecast_seccion_<n>_partial.parquet (checkpoints),
                       dashboard/   → artefactos precomputados del explorer
app/
  ingestor.py
  categories_selector.py
  forecasts.py             → agregación 3 niveles + ventanas por sección + RLS
  backend.py               → métricas, rankings, chart (sin Streamlit)
  dashboard_data.py        → path rápido (artefactos) + legacy
  dashboard_artifacts.py   → construye data/output/dashboard/
  dashboard.py             → UI Streamlit
settings.py
tests/
artifacts/validate_wmape_bottom_up.py   → validación numérica wMAPE
artifacts/wmape_insample_seccion1.ipynb → notebook de auditoría
```

**Principio offline vs online:** el cómputo pesado (agregación, features, EDP, fits RLS) vive en `forecasts.py` y corre **offline**. El dashboard **no reajusta modelos**: lee `forecast.parquet` y, si existen, los **artefactos** en `data/output/dashboard/` (series particionadas + `metrics.parquet`).

## `unique_id`

| Filtros | `unique_id` | Significado |
|---------|-------------|-------------|
| Sección | `1` | Total sección |
| + Tienda | `1\|\|T:00122` | Tienda (todos los SKU) |
| + SKU | `1\|\|S:SKU123` | SKU en todas las tiendas |
| + Tienda + SKU | `1\|\|T:00122\|\|S:SKU123` | Hoja |

Helpers: `settings.make_unique_id` / `split_unique_id` / `display_label`.

## Ventanas por sección (`SECCIONES`)

| Concepto | Regla |
|----------|--------|
| **train_end** | `SECCIONES[s].test_start` |
| **In-sample** | `[train_start, train_end]` |
| **Out-of-sample** | lunes ≥ `test_start` → `test_end` |
| **Solo forecast** | `test_end+1` → `test_end+28` (sin actuals) |

Ejemplos: sec. 23 `test_start=2025-11-30`; sec. 1 `test_start=2026-03-29`.

## Modelo jerárquico

### RLS real: sección y tienda

Ajuste sobre `log1p(y)` / `log1p(value)`. Reconstrucción con `expm1(intercept + efecto [+ residuo SES])`.

### Nivel derivado: SKU+tienda (sin re-fit)

1. Elección de coeficientes sección vs tienda por **WMAPE in-sample** (`_select_model_wmape`).
2. Efecto de drivers en log-space.
3. Residuo log + **SES no causal** (`SES_ALPHA`).
4. In-sample: `yhat = expm1(intercept + efecto)`; OOS/forecast: `+ SES`, recortado ≥ 0.
5. **Carry-forward de precios** (`asp` / `edp` / `discount`) al tramo solo-forecast.

## WMAPE (bottom-up)

Única definición en rankings y métricas de agregados:

1. Solo **hojas** `sec||T:tienda||S:sku`.
2. Excluir `forecast_only`, `y = 0`, `y`/`yhat` no finitos.
3. `abs_err = |y − ŷ|` por fila hoja.
4. \(\mathrm{wMAPE} = \sum \mathrm{abs\_err} / \sum |y|\)

| Nivel | Alcance de la suma |
|-------|--------------------|
| SKU+tienda | esa hoja |
| Tienda | todas las hojas de la tienda |
| Sección | todas las hojas de la sección |
| SKU puro | hojas de ese SKU en todas las tiendas |

**No** se usa el `yhat` de la serie agregada RLS de tienda/sección para el ranking.

Implementación: `backend.wmape_bottom_up` / `wmape_por_id`.  
Pipeline export: `RLSForecastRunner._compute_wmape` (por `period_type`).  
Validación: `python artifacts/validate_wmape_bottom_up.py`.

## Dashboard

- Path **rápido**: `data/output/dashboard/` (`index.json`, `metrics.parquet`, `series/seccion=*/…`). Builder: `python -m app.dashboard_artifacts` (también al final del pipeline).
- Path **legacy**: si no hay artefactos, calcula sobre el parquet completo.
- Unidad por defecto: **Valor ($)**.
- Filtros independientes Sección / Tienda / SKU; el último tocado acota al otro.
- Gráfico: actuals + línea punteada en solo-forecast.
- Rolling 28d: **desactivado** en el pipeline actual (no hay `yhat28` de producción).

### Ranking

Dos tablas (Tiendas / SKU):

| Código | Descripción | wMAPE (%) | Rotación | N puntos | % ≠0 |
|--------|-------------|-----------|----------|----------|------|

- **Orden: ASCENDENTE por wMAPE** (mejor → peor).
- **Tabla Tiendas**: siempre nodos tienda puros de la sección (no mezcla hojas).
- **Tabla SKU**: sin tienda → SKU puros (o agregación de hojas); con tienda → hojas de esa tienda.
- Excluye el nodo seleccionado en su eje y filas con `wmape == 0` / sin rotación.
- Rotación = \(\sum y\) (o `value` según unidad).

## Ejecución

```bash
python -m app.main
python -m app.main --run forecast --n-jobs 8
python -m app.main --run ingest
python -m app.main --run select
python -m app.main --run dashboard

python -m app.forecasts --n-jobs 8
python -m app.forecasts --n-jobs 8 --limit-series 200   # debug
python -m app.dashboard_artifacts                       # regenerar métricas/series UI
streamlit run app/dashboard.py

pytest tests/ -q
python artifacts/validate_wmape_bottom_up.py --seccion 1 --store 00063 --sku 127360
```

## Tests

| Suite | Contenido |
|-------|-----------|
| `test_wmape.py` | Fórmula cliente, bottom-up hoja/tienda/sección, `wmape_por_id`, NaN yhat |
| `test_backend.py` | Filtros cruzados, `ranking_table` (orden ASC, ejes), chart |
| `test_aggregator.py` | Agregación temporal / niveles dashboard |
| `test_forecasts_runner.py` | SES, derivación SKU+tienda |
| `test_forecasts_logspace.py` | Coherencia log-space / expm1 |
| `test_edp_batch.py` | EDP batched vs loop |
| `test_densify.py` | Panel denso, spine |
| `test_settings.py` | Horizontes, `unique_id` |
| `test_rolling28.py` | Métricas rolling residuales en backend (si hay columnas) |

```bash
uv add --dev pytest
uv run pytest tests/ -q
```

## Dependencias

`polars`, `numpy`, `plotly`, `streamlit`, `python-dateutil`, `tqdm`, `fastexcel`, módulo interno `rls_opt`.

## Panel denso

Spine por sección desde `section_horizons`. Cada `unique_id` × fecha del spine. Sin venta → `y=0`. El filtro `y=0` aplica **solo a métricas**, no al gráfico.
