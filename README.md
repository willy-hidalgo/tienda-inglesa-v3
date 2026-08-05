# Forecast RLS – Secciones 1 & 23

Pipeline de forecasting jerárquico con **Recursive Least Squares (RLS)** para retail (Tienda Inglesa).  
Niveles: **sección → tienda → SKU**.

## Arquitectura

```
data/input/          → archivos crudos (xlsx, dat, Valida_Profi*)
data/output/         → master.parquet, sales.parquet, selected.parquet,
                       forecast.parquet, wmape.parquet
app/
  ingestor.py
  categories_selector.py   → filtro secciones + locales de SECCIONES
  forecasts.py             → agregación 3 niveles + ventanas por sección + RLS
  dashboard.py             → Streamlit Explorer (paginación, sync, línea punteada)
settings.py
tests/
```

## Niveles y etiquetas

Jerarquía: **sección → tienda → SKU**.

| unique_id interno        | Código ranking | Descripción        |
|--------------------------|----------------|--------------------|
| `1`                      | `1`            | Sección 1          |
| `1\|\|00122`             | `00122`        | Nombre de tienda   |
| `1\|\|00122\|\|SKU123`   | `SKU123`       | DESCRIPCION SKU    |

Ranking: columnas **Código | Descripción | métrica | N puntos** (paginado de 5).

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

## Dashboard

- Unidad por defecto: **Valor ($)**.
- Rankings wMAPE y rotación: **todas** las filas, **5 por página**, con Anterior / Siguiente y selector de página.
- Click en fila → sincroniza selectores de la sidebar.
- Gráfico: área con actuals; **línea punteada** en zona solo-forecast.

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

`--n-jobs` / `FORECAST_N_JOBS` se propagan a `forecasts.py` (ProcessPool por `unique_id`).


```bash
python -m app.ingestor
python -m app.categories_selector
python -m app.forecasts --n-jobs 4
streamlit run app/dashboard.py
pytest tests/ -q
```

## Tests

Cubren helpers de fecha (domingo, lunes, bloques 28 días), horizontes por sección, etiquetas sin prefijo de sección, y agregación con descripciones.

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

Walk-forward por bloques de 28 días sobre **toda la historia** (desde el primer lunes ≥ primera fecha de la serie):

1. Se ajusta el modelo RLS con todos los actuals disponibles → **priors opción B**.
2. En cada bloque de 28 días se genera `yhat28` / `valuehat28`.
3. Se actualiza el modelo con los actuals del bloque y se avanza.

No modifica `yhat` / `valuehat`. Métricas `WMAPE₂₈` / `BIAS₂₈` en sección aparte del dashboard.
