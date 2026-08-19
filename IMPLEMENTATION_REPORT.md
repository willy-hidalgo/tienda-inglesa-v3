# Implementation report

## Alcance ejecutado

- Normalización a Python 3.13 (`pyproject.toml`, `.python-version`, `uv.lock`).
- Corrección de errores de sintaxis heredados en `settings.py`, `app/main.py` y forecasting.
- Refactor de `app/forecasts.py` a fachada; lógica movida a `app/forecasting/`.
- Separación de kernels Numba RLS a `rls_opt/kernels.py`.
- Validaciones RLS de producción: shapes, finitud, parámetros, priors y APIs no implementadas.
- Optimizador OOS causal por SKU-tienda con selección temporal de modelos y calibración estable.
- Trazabilidad `yhat_pre_leaf_optimizer` para comparar el modelo anterior y el optimizado en el mismo parquet.
- Heartbeat periódico de etapas largas y subprocess sin buffering.
- Tests nuevos para RLS y modelos causales.
- README consolidado.
- Limpieza de caches, virtualenv accidental, scripts diagnósticos obsoletos, prompt interno y archivo de ejemplo no referenciado.
- Conservación de los 4 notebooks Jupyter.

## Benchmark causal sobre datos incluidos

Fórmula exacta del cliente: filas hoja con `y=0` se eliminan antes de agregar tienda/sección.

| Sección | Benchmark | SKU-tienda | Tienda | Sección |
|---|---:|---:|---:|---:|
| 1 | SES 0.74 | 42.81% | 20.42% | 15.38% |
| 1 | optimizador | 39.14% | 16.95% | 11.67% |
| 23 | SES 0.74 | 38.34% | 24.91% | 23.92% |
| 23 | optimizador | 33.36% | 18.37% | 13.15% |

Mejora relativa contra SES 0.74:

- Sección 1: 8.58% SKU-tienda, 17.00% tienda, 24.08% sección.
- Sección 23: 12.98% SKU-tienda, 26.26% tienda, 45.05% sección.

El benchmark valida la nueva capa causal NumPy. La comparación exacta contra el RLS+SES previo del pipeline se obtiene, después de ejecutar forecasting en un entorno con dependencias completas, con:

```bash
python scripts/compare_oos_models.py
```

## Validación

- `python -m compileall`: PASS.
- `uv lock --check --offline`: PASS.
- `tests/test_rls_contract.py + test_leaf_models.py + test_settings.py`: 25 passed.
- Suite completa: 8 errores de colección en este runtime exclusivamente porque `polars` no está instalado. El entorno no tiene acceso de red para instalar dependencias.
