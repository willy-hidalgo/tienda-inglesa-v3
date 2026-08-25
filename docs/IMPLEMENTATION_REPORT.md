# Implementation report

## Alcance

Refactor y endurecimiento de la versión recibida el 19-08-2026, tomando ese ZIP
como única base. Se preservaron todos los notebooks Jupyter.

## Cambios principales

- Runtime normalizado a Python 3.13.
- Corrección de SyntaxError en `app/forecasts.py` y `app/main.py`.
- `forecasts.py` reducido a fachada/CLI; dominio separado bajo `app/forecasting/`.
- RLS separado en API (`solver.py`) y kernels Numba (`kernels.py`).
- Validación RLS estricta para shapes, parámetros y NaN/inf.
- Métricas RLS no implementadas fallan explícitamente en lugar de ignorarse.
- SES residual OOS corregido: estado final de train y congelamiento durante OOS.
- `SES_ALPHA` por defecto: 0.20 en lugar de 0.74.
- Guardrail robusto SKU-tienda con validación temporal y baseline por sección.
- Corrección de bias no modifica forecasts reemplazados por baseline.
- Progreso explícito del pipeline con porcentaje + logs/timers existentes.
- `DEMO_MODE` desactivado por defecto; activable con `TI_DEMO_MODE=1`.
- Dependencias directas corregidas (`python-dateutil`, `numpy`); removidas dependencias no usadas.
- Lockfile Python 3.14 retirado por incompatibilidad; debe regenerarse con Python 3.13.
- Tests migrados hacia módulos de dominio y agregados contratos RLS/baseline.
- README actualizado.

## Benchmark OOS independiente

Benchmark directo sobre CSV crudos, no sobre el RLS final. Se usa para calibrar
guardrails y comprobar que el SES fijo anterior era una mala referencia.

- Sección 1: `median_pos56` elegido en validación train; wMAPE OOS 0.5173.
- Sección 23: `weekday_pos8` elegido en validación train; wMAPE OOS 0.4635.
- SES directo alpha=0.74: 0.7021 (sec. 1), 0.6218 (sec. 23).

Ver `docs/OOS_BENCHMARK.md` para detalle y limitaciones.

## Validación ejecutada

- `python -m compileall -q .`: PASS.
- `test_rls_contract.py + test_robust_baseline.py + test_settings.py`: 24 PASS.
- Suite completa: no recolectable en este runtime porque `polars` no está instalado.
  El entorno no tiene acceso de red y `pip/uv` no pudieron descargar Polars.

No se declara la suite completa como verde hasta ejecutar `uv lock && uv sync --dev`
en un entorno Python 3.13 con acceso a paquetes y correr `pytest -q`.

## Archivos retirados

- `check_parquet.py`
- `check_ranking.py`
- `prompt.md`
- `data/input/ejemplo.xlsx` (sin referencias)
- `uv.lock` de Python 3.14
- `.pytest_cache/`, `__pycache__/`, `.pyc`, `.nbc`, `.nbi`

## Notebooks preservados

- `notebook/data_selection.ipynb`
- `notebook/nuevas_pruebas.ipynb`
- `notebook/tests.ipynb`
- `notebook/wmape_insample_seccion1.ipynb`
