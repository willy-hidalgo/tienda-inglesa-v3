# VALIDATION — Tienda Inglesa v11.3.1

## Performance/correctness delta v11.3.1

Acceptance gates: OOS sigue siendo los últimos 28 actuals; métricas oficiales
bottom-up no cambian; fallback alternativo solo puede seleccionarse con >=7
días con venta en su bloque causal de validación; `deses_robust_mean` solo se
evalúa si `deses_ses` no mejora al menos `LEAF_FALLBACK_MIN_IMPROVEMENT`;
intercept debe ser 1 en filas densificadas; dashboard artifact version=14.

## Objetivo

v11.2.1.1 corrige la alineación temporal, incorpora fallbacks robustos causales para
SKU+tienda de alto error y reduce trabajo redundante del pipeline.

## 1. Contrato temporal obligatorio

Para cada sección, usando la última fecha real disponible `last_actual`:

```text
OOS start      = last_actual - 27 días
OOS end        = last_actual
forecast start = last_actual + 1 día
forecast end   = forecast start + 27 días
```

No existen filas productivas `actual_extension` ni `observed_tail` después de
OOS. `train_start` se alinea hacia atrás desde OOS para que el entrenamiento
contenga un número entero de bloques de 28 días.

Ejemplo observado en la validación:

```text
last_actual   = 2026-04-30
train_end     = 2026-04-02
OOS           = 2026-04-03 → 2026-04-30
forecast_only = 2026-05-01 → 2026-05-28
```

## 2. Métrica oficial

La regla del cliente queda aplicada en todos los caminos oficiales:

```text
wMAPE = Σ|y-yhat| / Σ|y|   solo donde y != 0
BIAS  = Σ(yhat-y) / Σ|y|   solo donde y != 0
```

Los días sin venta no desaparecen del control: se reportan por separado en
`Forecast sin demanda`, `Impacto zero-demand` y `wMAPE All`.

## 3. Fallback causal para errores altos

El modelo normal sigue siendo:

```text
SES directo → nivel
RLS store/section mean-one → forma
forecast = nivel × driver_factor
```

Si el error histórico causal de una hoja supera el trigger configurado, se
comparan tres niveles sobre el último bloque histórico cerrado:

```text
1. direct_ses
2. deses_ses
3. deses_robust_mean
```

`deses_ses` divide primero los actuals por el factor RLS conocido, calcula SES
en la serie desestacionalizada y vuelve a aplicar el factor RLS al forecast.

`deses_robust_mean` desestacionaliza la historia completa disponible, limita
picos mediante una cota robusta basada en mediana/MAD y calcula la media diaria
incluyendo los días calendario sin venta como cero. Después reaplica el factor
RLS.

La elección OOS usa únicamente el bloque inmediatamente anterior a OOS. No hay
leakage. Para forecast-only sí puede usarse el OOS ya observado para volver a
elegir el nivel, porque esa información es conocida en ese origen.

## 4. Rendimiento / refactor

Cambios del hot path:

- RLS de tiendas en un batch; train y targets se particionan una sola vez.
- paralelización interna por UID en `fit_and_predict_sections`.
- eliminada la ruta productiva post-OOS `actual_extension/observed_tail`.
- estadísticas recent-28 reutilizadas; se elimina un re-scan/group-by por cada
  bloque histórico.
- auditoría descriptiva pesada eliminada del hot path; permanecen invariantes
  mínimas y `diagnose_leaf`/`validate_v11` para diagnóstico detallado.
- eliminado el sort global final de SKU+tienda por defecto.
- fallback robusto solo procesa hojas identificadas como de riesgo.

## 5. Pruebas ejecutadas en este runtime

```text
Archivos Python parseados/compilados : 45
Errores de sintaxis                   : 0
Tests sin dependencia Polars          : 35 passed
```

Este runtime no dispone de Polars, por lo que el benchmark completo debe correr
en el `.venv` productivo con Python 3.13.

## 6. Ejecución de aceptación

```powershell
pytest -q
python app\forecasts.py --n-jobs 8
python -m app.forecasting.validate_v11
python -m app.dashboard_artifacts
python -m app.dashboard_consistency
```

Resultado estructural requerido:

```text
AUDITORÍA MODELO v11.2.1.1: OK
AUDITORÍA DASHBOARD: OK
```

## 7. Gate estadístico

Comparar v11.2.1.1 contra la corrida v11.0.1 usando exactamente la misma data:

1. wMAPE Active OOS bottom-up por sección.
2. BIAS Active OOS por sección.
3. wMAPE por tienda.
4. distribución wMAPE SKU+tienda.
5. hojas de alto volumen.
6. SKU puro 586896 y otros casos con error extremo.
7. `Forecast sin demanda` y `Impacto zero-demand`.
8. tiempo por etapa y tiempo total.

No aceptar una mejora obtenida simplemente ocultando zero-demand: la métrica
official excluye `y=0` por definición del cliente, pero esos falsos positivos
siguen auditándose por separado.

## Performance gate v11.2.1.1

La versión 11.2.1 añade dos invariantes de escalabilidad:

1. `_rls` debe ser un `CPUDispatcher` de Numba y ejecutarse con
   `nopython=True, cache=True, nogil=True`. Si esta condición se pierde, el
   pipeline falla antes de lanzar los fits en vez de degradar silenciosamente
   a Python.
2. `app/forecasting/leaf_fallback.py` no puede contener un loop por
   `unique_id`. El gate de riesgo se calcula sobre el bloque causal de 28 días
   y solamente las hojas con wMAPE alto pasan por los candidatos robustos,
   todos vectorizados.

Benchmark de referencia de construcción (no sustituye el benchmark end-to-end):

```text
RLS 700 observaciones × 100 drivers
v11.1 sin JIT: ~2.44 s/fit
v11.2.1.1 JIT steady-state: ~0.02–0.03 s/fit
```

Comando de diagnóstico local:

```powershell
python -m app.forecasting.benchmark_rls_kernel
```


## v11.2.1

- Kernel RLS sin operadores BLAS dentro del hot path.
- Equivalencia numérica validada contra referencia NumPy.
- Benchmark validado con SciPy bloqueado: OK.
- `tests/test_v11_1_source_contract.py`: 8 passed.

## v11.2.2 — input preflight

Antes de RLS, `forecasts.py` debe garantizar `selected.parquet` sin trabajo
redundante. Gate estructural: (1) si selected existe no se ejecuta ninguna etapa
previa; (2) con master+sales se ejecuta solo selección; (3) con fuentes raw se
ejecuta ingesta+selección; (4) sin fuentes se falla antes de crear el LazyFrame.
Tests dedicados: `tests/test_preflight.py` (4 casos).

## v11.3.1 performance-only contracts

- Bias correction runs on `parent_res` only, never on the full leaf frame.
- Leaf fallback uses a slim analytical projection before repeated joins/group-bys.
- Parent block prediction uses Numba kernels for both base and AR paths.
- Full section checkpoints are disabled by default and remain opt-in.
- Numerical equivalence of new parent predictors vs Python reference: max abs diff ~1e-16 in local synthetic checks.
- Static tests: 28 passed; 49 Python files parse with 0 AST errors.
