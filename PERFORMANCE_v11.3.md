# Performance audit — Tienda Inglesa v11.3.1

## Baseline real v11.2.2 (28-08-2026)

- Pipeline RLS: 110.3 s.
- Sección 1 leaf: 58.5 s; fallback: 38.5 s.
- Sección 23 leaf: 25.6 s; fallback: 14.6 s.
- Dashboard artifacts: 77.8 s.
- RLS sección/tiendas: ~5.8 s total; ya no es el cuello de botella.

v11.3.1 ataca específicamente fallback y dashboard; no modifica el kernel RLS.

## Problemas encontrados en v11.1

### 1. RLS no estaba compilado con Numba

`rls_opt/kernels.py` importaba `jit`, pero `_rls`, `_rls_predict` y
`_numba_outer` no tenían decorador. `_numba_outer` ejecutaba un doble bucle
Python de tamaño `n_features × n_features` por observación y candidato RLS.
Con aproximadamente 100 drivers, el costo por actualización es O(p²) y la
ruta no era viable en producción.

v11.2.1.1 usa:

```python
@jit(nopython=True, cache=True, nogil=True)
```

El runner verifica que `_rls` sea un `numba.core.registry.CPUDispatcher` y
falla de inmediato si se pierde la aceleración.

Microbenchmark realizado en el runtime de construcción, 700 observaciones y
100 drivers:

```text
v11.1 sin JIT : 2.4413 s / fit
v11.2.1.1 con JIT : 0.02–0.03 s / fit steady-state
```

La primera compilación JIT cuesta algunos segundos; `cache=True` evita pagar
ese costo completo en cada ejecución posterior.

### 2. Fallback SKU+tienda tenía un loop Python por hoja

v11.1 hacía:

```text
risk_ids -> partition_by(unique_id) -> for cada hoja -> target y/value ->
reconstruir historia -> SES/robust mean -> score
```

El tiempo crecía aproximadamente con `n_hojas_riesgosas × historia × targets`.

v11.2.1.1 hace:

```text
último bloque 28d -> group_by global -> gate de riesgo ->
solo hojas riesgosas -> candidatos vectorizados Polars -> join final
```

No existe `partition_by(unique_id)` ni loop Python por hoja en
`leaf_fallback.py`.

El SES desestacionalizado usa la forma cerrada de la recurrencia sobre datos
sparse, por lo que no requiere densificar SKU×fecha:

```text
L_E = (1-a)^(E-B) L0 + a Σ (1-a)^(E-t) * (y_t / driver_factor_t)
```

La media robusta desestacionalizada usa mediana/MAD para limitar picos y divide
por días calendario, conservando cero-demanda implícita.

## Gates de validación

- Mismo contrato OOS final de 28 días.
- Mismos drivers RLS mean-one.
- `forecast = level × driver_factor` se conserva.
- Fallback OOS solo usa el bloque de validación anterior al OOS.
- Forecast-only puede validar con OOS porque ya está observado a ese origen.
- Métricas oficiales siguen excluyendo días `actual=0`; zero-demand se reporta
  separadamente.

## Validación disponible en este runtime

- 46 archivos Python parseados sin errores de sintaxis.
- 49 tests sin dependencia de Polars: passed.
- Kernel Numba 700×100: ~0.02–0.03 s/fit steady-state.

El pipeline Polars end-to-end debe medirse en el entorno productivo del usuario.


## v11.2.1 — kernel BLAS-free

El hot path RLS ya no usa `numpy @`/`dot` dentro de funciones Numba.
Se reemplazó por productos vector/matriz explícitos e in-place compilados por Numba.
Esto elimina la dependencia runtime de SciPy/BLAS para compilar el kernel y reduce temporales.

Validación local:

```text
shape              : 700 x 100
steady-state fit   : ~0.035 s
SciPy disponible   : NO (bloqueado intencionalmente en la prueba)
resultado          : OK
```

Equivalencia numérica vs implementación NumPy de referencia: error máximo ~1e-14 o menor.

## Log 2026-08-28 and v11.3.1 response

Observed v11.3.0:
- Pipeline RLS: 140.7 s (cold JIT included).
- Section 1 fallback: 22.9 s; section 23 fallback: 12.3 s.
- Section 1 bias correction: 13.0 s despite leaves being structurally exempt.
- Dashboard artifacts: 28.2 s (down from 77.8 s in prior run).

v11.3.1 removes the avoidable full-leaf bias pass, slims fallback working frames, Numba-compiles parent recursive block prediction, and disables full-section checkpoints by default.
