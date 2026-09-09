# Performance v12.6.0 — pooled LightGBM SHAPE SKU-total

## Evidencia causal previa a producción

Rolling PRE-OOS de 12 bloques, history=16, gamma elegido sin OOS:

| Panel | gain pooled leaf | recent-4 | win-rate | peor bloque | holdout |
|---|---:|---:|---:|---:|---:|
| sec1 Unidades | +0.79 pp | +0.84 pp | 100% | +0.37 pp | +0.63 pp |
| sec1 Valor | +0.70 pp | +0.86 pp | 100% | +0.25 pp | +0.74 pp |
| sec23 Unidades | +0.42 pp | +0.40 pp | 100% | +0.11 pp | +1.07 pp |
| sec23 Valor | +0.45 pp | +0.50 pp | 92% | -0.04 pp | +1.16 pp |

## Costo esperado

La capa entrena un LightGBM pooled por target/origen requerido por el selector
leaf. Para no repetir el trabajo, v12.6 precomputa una sola vez los bloques SKU
solapados usados por OOS, forecast-only y las ventanas históricas de selección.
No introduce modelos SKU×tienda. `n_jobs` del runner se reutiliza como
`num_threads` de LightGBM.

El costo incremental debe medirse en la corrida completa; el acceptance report
separa la calidad del shape y valida que ningún total SKU de 28 días cambió.
