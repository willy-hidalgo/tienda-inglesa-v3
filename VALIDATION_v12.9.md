# VALIDATION v12.9.0 — Walk-Forward Portfolio Selector

## Alcance

v12.9.0 retoma la optimización estadística exclusivamente en **Valor ($)**. La policy de Unidades permanece congelada con la evidencia de v12.8.6. Los generadores SES/RLS, SKU-total, LightGBM shape-only, occurrence, share 28/84 y la infraestructura multi-cadencia no cambian.

## Contrato causal

Para el escenario productivo de 28 días se materializan cinco bloques históricos cerrados. La policy de Valor utiliza tres pseudo-OOS walk-forward:

- fold 1: evalúa el bloque cerrado más reciente;
- fold 2: evalúa el segundo bloque cerrado;
- fold 3: evalúa el tercero.

Cuando se evalúa `meta_leaf`, cada fold entrena su meta-modelo únicamente con labels y features de bloques más antiguos que el fold que está siendo evaluado.

## Gates de promoción

`v12_all` solo puede desplazar a `v11_all` cuando:

- hay al menos 3 folds válidos;
- win-rate >= 2/3;
- mediana de gain wMAPE >= 0.25 pp;
- peor fold no degrada más de 4 pp;
- deterioro máximo de |BIAS| <= 2 pp;
- gain de utilidad bottom-up ponderada >= 0.30 pp.

`meta_leaf` debe pasar los mismos guards (con al menos 2 folds meta disponibles) y superar al mejor all-mode seguro por al menos 0.25 pp de utilidad.

Evidencia insuficiente o inestable implica `v11_all`.

## Invariantes

- Qty usa solo los 4 bloques de selección de v12.8.6 (`V129_QTY_FROZEN_SELECTION_BLOCKS=4`).
- OOS y forecast-only siguen siendo 28 días para comparación estadística.
- La calibración SKU-total permanece apagada.
- LightGBM conserva el total SKU 28d.
- `ARTIFACT_VERSION=20` por nuevas columnas diagnósticas.

## Acceptance report

La sección 23 muestra por sección:

- modo seleccionado y threshold;
- folds y win-rate;
- gain de cada fold;
- mediana y peor fold;
- gain wMAPE ponderado;
- gain de utilidad ponderado;
- deterioro máximo de |BIAS|;
- cantidad de folds meta;
- razón final de selección.
