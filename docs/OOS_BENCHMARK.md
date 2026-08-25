# OOS benchmark — guardrail SKU-tienda

Este benchmark se ejecutó directamente sobre los CSV de ventas de `data/input`,
filtrando las secciones y tiendas configuradas. No usa predicciones del RLS y,
por tanto, **no debe interpretarse como el wMAPE final del pipeline**. Su objetivo
es medir baselines temporales leakage-free y calibrar el guardrail de hojas.

La métrica replica la regla del proyecto: `sum(abs(y-yhat))/sum(abs(y))`, excluyendo
`y == 0`. Los baselines usan únicamente información anterior al inicio de OOS.
La elección de método por sección se hizo sobre los últimos 28 días del train,
antes de mirar el OOS.

| Sección | Ventana OOS | Método elegido en validación train | wMAPE OOS bottom-up | Mediana wMAPE SKU-tienda | P75 SKU-tienda |
|---|---|---|---:|---:|---:|
| 1 | 2026-03-30 → 2026-04-26 | `median_pos56` | 0.5173 | 0.3200 | 0.5000 |
| 23 | 2025-12-01 → 2025-12-07 | `weekday_pos8` | 0.4635 | 0.4444 | 0.6250 |

Rango de wMAPE por tienda con esos baselines:

- Sección 1: 0.4643–0.5794 (mediana 0.5158).
- Sección 23: 0.4130–0.5681 (mediana 0.4832).

Como diagnóstico, el SES directo sobre demanda con `alpha=0.74` produjo 0.7021
en sección 1 y 0.6218 en sección 23. Esto confirmó que el valor fijo anterior
era demasiado reactivo para las hojas y motivó bajar el default residual a 0.20
y añadir el guardrail robusto.

## Criterio de producción

El pipeline conserva RLS+drivers como candidato principal. En SKU-tienda, el
baseline robusto reemplaza el forecast futuro solo si supera al candidato RLS+SES
por el margen configurado sobre una cola de validación del train. Si no lo supera,
el RLS se conserva. Además, el forecast RLS queda limitado a un rango amplio
alrededor del baseline para evitar explosiones numéricas.

El OOS no actualiza el estado SES con actuals del propio OOS: el estado queda
congelado al cierre del train, que es la simulación correcta de un horizonte
multi-step.
