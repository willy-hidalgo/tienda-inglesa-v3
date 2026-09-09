# v12.9.6 — Long-Horizon Segment Stress Test

## Alcance

Diagnóstico únicamente. No modifica la decisión productiva v11/v12 de v12.9.4.

## Costo esperado

La corrida 28d materializa hasta 12 ventanas históricas en lugar de las 5 usadas
por la policy productiva. Por ello aumenta el tiempo de scoring histórico y el
uso temporal de CPU, aunque la policy final y el dashboard mantienen el mismo
esquema de artefactos (ARTIFACT_VERSION=20).

## Invariantes

- Qty sin cambios.
- Valor Sec.1 sin cambios.
- Valor Sec.23 conserva v12.9.4 como forecast productivo.
- OOS actual nunca entra al stress test.
- Solo los primeros `V12_SELECTION_BLOCKS` alimentan la selección productiva.
