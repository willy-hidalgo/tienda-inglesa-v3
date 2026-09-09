# Performance v12.5.0 — share 28d×84d

v12.5 no agrega modelos RLS ni nuevas expansiones por SKU×tienda. Reemplaza el
weekday-share de v12.4 por dos agregaciones causales de share (28 y 84 días) y
un blend 70/30. El occurrence gate existente se conserva.

## Evidencia PRE-OOS (12 rolling origins)

| Panel | gain pooled | recent-4 | win-rate | peor bloque |
|---|---:|---:|---:|---:|
| sec1 Unidades | +0.85 pp | +1.67 pp | 92% | -0.51 pp |
| sec1 Valor | +1.08 pp | +2.09 pp | 83% | -0.42 pp |
| sec23 Unidades | +1.53 pp | +0.53 pp | 83% | -0.40 pp |
| sec23 Valor | +1.55 pp | +0.48 pp | 92% | -0.67 pp |

El candidato ganó 4/4 paneles sin usar OOS para selección.

## Costo esperado

El costo incremental frente a v12.4 es bajo: dos group-by adicionales por target
para ventanas 28/84 dentro de `_block_candidate`. Se elimina el uso del weekday
share como peso de allocation, aunque las estadísticas DOW siguen calculándose
para occurrence. No se introduce ningún RLS nuevo ni bucles por SKU×tienda.
