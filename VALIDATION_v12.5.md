# Validación v12.5.0 — contrato share 28d×84d

## Contrato causal

- El share de 28 días usa exclusivamente fechas `< origin` y `>= origin-28d`.
- El share estable usa exclusivamente fechas `< origin` y `>= origin-84d`.
- Blend: `0.70*share28 + 0.30*share84`.
- El support continúa siendo el shared true-hurdle quantity/value de v12.3.
- El blend se normaliza dentro del support; un gate cerrado recibe share 0 cuando
  existe al menos una tienda abierta para el SKU/día.
- Si todos los pesos de las tiendas abiertas son cero, se usa uniforme dentro del
  support; si ningún gate abre, se conserva el fallback base para poder asignar
  el total SKU.
- El total SKU, su ensamble causal y el selector leaf anidado siguen intactos.

## Evidencia rolling PRE-OOS

`leaf_28x84` fue robust winner en los cuatro paneles con 12/12 bloques válidos.
Los gains pooled fueron +0.85 pp y +1.08 pp en sección 1, y +1.53 pp y +1.55 pp
en sección 23 (Unidades/Valor). Win-rate 83–92%; peor degradación -0.40 a -0.67 pp.

## OOS reservado observado antes de integrar

Manteniendo total SKU y occurrence support fijos, `leaf_28x84` mejoró:

- sec1 Unidades: 62.93% -> 61.85% (+1.08 pp)
- sec1 Valor: 62.66% -> 61.23% (+1.42 pp)
- sec23 Unidades: 58.05% -> 57.13% (+0.93 pp)
- sec23 Valor: 61.26% -> 60.77% (+0.49 pp)

Estos valores son del diagnóstico aislado de share. La aceptación productiva de
v12.5 debe hacerse sobre una corrida completa porque el selector leaf puede cambiar.
