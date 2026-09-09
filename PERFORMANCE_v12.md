# PERFORMANCE v12.4.0

La corrida real v12.2.0 midió:

```text
Pipeline RLS: 94.2 s
challenger sec 1: 9.5 s
challenger sec 23: 4.3 s
Dashboard artifacts: 28.0 s
```

v12.3 agrega un cuarto bloque cerrado para separar discovery (3 bloques anteriores) de confirmation (bloque más reciente). Esto agrega una construcción/puntuación histórica por sección, pero no introduce loops Python por SKU ni por SKU+tienda: el hot path sigue vectorizado con Polars `group_by`, `join` y window expressions.

La selección conjunta quantity/value reutiliza los scores ya calculados y tiene costo marginal. El shared occurrence gate reemplaza dos decisiones booleanas independientes por una decisión común y tampoco aumenta la complejidad asintótica.

Expectativa de runtime: incremento moderado frente a 94.2 s, concentrado en `_block_candidate` + `_score_closed_block` del cuarto bloque. La prioridad sigue siendo calidad estadística; no se recortará el backtest anidado antes de medir su efecto OOS.

## Controles de rendimiento

- No conservar candidatos históricos completos después de puntuarlos.
- Reutilizar `sku_daily_all`, primera fecha SKU y primera fecha SKU+tienda entre orígenes.
- Mantener occurrence/share vectorizado.
- No reintroducir checkpoints intermedios por sección.


## v12.4 — costo del ensamble SKU-total

El ensamble añade scoring vectorizado de cinco trayectorias SKU sobre tres
bloques cerrados antes de cada origen. No introduce loops Python por SKU ni por
SKU+tienda; los únicos loops son sobre el número fijo de bloques/métodos. Los
candidatos se construyen con `group_by`, `join` y expresiones horizontales de
Polars. El objetivo de runtime sigue siendo mantener el pipeline en el orden de
1–2 minutos para la muestra actual. La corrida real es necesaria para medir el
costo exacto porque este entorno no dispone de Polars/fastexcel offline.
