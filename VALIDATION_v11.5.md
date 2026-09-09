# Validación v11.5.0

## Motivo del cambio

v11.4.0 mejoró el BIAS, pero la aceptación OOS siguió alrededor de 58–63% wMAPE. El patrón más fuerte aparece en sección 23, cuyo OOS cruza diciembre/Navidad/Año Nuevo. Hasta v11.4 el driver leaf se forzaba a media exactamente 1 dentro de cada horizonte de 28 días: podía redistribuir ventas entre fechas, pero no trasladar un uplift comercial que afectara el nivel medio de todo el bloque.

## Hipótesis v11.5

Para cada padre RLS (tienda o sección) se construyen dos candidatos causales:

- `shape_only`: comportamiento v11.4, factor diario con media 1.
- `level_shape`: misma forma diaria multiplicada por el cambio de nivel que el RLS padre anticipa frente al nivel real del bloque inmediatamente anterior.

El multiplicador de nivel para el bloque b es:

`mean(parent_forecast_b) / mean(parent_actual_{b-1})`

Solo usa información disponible al origen del forecast. Se limita al mismo máximo cambio de bloque ya definido por `LEAF_REGIME_TREND_MAX_STEP_RATIO` para evitar que una inestabilidad numérica del padre se convierta en un salto no auditable.

## Selección causal

Cada hoja compite entre:

1. store + shape_only
2. store + level_shape
3. section + shape_only
4. section + level_shape

El bloque objetivo nunca elige su propio candidato. La selección usa wMAPE acumulado de bloques anteriores, manteniendo el contrato expanding-28.

## Invariantes

- `forecast_raw = ses_level * driver_factor` siempre.
- `shape_only`: media del driver = 1.
- `level_shape`: media del driver = `driver_level_factor`.
- OOS y forecast-only conservan exactamente 28 fechas.
- padre siempre store/section, strength=1.
- forecast-only puede reutilizar la forma OOS si el padre futuro es plano, pero conserva el level-factor causal pronosticado para el horizonte futuro.

## Gate estadístico

Comparar contra v11.4.0:

- wMAPE Active OOS por sección/unidad.
- BIAS Active OOS por sección/unidad.
- zero-demand / forecast sin demanda.
- contribuyentes principales al error absoluto.
- calidad OOS de `shape_only` vs `level_shape`.

No promover si level_shape reduce BIAS pero deteriora materialmente wMAPE/zero-demand, o si no aporta mejora clara en sección 23.
