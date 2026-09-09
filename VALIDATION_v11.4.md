# Validación v11.4.0

## Motivo del cambio

La aceptación de v11.3.1 pasó las auditorías estructurales y de dashboard, pero no la aceptación estadística:
- wMAPE OOS Active por sección: ~59%–65%.
- BIAS OOS fuertemente negativo, especialmente sección 23.
- muchos SKU+tienda Active cercanos a 100% de wMAPE.

El código de v11.3.1 estimaba el nivel SES/fallback sobre días calendario incluyendo cero demanda, mientras la métrica oficial excluye actual=0. v11.4 alinea la selección del nivel y del padre con el soporte de la métrica oficial.

## Cambios estadísticos

1. Alpha SES y padre RLS se puntúan solo en días con actual>0.
2. `deses_ses` usa SES en tiempo de eventos positivos sobre `actual / driver_factor`.
3. `deses_robust_mean` usa media de magnitudes positivas desestacionalizadas tras winsorizar picos.
4. Los fallbacks exigen al menos 14 observaciones positivas históricas.
5. OOS sigue siendo estrictamente causal y de 28 días; forecast-only empieza al día siguiente.
6. Zero-demand sigue auditado por separado; no se oculta un posible incremento de falsos positivos.

## Validación local disponible

- 50 archivos Python: AST/sintaxis OK.
- 28 tests de contrato fuente/RLS/performance/positive-metric: PASS.
- No se ejecutó el pipeline Polars completo en el runtime de construcción por no disponer de Polars.

## Gate de aceptación en datos reales

Ejecutar:

```bash
uv sync
uv run python app/forecasts.py --n-jobs 8
uv run python -m app.forecasting.validate_v11
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
```

No promover v11.4 si:
- wMAPE Active empeora frente a v11.3.1;
- |BIAS| no mejora materialmente;
- zero-demand crece de forma desproporcionada;
- el tiempo total vuelve a orden de decenas de minutos.
