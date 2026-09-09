# Validación v12.7.0 — calibración causal de nivel y dashboard diagnóstico

## Objetivo

v12.7 corrige dos puntos observados en la corrida completa de v12.6: subpronóstico
del nivel SKU-total y selección v11/v12 insuficientemente alineada con BIAS y
estabilidad. No cambia el contrato de LightGBM shape-only ni el store-share.

## Contratos implementados

1. **Calibración causal SKU-total**: el factor se estima solo con bloques de 28 días
   completamente cerrados antes del origen objetivo. Se usa mediana robusta,
   recencia, shrinkage hacia 1.0, mínimo de historia y clip configurable.
2. **Orden del pipeline**: SKU-total base → calibración → LightGBM shape → hurdle →
   share tienda. La capa LightGBM sigue preservando exactamente el total de 28 días.
3. **Selector conjunto**: discovery/confirmation exige mejora de wMAPE y de una
   utilidad `wMAPE + λ|BIAS|`, además de estabilidad entre bloques y recent win.
4. **Oracle diagnóstico**: el acceptance report calcula una cota no causal eligiendo
   v11/v12 con verdad OOS por leaf. Nunca se usa para producir forecasts.
5. **Dashboard**: artifact version 15, candidatos v11/v12 persistidos, periodos
   in-sample/OOS/forecast-only separados y comparativa final/v11/v12/oracle.

## Validación disponible sin datos productivos

- compilación Python de módulos modificados;
- tests de contrato v12.7;
- tests históricos de v12.1–v12.6;
- auditoría de invariantes en `app.forecasting.validate_v12` cuando exista el
  `forecast.parquet` de la nueva corrida.

## Aceptación con datos reales

Después de ejecutar el pipeline completo:

```bash
uv run python -m app.forecasting.validate_v12
uv run python -m app.dashboard_artifacts
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
uv run streamlit run app/dashboard.py
```

La versión no debe darse por mejor estadísticamente hasta observar las secciones
19 y 20 del acceptance report, además de wMAPE/BIAS OOS por sección. Un resultado
peor no invalida la implementación: permite aislar si el problema está en nivel,
shape, allocation o selector.
