# VALIDATION v11.6

Validaciones locales disponibles sin la data del cliente:

- `python -m compileall -q app rls_opt settings.py`: OK.
- 27 tests de contrato/fuente v11.1–v11.6: PASSED.
- Contrato causal: factor del bloque b usa únicamente b-13 y b-14.
- Producción: `level_shape` deshabilitado; parent RLS conserva `shape_only`.

La aceptación estadística debe ejecutarse con la data real mediante:

```bash
uv run python app/forecasts.py --n-jobs 8
uv run python -m app.forecasting.validate_v11
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
```

## v11.6.1

- Invariante online final: `mean(forecast)/SES = parent_level_factor × sku_seasonal_multiplier`.
- Invariante offline: `forecast = SES × driver_factor × sku_seasonal_multiplier`.
- `forecast_run_status.json` evita validar artefactos de una corrida anterior después de un fallo.
- Acceptance report exige que `forecast.parquet`, `index.json` y `metrics.parquet` correspondan a la misma corrida.
