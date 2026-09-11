# Tienda Inglesa Forecast v13.3.3

Modelo productivo: **SES leaf + RLS parent + transición estacional YoY causal por SKU+Tienda** para todos los SKU+Tienda de secciones 1 y 23.

Estado de modelos alternativos: evaluados, sin rutas activas. El Exception Lab queda como herramienta diagnóstica offline.

## Contratos productivos

- Secciones: 1 y 23.
- Universo: todos los SKU+Tienda elegibles; sin `select_best_skus`.
- Horizonte: OOS 28 días y forecast-only 28 días.
- Cadencias: 1/7/14/28 días.
- Métricas oficiales: wMAPE y BIAS con soporte `y != 0`.
- OOS no participa en tuning.
- El estado puede actualizarse causalmente al cierre de cada bloque.
- Dashboard: tablas con formatter global y coma como separador de miles para numéricos no %.
- Exception routing productivo: eliminado; rutas activas = 0.

## Comandos

```bash
uv run python -m app.forecasting.contract_gate
uv run pytest -q

uv run python -m app.forecasts --all-update-blocks --n-jobs 8
uv run python -m app.forecasting.validate_v13 --all-update-blocks
uv run python -m app.dashboard_consistency --all-update-blocks
uv run python -m app.forecasting.regression_gate --all-update-blocks
uv run python -m app.dashboard_artifacts --all-update-blocks
uv run python main.py --run dashboard
```

## Notas

El gate de regresión todavía puede señalar pocos casos localizados: incoherencia entre cadencias, reactivaciones long-gap y el centinela `478160`. No justifican activar otro modelo; deben tratarse como mejoras puntuales del SES+RLS/gap handling.


v13.3.3 añade un único ajuste estadístico productivo: un factor YoY causal y acotado por hoja. Para cada mes objetivo compara la mediana positiva del mismo mes del año anterior con el mes inmediatamente previo de ese mismo año histórico (por ejemplo, abril-2026 usa abril-2025 / marzo-2025). El ratio se contrae hacia 1 según soporte y queda en [0.50, 1.50]. Sin soporte suficiente vale 1.0.

El factor es independiente de la cadencia, no usa actuals OOS/forecast-only y se aplica multiplicativamente al factor RLS antes de la recurrencia SES, preservando una sola familia productiva.
