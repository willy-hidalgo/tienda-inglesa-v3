# Changelog

## v13.3.3
### Seasonal YoY leaf correction
- Mantiene una única familia productiva: SES leaf + RLS parent.
- Añade un factor estacional YoY causal específico de SKU+Tienda para corregir transiciones de nivel recurrentes que el SES reciente y el parent agregado no capturan.
- Para un mes objetivo M/Y usa únicamente la historia M/(Y-1) versus el mes precedente del mismo año histórico; abril-2026, por ejemplo, usa abril-2025 / marzo-2025.
- Nivel histórico robusto: mediana de días positivos. Requiere >=7 positivos en ambos meses; confiabilidad completa a 14; shrink lineal hacia 1.0.
- Guard productivo: factor en [0.50, 1.50]. Sin soporte suficiente: 1.0.
- El factor es independiente de la cadencia 1/7/14/28, no usa actuals del OOS/forecast-only objetivo y deja trazas `leaf_yoy_*` + `leaf_total_factor_*`.
- `ARTIFACT_VERSION=33`.

## v13.3.2
### Hotfix causal de gap OOS/forecast-only
- Corrige limpieza incompleta de `runner.py`: no quedan imports ni llamadas a `exception_routing`.
- El staleness por gap deja de quedar congelado en el primer origen OOS; se recalcula en cada origen de bloque desde la última venta positiva cerrada, sin mutar el estado SES.
- Objetivo: permitir que 1d/7d/14d reaccionen a secuencias sin venta y que forecast-only no herede mecánicamente el nivel alto previo al OOS.
- No cambia alpha, parent, lambda/dynamics, métricas ni familia productiva SES+RLS.


- Limpieza productiva: se elimina exception routing activo y sus comandos productivos.
- El modelo productivo queda como SES+RLS puro para todos los SKU+Tienda.
- `exception_lab.py` queda únicamente como laboratorio offline.
- `HARD_FAILURE_CHAMPION` se reclasifica semánticamente como `EXTREME_RELATIVE_ERROR`.
- Se conserva trazabilidad neutral `exception_model_* = ses_rls_champion` y `exception_routing_applied_* = False` para compatibilidad de artefactos/dashboard.
- Documentación actualizada con contratos y tratamiento de SKU+Tienda inactivos/dormantes.

## v13.3.1

- Evaluación de exception routing forward-only.
- Resultado de OOS: 2 candidatos actuales vetados; 0 rutas activas.

### v13.3.3 hotfix — warm-up YoY trace identity

- Corrected `leaf_total_factor_y/value` on warm-up rows to `1.0`, matching the SES kernel, which already forecasts warm-up with no RLS/YoY factor.
- Fixes false `validate_v13` failures for `yhat_raw/valuehat_raw` identity and `leaf_total_factor == driver_factor * leaf_yoy_factor`.
- No statistical forecast change outside trace/export consistency; OOS and forecast-only values are unchanged by this hotfix.
