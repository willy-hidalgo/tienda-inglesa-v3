# RLS expanding-28 y métricas

## Requisito cliente

Modo por defecto:

1. Días 1-28: actuals usados para estimar el estado RLS.
2. Ese estado pronostica días 29-56.
3. Al cierre del día 56, los actuals 29-56 ya fueron incorporados recursivamente.
4. El estado de cierre de 56 pronostica 57-84.
5. Se repite el proceso en bloques de 28 días.
6. OOS usa el estado disponible al inicio de su bloque, sin leakage.
7. Forecast-only usa el último estado disponible después de todos los actuals conocidos.

La implementación usa un único recorrido RLS y snapshots de coeficientes en
28, 56, 84, ...; no reentrena desde cero. Esto es equivalente al esquema
expansivo requerido y mantiene el desempeño.

## Settings

```python
RLS_FIT_MODE = "expanding_28"   # default; "current" restaura el modo anterior
RLS_BLOCK_DAYS = 28
METRICS_MODE = "rolling_28"     # default; "current" restaura métricas anteriores
FAST_LEAF_DRIVER_EFFECTS = True
```

## SKU-tienda

Para mantener escalabilidad no se ejecuta RLS por cada hoja. El nivel robusto
rápido de cada SKU-tienda se multiplica por el perfil temporal relativo del RLS
de su tienda; si no existe, usa sección. Por ello OOS y forecast-only ya no son
líneas planas: calendario, feriados, precio/promoción y otros drivers que
afecten al modelo padre alteran el forecast diario de la hoja.

## Métricas

Las filas RLS contienen:

- `rls_block`
- `rls_train_days`
- `rls_metric_eligible`

En `METRICS_MODE="rolling_28"` sección y tienda calculan wMAPE/BIAS con forecasts
emitidos por los bloques rolling. Los primeros 28 días son warm-up y no se
puntúan. `METRICS_MODE="current"` conserva el comportamiento anterior.
