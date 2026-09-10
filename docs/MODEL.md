# Modelo productivo v13

## 1. RLS en Sección y Tienda

Cada Sección y cada Tienda se modelan con RLS usando la librería `rls_opt` incluida en el repositorio.

El RLS trabaja con drivers construidos causalmente, incluyendo señales calendario/comerciales disponibles. La historia es expansiva: a medida que se cierran bloques, el modelo puede incorporar la información observada para el siguiente origen.

La selección interna de especificación RLS se hace con errores acumulados en **bloques anteriores**; nunca usa el resultado del bloque objetivo para decidir ese mismo bloque.

## 2. Nivel SKU+Tienda mediante SES

### Warm-up

Para cada hoja se toman los primeros 28 días calendario disponibles y se calcula:

```text
L0_qty   = median(y | y > 0)
L0_value = median(value | y > 0 y value > 0)
```

La mediana evita dos problemas:

- dilución por días sin venta;
- sensibilidad excesiva a picos puntuales.

El warm-up es inicialización del mismo modelo SES+RLS; no una familia estadística distinta.

### Recurrencia

Después del warm-up, el SES usa toda la historia disponible. Desde v13.2.11, los días sin venta sí hacen avanzar causalmente el estado cuando la misma Sección+Tienda fue observable ese día; días sin evidencia de operación/datos no se convierten en cero. En v13.2.10 la hoja **no usa** la separación intercepto/no-intercepto de los coeficientes RLS, porque esa descomposición puede cambiar mucho bajo colinealidad aunque el forecast agregado del parent sea estable. La transferencia usa una magnitud identificable y centrada:

```text
nivel_parent_t = mediana de actuals positivos del parent en los 28 días previos cerrados
raw_t          = log1p(forecast_RLS_parent_t)
referencia_t   = log1p(nivel_parent_t)
relativo_t      = raw_t - referencia_t
centro_t        = mediana causal reciente de relativo_t
efecto_leaf_t  = clip(relativo_t - centro_t, log(0.50), log(2.00))
obs_ajustada   = expm1(log1p(actual_leaf) - efecto_leaf_t)
obs_robusta    = clip(obs_ajustada, 0.50*L_{t-1}, 2.00*L_{t-1})
L_t            = alpha * obs_robusta + (1-alpha) * L_{t-1}
cap_t          = 2.00 * max(actual positivo en bloques cerrados previos)
forecast_leaf  = min(max(expm1(log1p(L_t_origen)+efecto_leaf_t), 0), cap_t)
```

La referencia se fija al inicio de cada bloque de actualización. Por tanto, 1d puede incorporar el día cerrado anterior, 7d cada semana, etc.; ningún actual del bloque objetivo interviene en su propio forecast. Si un bloque completo cambia de offset por más de `LEAF_DRIVER_REFERENCE_SHIFT_FACTOR`, ese offset persistente se centra usando el propio path de forecasts conocido al origen y no se transfiere como nivel leaf. El coeficiente no-intercepto bruto se conserva únicamente como traza de auditoría.

Desde v13.2.10 la recurrencia SES es robusta: un único positivo extremo no puede mover el estado más allá del rango causal `[0.50x, 2.00x]` respecto del estado previo antes de aplicar alpha. Además, tras 7 positivos cerrados, el forecast final queda protegido por un tope de `2.0x` el máximo positivo observado antes del bloque. El tope se calcula en el origen, se persiste en `leaf_forecast_cap_y/value` y nunca usa actuals del bloque objetivo.

El alpha se selecciona causalmente entre candidatos usando **solo historia in-sample cerrada**. Al primer origen OOS se congela explícitamente la selección de alpha, parent y `dynamics/lambda`; los actuals OOS pueden actualizar el estado operativo para un origen posterior de la misma cadencia, pero nunca re-seleccionan la configuración.

## 3. Selección de parent: Tienda vs Sección

Para cada target (Unidades y Valor) se mantiene el error histórico del RLS de Tienda y del RLS de Sección.

Al inicio de cada bloque:

```text
si wMAPE_store <= wMAPE_section → parent = store
si wMAPE_section < wMAPE_store → parent = section
```

Si aún no existe historia suficiente, se usa el fallback configurado y disponible; por defecto, Tienda.

La decisión nunca usa el OOS actual del bloque objetivo.

## 4. Identidad del forecast leaf

La hoja usa el forecast RLS del parent relativo al nivel causal del mismo parent:

```text
driver_effect_raw       = log1p(forecast_RLS_parent)
driver_effect_reference = log1p(nivel_parent_causal)
driver_effect_center    = offset causal persistente del relativo identificable
efecto_RLS_leaf         = clip(driver_effect_raw - driver_effect_reference - driver_effect_center, log(0.50), log(2.00))
log(1 + forecast_leaf)  = log(1 + nivel_SES) + efecto_RLS_leaf
```

`driver_effect_coef_raw` conserva la contribución no-intercepto original del RLS solo para auditoría y **no entra** en el forecast leaf.

por tanto:

```text
forecast_leaf_raw = max(
    expm1(log1p(nivel_SES) + efecto_RLS_leaf),
    0
)
```

El guard equivale a un factor RLS transferido dentro de `[0.50, 2.00]`. Su función es impedir que un offset arbitrario o una explosión de coeficientes del parent destruya el nivel SES. El `max(..., 0)` sigue siendo la restricción física de soporte no negativo. La identidad es la misma en in-sample, OOS y forecast-only.

## 5. In-sample, OOS y forecast-only

### In-sample

No es un fitted-value arbitrario. Es walk-forward histórico: cada bloque se pronostica con el estado existente al inicio de ese bloque.

### OOS

28 días reservados para evaluación. Usa exactamente SES+RLS con el estado disponible al origen OOS.

### Forecast-only

28 días posteriores sin actuals. Usa la misma familia; no actualiza el estado dentro de ese horizonte porque todavía no existen ventas reales.

## 6. Métricas oficiales

Se calculan solo en puntos con `y != 0`:

```text
wMAPE = Σ|y - ŷ| / Σ|y|
BIAS  = Σ(ŷ - y) / Σ|y|
```

- menor wMAPE es mejor;
- BIAS < 0 indica subestimación;
- BIAS > 0 indica sobreestimación.

Estas métricas gobiernan selección, ranking y KPI oficiales.

## 7. Métricas all-points

Como prueba ácida:

```text
wMAPE_all = Σ_todos |y - ŷ| / Σ_todos |y|
BIAS_all  = Σ_todos (ŷ - y) / Σ_todos |y|
```

Los puntos `y=0` no aumentan el denominador pero sí pueden aportar error/sesgo. Son solo diagnóstico y no alteran decisiones productivas.

## 8. Restricciones del modelo

El modelo debe mantener simultáneamente:

- causalidad;
- misma familia en todas las fases;
- interpretación simple;
- velocidad;
- consumo de memoria controlado;
- Polars como motor tabular;
- ausencia de capas experimentales fuera de SES+RLS durante la optimización vigente.
## 9. Optimización estadística v13.1

La optimización no introduce otra familia de modelos. Solo compara parámetros dentro del mismo contrato:

- SES: `LEAF_SES_ALPHA_CANDIDATES`;
- RLS: `RLS_FORGETTING_FACTOR_CANDIDATES`;
- dinámica RLS: `base` y `ar`, cuando AR está habilitado.

Cada candidato se puntúa en bloques cerrados con wMAPE oficial (`y != 0`). El candidato del bloque objetivo se sigue eligiendo con información acumulada de bloques anteriores. El diagnóstico adicional conserva por separado el OOS actual como holdout.

El screening de drivers trabaja únicamente con columnas ya presentes. En esta primera fase anula la contribución de cada grupo sobre el camino RLS ajustado para detectar señales útiles, débiles o potencialmente perjudiciales. Para weekday/month/price/holiday no equivale todavía a una ablation refiteada y no modifica el modelo; los grupos dudosos se someterán a esa prueba en la fase siguiente. La parte autoregresiva sí se compara contra el RLS base refiteado con la misma lambda, porque ese candidato ya existe en el runner.


## 10. Phase 2: búsqueda diagnóstica previa a promoción

Phase 2 evalúa alphas SES y lambdas RLS adicionales con una máscara explícita que impide que los candidatos diagnósticos modifiquen el forecast durante la evaluación.

Para drivers se usa refit exacto del mismo RLS. La convención del reporte es:

```text
wmape_improvement = wMAPE_actual - wMAPE_challenger
```

Por tanto, un valor positivo significa que la acción probada mejora el RLS actual. La acción puede ser `remove` para un grupo existente o `add` únicamente para el grupo de precio ya existente en el diseño de Unidades y testeado sobre Valor ($).

La elegibilidad del target Valor sigue el contrato del negocio: un punto entra si `y>0`, incluso cuando `value==0`.


## 11. Baseline productivo v13.2

La promoción manual posterior a Phase 2 mantiene exactamente la identidad SES+RLS y cambia únicamente parámetros/drivers existentes:

- SES productivo incluye `alpha=0.30`;
- RLS productivo incluye `lambda=0.990` y `0.9975`;
- `lambda=1.0` permanece fuera de la selección productiva;
- Valor ($) del nodo de Sección 1 usa el mismo RLS con el grupo de precio existente; los demás nodos Valor continúan sin ese grupo.

Estas decisiones no crean rutas especiales para OOS o forecast-only. Un parámetro/driver promovido se usa mediante el mismo walk-forward en in-sample, OOS y forecast-only según la selección causal disponible en cada origen.


## Gobierno de tuning v13.2.1

La estructura del modelo no cambia. Los cambios de parámetros o drivers del mismo RLS se seleccionan con historia cerrada y se validan después en OOS. Esta separación impide usar el holdout para optimizar y mantiene la misma composición SES+RLS en in-sample, OOS y forecast-only.


## Promociones node-specific v13.2.2

Las promociones de drivers no crean modelos ni variables nuevas. `RLS_DRIVER_GROUP_EXCLUSIONS` permite remover grupos **ya existentes** para un `unique_id` y target cuando cumplen dos etapas: candidato por historia cerrada y validación posterior en OOS holdout.

Promociones vigentes:

- `1||T:00211`: remover `month` en Unidades y Valor ($);
- `23||T:00006`: remover `price` en Unidades.

La misma selección de columnas se utiliza en bloques in-sample, OOS y forecast-only. Phase 2 conserva un challenger de add-back para que una exclusión promovida pueda ser auditada posteriormente con el mismo RLS.

## Diagnóstico de efecto faltante v13.2.3

La optimización residual no modifica el modelo. Para una hoja SKU+Tienda, dado el nivel SES `L`, el actual `y` y el efecto RLS aplicado `e`, se calcula el efecto que habría sido necesario para reproducir el actual con la misma identidad del modelo:

`required_effect = log1p(y) - log1p(L)`

`unexplained_effect = required_effect - e`

Esta diferencia no se aplica al forecast. Solo cuantifica la parte del movimiento observado que no fue explicada por los drivers actuales. Si aparece de forma repetida y estructurada por fecha, weekday, mes o tienda, sirve como evidencia para revisar los drivers existentes o, previa aprobación, proponer información adicional.


## Calibración diagnóstica del efecto RLS v13.2.4

Producción conserva exactamente `gamma=1.0` en la identidad SES+RLS. La Phase 4 solo evalúa, sobre forecasts causales ya emitidos, si la intensidad del efecto RLS parece demasiado fuerte o demasiado débil:

`log1p(forecast_gamma) = log1p(nivel_SES) + gamma * efecto_RLS`

La búsqueda de `gamma` es exclusivamente diagnóstica y utiliza historia cerrada para crear candidatos; OOS únicamente valida o veta. No existe un parámetro productivo `LEAF_DRIVER_STRENGTH` ni una ruta alternativa por período. Cualquier promoción futura tendría que ser explícita y mantener la misma ecuación para in-sample, OOS y forecast-only.

La misma fase compara el forecast productivo contra un contrafactual SES-only (`gamma=0`) para cuantificar si los drivers existentes reducen o aumentan el error en los puntos extremos. Esta atribución no altera el forecast.


## Selección diagnóstica del parent en contexto leaf v13.2.5

Producción conserva la regla vigente de un único parent RLS por target y bloque. La Phase 5 agrega una auditoría contrafactual dentro de la **misma** familia: para cada SKU+Tienda reutiliza exactamente `ses_level` y `ses_alpha` productivos y reconstruye el forecast con el efecto RLS de Tienda o de Sección ya calculado. No se refitea un nuevo modelo leaf.

Las trazas comparadas son `current_policy`, `store`, `section` y `ses_only`. `ses_only` equivale a aplicar efecto cero al mismo nivel SES y solo sirve para atribución; nunca puede ser parent productivo. Los candidatos `store`/`section` se puntúan sobre `y>0` con el mismo redondeo final que las métricas oficiales.

La recomendación se construye solo con bloques cerrados: mejora >=1 pp frente a `current_policy`, >=4 folds, >=60% de folds no peores y deterioro histórico de |BIAS| <=1 pp. El OOS actual se consulta después y solo valida/veta. Esta fase todavía no cambia `parent_model_y`/`parent_model_value` en `forecast.parquet`.

La auditoría complementaria de `zero_rate` usa `y<=0` para describir ocurrencia por hoja, weekday, mes y bloque. Es diagnóstica: no altera la definición oficial de wMAPE/BIAS ni introduce una capa de occurrence/share.

## 12. Baseline vigente v13.2.11

La configuración productiva de parámetros permanece congelada en el baseline v13.1.1:

- alphas SES productivos: `0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80`;
- lambdas RLS productivos: `0.970, 0.985, 0.995`;
- Valor ($) no incorpora price de forma productiva;
- no hay exclusiones node-specific productivas.

La corrección v13.2.11 conserva los guards de v13.2.10 y añade SES gap-aware con días cero observables por Sección+Tienda, reactivación causal tras gaps largos y una ventana histórica canónica idéntica entre 1d/7d/14d/28d. Las promociones históricas documentadas en v13.2.0/v13.2.2 permanecen como antecedentes experimentales, no como configuración vigente.
