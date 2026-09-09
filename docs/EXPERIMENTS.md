# Experimentos históricos y estado de gobierno

## Estado actual

La optimización estadística está **reactivada** desde v13.1.0, pero bajo un contrato estricto:

1. solo se optimiza SES + RLS;
2. la misma familia sigue generando in-sample, OOS y forecast-only;
3. tuning usa exclusivamente bloques históricos cerrados;
4. el OOS actual es holdout final;
5. no se crean/promueven drivers nuevos sin aprobación explícita;
6. no existe promoción automática desde los diagnósticos.

### Fase activa

- SES: comparar `alpha` candidatos manteniendo fijo el RLS parent.
- RLS: comparar `lambda` y dinámica `base/ar` dentro de la librería RLS existente.
- Drivers: screening de contribución de los drivers actuales. Los grupos que resulten dudosos pasan después a una ablation refiteada del **mismo RLS**; no se sustituye el modelo.

Los artefactos se guardan en `statistical_optimization/` y el resumen textual separa explícitamente historia de tuning y OOS holdout.

## Qué se investigó en v12

Durante v12 se evaluaron, entre otros:

- occurrence/share;
- SKU-total ensembles;
- LightGBM shape-only;
- sparse-demand rescue;
- calibraciones de nivel;
- meta-selectores v11/v12;
- residual selectors;
- high-impact level rescue;
- fast causal selectors y budgets adaptativos;
- diagnósticos SKU-total agregados.

Esos experimentos fueron útiles para entender sesgo, sparse demand, selección y estabilidad. Sin embargo, el conjunto productivo terminó siendo demasiado complejo y podía producir una historia in-sample distinta de OOS/forecast-only.

## Decisión v13

Los experimentos anteriores **no forman parte del código productivo v13**. No se mantienen módulos de forecast legacy para poder reactivarlos accidentalmente.

La optimización vigente respeta estas reglas:

1. todo tuning parte del contrato SES+RLS vigente;
2. se conserva la misma familia en las tres fases;
3. solo se usa información causal para seleccionar;
4. las mejoras deben ser repetidas en backtests y confirmadas en holdout;
5. BIAS y velocidad permanecen como restricciones;
6. no se promueve ninguna lógica que no pueda explicarse claramente a un usuario de negocio.

## Regla documental

Los experimentos futuros se documentarán aquí y en `CHANGELOG.md`; no se crearán archivos Markdown por parche/versiones individuales.

## Phase 2 — refinamiento de parámetros y refit exacto de drivers

El diagnóstico Phase 1 mostró dos señales que justifican ampliar la búsqueda sin cambiar de modelo:

- varios RLS ganadores históricos se ubican en `lambda=0.995`, borde superior de la grilla productiva;
- en SES, algunos ganadores históricos están en `alpha=0.005`, borde inferior de la grilla productiva.

Por ello `--optimization-phase2` agrega candidatos **solo diagnósticos** alrededor de esos bordes. Esos candidatos se puntúan con la misma historia causal, pero nunca pueden ser seleccionados por el forecast productivo hasta una promoción explícita.

Además se reemplaza el screening aproximado por un refit exacto del **mismo RLS**:

1. se toma el mejor candidato histórico `dynamics/lambda` del nodo;
2. se refitea el mismo RLS removiendo un grupo de drivers existente;
3. se evalúan los mismos bloques cerrados;
4. para Valor ($) se prueba adicionalmente agregar el grupo de precio ya disponible;
5. se reporta mejora de wMAPE, share de folds ganadores y cambio en |BIAS|.

No se promueve ningún cambio automáticamente. Los resultados de Phase 2 son evidencia para una decisión posterior.

## Phase 3 — promociones manuales v13.2

La revisión de Phase 2 permitió promover parámetros sin cambiar la arquitectura:

1. `alpha=0.30` se incorpora al SES productivo. La señal fue consistente para Sección 23 en historia y holdout.
2. `lambda=0.990` y `0.9975` se incorporan al RLS productivo. `lambda=1.0` continúa como diagnóstico porque la ausencia total de olvido mostró resultados mixtos entre nodos/secciones.
3. Los drivers de precio existentes se habilitan en **Valor ($) únicamente para Sección 1**. El resto de nodos conserva la especificación previa.
4. No se promueve `alpha=0.0025`: aunque mejora el agregado histórico de Sección 1, el holdout de Unidades empeora de forma material, por lo que se mantiene como diagnóstico.

El refit Phase 2 ahora se reporta por `unique_id`; esta corrección es necesaria para decidir futuras inclusiones/remociones por tienda sin mezclar nodos distintos.

La siguiente iteración debe volver a ejecutar `--optimization-phase2` sobre el baseline v13.2 y revisar principalmente:

- qué tiendas concretas justifican añadir precio en Valor ($);
- qué tiendas concretas de Sección 23 justifican retirar price/month en Unidades;
- si `holiday_other` tiene evidencia suficiente para conservarse en cada nodo;
- si los residuos extremos persisten después de los cambios de parámetros, antes de proponer cualquier driver nuevo.


## Phase 4 — validación holdout de candidatos (v13.2.1)

La promoción de drivers pasa a dos etapas explícitas. Primero, `candidate_add`/`candidate_remove` se determina únicamente con bloques históricos cerrados. Después se consulta el OOS actual como holdout independiente. Un holdout negativo veta el cambio; un holdout favorable solo lo valida, pero la promoción sigue siendo manual. El artefacto `driver_refit_validation.parquet` conserva esta trazabilidad.


## Phase 5 — promociones holdout validadas (v13.2.2)

El reporte v13.2.1 produjo tres `validated_candidate`: quitar `month` en `1||T:00211` para ambos targets y quitar `price` en `23||T:00006` para Unidades. Esos cambios se promueven manualmente. Los candidatos con `holdout_veto` permanecen fuera de producción.

El diagnóstico futuro prueba el **add-back** de una exclusión promovida con el mismo RLS y la misma lambda/dinámica seleccionada históricamente; así una decisión puede reevaluarse sin introducir una nueva familia de modelos.

## Phase 6 — diagnóstico residual causal SKU+Tienda (v13.2.3)

No se promueven nuevos parámetros ni drivers desde el reporte v13.2.2. Las exclusiones ya promovidas se reauditan mediante add-back del mismo RLS y permanecen justificadas; los candidatos nuevos no superan el holdout.

La siguiente fase analiza residuos extremos del forecast productivo, no una familia alternativa. Se usa una escala causal basada en el estado SES disponible al origen y se calcula cuánto efecto RLS adicional habría sido necesario para reconciliar el nivel SES con el actual. Se agregan señales por tienda, fecha, weekday, mes y parent para distinguir: (a) régimen/local, (b) patrón calendario insuficiente, (c) posible evento transversal no representado y (d) driver comercial/price insuficiente.

También se genera `promotion_reaudit.parquet`, que vuelve a probar por add-back cada exclusión ya promovida después de la nueva selección de lambda/dynamics. Esto permite detectar interacciones de segundo orden sin usar OOS para tuning.

Los artefactos residuales son `leaf_residual_extremes.parquet`, `leaf_residual_summary.parquet` y `residual_signal_summary.parquet`. Para controlar RAM se usa OOS completo + 168 días recientes de historia cerrada y los targets se procesan secuencialmente. Ninguna señal modifica producción automáticamente; cualquier driver nuevo requiere aprobación explícita.


## Phase 7 — calibración de intensidad y recurrencia calendario (v13.2.4)

El reporte v13.2.3 muestra que el error extremo de hojas sigue siendo muy superior al error de varios parents RLS y que, en algunas fechas, la concentración es transversal. Antes de proponer drivers nuevos se separan dos hipótesis dentro de la arquitectura vigente.

Primero se evalúa la intensidad del efecto RLS en la hoja con una grilla diagnóstica `gamma`. `gamma=1` reproduce producción, `gamma=0` equivale a observar el nivel SES sin efecto parent y los valores intermedios/superiores permiten detectar sobre-reacción o sub-reacción. El candidato se define solo con historia cerrada, incluyendo consistencia por folds, y OOS se consulta posteriormente como holdout.

Segundo se incorpora una taxonomía del residuo: participación del error de días `y=0`, error oficial de días con venta positiva y proporción de extremos en que el driver ayuda frente al SES-only. Esto permite diferenciar un problema de intensidad del driver de uno de nivel/ocurrencia.

Finalmente se compara cada `MM-DD` del OOS con el mismo día de años históricos cerrados. Este análisis es calendario-diagnóstico, no un nuevo driver. Su objetivo es determinar si una concentración como fin de año es recurrente y ya debería ser capturable por los drivers existentes, o si corresponde a un cambio de régimen/operación que no debe inferirse a partir del holdout.


## Phase 8 documental — selección parent en contexto SKU+Tienda (v13.2.5)

El diagnóstico v13.2.4 mostró que, en varias tiendas, el RLS agregado puede tener métricas razonables mientras su efecto apenas ayuda en extremos leaf. Antes de añadir drivers nuevos se evalúa si el problema es **qué parent existente se transfiere a cada hoja**.

La nueva auditoría compara `current_policy`, `store` y `section` usando el mismo SES/alpha productivo; añade `ses_only` únicamente para medir valor incremental del driver. La recomendación nace con historia cerrada y el OOS solo valida/veta. No se promueve ningún switch automáticamente.

En paralelo se guarda `zero_rate` por hoja y patrones de cero por weekday/mes/bloque. Esta evidencia no entra al modelo; sirve para separar error de magnitud positiva de error de ocurrencia sin reintroducir occurrence/share.
