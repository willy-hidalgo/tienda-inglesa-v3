# Tienda Inglesa Forecasting

## v12.9.10 — Value Sparse Rescue ACTIVE Promotion Gate

v12.9.10 conserva el selector productivo validado de v12.9.7 y promueve el rescue segmentado de Valor ($) de v12.9.9 **solo** cuando pasa un replay causal sobre el cohort oficial ACTIVE. El OOS actual nunca participa en la decisión de promoción. Cantidades permanece sin cambios. Ver `V12.9.10_VALUE_SPARSE_PROMOTION.md`.


## v12.9.8 — sparse-demand bias diagnostic

La política productiva de v12.9.7 queda congelada. v12.9.8 añade un challenger causal,
solo diagnóstico, para el sesgo negativo de hojas con 7–13 días de venta y corrige el
cierre de corrida para registrar `forecast.parquet` como `success` antes de construir
los artefactos pesados del dashboard. Ver `PERFORMANCE_v12.9.8.md`.

## v12.9.0 — Walk-Forward Portfolio Selector (Valor)

La optimización estadística se retoma **solo en Valor ($)** sobre la infraestructura multi-cadencia v12.8.6. Unidades queda congelado con la policy de v12.8.6.

- Valor usa 5 bloques históricos cerrados para construir 3 folds pseudo-OOS walk-forward.
- Cada fold evalúa `v11_all`, `v12_all` y, cuando hay historia suficiente, `meta_leaf` sin usar actuals del target.
- La evaluación es bottom-up y ponderada naturalmente por volumen mediante los denominadores de wMAPE.
- `v12_all` solo se promueve con win-rate mínimo 2/3, mediana de gain positiva, peor fold acotado, deterioro de |BIAS| <= 2 pp y gain mínimo de utilidad ponderada.
- `meta_leaf` debe cumplir los mismos guards y superar al mejor all-mode seguro por margen adicional.
- Evidencia insuficiente o inestable => `v11_all`.
- Acceptance report añade la sección 23 con folds, win-rate, median/worst/weighted gain, BIAS y razón de decisión.
- `--skip-existing` ya no reutiliza forecasts de otra `APP_VERSION`; una versión estadística nueva obliga a recalcular el forecast, aunque los artifacts sí pueden repararse sin refit cuando el forecast pertenece a la versión actual.

```text
APP_VERSION      = 12.9.0
ARTIFACT_VERSION = 20
Python           = >=3.13,<3.14
```

Para validar primero la referencia productiva:

```bash
uv run python app/forecasts.py --update-block-days 28 --n-jobs 8
uv run python -m app.forecasting.validate_v12 --update-block-days 28
uv run python -m app.dashboard_artifacts --update-block-days 28
uv run python -m app.dashboard_consistency --update-block-days 28
uv run python -m app.forecasting_acceptance_report --update-block-days 28
```

## v12.8.6 — hotfix artefactos multi-cadencia

- Cada escenario usa su propio `block_XXd/dashboard`; los artefactos 1d/7d/14d/28d ya no se sobrescriben.
- `--all-update-blocks --skip-existing` reutiliza forecasts existentes y reconstruye **solo** artefactos faltantes/desactualizados.
- `python -m app.dashboard_artifacts --all-update-blocks --skip-existing` genera únicamente los artefactos pendientes.
- Streamlit no cae a modo legacy para un escenario multi-cadencia sin artefactos: evita cargar el parquet completo y muestra el comando exacto de reparación.


# Tienda Inglesa — Forecasting híbrido v12.9.0 · multi-cadencia memory-safe

## v12.8.6 — Multi-block memory-safe pipeline

Esta versión **no modifica la lógica estadística**. Mantiene congelada la optimización y corrige el consumo de RAM de los escenarios 1/7/14 días. Para cadencias menores a 28d los estados leaf por origen se derraman a parquet temporal en `data/output/_spill`, los resultados de cada sección se escriben antes de continuar y los artefactos del dashboard se construyen recién después de liberar el forecast completo de memoria.

El paralelismo se limita automáticamente por seguridad: 1d→1 worker, 7d→2, 14d→4 y 28d→8. Un `--n-jobs` mayor se reduce al cap correspondiente. `--all-update-blocks --skip-existing` permite reanudar una precarga sin repetir escenarios ya materializados.

### Hotfix heredado v12.8.4 — horizonte OOS independiente de la cadencia

La cadencia de actualización (`1/7/14/28` días) controla únicamente cada cuánto se cierra un bloque y se incorporan nuevas observaciones al ajuste. El horizonte comparable de `out_sample` y `forecast_only` permanece fijo en 28 días.

Aplicación de forecasting jerárquico para secciones, tiendas y SKU+tienda.

```text
APP_VERSION      = 12.9.0
ARTIFACT_VERSION = 20
Python           = >=3.13,<3.14
```




## v12.8.4 — Demo multi-cadencia de actualización

Esta versión **congela la optimización estadística** y añade una capacidad de demostración: precalcular y visualizar cuatro escenarios donde el modelo incorpora bloques cerrados cada **1, 7, 14 o 28 días**. Los cuatro escenarios mantienen **el mismo OOS de 28 días y el mismo forecast-only de 28 días**, de modo que sus wMAPE/BIAS son comparables; solo cambia la cadencia con la que los actuals de cada bloque cerrado pasan al histórico expansivo.

Los resultados se guardan en `data/output/update_blocks/block_01d`, `block_07d`, `block_14d` y `block_28d`. Streamlit solo cambia entre artefactos precalculados: **no reentrena al mover el selector**.

Comandos principales:

```bash
uv run python app/forecasts.py --all-update-blocks --n-jobs 8 --skip-existing
uv run python -m app.dashboard_artifacts --all-update-blocks
uv run python -m app.dashboard_consistency --all-update-blocks
uv run streamlit run app/dashboard.py
```

También se puede calcular un único escenario con `--update-block-days 7` (o 1/14/28). El escenario de **28 días reproduce la baseline v12.8.3**.

## v12.8.3 — Value Portfolio Safety

v12.8.3 mantiene **sin cambios** los generadores, SKU-total, LightGBM shape-only, occurrence/share y la política de **Unidades** de v12.8.2. Solo endurece la decisión de **Valor ($)** entre `v11_all`, `v12_all` y `meta_leaf` usando información de bloques 28d ya cerrados:

- `v12_all` exige dominancia de wMAPE sin deteriorar `|BIAS|` más allá de la tolerancia.
- La señal `v12_all` debe confirmarse en los **dos bloques cerrados más recientes**.
- Un all-mode v12 requiere cobertura suficiente del guard de BIAS; la cobertura mínima se toma causalmente de la grilla `50/55/60/65%`.
- `meta_leaf` solo desplaza al mejor all-mode seguro si mejora su utilidad por un margen mínimo.
- Evidencia insuficiente => fallback seguro `v11_all`.

El dashboard conserva el hotfix de rendimiento: `metrics.parquet` sigue siendo la fuente oficial de wMAPE/BIAS OOS y el diagnóstico pesado se carga bajo demanda. El esquema agrega trazas de Value Safety, por lo que `ARTIFACT_VERSION=18`.

## v12.8.2 — adaptive causal meta-policy por sección × target

v12.8.2 conserva intactos los generadores de v12.8.1: SKU-total sin calibrar
(v12.6 baseline), LightGBM shape-only, true-hurdle compartido y store-share
28d×84d. La única intervención estadística está en la **política de selección**.

La corrida v12.8 mostró que el meta-modelo sí identifica hojas donde v12 gana,
pero un threshold global de 0.55 y el legacy section gate seguían dejando gran
parte del oracle gap sin capturar; en Sec.23/Unidades incluso bloquearon todas
las selecciones aunque `v12_all` era mejor que `v11_all`.

La nueva política es 100% causal y target-specific:

- un meta-modelo **más antiguo** aprende el ganador del bloque 2 usando solo
  bloques 3/4 y predice el bloque cerrado 1 con features 2/3/4;
- el bloque 1, ya cerrado antes del target, elige el mejor modo entre
  `v11_all`, `v12_all` y `meta_leaf`;
- si gana `meta_leaf`, el threshold se selecciona entre
  `0.45 / 0.50 / 0.55 / 0.60 / 0.65` usando ese mismo bloque forward;
- la utilidad sigue siendo `wMAPE + 0.25*|BIAS|` y existe un guard explícito de
  deterioro absoluto de BIAS de 3 pp, tanto a nivel leaf como portfolio;
- el modelo productivo se desplaza un bloque: aprende el ganador del bloque 1
  con features 2/3/4 y puntúa el target con 1/2/3;
- el legacy portfolio gate continúa materializado **solo como diagnóstico** y
  ya no puede vetar una decisión del meta-policy;
- Qty y Valor eligen política, threshold y familia de magnitud de forma
  independiente; occurrence/share sigue compartido.

La calibración de nivel v12.7 permanece como challenger diagnóstico y sigue
apagada por defecto. LightGBM continúa shape-only y debe preservar exactamente
el total SKU de 28 días.

El dashboard usa `ARTIFACT_VERSION=17` y muestra modo causal, threshold aprendido,
P(v12 gana), gain histórico de utilidad de la policy, BIAS guard y los escenarios
final/v11/v12/oracle.

Comandos después de generar el forecast:

```bash
uv run python -m app.forecasting.validate_v12
uv run python -m app.dashboard_artifacts
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
uv run streamlit run app/dashboard.py
```


## v12.7.0 — calibración causal de nivel + selector por utilidad + dashboard diagnóstico

v12.7 conserva el ensamble SKU-total v12.4, el store-share 28d×84d v12.5 y
la capa LightGBM shape-only v12.6. La intervención se concentra en el nivel y
en la selección, motivada por el fuerte BIAS negativo observado en la corrida
productiva v12.6.

Antes de LightGBM se aplica una calibración causal al total SKU de 28 días. El
factor se aprende exclusivamente con bloques cerrados anteriores, combina
mediana histórica y bloque reciente, se contrae hacia 1.0 según la cantidad de
historia y se limita al rango configurado. LightGBM sigue cambiando únicamente
el shape diario y mantiene exactamente el nuevo total calibrado de 28 días.

El selector leaf ahora exige simultáneamente mejora de wMAPE y mejora de una
utilidad que penaliza |BIAS|, además de confirmación reciente y estabilidad entre
bloques. Quantity/value mantienen selección conjunta. El acceptance report añade:

- sección 19: efecto de la calibración SKU-total;
- sección 20: oracle diagnóstico v11/v12 por leaf, **no causal**, para medir el
  gap atribuible al selector sin confundirlo con generación del forecast.

El dashboard v12.7 usa `ARTIFACT_VERSION=15`, conserva candidatos v11/v12 en los
artefactos, separa in-sample/OOS/forecast-only, muestra wMAPE y BIAS, y agrega un
panel OOS bottom-up Active con final/v11/v12/oracle. Si encuentra artefactos viejos
no se bloquea: abre el `forecast.parquet` actual en modo legacy y solicita
regenerarlos para recuperar velocidad.

Comandos recomendados después de generar el forecast:

```bash
uv run python -m app.forecasting.validate_v12
uv run python -m app.dashboard_artifacts
uv run python -m app.dashboard_consistency
uv run python -m app.forecasting_acceptance_report
uv run streamlit run app/dashboard.py
```

La promoción estadística de v12.7 depende de la corrida completa con datos reales;
la implementación no presupone que la calibración o el selector deban mejorar OOS.


## v12.6.0 — pooled LightGBM SHAPE a nivel SKU-total

v12.6 mantiene intactos el ensamble causal de total SKU, el total de 28 días,
el shared true-hurdle y el store-share 28d×84d de v12.5. Añade una única capa
supervisada pooled por sección/target para corregir exclusivamente la forma
diaria del SKU agregado. Nunca se ajusta LightGBM a SKU×tienda.

El target supervisado es el residual de shape normalizada entre actual y el
forecast SKU base. Las features son causales: forecast base, lags
28/56/84/364/365, shares de esos lags, estados recientes, posición del horizonte,
DOW/mes, eventos configurados, price-state histórico y categoría comercial.
El modelo usa 16 bloques cerrados de 28 días y parámetros congelados del
rolling PRE-OOS. `gamma=1.0` fue el ganador PRE-OOS en los cuatro paneles.

Después de predecir el residual, la trayectoria se renormaliza por SKU para
preservar exactamente el total de 28 días del ensamble base. Por tanto, la capa
solo puede modificar la forma temporal. El selector leaf v11/v12 sigue usando
discovery + confirmación cerrada y se recalcula sobre el challenger ya corregido.

Evidencia del diagnóstico holdout previo a integración productiva:

- sec1 Unidades: +0.63 pp leaf;
- sec1 Valor: +0.74 pp leaf;
- sec23 Unidades: +1.07 pp leaf;
- sec23 Valor: +1.16 pp leaf.

Los picks fueron hechos solo con PRE-OOS y mejoraron 4/4 paneles en el holdout.
La aceptación final de v12.6 debe hacerse sobre una corrida completa porque el
selector leaf puede cambiar al recibir el nuevo challenger.

## v12.5.0 — store-share causal por recencia 28d×84d

Los diagnósticos v12.4 separaron el error de allocation en occurrence y share.
Manteniendo fijo el total SKU, perfeccionar solo el share dentro del support
productivo redujo el wMAPE entre 6.35 y 10.41 pp, mientras eliminar el hurdle
cambió apenas -0.10 a +0.25 pp. Por tanto, v12.5 mantiene intactos el total SKU,
el shared true-hurdle y el selector anidado, y cambia únicamente el estimador de
share.

El nuevo share productivo es completamente causal y muy barato:

```text
share_raw = 0.70 * share_SKU×tienda_28d + 0.30 * share_SKU×tienda_84d
share     = normalize(share_raw | occurrence_support)
forecast  = forecast_SKU_total * share
```

No se usa el weekday share para repartir magnitud; el DOW se conserva únicamente
en occurrence. Si hay support abierto pero todos los pesos quedan en cero, se
reparte uniformemente entre las tiendas abiertas. Si ningún gate abre, se mantiene
el fallback base documentado para conservar la identidad de suma del total SKU.

La decisión se tomó sin usar OOS para seleccionar el candidato. En 12 bloques
PRE-OOS, `leaf_28x84` fue el robust winner en los cuatro paneles: gain pooled
+0.85/+1.08 pp en sección 1 (U/V) y +1.53/+1.55 pp en sección 23, con win-rate
83–92% y peor bloque entre -0.40 y -0.67 pp. En el OOS reservado también mejoró
los cuatro paneles (+1.08, +1.42, +0.93 y +0.49 pp respectivamente).

Los priors categoría/subcategoría/familia/subfamilia permanecen fuera de
producción: fueron estables pero con ganancia menor y no superaron al blend de
recencia. RLS SKU y RLS jerárquicos tampoco se incorporan a v12.5.


## v12.4.0 — ensamble causal del total SKU

El diagnóstico de descomposición de v12.3 mostró que el mayor cuello de botella
ya no era el selector leaf: con total SKU ORACLE, el wMAPE bajaba a ~40–49%,
mientras `v12_SKU + ORACLE_share` seguía ~47–55%. Por eso v12.4 mantiene
intactos el true-hurdle, store-share y selector anidado v12.3, y reemplaza el
único modelo fijo de magnitud SKU por un ensamble causal de seis candidatos:

- `v11_sum`: suma SKU del incumbent leaf;
- `recent_weekday`: nivel reciente 28/84 × perfil de día de semana;
- `same_weekday_4`: promedio del mismo weekday en lags 28/35/42/49;
- `lag28`: seasonal-naive de 28 días;
- `annual_scaled`: trayectoria 364d escalada sin dilución;
- `annual_blend`: blend reciente + trayectoria escalada 364d.

Para cada origen de 28 días, el submodelo SKU se selecciona usando solo tres
bloques cerrados anteriores (mínimo dos válidos). El score es wMAPE pooled con
un peso pequeño al bloque más reciente; no usa actuals del bloque objetivo.
Quantity y value pueden elegir distinto submodelo de magnitud, pero la decisión
leaf v11/v12 y el hurdle de ocurrencia continúan siendo conjuntos.

## v12.3.0 — selector temporal anidado + selección conjunta quantity/value

La corrida v12.2 mostró una mejora real en sección 23 y una mejora pequeña en
Unidades sección 1, pero `Valor ($)` sección 1 todavía quedó 0.20 pp peor que el
incumbent. El contrafactual sobre exactamente las hojas seleccionadas confirmó
que el problema no era el true hurdle: el subconjunto seleccionado ganó en
Unidades pero perdió en Valor. Además, los grandes gains históricos del portfolio
se midieron sobre los mismos bloques usados para descubrir las hojas, creando una
selección optimista.

v12.3 separa explícitamente **discovery** y **confirmation**:

1. Se usan 4 bloques cerrados de 28 días. Los bloques 2–4 son discovery; el bloque
   1 (el más reciente) queda fuera de los estadísticos pooled/wins de discovery.
2. Una hoja debe pasar discovery y volver a ganar en el bloque 1 de confirmación.
3. Por defecto `V12_REQUIRE_JOINT_TARGET_WIN=True`: cantidad y valor deben aprobar
   simultáneamente; la familia leaf ya no puede cambiar solo para un target.
4. Los portfolio guards de cantidad y valor se calculan sobre el mismo conjunto
   conjunto y ambos deben confirmar.
5. Occurrence pasa a un único **shared true hurdle** para cantidad y valor; los
   shares/magnitudes posteriores al gate siguen siendo específicos por target.

El acceptance report ahora calcula medianas de estabilidad **solo sobre las hojas
seleccionadas** y agrega una sección 15 de consistencia conjunta.


## v12.2.0 — true hurdle + recent-win guard

La corrida v12.1.1 confirmó que la selección multi-bloque es mucho más segura que
v12.0, pero todavía dejó una regresión en `Valor ($)` sección 1 (58.40% → 58.86%)
y ganancias muy pequeñas en los otros tres portfolios. También expuso una
incoherencia semántica en el challenger: aunque occurrence debía actuar solo como
**gate**, `_alloc_weight` todavía multiplicaba `store_share × occurrence_prob`.
Eso doble-penaliza tiendas intermitentes precisamente en días positivos, mientras
la métrica oficial excluye actual=0.

v12.2 hace dos cambios acotados y causales:

1. **True hurdle**: occurrence abre/cierra una tienda; una vez abierta, el share no
   se multiplica por la probabilidad. Los shares de las tiendas abiertas se
   renormalizan a 1. Si ninguna tienda abre, se conserva el fallback de shares base.
2. **Recent-win guard**: además de 2 victorias en 3 bloques y mejora pooled ≥2 pp,
   el bloque cerrado más reciente debe mejorar ≥2 pp. El portfolio de la sección
   también debe ganar en el bloque más reciente. Esto impide que dos victorias
   antiguas activen v12 después de un cambio de régimen.

El acceptance report corrige además la sección 13 (en v12.1.1 no se imprimía porque
`_oos_leaf_methods()` no proyectaba las columnas nuevas) y agrega la sección 14,
que compara **v11 vs v12 sobre exactamente las mismas hojas seleccionadas**. Así se
distingue una selección realmente útil de una simple diferencia de composición.

## v12.0.0 — challenger SKU-total + occurrence/store-share

La serie v11.6.1 queda congelada como incumbent. Los diagnósticos de reconciliación
mostraron que conocer perfectamente el total de tienda no corrige el error leaf y que
incluso un total diario SKU perfecto deja un wMAPE alto: el cuello estadístico está en
la distribución SKU×tienda y en la ocurrencia diaria por celda.

v12 agrega un challenger independiente, interpretable y causal:

1. **SKU total diario**: nivel de los últimos 84 días × perfil de día de semana,
   opcionalmente mezclado con una trayectoria 364d escalada solo con historia previa.
2. **Occurrence gate**: propensión de venta por SKU+tienda y día de semana, con
   shrinkage hacia la frecuencia global de los últimos 84 días.
3. **Store share**: participación de tienda por SKU, con share de día de semana
   contraído al share global; los shares se normalizan a 1 por SKU/fecha.
4. **Forecast leaf**: `forecast_sku_total × store_share`. En v12.0/v12.1 la
   probabilidad se usó además como peso dentro del share; v12.2 corrige esa
   implementación para que occurrence sea un hurdle puro.
5. **A/B causal**: v11 sigue siendo incumbent. Para OOS, v12 solo reemplaza una hoja
   cuando mejora el score en el bloque cerrado inmediatamente anterior; forecast-only
   puede usar el OOS ya observado para la selección. Nunca se mira el actual del bloque
   objetivo.

El forecast conserva ambos candidatos y añade diagnósticos `leaf_model_family_*`,
`v12_selected_*`, `v12_sku_forecast_*`, `v12_store_share_*`,
`v12_occurrence_prob_*` y métricas del bloque de validación. El reporte de aceptación
añade las secciones 11 (calidad por familia) y 12 (A/B contrafactual v11/v12/final).

Baseline congelado v11.6.1 Active OOS:

```text
Unidades sec 1   59.14%
Unidades sec 23  60.13%
Valor sec 1      58.40%
Valor sec 23     61.14%
```

La aceptación de v12 exige mejora material frente a esos cuatro valores y control de
`forecast sin demanda`; no se considera aprobada por construcción.


## v11.6.1 — corrección de invariante estacional + artefactos stale

La primera corrida v11.6.0 activó correctamente `sku_yoy_seasonal`, pero el
invariante de producción seguía comprobando la identidad anterior
`mean(forecast)/SES = parent_level_factor`. Con el nuevo candidato la identidad
correcta es `parent_level_factor × sku_seasonal_multiplier`; por eso la corrida
se detenía aun cuando los ejemplos mostraban precisamente el multiplicador SKU.

v11.6.1 corrige el invariante online y el validador offline. Además incorpora
`forecast_run_status.json`: una corrida fallida marca el estado `failed`, y
`validate_v11`, `dashboard_consistency` y `forecasting_acceptance_report` se
niegan a auditar el `forecast.parquet` anterior como si perteneciera a la nueva
versión. El último forecast bueno se conserva, pero queda explícitamente stale.

## v11.6.0 — estacionalidad anual específica por SKU (364 días)

La aceptación OOS de v11.5 mostró que `level_shape` del padre reducía BIAS en
sección 23, pero empeoraba wMAPE y generaba sobreforecast severo en algunas
tiendas. La señal decisiva fue que los mismos SKU (por ejemplo 520078 y 10735)
aparecían entre los mayores errores en varias tiendas con subforecast muy
similar: el efecto faltante es específico del producto, no del padre tienda.

v11.6 vuelve a `shape_only` como modo padre productivo y agrega un candidato
`sku_yoy_seasonal` causal y vectorizado. Para cada bloque de 28 días calcula,
a nivel SKU agregado sobre tiendas, el cambio observado entre los bloques
`b-13` y `b-14` (13×28 = 364 días), con shrinkage por cantidad de observaciones
positivas y clip robusto. El candidato compite contra el forecast vigente usando
solo el bloque de validación precedente; nunca usa actuals del bloque objetivo.

El reporte de aceptación agrega una sección específica para el factor estacional
SKU 364d y conserva wMAPE/BIAS, distribución de errores, métodos leaf y modos de
driver padre.


## v11.5.0 — driver padre con uplift de nivel causal

La evidencia OOS de v11.4 mostró que corregir el SES para días positivos no era suficiente: sección 23 siguió con BIAS fuertemente negativo durante un horizonte diciembre–Año Nuevo. v11.5 deja de imponer que todo efecto RLS leaf sea necesariamente mean-one.

Cada hoja elige causalmente entre `shape_only` y `level_shape` para padre tienda/sección. `level_shape` usa el forecast medio del padre RLS relativo al actual medio del bloque inmediatamente anterior, preservando tanto la forma diaria como un cambio de nivel comercial anticipado. El modo actual `shape_only` sigue disponible como candidato, por lo que el uplift no se fuerza.

## v11.4.0 — alineación del nivel leaf con la métrica oficial

La aceptación de v11.3.1 mostró wMAPE OOS de 59–65% y BIAS fuertemente negativo.
La causa estructural era una inconsistencia objetivo/modelo: la métrica oficial excluye
días con `actual=0`, mientras SES y los fallbacks estimaban nivel calendario incluyendo
ceros. v11.4 mantiene el estado SES calendario como baseline, pero:

- selecciona alpha y padre RLS usando solo días con venta positiva, igual que wMAPE/BIAS oficial;
- `deses_ses` estima el nivel desestacionalizado en tiempo de eventos positivos;
- `deses_robust_mean` promedia solo magnitudes positivas desestacionalizadas tras winsorizar picos;
- exige al menos 14 observaciones positivas históricas para activar esos niveles;
- mantiene causalidad estricta: OOS se selecciona solo con el bloque previo;
- mantiene los indicadores zero-demand por separado para detectar sobreforecast en días sin venta.

El cambio no modifica las ventanas OOS/forecast-only ni la arquitectura RLS Numba.

## v11.3.1 — hot path parent-only + fallback slim

La corrida real del 28-08-2026 confirmó que el kernel RLS ya no es el cuello de
botella (0.5–0.6 s sección y 2.0–2.6 s para 8 tiendas). Los costos dominantes
quedaron en fallback leaf y materialización del dashboard. v11.3.0 aplica:

- fallback secuencial: `direct_ses → deses_ses → deses_robust_mean`; la media
  robusta se calcula solo cuando SES desestacionalizado no mejora lo suficiente;
- fallback solo con bloque de validación Active (>=7 días con venta), evitando
  ajustes costosos/no defendibles sobre series sparse;
- refit final solo del método realmente seleccionado;
- `intercept=1` garantizado también en días densificados;
- dashboard lee solo la proyección de columnas que utiliza, no las 136 columnas
  del forecast;
- lookup O(n) para descripciones de SKU puro (elimina búsqueda O(SKU×IDs));
- exportación Excel parsea `unique_id` vectorialmente en Polars.

No cambia la ecuación RLS ni el contrato OOS/forecast-only. La nueva lógica de
fallback implementa literalmente la secuencia solicitada: el promedio robusto
solo se prueba si el SES sobre la serie desestacionalizada no funciona.


### v11.3.1: optimizaciones sobre log real

- Bias correction se aplica **antes** de concatenar SKU+tienda y solo sobre sección/tienda.
  Las hojas nunca reciben bias post-hoc, por lo que ya no se recorren millones de filas para dejarlas intactas.
- Fallback robusto trabaja sobre una proyección analítica de ~13 columnas, no sobre el frame completo de 100+ columnas.
- Predicción RLS por bloques (base/AR) pasa a kernels Numba compilados, evitando `np.concatenate`, `@` y loops Python por día/candidato.
- Los kernels de predicción se precalientan antes del pool de tiendas para evitar contención JIT entre threads.
- Checkpoints completos por sección quedan opt-in (`WRITE_SECTION_CHECKPOINTS=False`) porque escriben millones de filas y no son necesarios en el hot path normal.

Estos cambios son de ejecución/arquitectura: no cambian las fórmulas RLS, SES, drivers ni las reglas de selección/fallback.

## v11.2.2 — preflight autocontenido de datos

`app/forecasts.py` ya no asume que `data/output/selected.parquet` existe. Antes
de iniciar RLS ejecuta un preflight idempotente: reutiliza `selected.parquet` si
existe; si faltó únicamente la selección la reconstruye desde `master.parquet` y
`sales.parquet`; y en una instalación limpia puede reconstruir ingesta + selección
desde `data/input/`. Si faltan fuentes, termina inmediatamente con una lista clara
de archivos requeridos. El cambio no modifica la lógica estadística.

También se eliminó el fallback `LazyFrame.columns` en `_lf_columns`: con Polars
>=1.43 se usa exclusivamente `collect_schema()`, evitando el PerformanceWarning y
la doble resolución de schema observada cuando faltaba el parquet.

## Cambios v11.2.1.1

v11.2.1.1 corrige cuatro problemas observados en la validación productiva:

1. **OOS correctamente alineado:** el OOS es siempre los últimos 28 días con actuals. No existen actuals posteriores clasificados como `actual_extension` o `observed_tail`. Forecast-only empieza al día siguiente.
2. **Fallback robusto para hojas de alto error:** si el último bloque histórico causal tiene wMAPE alto, compiten `SES directo`, `SES sobre serie desestacionalizada` y `media robusta desestacionalizada con picos winsorizados`. El OOS nunca se usa para elegir su propio modelo.
3. **wMAPE/BIAS oficial:** se calcula únicamente sobre días con venta real `y != 0`, según la regla del cliente. Los falsos positivos en días sin venta permanecen visibles en `Forecast sin demanda`, `Impacto zero-demand` y `wMAPE All`.
4. **Rendimiento:** RLS de tiendas se ejecuta en batch con una sola partición de train/targets; se elimina la ruta productiva post-OOS legacy, se reutilizan estadísticas de bloques ya calculadas, se quita el diagnóstico pesado del hot path y el sort global de hojas pasa a ser opcional.

Los factores RLS siguen normalizados a media 1 y el contrato final sigue siendo `forecast = level × driver_factor`.

## v11.2.1.1 — corrección de escalabilidad

La v11.2.1.1 corrige dos hot paths que hacían inviable la ejecución a escala:

1. `rls_opt/kernels.py` vuelve a ejecutar `_rls`, `_rls_predict` y `_numba_outer`
   como kernels Numba `nopython=True, cache=True, nogil=True`. El runner falla
   explícitamente si `_rls` deja de ser un `CPUDispatcher`; no existe fallback
   silencioso a Python.
2. Los fallbacks `deses_ses` / `deses_robust_mean` dejan de particionar y recorrer
   cada SKU+tienda en Python. Primero se hace un gate barato sobre el último
   bloque causal de 28 días; solo hojas con wMAPE alto construyen candidatos y
   todo se resuelve con operaciones vectorizadas Polars.

Microbenchmark del kernel (700 observaciones × 100 drivers, este runtime):

```text
v11.1 Python kernel : ~2.44 s / fit
v11.2.1.1 Numba kernel  : ~0.02–0.03 s / fit (steady state)
```

El microbenchmark no sustituye el tiempo end-to-end con datos reales, pero
elimina el cuello de botella O(p²) que se estaba ejecutando en Python.

## Objetivos de diseño

La versión 11 simplifica la ruta estadística de SKU+tienda. La regla central es:

```text
SES puro                         → NIVEL
RLS tienda o RLS sección         → FORMA
factor RLS normalizado a media 1 → no cambia el nivel medio
```

No existe una ruta alternativa SKU+tienda, no existe `parent="none"` y no se
aplican strengths parciales de drivers.

Prioridades:

1. minimizar wMAPE OOS bottom-up;
2. mantener BIAS bajo;
3. evitar que picos transitorios eleven el nivel SES;
4. permitir que caídas reales reduzcan el nivel;
5. conservar los efectos temporales de RLS;
6. evitar leakage OOS;
7. mantener ejecución escalable.

## Flujo de la aplicación

```text
Archivos fuente
     │
     ▼
ingestor.py
     │
     ▼
categories_selector.py
     │
     ▼
selected.parquet
     │
     ▼
forecasting/pipeline.py
     │
     ├── sección  → RLS expanding-28
     ├── tienda   → RLS expanding-28
     └── SKU+tienda
            ├── SES puro → nivel
            └── mejor RLS store/section → forma
     │
     ▼
forecast.parquet
     │
     ├── dashboard_artifacts.py
     ├── dashboard_consistency.py
     └── forecasting/validate_v11.py
     │
     ▼
dashboard.py
```

## Contrato temporal

El entrenamiento conserva bloques expanding de 28 días. En v11.2.1.1 el bloque OOS se ancla al final de los datos, no a una fecha estática antigua:

```text
train ... bloques completos de 28 días
OOS           = últimos 28 días con actuals
forecast-only = 28 días desde last_actual + 1
```

Para conservar exactamente el contrato 28×28, `train_start` se alinea **hacia atrás desde el inicio del OOS**. Se pueden descartar como máximo 27 días iniciales de historia; nunca se desplaza el OOS hacia atrás dejando actuals posteriores.

Ejemplo con `last_actual = 2026-04-30`:

```text
train_end      = 2026-04-02
OOS            = 2026-04-03 → 2026-04-30
forecast-only  = 2026-05-01 → 2026-05-28
```

`actual_extension` y `observed_tail` quedan fuera de la ruta productiva v11.2.1.1.

## RLS — sección y tienda

`RLS_FIT_MODE = "expanding_28"`.

Cada bloque decide causalmente entre:

```text
RLS base
RLS + AR
```

Los drivers AR usan:

```text
lag 7
lag 28
rolling mean 7
rolling mean 28
```

y el forecast AR se construye recursivamente; nunca utiliza actuals del bloque
que se está pronosticando.

También se evalúan los forgetting factors configurados:

```python
RLS_FORGETTING_FACTOR_CANDIDATES = (0.970, 0.985, 0.995)
```

La elección para un bloque utiliza únicamente errores de bloques anteriores.

## SKU+tienda v11

Solo existe una ruta productiva:

```python
FAST_LEAF_MODE = True
```

Si se intenta desactivarla, el pipeline falla explícitamente.

### 1. Warm-up

Los primeros 28 días de cada hoja inicializan el nivel:

```text
L0 = Σ actuals primeros 28 días / 28
```

Esta definición se conserva porque fue la versión aprobada para evaluación.

### 2. SES puro

Después del warm-up:

```text
L_t = alpha*y_t + (1-alpha)*L_(t-1)
```

Los candidatos son:

```python
LEAF_SES_ALPHA_CANDIDATES = (
    0.005, 0.01, 0.02, 0.05, 0.10,
    0.20, 0.40, 0.60, 0.70, 0.80,
)
```

### 3. Selección de alpha

No existe acumulador exponencial infinito del error.

El score utiliza una ventana causal finita de los tres últimos bloques
evaluados:

```text
B0 = 60%
B1 = 30%
B2 = 10%
B3 =  0%
```

El score se calcula desde numeradores y denominadores:

```text
finite_wMAPE = Σ w_b*AE_b / Σ w_b*|Y|_b
finite_BIAS  = Σ w_b*SE_b / Σ w_b*|Y|_b

score = finite_wMAPE + 0.20*|finite_BIAS|
```

Lo sucedido antes de la ventana desaparece completamente del score.

## Régimen de nivel

El régimen NO cambia la ecuación SES. Solo define qué hacer ante una ruptura
evidente del nivel.

Se calculan medias de bloques reales de 28 días:

```text
B0 = último bloque completo
B1 = anterior
B2 = anterior
B3 = anterior
```

### Nivel estructural robusto

```text
structural = mediana(B1, B2, B3, B4, B5)
```

B0 queda fuera de la referencia que lo evalúa.

La ventana estructural tiene un corte duro de 5 bloques = 140 días.

### Tendencia

Para considerar una tendencia se exigen tres transiciones consecutivas:

```text
trend_up:
B3 → B2 → B1 → B0 crecientes

trend_down:
B3 → B2 → B1 → B0 decrecientes
```

Además, un salto individual no puede superar:

```python
LEAF_REGIME_TREND_MAX_STEP_RATIO = 2.0
```

Esto evita interpretar uno o dos picos recientes como una tendencia estructural.

### Transient up

Si:

```text
B0 > structural * 1.75
```

con historia suficiente y sin tendencia persistente:

```text
regime = transient_up
```

Antes de emitir el bloque siguiente:

```text
SES level ← structural
status    ← SES_RESET_TRANSIENT_UP
```

El forecast continúa siendo SES puro; la mediana solo reinicializa el estado
ante una ruptura transitoria explícitamente detectada.

### Transient down

Si:

```text
B0 < structural / 1.75
```

NO se reinicializa SES hacia el nivel histórico alto.

En su lugar se usa una trayectoria SES más reactiva:

```python
LEAF_TRANSIENT_DOWN_ALPHA = 0.20
status = SES_REACTIVE_TRANSIENT_DOWN
```

Esto evita que una hoja cuya demanda colapsó quede pegada a un nivel antiguo.

## 4. RLS obligatorio en SKU+tienda

Después de fijar el nivel SES, compiten solamente:

```text
RLS tienda
RLS sección
```

El padre con menor wMAPE causal acumulado en bloques anteriores gana.

No existen:

```text
parent = none
strength = 25%
strength = 50%
strength = 75%
```

El strength productivo es siempre:

```python
LEAF_PARENT_DRIVER_STRENGTH = 1.0
```

## 5. RLS modifica forma, no nivel

Para el padre seleccionado se construye:

```text
driver_factor
```

y se normaliza por horizonte para cumplir:

```text
mean(driver_factor) = 1
```

El rango permitido es:

```python
FAST_LEAF_DRIVER_FACTOR_CLIP = (0.20, 5.00)
```

El recorte se aplica escalando desviaciones alrededor de 1 y luego
renormalizando, no mediante clipping independiente que altere la media.

Forecast final:

```text
forecast = SES_level * driver_factor
```

La aplicación valida que:

```text
mean(forecast_raw) ≈ SES_level
```

por bloque.

## Causalidad de drivers

### OOS

Los drivers de precio OOS NO se calculan desde los actuals OOS.

OOS recibe el último estado de precio conocido en train y los drivers de
calendario conocidos al origen del forecast.

### Forecast-only

En v11.2.1.1 no existe una zona intermedia entre OOS y forecast-only: el OOS termina
en `last_actual` y forecast-only empieza exactamente al día siguiente. El estado
de precio disponible al origen futuro incorpora únicamente información ya
observada hasta `last_actual`; SES y RLS se actualizan solo con bloques cerrados
y nunca con datos futuros.

## Métricas OOS

Las métricas oficiales se calculan bottom-up desde SKU+tienda.

### Cohortes

Usando los 28 días OOS:

```text
zero-demand : 0 días con venta
sparse      : 1..6 días con venta
active      : >=7 días con venta
```

Configurable:

```python
OOS_METRIC_MODE = "active"
OOS_ACTIVE_MIN_NONZERO_DAYS = 7
```

### wMAPE Active — métrica principal

```text
wMAPE Active =
Σ|y-yhat| en días con y != 0
----------------------------
Σ|y| en días con y != 0
```

BIAS:

```text
BIAS Active =
Σ(yhat-y) en días con y != 0
---------------------------
Σ|y| en días con y != 0
```

Las agregaciones sección, tienda y SKU puro se reconstruyen desde los
componentes exactos de las hojas Active.

### Sparse y zero-demand

Sparse permanece visible y se reporta aparte.

Si una hoja tiene:

```text
Σ|y| OOS = 0
```

su wMAPE individual es:

```text
N/A
```

No se elimina su error. Se muestra:

```text
Forecast sin demanda
Impacto zero-demand
wMAPE All
```

`wMAPE All` conserva esos falsos positivos en el numerador.

## Ranking SKU

El ranking conserva un filtro más estricto:

```python
RANKING_SKU_MIN_NONZERO_POINTS = 15
```

Una hoja puede pertenecer a la cohorte Active para la métrica agregada y no
cumplir el criterio del ranking.

## Auditorías obligatorias

### Integridad del dashboard

```powershell
python -m app.dashboard_artifacts
python -m app.dashboard_consistency
```

Resultado requerido:

```text
AUDITORÍA DASHBOARD: OK
```

### Contrato del modelo v11

```powershell
python -m app.forecasting.validate_v11
```

Valida, entre otros:

```text
parent_model ∈ {store, section}
driver_strength = 1
mean(driver_factor) = 1
28 días OOS
28 días forecast-only
forecast-only sin actuals
forecast_raw = SES_level * driver_factor
```

Resultado requerido:

```text
AUDITORÍA MODELO v11.2.1.1: OK
```

### Diagnóstico de una hoja

```powershell
python -m app.forecasting.diagnose_leaf --uid "1||T:00001||S:299994"
```

Para ver las 28 filas además del resumen:

```powershell
python -m app.forecasting.diagnose_leaf --uid "1||T:00001||S:299994" --all-rows
```

Campos principales:

```text
ses_regime_class_value
ses_structural_level_value
ses_model_reference_block
ses_block_b0_value
ses_block_b1_value
ses_block_b2_value
ses_block_b3_value
ses_alpha_value
ses_alpha_unconstrained_value
ses_guard_status_value
ses_level_value
parent_model_value
driver_strength_value
driver_factor_value
pure_ses_wmape_value
pure_ses_bias_value
```

## Ejecución recomendada

Crear el entorno con Python 3.13:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

Tests:

```powershell
pytest -q
```

Pipeline:

```powershell
python app\forecasts.py --n-jobs 8
```

Auditorías:

```powershell
python -m app.forecasting.validate_v11
python -m app.dashboard_artifacts
python -m app.dashboard_consistency
```

Dashboard:

```powershell
streamlit run app\dashboard.py
```

También puede utilizarse el menú principal:

```powershell
python -m app.main
```

que muestra:

```text
=== Pipeline Tienda Inglesa v11.2.1.1 ===
```

## Criterio de aceptación

No declarar una mejora por uno o dos SKU aislados.

Revisar como mínimo:

```text
wMAPE Active OOS sección
BIAS Active OOS sección
wMAPE por tienda
distribución wMAPE SKU+tienda
sum_abs_error de hojas de alto volumen
wMAPE All
Forecast sin demanda
Impacto zero-demand
regímenes transient_up/down
selección store/section
```

Casos de control históricos:

```text
299994
261298
594880
587833
567332
483046
58905
```

Una reducción de wMAPE obtenida únicamente filtrando observaciones no se
considera una mejora estadística.

## Jupyter

Los notebooks existentes se conservan para análisis y validación.


## Hotfix v11.2.1

El kernel RLS de Numba es ahora BLAS-free: no requiere SciPy para compilarse ni ejecutarse.
Después de copiar esta versión, usar `uv sync` y luego `uv run python -m app.forecasting.benchmark_rls_kernel`.


## Entorno con uv (v11.2.1 packaging fix)

Para el pipeline y dashboard:

```bash
uv sync
```

`matplotlib` e `ipykernel` no forman parte del runtime de `app`/`rls_opt`; se instalarán solo si se necesita trabajar con notebooks:

```bash
uv sync --group notebook
```

Antes de ejecutar el pipeline completo, validar el kernel RLS:

```bash
uv run python -m app.forecasting.benchmark_rls_kernel
```
### Dashboard hotfix v12.8.2

El dashboard rápido usa `metrics.parquet` como única fuente de verdad para los KPI wMAPE/BIAS OOS bottom-up Active. La serie agregada del gráfico ya no se usa para recalcular esa métrica. Esto corrige falsas alertas de inconsistencia (por ejemplo, ranking ~57.9% vs KPI agregado ~82.8%) y elimina la carga de todas las hojas SKU×tienda durante el arranque normal. El diagnóstico avanzado Final/v11/v12/Oracle se carga bajo demanda desde un toggle. Ese hotfix queda incorporado en v12.8.3; como esta versión añade columnas de Value Safety, ahora se requieren artefactos v19 para la demo multi-cadencia.

## v12.9.1 — Volume-Weighted Regime-Aware Expected Gain Selector

Optimización estadística acotada a **Valor ($)** en la cadencia productiva de 28 días.
Qty y los generadores base permanecen congelados. El selector leaf SKU×tienda se
entrena con pseudo-OOS históricos estrictamente cerrados y un target continuo de
ganancia de utilidad v12 vs v11. El ajuste ridge usa `sample_weight` proporcional
al denominador de wMAPE para alinear el aprendizaje con el KPI bottom-up.

La promoción de v12 exige expected gain mínimo, confianza local, estabilidad de
ganancia histórica, guard de BIAS y un guard secundario de portfolio. Si la
evidencia no alcanza los mínimos, el leaf conserva v11. Para 1/7/14 días se
mantiene la policy anterior hasta validar primero el escenario 28d.

Acceptance agrega:
- sección 24: selector expected-gain, share, expected/realized gain y buckets de calibración;
- sección 25: top 50 contribuidores de error con v11, v12, final y oracle.

## v12.9.2 — Calibrated Expected-Impact + Asymmetric Risk Guards

Esta versión corrige el experimento v12.9.1 sin modificar Qty ni los generadores base.
El selector de `Valor ($)` permanece limitado al escenario productivo de 28 días.

Cambios estadísticos:

- conserva el ridge leaf ponderado por denominador como **raw score**, pero deja de
  interpretar su magnitud como ganancia esperada directa;
- construye una calibración **strictly-temporal out-of-fold**: un pseudo-OOS solo
  puede ser predicho por modelos entrenados con labels más antiguos;
- transforma el raw score en 5 buckets y usa la ganancia realizada histórica del
  bucket como `calibrated_gain` conservador;
- `confidence` queda acotado a `[0,1]` y combina ganancia calibrada, error OOF y
  frecuencia histórica de pérdidas;
- añade downside histórico por bucket y por leaf (`loss_rate`, `p10`);
- aplica requisitos más estrictos al top 10% y top 2% de hojas por volumen;
- ordena candidatos por expected-impact ajustado por downside y aplica un
  presupuesto máximo de volumen antes del guard agregado de portfolio;
- fallback sigue siendo v11 cuando la evidencia es insuficiente.

La auditoría de metadata expected-impact se limita a hojas donde la capa aplica;
las hojas sparse/zero pueden llevar metadata nula sin considerarse corrupción.
Una selección real de v12, en cambio, exige autorización explícita del budget y
portfolio guard.

Diagnóstico de aceptación:

- sección 24: raw gain → calibrated gain → realized gain por bucket OOF;
- sección 25: top contributors con selección, oracle, raw/calibrated gain,
  confidence, budget, percentil de impacto y loss-rates.

Configuración principal:

```python
V1292_VALUE_EXPECTED_IMPACT_ENABLED = True
V1292_VALUE_PORTFOLIO_MAX_VOLUME_SHARE = 0.45
V1292_VALUE_PORTFOLIO_MIN_EXPECTED_GAIN = 0.0025
```

`ARTIFACT_VERSION` permanece en 20 porque no cambia el contrato de los artefactos
slim del dashboard; `APP_VERSION=12.9.2` invalida correctamente forecasts previos.

## v12.9.3 — Hierarchical Segment Walk-Forward Selector

La decisión productiva de **Valor ($)** para la cadencia canónica de 28 días deja de depender del ridge leaf de v12.9.1/2. El ridge/calibrated expected-impact permanece únicamente como diagnóstico.

El selector productivo usa evidencia causal de bloques cerrados y una jerarquía interpretable:

1. `section × level_method × seasonal_flag × sales_days_bucket`
2. `section × level_method × seasonal_flag`
3. `section × level_method`
4. `section`
5. fallback conservador a `v11`

Cada segmento necesita al menos 3 folds válidos, win-rate mínimo 2/3, ganancia wMAPE bottom-up positiva y estable, downside acotado y BIAS controlado. El nivel sección exige una ganancia mucho más fuerte (3 pp) para autorizar una promoción masiva; de lo contrario se conserva una mezcla de segmentos con budget de volumen.

`v1292_impact_percentile_value` queda normalizado: **1.0 = máximo impacto; valores cercanos a 0 = menor impacto**.

El acceptance report agrega la sección 26 con distribución por nivel jerárquico, share de hojas/volumen v12, razones de fallback y segmentos de mayor impacto.


## v12.9.4 — Sec23 Volume-Risk Control

- Congela Qty y Valor Sec.1 en la policy v12.9.3.
- Valor Sec.23 usa budget duro de 25% de volumen, máximo 8% por segmento.
- Hojas top 1%/5% de impacto requieren evidencia walk-forward excepcional.
- El budget es estricto: una hoja solo entra si su volumen completo cabe.
- La exposición de segmentos se asigna por gain/estabilidad/downside; dentro de cada segmento se evita concentrar primero las hojas gigantes.
- Metadata segment-selector nula es válida cuando el selector no aplica; toda hoja Valor seleccionada v12 debe tener evidencia completa.

## v12.9.7 — Long-Horizon Segment Stress Test

Esta versión no cambia la policy productiva de v12.9.4. Amplía exclusivamente la
infraestructura de evaluación histórica de Valor ($), 28d:

- materializa hasta 12 bloques cerrados de 28 días previos al OOS;
- mantiene la selección productiva congelada sobre su pool original;
- reconstruye el régimen histórico de cada bloque (método de nivel, estacionalidad
  y bucket de días con venta) en lugar de reutilizar la etiqueta del OOS actual;
- calcula por segmento/hierarquía: folds, win-rate, mediana, p25, p10, worst,
  desviación estándar, máximo de pérdidas consecutivas, BIAS p90 y gain BU ponderado;
- define `stress-pass` solo como diagnóstico con mínimo 8 folds;
- el OOS actual queda completamente fuera del stress training/evaluation y se usa
  únicamente como test final en el acceptance report.

Nueva sección de aceptación:

`=== 28) v12.9.7 LONG-HORIZON SEGMENT STRESS TEST (DIAGNÓSTICO, NO PRODUCTIVO) ===`

## v12.9.7 — Sec23 L4 Long-Horizon Stress-Gated Selector

- Qty and Value Sec.1 remain frozen on the v12.9.5 production path.
- Value Sec.23 starts from v11 and only promotes leaves supported by long-horizon specific segment evidence.
- Productive levels: L4; L3 is allowed only as backoff when L4 lacks sufficient evidence. L2/L1 never authorize Sec.23.
- A sufficiently evidenced specific FAIL is terminal and cannot be overridden by broader evidence.
- Minimum history: 8 folds; win-rate >= 75%; median > 0; p25 > 0; p10 >= -1 pp; bias p90 worsen <= 1.5 pp; max consecutive losses <= 2.
- Top 5% and top 1% impact leaves require non-negative p10 gain.
- A 15% section-volume budget remains only as a secondary safety net.
- Acceptance report section 29 audits eligible/pass/fail/insufficient L4 evidence, selected volume, realized OOS gain, and top selected/rejected segments.

## v12.9.7 — Multi-Cutoff Temporal Robustness Audit

- Production forecast policy is frozen from the promoted v12.9.6 behavior.
- Adds a diagnostic replay of the Sec23 Value L4 stress-gated policy over historical pseudo-OOS cutoffs.
- Each cutoff is trained strictly on older 28-day blocks; the current OOS is excluded.
- Reports fold gains, selected-volume shares, BIAS deterioration, win rate, median/worst gain and a diagnostic replay-pass flag.
- No new selector is authorized by the v12.9.7 replay layer.

- **v12.9.11** adds a diagnostic-only Sec23 Value residual selector-gap challenger with leaf-specific long-horizon evidence, 10% exposure cap and strict multi-cutoff replay. v12.9.10 remains productive.

### Dashboard v12.9.11 — verificación UX/performance

Hotfix de dashboard sin cambios estadísticos: slider de cadencia, rankings alineados, horizontes por sección, no-activos, métricas in-sample, outliers y optimización de refresco. Ver `V12.9.11_DASHBOARD_VERIFICATIONS.md`.
