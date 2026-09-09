## 13.2.6 — Emergency statistical rollback / regression freeze

- Restores the **productive statistical configuration of v13.1.1** after end-to-end regressions were observed in later v13.2 promotions (notably abnormal 1d OOS spikes and depressed 28d leaf levels).
- Productive SES alpha grid returns to v13.1.1; `alpha=0.30` is diagnostic-only again.
- Productive RLS lambda grid returns to `(0.970, 0.985, 0.995)`; `0.990`, `0.9975`, and `1.0` are diagnostic-only.
- Removes productive Value-price promotion on Section 1 and the node-specific driver exclusions introduced in v13.2.2.
- Keeps the current dashboard, `main.py`, all-block preparation, audits, and optimization diagnostics.
- This is a stability release: future product changes must clear a 1d/7d/14d/28d regression gate before promotion.

## 13.2.5 — Phase 5 parent RLS por SKU+Tienda (diagnóstico causal)

## v13.2.5-r1 — entrega estable dashboard multi-cadencia

- Sin cambios estadísticos ni de forecast respecto de v13.2.5. `APP_VERSION` permanece en 13.2.5 para poder reutilizar corridas vigentes de la misma versión.
- Añade `python -m app.dashboard_ready` y `main.py --run dashboard-ready-all`: prepara/repara secuencialmente 1d/7d/14d/28d, reutiliza solo corridas vigentes, reconstruye artefactos y ejecuta auditoría de modelo + dashboard antes de declarar la interfaz lista.
- El dashboard ya no ofrece escenarios stale/incompletos en el selector; solo muestra forecasts de la APP_VERSION actual con artefactos sincronizados.
- La ruta estable no ejecuta diagnósticos de optimización. La optimización queda congelada para retomarse después de la entrega operativa.


- No cambia el forecast productivo ni incorpora modelos/drivers nuevos. Producción sigue usando SES + un parent RLS existente con `gamma=1`.
- Añade diagnóstico leaf-specific `current_policy` vs `store` vs `section`, reutilizando exactamente el mismo nivel SES y alpha emitidos por producción; `ses_only` se persiste solo como referencia diagnóstica y nunca puede ser promovido.
- Un switch Section/Store solo nace por historia cerrada si mejora al menos 1 pp de wMAPE, tiene >=4 folds, gana/empata >=60% de folds y deteriora |BIAS| histórico <=1 pp. El OOS actual únicamente valida/veta.
- Nuevos artefactos: `leaf_parent_candidates.parquet`, `leaf_parent_summary.parquet`, `leaf_parent_section_summary.parquet`, `leaf_parent_recommendation.parquet`, `leaf_zero_rate_summary.parquet` y `leaf_zero_rate_patterns.parquet`.
- La auditoría de venta cero registra zero-rate por SKU+Tienda y patrones por weekday/mes/bloque; no entra al wMAPE/BIAS oficial ni a la selección del modelo.
- El detalle diario alternativo no se escribe en `forecast.parquet`; solo se conservan agregados leaf×período×candidato para proteger RAM y el esquema del dashboard permanece en `ARTIFACT_VERSION=22`.
- `APP_VERSION=13.2.5`.

## 13.2.4 — calibración diagnóstica del efecto RLS y recurrencia calendario

- No cambia el forecast productivo ni incorpora drivers/modelos nuevos; `gamma=1.0` sigue siendo la identidad SES+RLS usada en producción.
- Añade Phase 4 diagnóstica para evaluar la intensidad del efecto RLS en hojas mediante `log1p(forecast)=log1p(nivel_SES)+gamma*efecto_RLS`; la recomendación se genera solo con historia cerrada y OOS únicamente valida/veta.
- La taxonomía residual separa error de días con `y=0`, error oficial con venta positiva y cuantifica si el efecto RLS ayuda o empeora cada discrepancia frente a un forecast SES-only.
- Añade recurrencia por `MM-DD` contra años históricos cerrados para distinguir patrones calendario recurrentes de cambios de régimen/ceros excepcionales, sin crear variables productivas.
- Nuevos artefactos: `driver_strength_candidates.parquet`, `driver_strength_summary.parquet`, `driver_strength_recommendation.parquet`, `calendar_day_yearly.parquet` y `calendar_day_recurrence.parquet`.
- Los reportes Phase 3 incorporan `wmape_all_points`, participación del error en días `y=0`, share de extremos de cero-venta y share de extremos donde el driver reduce el error.
- `APP_VERSION=13.2.4`; `ARTIFACT_VERSION=22` se mantiene.

## 13.2.3 — diagnóstico causal de residuos extremos SKU+Tienda

- No se promueve ningún parámetro ni driver nuevo: el reporte v13.2.2 confirma las exclusiones `1||T:00211 - month` y `23||T:00006 Unidades - price`, mientras los nuevos candidatos históricos quedan vetados o en revisión.
- `--optimization-phase2` incorpora una Phase 3 puramente diagnóstica sobre el forecast ya emitido: identifica discrepancias extremas Actual vs Forecast en hojas SKU+Tienda con la misma regla factor/gap usada por el dashboard.
- Para evitar usar el OOS al definir materialidad, la escala residual es causal: `max(nivel SES al origen, nivel inicial robusto, 1)`.
- Se calcula el `required_driver_effect` que habría reconciliado el nivel SES con el actual y el `unexplained_driver_effect = required - current`; esto cuantifica información no capturada sin inventar ni refitear drivers.
- Añade `promotion_reaudit.parquet`: toda exclusión de drivers ya promovida se vuelve a probar mediante add-back del mismo RLS después de que lambda/dynamics hayan sido re-seleccionados; esto detecta interacciones de segundo orden sin revertir nada automáticamente.
- Nuevos artefactos residuales: `leaf_residual_extremes.parquet`, `leaf_residual_summary.parquet` y `residual_signal_summary.parquet`, con señales por tienda, fecha, weekday, mes y parent.
- `optimization_summary.txt` añade la sección 8 con stores/targets OOS prioritarios, concentración de error extremo y fechas con señal transversal; sirve para decidir si hace falta proponer un driver nuevo antes de incorporarlo.
- El encabezado de `optimization_summary.txt` deja de estar hardcodeado en v13.2.1 y usa `settings.APP_VERSION`.
- Para proteger RAM, el análisis residual usa OOS completo + 168 días recientes de historia cerrada y procesa Unidades/Valor secuencialmente; el modelo productivo sigue usando toda la historia. Los resúmenes se calculan sobre toda esa ventana y solo el detalle persistido se limita a 50,000 extremos.
- `APP_VERSION=13.2.3`; `ARTIFACT_VERSION=22` se mantiene.

## 13.2.2 — promociones holdout + interfaz operativa completa

- Promoción manual, misma familia RLS: `1||T:00211` excluye `month` en Unidades y Valor ($), ambos validados por historia cerrada y OOS holdout.
- Promoción manual, misma familia RLS: `23||T:00006` excluye `price` en Unidades, también validado por historia + holdout.
- Los candidatos vetados por OOS no se incorporan. No se crean drivers ni modelos nuevos.
- Phase 2 puede hacer add-back de un grupo excluido y refitear el mismo RLS para reauditar la promoción.
- `main.py`/`app.main` exponen desde una sola interfaz todos los procesos operativos: ingestión, selección, forecast por bloque/todos, optimización Phase 2, artefactos, validaciones, aceptación, pytest y dashboard.
- Los procesos de un bloque permiten elegir 1/7/14/28d y los forecasts conservan `--n-jobs`; los subprocesos se ejecutan siempre con la raíz del proyecto como `cwd`.
- En el ranking SKU+Tienda, `SKU descripción` pasa de ancho grande a medio y las columnas `Tienda`, `SKU`, `wMAPE` y `BIAS` se compactan para mantener visibles las métricas oficiales.
- `APP_VERSION=13.2.2`; `ARTIFACT_VERSION=22` se mantiene.

## 13.2.2 — corrección de auditoría de identidad SES+RLS

- `validate_v13` reconstruye ahora exactamente el forecast productivo con su soporte no negativo: `forecast_raw = max(expm1(log1p(nivel_SES) + efecto_RLS), 0)`.
- Corrige un falso ERROR de identidad cuando un efecto RLS negativo llevaba la transformación algebraica por debajo de cero mientras el kernel productivo almacenaba correctamente `0.0`.
- Se alinearon `docs/MODEL.md`, `docs/VALIDATION.md` y el Excel de auditoría generado por el dashboard con la misma identidad.
- No cambia forecasts, parámetros, drivers, métricas, artefactos ni `APP_VERSION`; los outputs v13.2.2 existentes siguen siendo válidos.

## 13.2.1 — validación holdout de candidatos SES+RLS

- No modifica la lógica productiva de v13.2.0: misma familia SES+RLS, mismos alpha/lambda productivos y mismas promociones manuales de drivers.
- Añade `driver_refit_validation.parquet`: un cambio de driver debe ser candidato por historia cerrada antes de mirar OOS.
- El OOS actual solo valida/veta; nunca crea candidatos ni participa en tuning.
- El reporte muestra los candidatos históricos completos con mejora OOS y cambio de |BIAS|.
- Corrige la presentación truncada de holdout SES/RLS: ahora se muestran los mejores candidatos por cada sección/target (y nivel en RLS), evitando que grupos posteriores desaparezcan del TXT.
- `APP_VERSION=13.2.1`; `ARTIFACT_VERSION=22` se mantiene.

# Changelog consolidado

## 13.2.0 — promoción manual Phase 3 dentro de SES+RLS

- Se revisó `optimization_summary.txt` de Phase 2; no existe promoción automática.
- SES: `alpha=0.30` pasa a la grilla productiva. Mostró mejora histórica y holdout consistente en Sección 23. Los alphas `0.0025`, `0.0075` y `0.50` permanecen diagnósticos.
- RLS: `lambda=0.990` y `0.9975` pasan a la grilla productiva para ampliar la selección causal por nodo; `lambda=1.0` permanece diagnóstico por evidencia mixta, especialmente en Sección 23.
- Valor ($): se habilitan los drivers de precio existentes (`asp`, `edp`, `discount`) únicamente para el nodo de **Sección 1** (`unique_id="1"`), donde el refit histórico mejoró wMAPE y |BIAS|. No se cambia el diseño de tiendas ni de Sección 23.
- El refit de drivers ahora se resume por `unique_id`; se corrige la ambigüedad que mezclaba nodos distintos bajo el mismo `section/node_level`.
- No se incorporan nuevos drivers ni familias de modelos. La selección sigue siendo histórica/causal; OOS actual sigue siendo holdout.
- `APP_VERSION=13.2.0`; `ARTIFACT_VERSION=22` permanece porque el esquema del dashboard no cambia.
- Polars continúa como único motor tabular; Pandas no se incorpora.


## 13.1.1 — optimización estadística Phase 2 (diagnóstico, sin promoción)

- Añade `--optimization-phase2`; la grilla ampliada y los refits son diagnósticos y no participan en la selección productiva.
- SES: amplía solo la grilla diagnóstica con alphas adicionales; la selección productiva sigue limitada a `LEAF_SES_ALPHA_CANDIDATES`.
- RLS: amplía solo la grilla diagnóstica de `lambda`; la selección productiva sigue limitada a `RLS_FORGETTING_FACTOR_CANDIDATES`.
- Drivers: ejecuta refit exacto del mismo RLS al remover `weekday`, `month`, `holiday_other` y `price` cuando ya están presentes.
- Valor ($): prueba causalmente la inclusión de los drivers de precio ya existentes (`asp`, `edp`, `discount`) sin promoverlos.
- AR: la remoción se compara contra el candidato `base` con la misma lambda.
- Corrige el soporte de tuning de Valor ($): la elegibilidad contractual es `y>0`, aunque `value==0`; esos puntos aportan error al numerador y cero al denominador.
- OOS actual continúa siendo holdout; no participa en la recomendación histórica.
- `APP_VERSION` sube a `13.1.1` porque se corrige la elegibilidad de tuning de Valor ($) y esa corrección puede cambiar la selección causal de alpha cuando existe `y>0` con `value==0`. `ARTIFACT_VERSION=22` se mantiene porque el esquema del dashboard no cambia.
- Polars sigue siendo el único motor tabular; no se incorpora Pandas.


## 13.1.0 — revisión de diagnóstico dashboard y trazabilidad

- Outliers visuales redefinidos como discrepancias materiales Actual vs Forecast, con regla simétrica basada en factor y escala robusta de la serie.
- SKU+Tienda incorpora EDP observado en segundo eje; forecast-only usa carry-forward del último EDP observado. Es diagnóstico y no altera el RLS productivo.
- La auditoría Excel SKU+Tienda detalla actual, EDP/ASP, nivel inicial, nivel SES, actual desestacionalizado, parent RLS, efecto/factor conjunto de drivers, reconstrucción del forecast, errores y contribuciones/cálculo acumulado de wMAPE/BIAS.
- No se modifica la arquitectura SES+RLS, la selección causal ni los parámetros productivos.
- Polars continúa como único motor tabular; no se incorpora Pandas.


## 13.1.0 — optimización estadística dentro de SES+RLS

- Reactiva la optimización estadística sin cambiar la arquitectura productiva.
- Añade diagnósticos opcionales `--optimization-diagnostics`; por defecto el forecast mantiene el mismo hot path.
- SES: registra desempeño por bloque de cada `alpha` candidato usando el mismo parent RLS y la misma recurrencia causal.
- RLS: registra desempeño por bloque de cada combinación existente `dynamics × lambda` a nivel Sección/Tienda.
- Drivers: añade screening de contribución de los grupos existentes (`weekday`, `month`, `price`, `holiday_other` y `autoregressive` cuando aplica) sin crear ni promover nuevos drivers.
- OOS actual se reporta como holdout final y no participa en la recomendación de tuning.
- No existe promoción automática de parámetros/drivers.
- Dashboard y `ARTIFACT_VERSION=22` se mantienen operativos sin cambios de lógica.
- Polars sigue siendo el único motor tabular; no se incorpora Pandas.


## 13.0.0 — corrección de contratos de validación
- Dashboard: las métricas diagnósticas que incluyen `y=0` se alinean ahora con las columnas oficiales: In-sample a la izquierda y OOS a la derecha, con BIAS debajo de su wMAPE correspondiente. Cambio exclusivamente visual; no modifica cálculos ni rankings.

- El contrato de documentación ahora valida únicamente los siete documentos canónicos (`README.md`, `CHANGELOG.md` y `docs/*.md` esperados); README operativos en `data/` u otras carpetas funcionales no se consideran documentación principal.
- El test de fallback EDP ahora parchea directamente `app.forecasting.pipeline.decompose_price`, que es la dependencia real usada por producción; se eliminó la fachada de compatibilidad basada en `sys.modules`.
- El contrato de documentación ignora entornos virtuales y artefactos de build al verificar los siete Markdown del proyecto.
- Sin cambios en la arquitectura RLS+SES, métricas, dashboard ni formato de artefactos.

## 13.0.0 — reset de arquitectura productiva

Cambio mayor motivado por una inconsistencia conceptual detectada en el forecast leaf: la trayectoria in-sample y las trayectorias OOS/forecast-only podían provenir de mecanismos diferentes, dificultando su explicación y auditoría.

### Arquitectura productiva

- Sección y Tienda continúan usando RLS expansivo con `rls_opt`.
- SKU+Tienda se simplifica a una única familia **SES + RLS parent**.
- Nivel inicial SES: **mediana de magnitudes positivas** en el warm-up inicial de 28 días calendario.
- SES recursivo sobre toda la historia disponible; los días `y=0` mantienen el estado de magnitud en lugar de reducirlo.
- Las observaciones positivas posteriores se ajustan inversamente por el efecto del parent RLS antes de actualizar el estado SES.
- Parent Sección/Tienda elegido exclusivamente por wMAPE oficial acumulado en bloques cerrados previos.
- In-sample, OOS y forecast-only usan el mismo kernel y la misma identidad algebraica.
- OOS y forecast-only permanecen en 28 días.
- Cadencias disponibles: 1/7/14/28d.

### Código retirado

Se eliminaron físicamente del árbol productivo módulos y ramas asociados a la arquitectura v12 que ya no forman parte del contrato:

- occurrence/share productivo;
- SKU-total ensembles;
- LightGBM shape;
- sparse rescue;
- meta-selectores y residual selectors;
- level-rescue / fast-selector diagnostics;
- backtests y validators legacy específicos de v11/v12;
- código muerto, flags y columnas productivas asociados.

### Dashboard

- `ARTIFACT_VERSION = 22`.
- soporte multi-bloque 1/7/14/28d;
- métricas oficiales y all-points precalculadas;
- ranking siempre basado en wMAPE oficial;
- auditoría Excel bajo demanda para SKU+Tienda;
- trazas v13: nivel inicial, SES, alpha, parent, wMAPE del parent, driver effect y bloque/origen.

### Métricas

Sin cambio contractual:

- wMAPE/BIAS oficiales excluyen `y=0`;
- all-points incluyen `y=0` solo como diagnóstico;
- selección y rankings usan exclusivamente la métrica oficial.

### Ingeniería

- procesamiento tabular con Polars;
- sin dependencia productiva de pandas;
- memory-safe spill por sección conservado;
- kernels numéricos Numba conservados para RLS/SES;
- documentación reducida a siete Markdown vigentes.

### Gobierno

La optimización estadística experimental permanece pausada. Esta versión prioriza coherencia, trazabilidad y cumplimiento del diseño original.

## Historia v12 — resumen

v12 introdujo y evaluó occurrence/share, SKU-total, LightGBM shape, sparse rescue y múltiples selectores/guards. Esos experimentos aportaron evidencia diagnóstica, pero la arquitectura acumuló capas que terminaron separando la narrativa histórica del forecast productivo. v13 no conserva esas capas como código productivo.

Los detalles históricos relevantes quedan resumidos en `docs/EXPERIMENTS.md`; no se mantienen documentos separados por parche.
