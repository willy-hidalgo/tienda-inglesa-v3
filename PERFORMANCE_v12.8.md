# Performance v12.8.6 — Memory-safe multi-cadencia

## Estrategia de memoria

El cuello de botella de 1d no era el tamaño final del forecast, sino la coexistencia en RAM de cientos de estados leaf por origen, dos secciones completas y una segunda lectura del forecast para dashboard artifacts. v12.8.6 elimina esos tres picos:

1. `chosen_state` por origen → parquet temporal en disco; join final lazy/streaming.
2. Resultado de cada sección → `_section_spill`; se libera antes de la siguiente sección.
3. `forecast.parquet` se guarda, se liberan DataFrames y recién entonces se construyen artifacts.
4. Paralelismo adaptativo limita copias simultáneas: 1/2/4/8 workers para 1/7/14/28d.

El costo esperado es mayor I/O y algo más de tiempo para 1d/7d, a cambio de mantener el pico de RAM mucho más acotado. Los spills se eliminan al consolidar correctamente.

---

# Performance histórica v12.8.4 — Demo multi-cadencia

## Objetivo de rendimiento

La UI no recalcula modelos ni KPI. Cada cadencia tiene `forecast.parquet` y artefactos slim propios; `metrics.parquet` sigue siendo la fuente oficial de wMAPE/BIAS. El cambio de selector solo invalida caches por path/mtime y carga la serie seleccionada.

El costo de cómputo se paga una sola vez al ejecutar `--all-update-blocks`; el costo de disco crece aproximadamente linealmente con cuatro escenarios, mientras labels/index son marginales frente a forecasts/series.


---

# Performance v12.8.3

## Intervención v12.8.3

Cambio limitado a **Value Portfolio Safety**. Unidades conserva exactamente la policy v12.8.2. Para Valor, `v12_all` requiere dominancia wMAPE/BIAS, 2/2 confirmaciones recientes y cobertura causal del guard BIAS; `meta_leaf` debe superar al mejor all-mode seguro por margen mínimo. Forecast generators, LightGBM shape, occurrence y shares quedan congelados.


## Baseline productivo observado en v12.8.0

- Sec 1 Unidades: final 58.62%, oracle 52.57%, gap 6.05 pp.
- Sec 23 Unidades: final 60.13%, oracle 55.57%, gap 4.56 pp.
- Sec 1 Valor: final 58.15%, oracle 52.60%, gap 5.55 pp.
- Sec 23 Valor: final 60.75%, oracle 56.56%, gap 4.18 pp.

El meta-selector v12.8 ya mostró señal real en las hojas seleccionadas, pero siguió siendo conservador. El caso crítico fue Sec.23/Unidades: `selected=0/4290` aunque `v12_all=59.36%` superó a `v11_all=60.13%`; el legacy portfolio gate quedó `False` y anuló el meta-modelo.

## Intervención v12.8.2

No se modifica ningún forecast generator. Solo se cambia la policy de selección:

- threshold adaptativo por sección × target;
- elección causal entre `v11_all`, `v12_all` y `meta_leaf`;
- legacy gate pasa a diagnóstico, no veto;
- guard de BIAS leaf y portfolio de 3 pp;
- utilidad de selección = `wMAPE + 0.25*|BIAS|`;
- calibración SKU-total sigue apagada;
- LightGBM shape, occurrence y share quedan congelados.

## Objetivo

Capturar una fracción mayor del oracle gap de 4.2–6.1 pp sin aceptar mejoras de wMAPE que deterioren materialmente el BIAS. El objetivo exploratorio sigue siendo capturar al menos 20–30% del gap observable (~1–2 pp donde exista señal temporal estable).

## Coste esperado

Por sección/target se añade un segundo fit logístico pequeño para calibrar la policy en el bloque cerrado 1. El coste es marginal frente al LightGBM pooled y al pipeline leaf.

## v12.9.0

- Qty congelado con policy v12.8.6 (4 bloques).
- Value Portfolio Selector walk-forward con 5 bloques / 3 folds pseudo-OOS.
- Guardias: win-rate, mediana gain, worst-fold, |BIAS| y utilidad bottom-up ponderada por volumen.
- `ARTIFACT_VERSION=20`; acceptance añade sección 23.
