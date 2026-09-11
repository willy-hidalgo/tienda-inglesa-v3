# Arquitectura v13

## Objetivo de diseño

Una única historia estadística desde el pasado hasta el futuro:

```text
Datos históricos
    │
    ├── RLS Sección ──────┐
    │                     ├── elegir parent por wMAPE histórico causal
    ├── RLS Tienda ───────┘                 │
    │                                       ▼
    └── SKU+Tienda ── mediana inicial → SES de magnitud → + efecto RLS relativo
                                                     (forecast parent / nivel causal + guard)
                                                    │
                                                    ▼
                                  in-sample / OOS / forecast-only
                                  MISMA familia, distinto origen
```

## Pipeline

1. Ingesta y selección de secciones.
2. Resolución de horizontes.
3. Construcción de calendario y drivers.
4. RLS expansivo de Sección.
5. RLS expansivo de Tiendas.
6. Construcción leaf SKU+Tienda:
   - warm-up robusto;
   - selección causal de parent;
   - selección causal de alpha SES;
   - generación walk-forward con un único kernel.
7. Métricas y exportaciones.
8. Escritura de `forecast.parquet`.
9. Construcción automática de artefactos rápidos del dashboard.

## Separación de responsabilidades

### RLS Sección/Tienda

Modela efectos agregados de calendario/comerciales y dinámica temporal. El engine usa `rls_opt`, con historia expansiva y selección de especificación sobre bloques previos cerrados.

### SES SKU+Tienda

Modela la magnitud propia de la hoja. El estado inicial es robusto a ceros y picos; después recorre toda la historia disponible.

### Parent RLS

Aporta la dinámica agregada. No sustituye el nivel leaf. Se elige entre Tienda y Sección según el wMAPE oficial histórico disponible antes del bloque objetivo. La hoja no transfiere la separación intercepto/no-intercepto de coeficientes. Usa el forecast RLS del parent relativo a la mediana positiva causal del mismo parent en los 28 días cerrados previos y lo acota a un factor `[0.50, 2.00]`. La contribución no-intercepto original se conserva solo como traza de auditoría; una explosión de coeficientes no puede redefinir el nivel SES.

## Misma familia temporal

La fase temporal cambia, no el modelo:

- `in_sample`: walk-forward histórico;
- `out_sample`: 28 días reservados para evaluación;
- `forecast_only`: 28 días posteriores sin actuals.

Cada bloque usa el estado congelado en su origen. Los actuals dentro de un bloque no alteran el forecast de ese mismo bloque; solo pueden actualizar el estado para bloques posteriores.

## Multi-bloque

La cadencia de actualización puede ser 1/7/14/28 días. Cada escenario es autocontenido en:

```text
data/output/update_blocks/block_XXd/
```

OOS y forecast-only siguen siendo 28 días para mantener comparabilidad.

## Performance y memoria

- Polars para operaciones tabulares.
- Numba para kernels numéricos RLS/SES.
- procesamiento por sección;
- spill temporal por sección cuando corresponde;
- predicate/projection pushdown en artefactos y auditorías;
- dashboard sin entrenamiento ni agregaciones pesadas en el hot path.

## Dependencias retiradas de la arquitectura

No existe una segunda ruta productiva basada en occurrence/share, LightGBM, ensembles SKU-total, sparse rescue o meta-selectores. Esto reduce costo, RAM, superficie de errores y complejidad explicativa.
## Diagnósticos de optimización

Con `--optimization-diagnostics`, el pipeline conserva el forecast productivo y escribe sidecars compactos en `statistical_optimization/`. Los diagnósticos se recolectan mientras SES/RLS ya están evaluando sus candidatos, evitando una segunda familia de modelos y limitando el costo adicional. El dashboard no lee estos archivos y su performance interactiva no cambia.



## v13.3.3: transición YoY leaf

La producción sigue una sola arquitectura. La hoja combina el nivel SES, el movimiento relativo del parent RLS y un factor YoY causal/acotado de la propia hoja. No existe routing a otra familia de modelos.
