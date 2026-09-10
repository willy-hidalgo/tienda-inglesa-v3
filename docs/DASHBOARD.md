# Dashboard v13

## Propósito

El dashboard es una capa de consumo y auditoría. **No entrena modelos** ni recalcula métricas pesadas al navegar.

## Inicio

```bash
uv run streamlit run app/dashboard.py
```

## Preparar todos los escenarios

La forma recomendada para tener el selector multi-bloque completamente operativo es:

```bash
uv run python app/forecasts.py --all-update-blocks --n-jobs 8
```

Cada escenario genera sus artefactos al finalizar.

Si los forecasts ya existen:

```bash
uv run python -m app.dashboard_artifacts --all-update-blocks
```

## Escenarios

El sidebar detecta automáticamente los forecasts existentes para:

- 1d
- 7d
- 14d
- 28d

El cambio de escenario solo cambia el archivo/artefacto precalculado; no dispara entrenamiento.

## Métricas visibles

### Oficiales

- wMAPE
- BIAS

Excluyen `y=0` y son la base del ranking.

### Prueba ácida

Toggle `Mostrar métricas incluyendo y=0`:

- wMAPE incl. y=0
- BIAS incl. y=0

El toggle cambia únicamente visibilidad. Nunca cambia el orden del ranking.

En el ranking **SKU+Tienda**, las columnas `Tienda`, `SKU`, `wMAPE` y `BIAS` se mantienen compactas y `SKU descripción` usa ancho medio. El objetivo es que las métricas oficiales permanezcan visibles en pantallas de escritorio sin que una descripción larga desplace las columnas de error.

## Gráficos

La serie leaf muestra:

- Actual in-sample
- Forecast in-sample
- Actual OOS
- Forecast OOS
- Forecast-only

En v13 las tres trayectorias de forecast provienen de la misma familia SES+RLS. Esto permite interpretar continuidad entre historia y futuro.

## Auditoría Excel SKU+Tienda

En el detalle de una hoja se puede preparar y descargar un Excel bajo demanda.

Incluye:

- resumen de la serie;
- resumen por bloque;
- detalle diario;
- fórmulas y definiciones.

Trazas disponibles:

- nivel inicial;
- nivel SES;
- alpha SES;
- parent elegido para Unidades/Valor;
- wMAPE histórico del parent;
- efecto/factor RLS;
- bloque y días de entrenamiento;
- elegibilidad de métrica.

La auditoría lee solo una `unique_id` mediante predicate/projection pushdown. No amplía el hot path del dashboard.

## Artefactos

`ARTIFACT_VERSION = 24`.

Cada directorio `dashboard/` contiene:

- `index.json`
- `labels.parquet`
- `metrics.parquet`
- `metrics_in_sample.parquet`
- `series/seccion=<s>/data.parquet`

Los artefactos incluyen fingerprint del forecast de origen; si quedan stale, el dashboard no los sirve como válidos.

## Validación

```bash
uv run python -m app.dashboard_consistency --all-update-blocks
```

La auditoría verifica versión, fingerprint, unicidad, horizonte y consistencia bottom-up de métricas.
## Relación con la optimización estadística

Los diagnósticos de tuning no se calculan ni se cargan dentro de Streamlit. El dashboard continúa usando artefactos precalculados v22 y conserva exactamente la misma lógica de métricas/rankings. Por tanto, activar `--optimization-diagnostics` en una corrida de forecast no añade costo a la navegación del dashboard.


## Discrepancias Actual vs Forecast y EDP

El toggle de outliers no busca extremos de cada serie por separado. Marca **discrepancias entre Actual y Forecast** con una regla simétrica: `factor_gap >= 4` y error absoluto al menos 15% del nivel típico de la serie, o error absoluto al menos 3 veces ese nivel. El nivel típico es la mediana positiva de actuals; se usa un floor de 5% para evitar que valores muy pequeños generen ratios artificiales.

En SKU+Tienda se muestra EDP en un segundo eje. El tramo histórico/OOS es una descomposición ex-post del precio observado de esa hoja y forecast-only usa carry-forward del último EDP observado. Es diagnóstico; no significa que el modelo leaf use EDP propio como driver: el forecast sigue usando el efecto RLS del parent Sección/Tienda.

Para agregaciones se recomienda: (1) SKU: EDP ponderado por unidades entre tiendas; (2) Tienda/Sección: índice EDP de mix fijo base 100, con ponderaciones de unidades de un período base, para aislar movimiento de precio de cambios de mix.
