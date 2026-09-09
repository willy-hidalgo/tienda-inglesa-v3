# PERFORMANCE v11.6

- Mantiene kernels Numba/BLAS-free y hot path de v11.3.1.
- `level_shape` queda deshabilitado en producción tras el A/B v11.5.
- `sku_yoy_seasonal` se calcula con group-bys Polars por SKU×bloque; no hay loop Python por hoja.
- El factor usa 13 bloques de 28 días (=364 días), shrinkage por cobertura y clip configurable.
- La selección del candidato usa el bloque de validación previo; OOS/forecast-only siguen causales.

- v11.6.1: el invariante estacional usa `parent_factor × sku_multiplier`; sin costo material adicional.
- Se eliminan columnas temporales `_fb_mult_*` del resultado final.
