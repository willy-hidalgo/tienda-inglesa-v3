# Rendimiento v11.4.0

v11.4 no revierte las optimizaciones de v11.3.1:
- kernel RLS Numba sin SciPy/BLAS obligatorio;
- bias correction solo en padres;
- fallbacks sobre frame slim;
- dashboard con proyección de columnas;
- checkpoints por sección desactivados por defecto.

El cambio de fallback sigue siendo vectorizado. La nueva lógica opera solo sobre observaciones positivas y, por tanto, procesa menos filas que la historia calendario completa.

Referencia v11.3.1 en la muestra del cliente:
- Pipeline RLS: 64.9 s.
- Dashboard artifacts: 24.5 s.
- Corrida completa: ~1m44s.

La aceptación v11.4 debe conservar el mismo orden de magnitud.
