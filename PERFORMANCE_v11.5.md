# Rendimiento v11.5.0

v11.5 conserva las optimizaciones de v11.3.1/v11.4: Numba RLS, fallback slim vectorizado, bias correction solo padres y dashboard proyectado.

El nuevo `level_shape` no ajusta RLS adicionales. Reutiliza los pronósticos padre ya calculados y añade group-by/join pequeños sobre 9 series padre por sección. El costo adicional esperado está en el scoring leaf de 4 parent-modes en vez de 2 parent candidates; debe mantenerse en el mismo orden de magnitud que v11.4.

Baseline observado v11.4.0: Pipeline RLS 78.6s en la muestra del cliente. La aceptación de rendimiento de v11.5 no debe volver a tiempos de decenas de minutos.
