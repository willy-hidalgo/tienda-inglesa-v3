# Performance v12.7.0

v12.7 evita introducir un nuevo modelo pesado. La calibración SKU-total usa
estadísticas robustas de pocos bloques cerrados y el selector añade solo agregados
de error/BIAS; el costo esperado es pequeño frente a LightGBM y al procesamiento
leaf existente.

El dashboard rápido aumenta `ARTIFACT_VERSION` de 14 a 15 porque conserva columnas
de candidatos v11/v12 y metadatos de calibración/selección. Esto incrementa el
tamaño de las series del dashboard, pero evita releer el forecast completo para
comparativas. Si los artefactos no están sincronizados, el dashboard cae a modo
legacy en vez de bloquear la visualización.

Los tiempos productivos deben medirse en la próxima corrida real; este documento
no atribuye una mejora de rendimiento sin medición.
