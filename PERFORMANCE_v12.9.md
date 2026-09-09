# PERFORMANCE v12.9.0

La intervención añade un quinto bloque histórico de scoring para habilitar tres folds walk-forward de Valor. Por ello la corrida 28d puede ser algo más lenta que v12.8.6, principalmente por un candidato histórico adicional y por el entrenamiento/evaluación de meta-modelos temporales pequeños.

No cambia la arquitectura memory-safe de v12.8.6. Los escenarios multi-cadencia siguen particionados por `block_XXd` y los artefactos se generan por escenario.

Para la primera validación estadística se recomienda ejecutar únicamente 28d. No usar `--skip-existing` para reutilizar un forecast v12.8.6: v12.9.0 detecta la APP_VERSION anterior como stale y recalcula el modelo.
