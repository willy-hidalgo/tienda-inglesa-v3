# Experimentos

## Exception Model Lab — v13.3.2

Estado: diagnóstico solamente. No hay modelos alternativos activos en producción.

Resultado de la evaluación actual:

- Candidatos estadísticos actuales: 2 leaf-target (`1||T:00001||S:599068`, Unidades y Valor).
- Ambos fueron vetados por el OOS independiente.
- Rutas activas productivas: 0.
- `last_positive_naive`, `positive_median_28`, `positive_ses_a20`, Croston/TSB/Hurdle no se activan productivamente.
- Los 5 casos extremos detectados son errores relativos en series dormantes; no son explosiones de forecast.

Tratamiento recomendado:

1. Mantener SES+RLS como modelo productivo único.
2. Usar `exception_lab.py` solo como benchmark offline.
3. No usar OOS para escoger modelos; solo para vetar candidatos históricos.
4. Separar `dormant/no_positive` de rankings de mejora, sin eliminarlos del forecast.
