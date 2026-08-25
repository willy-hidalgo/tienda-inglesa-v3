# VALIDATION — Tienda Inglesa v9.2

## Release

- APP_VERSION: 9.2
- pyproject: 9.2.0
- dashboard ARTIFACT_VERSION: 7
- Python objetivo: >=3.13,<3.14

## Corrección principal

v9.2 evita que el último bloque de 28 días, potencialmente contaminado por un shock, participe en la definición de su propia referencia estructural.

La mediana estructural usa hasta 6 bloques anteriores, empezando en `block -2`.

Se agregan alphas SES de memoria larga `0.0002` y `0.001`.

El nivel final sigue siendo siempre una trayectoria SES pura existente.

## Regresiones con datos reales incluidos

### 587833 @ 00001

- último bloque train ≈ 1430.25
- mediana estructural previa ≈ 259.7
- shock alcista: sí
- candidato SES alpha 0.0002 ≈ 324.3
- actual OOS medio observado ≈ 301.3

### 299993 @ 00001

- último bloque train ≈ 3338.19
- mediana estructural previa ≈ 290.7
- shock alcista: sí
- candidato SES alpha 0.0002 ≈ 279.8
- actual OOS medio observado ≈ 430.8

### Casos de control

`483046` y `58905` no superan el umbral de shock alcista respecto de su régimen estructural, por lo que esta rama no debería alterar su selección normal.

## Validación requerida en entorno productivo

```powershell
pytest -q
python app\forecasts.py --n-jobs 8
python -m app.dashboard_artifacts
python -m app.dashboard_consistency
streamlit run app\dashboard.py
```

Diagnóstico prioritario:

```powershell
python -m app.forecasting.diagnose_leaf --uid "1||T:00001||S:587833"
```

Revisar:

```text
ses_level_value
ses_structural_block_median_value
ses_upward_shock_value
ses_alpha_value
driver_strength_value
driver_factor_value
valuehat
```

Para `587833 @ 00001` se espera que el nivel SES en el origen OOS quede mucho más próximo al régimen estructural (~260–330) que al bloque de shock (~1430).

## Limitación del runtime de entrega

Este entorno no contiene Polars y no tiene acceso de red para instalarlo. Los tests que importan Polars deben ejecutarse en el entorno productivo del proyecto y no se contabilizan como aprobados aquí.

## Quality gate ejecutado antes de empaquetar

```text
61 archivos Python
0 errores de sintaxis
0 patrones Polars with_columns/filter de riesgo
147 tests de contrato/regresión ejecutables
147 passed
9 módulos funcionales Polars diferidos al entorno productivo
```
