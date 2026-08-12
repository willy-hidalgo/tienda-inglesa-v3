# Issue: Diferenciación de Suavización Exponencial entre Cantidad y Precio

## 🔴 Problema Identificado

El código actualmente:

### ✅ HACE BIEN:
```python
# Modelo RLS para cantidad (log1p(y))
model_y = RecursiveLeastSquaresRegression(...)
model_y.fit(x=X_y, y=log_y, priors=priors_y)
coef_y = model_y.final_coef_

# Modelo RLS para precio (log1p(value))  
model_p = RecursiveLeastSquaresRegression(...)
model_p.fit(x=X_p, y=log_price, priors=priors_p)
coef_p = model_p.final_coef_

# Efectos separados:
effect_y = drivers · coef_y  # Efecto en CANTIDAD
effect_v = drivers · coef_p  # Efecto en PRECIO (diferente!)
```

### ❌ PROBLEMA:
```python
# En compute_derived_forecasts():
# Ambos usan el MISMO alpha (0.1)

sub = self._apply_causal_ses(
    sub, "_y_neto", "_y_neto_hat", alpha=settings.SES_ALPHA
)  # ← 0.1
sub = self._apply_causal_ses(
    sub, "_v_neto", "_v_neto_hat", alpha=settings.SES_ALPHA
)  # ← 0.1 (MISMO!)
```

## 📊 Por Qué es Problemático

### Escala y Volatilidad Diferentes

| Variable | Magnitud Típica | Volatilidad | Comportamiento |
|----------|-----------------|-------------|-----------------|
| Cantidad (y) | 1-1000 unidades | Moderada | Cambios discretos |
| Precio (value) | $100-10,000 | Baja-Moderada | Cambios suave/porcentual |
| Residuo (y_neto) | -500 a +500 | Pequeña escala | Requiere α sensible |
| Residuo (v_neto) | -$5,000 a +$5,000 | Gran escala | Requiere α menos sensible |

**Impacto:**
- `alpha=0.1` en _y_neto: Cada día nuevo pesa 10% → **Muy reactivo**
- `alpha=0.1` en _v_neto: Cada día nuevo pesa 10% → **Insuficientemente reactivo**

### Ejemplo Concreto

**Escenario 1: Día con promoción**

```
Cantidad:
  Actual:    100 unidades
  Esperado:  50 unidades (por driver)
  y_neto:    50 residual
  SES (α=0.1): 50 → pesa 10% en next day
  Efecto:    ✓ Captura bien la anomalía

Precio:
  Actual:    $5,000
  Esperado:  $4,500 (por driver)
  v_neto:    $500 residual
  SES (α=0.1): $500 → pesa 10% en next day
  Efecto:    ✗ Sub-captura la anomalía de precio
```

## ✅ Solución Propuesta

### Opción 1: Alphas Diferenciados (RECOMENDADO)

```python
# En settings.py:
SES_ALPHA_QUANTITY = 0.15  # Más reactivo (escala pequeña)
SES_ALPHA_VALUE = 0.05  # Menos reactivo (escala grande)

# En forecasts.py - compute_derived_forecasts():
sub = self._apply_causal_ses(
    sub, "_y_neto", "_y_neto_hat", alpha=settings.SES_ALPHA_QUANTITY
)
sub = self._apply_causal_ses(
    sub, "_v_neto", "_v_neto_hat", alpha=settings.SES_ALPHA_VALUE
)
```

**Ventajas:**
- ✓ Modelado más realista de dinámicas diferentes
- ✓ Fácil de tunar y documentar
- ✓ Respeta magnitudes diferentes de variables

### Opción 2: Alphas Auto-Sintonizados (AVANZADO)

```python
def compute_derived_forecasts(self, panel, coefs, alpha_config=None):
    """
    Calcula alpha automático basado en volatilidad de residuos.
    """
    alpha_config = alpha_config or {}
    
    for seccion, (coef_y_full, coef_p_full) in coefs.items():
        # ... cálculo de efectos ...
        
        # Auto-sintonizar alpha_y
        if "_y_neto" in sub.columns and "alpha_quantity" not in alpha_config:
            vol_y = sub["_y_neto"].std()
            alpha_y = min(0.2, vol_y / (vol_y + 100))  # Proporción a volatilidad
        else:
            alpha_y = alpha_config.get("alpha_quantity", 0.1)
        
        # Auto-sintonizar alpha_v
        if "_v_neto" in sub.columns and "alpha_value" not in alpha_config:
            vol_v = sub["_v_neto"].std()
            alpha_v = min(0.1, vol_v / (vol_v + 1000))
        else:
            alpha_v = alpha_config.get("alpha_value", 0.1)
        
        sub = self._apply_causal_ses(sub, "_y_neto", "_y_neto_hat", alpha=alpha_y)
        sub = self._apply_causal_ses(sub, "_v_neto", "_v_neto_hat", alpha=alpha_v)
```

**Ventajas:**
- ✓ Se adapta automáticamente a cada serie
- ✓ Más sofisticado y flexible
- ✗ Más complejo de debuggear

### Opción 3: Normalizar antes de SES (ALTERNATIVA)

```python
# Antes de SES, normalizar a media=0, std=1
y_neto_norm = (sub["_y_neto"] - sub["_y_neto"].mean()) / sub["_y_neto"].std()
v_neto_norm = (sub["_v_neto"] - sub["_v_neto"].mean()) / sub["_v_neto"].std()

# Aplicar SES a datos normalizados
y_neto_hat_norm = ses(y_neto_norm, alpha=0.1)
v_neto_hat_norm = ses(v_neto_norm, alpha=0.1)

# Desnormalizar
y_neto_hat = y_neto_hat_norm * sub["_y_neto"].std() + sub["_y_neto"].mean()
v_neto_hat = v_neto_hat_norm * sub["_v_neto"].std() + sub["_v_neto"].mean()
```

**Ventajas:**
- ✓ Un solo alpha, pero aplicado a escala normalizada
- ✓ Robusto a cambios de magnitud
- ✗ Requiere manejo de NaN en std

## 📋 Pasos de Implementación

### Paso 1: Agregar parámetros a settings.py

```python
# Suavización exponencial: alphas separados por variable
SES_ALPHA_QUANTITY = 0.15  # α para cantidad/unidades
SES_ALPHA_VALUE = 0.05  # α para precio/valor
# Si ambos son None, usa Opción 3 (auto-normalización)
SES_NORMALIZE_BEFORE_SMOOTHING = True
```

### Paso 2: Modificar signature de compute_derived_forecasts

```python
def compute_derived_forecasts(
    self,
    panel: pl.DataFrame,
    coefs: dict[str, tuple[np.ndarray, np.ndarray]],
    alpha_quantity: float | None = None,
    alpha_value: float | None = None,
) -> pl.DataFrame:
```

### Paso 3: Actualizar llamada en _run_section

```python
res_derived = runner.compute_derived_forecasts(
    panel_all,
    coefs,
    alpha_quantity=settings.SES_ALPHA_QUANTITY,
    alpha_value=settings.SES_ALPHA_VALUE,
)
```

### Paso 4: Documentar en README

Agregar sección explicando por qué alphas son diferentes y cómo afecta predicciones.

## 🧪 Validación

Para verificar si el cambio es beneficioso:

```python
# Test: Comparar métrica WMAPE antes/después
# Con alpha_quantity=0.15, alpha_value=0.05

wmape_actual_mejor = wmape_actual < wmape_antes
bias_actual_mejor = abs(bias_actual) < abs(bias_antes)

# Debería ver:
# - WMAPE en cantidad: mejora o similar
# - WMAPE en precio: mejora
# - Bias reducido en precio especialmente
```

## 🎯 Recomendación

**Implementar Opción 1 (Alphas Diferenciados):**

1. ✅ Es simple y claro
2. ✅ Respeta completamente la diferencia entre variables
3. ✅ Fácil de documentar y tunar
4. ✅ Sin complejidad computacional extra

Propone:
- `SES_ALPHA_QUANTITY = 0.15` (para cantidad)
- `SES_ALPHA_VALUE = 0.05` (para precio)

Estos valores se pueden ajustar después con data real.

## 🔗 Código Relacionado

- [app/forecasts.py#827](app/forecasts.py#L827) - `_apply_causal_ses()`
- [app/forecasts.py#839](app/forecasts.py#L839) - `compute_derived_forecasts()`
- [settings.py#57](settings.py#L57) - `SES_ALPHA`

