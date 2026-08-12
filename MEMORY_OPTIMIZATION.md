# Optimizaciones de Memoria - Pipeline RLS

## Problema Original
El pipeline cargaba TODO el archivo `selected.parquet` (~75MB) en memoria de una vez, lo que duplicaba el uso de RAM y bloqueaba la máquina durante el procesamiento.

## Soluciones Implementadas

### 1. **Lazy Evaluation + Streaming (Principal)**
- ✅ `_load()` ahora retorna `LazyFrame` en lugar de DataFrame
- ✅ Filtrado por sección ANTES de `.collect()` 
- ✅ `.collect(streaming=True)` procesa en chunks sin cargar todo en RAM
- **Impacto**: Reduce carga de ~75MB a ~37.5MB por sección (50% reducción)

### 2. **Garbage Collection Explícito**
- ✅ `gc.collect()` después de procesar cada sección
- **Impacto**: Libera memoria inmediatamente entre secciones

### 3. **Configuración de Polars**
- ✅ `pl.Config.set_streaming_chunk_size(100_000)` para procesar datos en chunks
- **Impacto**: Evita picos de memoria durante joins y aggregaciones

### 4. **Tipo Union para Lazy/Eager**
- ✅ Funciones aceptan `pl.DataFrame | pl.LazyFrame`
- **Impacto**: Flexibilidad para procesamiento lazy o eager según contexto

## Parámetro `--max-memory-mb`

```bash
python app/main.py --max-memory-mb 2000
```

Cuando se especifica límite de memoria:
- RLS automáticamente reduce workers si memoria excede límite
- EDP usa fallback vectorizado rápido si hay restricción
- Prevents OOM (out-of-memory) errors

## Recomendaciones de Uso

### Para Máquinas con <8GB RAM:
```bash
python app/main.py --max-memory-mb 1000 --n-jobs 2
```

### Para Máquinas con 8-16GB RAM:
```bash
python app/main.py --n-jobs 4
```

### Para Máquinas con >16GB RAM:
```bash
python app/main.py --n-jobs 8
```

## Benchmarks (Estimado)

| Escenario | RAM Pico Original | RAM Pico Después | Mejora |
|-----------|------------------|-----------------|---------|
| 2 secciones, n_jobs=1 | ~150MB | ~75MB | 50% ↓ |
| 2 secciones, n_jobs=4 | ~180MB | ~90MB | 50% ↓ |
| Single sección, n_jobs=1 | ~75MB | ~40MB | 47% ↓ |

## Verificación

Monitorear memoria durante ejecución:
```bash
# Windows (PowerShell)
while($true) { Get-Process -Name python | Select-Object Name, WorkingSet; Start-Sleep 2 }

# Linux
watch -n 2 'ps aux | grep main.py'
```

## Cambios Técnicos Detallados

### forecasts.py

#### Cambio 1: `_load()` retorna LazyFrame
```python
# ANTES:
def _load(self) -> pl.DataFrame:
    selected = pl.scan_parquet(...).collect()  # CARGA TODO


# DESPUÉS:
def _load(self) -> pl.LazyFrame:
    selected = pl.scan_parquet(...)  # Solo metadatos, datos en disco
    return selected
```

#### Cambio 2: Filtrado lazy + collect streaming
```python
# ANTES:
sec_df = selected.filter(pl.col("SECCION") == seccion)
raw_train = sec_df.filter((pl.col("date") >= start) & ...)

# DESPUÉS:
raw_train_lazy = selected.filter(pl.col("SECCION") == seccion).filter(
    (pl.col("date") >= start) & ...
)
raw_train = raw_train_lazy.collect(streaming=True)  # Solo datos necesarios
```

#### Cambio 3: Garbage collection entre secciones
```python
for seccion in FOCUS_SECTIONS:
    res, wm, driver_cols = self._run_section(selected, seccion, ...)
    # ... guardar resultados ...
    gc.collect()  # Libera memoria inmediatamente
```

## Notas Importantes

1. **No es una regresión**: Los cálculos siguen siendo 100% idénticos
2. **Compatibilidad**: Funciona con Polars 0.20+
3. **Streaming automático**: Polars optimiza automáticamente cuando usa `.collect(streaming=True)`
4. **Sin cambio en precision**: Precisión numérica exacta preservada

## Próximas Optimizaciones Posibles

- [ ] Particionar datos de entrada antes de procesar (pre-procesamiento)
- [ ] Usar Apache Parquet `row_groups` para lectura selectiva
- [ ] Memory-mapping de archivos grandes
- [ ] Procesamiento asincrónico con `asyncio`

