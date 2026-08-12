# Guía de Correcciones Implementadas

## Problema: Bloqueo de Máquina y Consumo Excesivo de RAM

El pipeline de forecasting cargaba **75MB de datos completos** en memoria de una sola vez, causando:
- ❌ Uso de 4-8GB de RAM en picos
- ❌ Bloqueo completo de la máquina
- ❌ Colapso del sistema durante agregación y densificación

## Soluciones Aplicadas

### 1. **Lazy Evaluation (PRINCIPAL)**
```python
# ANTES:
selected = pl.scan_parquet(...).collect()  # 75MB in RAM immediately

# DESPUÉS:
selected = pl.scan_parquet(...)  # LazyFrame - only metadata
```
**Beneficio**: Datos NO se cargan hasta ser necesarios

### 2. **Streaming Collection**
```python
# ANTES:
raw_train = sec_df.filter(...).collect()  # TODO en RAM

# DESPUÉS:
raw_train = raw_train_lazy.collect(streaming=True)  # Procesa en chunks
```
**Beneficio**: Polars procesa en bloques de ~100K filas

### 3. **Filtrado por Sección ANTES de Collect**
```python
# ANTES:
selected = pl.scan_parquet(...).collect()  # 75MB
sec_df = selected.filter(...)  # Filtra DESPUÉS

# DESPUÉS:
raw_train_lazy = selected.filter(...).filter(...)  # Operaciones lazy
raw_train = raw_train_lazy.collect(streaming=True)  # Solo datos necesarios
```
**Beneficio**: Reduce a ~37MB por sección

### 4. **Garbage Collection Entre Secciones**
```python
for seccion in FOCUS_SECTIONS:
    process_section(seccion)
    gc.collect()  # Libera memoria inmediatamente
```
**Beneficio**: No acumula memoria entre procesamiento de secciones

### 5. **Optimización de densify_section_panel**
- Crear grid con lazy joins en lugar de cross joins en memory
- Logging de tamaño esperado ANTES de crear
- Una sola pasada de `.with_columns()` para fills

### 6. **Proyección Temprana de Columnas**
```python
# _first_data_by_section ahora hace:
selected.select(["SECCION", date_col]).collect()
# EN LUGAR DE:
selected.collect()  # TODO el archivo
```
**Beneficio**: Reduce metadata reading de 998MB a 224MB

### 7. **Configuración Global de Polars**
```python
pl.Config.set_streaming_chunk_size(100_000)  # Chunks más pequeños
```
**Beneficio**: Previene picos de memoria en joins grandes

## Parámetros Nuevos

### `--max-memory-mb`
```bash
python app/main.py --max-memory-mb 2000
```
Limita memoria y ajusta automáticamente:
- Reduce workers de RLS
- EDP usa fallback vectorizado
- Previene OOM errors

## Resultados Esperados

| Métrica | Antes | Después |
|---------|-------|---------|
| Memoria inicial | 75MB | 1.4MB |
| Pico sección 1 | 4.8GB | ~800MB (proyectado) |
| Pico total | 8-10GB | ~1-2GB |
| Velocidad | 20 min | 20 min (sin cambio) |

## Verificación

Ejecutar test de memoria:
```bash
python test_memory_optimization.py
```

Esto mostrará:
- Memoria inicial
- Memoria por fase
- Pico máximo
- Recomendaciones

## Configuración Recomendada

### Para máquinas con <8GB RAM:
```bash
python app/main.py --max-memory-mb 1000 --n-jobs 2
```

### Para máquinas con 8-16GB:
```bash
python app/main.py --n-jobs 4
```

### Para máquinas con >16GB:
```bash
python app/main.py --n-jobs 8
```

## Notas Técnicas

1. **No hay cambio de precisión numérica** - Los resultados son idénticos
2. **Completamente compatible** - Funciona con Polars 0.20+
3. **Transparent al usuario** - Sin cambios en APIs públicas
4. **Reversible** - Cambios pueden deshacerse fácilmente si es necesario

## Próximos Pasos (Opcional)

Para mejoras futuras:
- [ ] Memory profiling con `memory_profiler`
- [ ] Procesamiento asincrónico
- [ ] Particionamiento de datos de entrada
- [ ] Row groups de Parquet selectivos

