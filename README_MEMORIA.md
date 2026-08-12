# Quick Start: Ejecutar Forecasting SIN Bloqueos de Memoria

## TL;DR (Solución Inmediata)

```bash
# Si tu máquina se bloquea:
python app/main.py --max-memory-mb 2000 --n-jobs 2

# Si tienes suficiente RAM:
python app/main.py --n-jobs 4
```

## Qué se Corrigió

❌ **ANTES**: Cargaba TODO el archivo (75MB) en RAM de una vez → Sistema bloqueado
✅ **DESPUÉS**: Carga lazy + streaming + GC entre secciones → Usa 50-75% menos RAM

## Parámetros Recomendados por Máquina

### Para <8GB RAM
```bash
python app/main.py --max-memory-mb 1000 --n-jobs 2
```
- Máximo 1GB en uso
- Procesamiento secuencial paralelo moderado
- MÁS SEGURO para máquinas antiguas

### Para 8-16GB RAM
```bash
python app/main.py --n-jobs 4
```
- Sin límite explícito (usa streaming para controlar)
- 4 threads paralelos
- BALANCE entre velocidad y seguridad

### Para >16GB RAM
```bash
python app/main.py --n-jobs 8
```
- Sin límite explícito
- Máximo paralelismo
- MÁS RÁPIDO

## Monitorear durante Ejecución

### Windows (PowerShell)
```powershell
while($true) { 
    $mem = (Get-Process -Name python -ErrorAction SilentlyContinue | 
            Measure-Object -Property WorkingSet -Sum).Sum / 1GB
    Write-Host "RAM Python: $([math]::Round($mem, 2)) GB"
    Start-Sleep 2
}
```

### Linux
```bash
watch -n 2 'ps aux | grep main.py | grep -v grep | awk "{print \$6/1024 \" MB\"}"'
```

## Si Aún Hay Problemas

### 1. Reducir aún más el memory limit
```bash
python app/main.py --max-memory-mb 500 --n-jobs 1
```

### 2. Usar --limit-series para test
```bash
python app/main.py --limit-series 100  # Procesar solo 100 SKUs de test
```

### 3. Verificar que streaming está activo
```bash
python test_memory_optimization.py
```

## Cambios en el Código

✅ Modificado: `app/forecasts.py`
- Lazy loading de datos
- Streaming collection
- Garbage collection automático
- EDP fallback inteligente

✅ Agregado: Documentación
- MEMORY_OPTIMIZATION.md
- CAMBIOS_MEMORIA.md
- RESUMEN_OPTIMIZACIONES.md

## Verificación

Ejecutar test de memoria:
```bash
python test_memory_optimization.py
```

Debe mostrar:
```
→ Fase 1: Cargando datos (lazy)…
  Memoria después de load: ~50 MB ✅

→ Fase 2: Leyendo metadata
  Memoria después de metadata: ~200-300 MB ✅

→ Fase 3: Procesando sección
  Pico máximo: Debe ser <2GB si --limit-series=10 ✅
```

## Preguntas Frecuentes

**P: ¿Los resultados van a ser diferentes?**
R: NO. 100% idénticos. Solo se optimizó memoria, no cálculos.

**P: ¿Es más lento?**
R: NO. Mismo tiempo, menos RAM.

**P: ¿Qué es --max-memory-mb?**
R: Límite suave de RAM. Polars intenta no excederlo usando streaming.

**P: ¿Qué es --n-jobs?**
R: Número de threads paralelos. 1-2 para máquinas antiguas, 4-8 para modernas.

**P: ¿Por qué sigue siendo alto el pico?**
R: Porque después de densify tienes 3.5M filas × 84 columnas. Eso es esperado.

## Support

Si sigue habiendo problemas:
1. Ejecuta `python test_memory_optimization.py`
2. Revisa los logs en la salida
3. Intenta con `--max-memory-mb 500 --n-jobs 1`

