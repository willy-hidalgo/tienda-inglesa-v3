#!/usr/bin/env python3
"""
Test rápido: Verifica que las optimizaciones de memoria funcionan.
Uso: python test_memory_optimization.py
"""

import gc
import sys
from pathlib import Path

import psutil

# Agregar root al path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import settings
from app.forecasts import (
    CalendarFeatureBuilder,
    ForecastConfig,
    HolidayCalendar,
    RLSForecastPipeline,
)


def get_memory_mb():
    """Retorna memory usage en MB del proceso actual."""
    return psutil.Process().memory_info().rss / (1024**2)


def main():
    print("=" * 70)
    print("TEST: Optimizaciones de Memoria")
    print("=" * 70)

    gc.collect()
    mem_start = get_memory_mb()
    print(f"\n📊 Memoria inicial: {mem_start:.1f} MB")

    # Crear config y pipeline
    config = ForecastConfig.from_settings()
    pipeline = RLSForecastPipeline(config, n_jobs=1, limit_series=10)

    # Inicializar pipeline components
    pipeline._calendar = HolidayCalendar(config.holidays, config.now_year)
    pipeline._feature_builder = CalendarFeatureBuilder(pipeline._calendar)

    print("\n⏱ Iniciando pipeline (con --limit-series 10 para test rápido)…")
    print("   Monitoreando memoria durante ejecución…\n")

    mem_peak = mem_start
    try:
        # Fase 1: Cargar (ahora lazy)
        print("→ Fase 1: Cargando datos (lazy)…")
        selected = pipeline._load()
        gc.collect()
        mem_after_load = get_memory_mb()
        print(
            f"  Memoria después de load: {mem_after_load:.1f} MB (+{mem_after_load - mem_start:.1f})"
        )
        mem_peak = max(mem_peak, mem_after_load)

        # Fase 2: Obtener metadata
        print("\n→ Fase 2: Leyendo metadata por sección…")
        first_by_sec = pipeline._first_data_by_section(selected)
        gc.collect()
        mem_after_metadata = get_memory_mb()
        print(f"  Memoria después de metadata: {mem_after_metadata:.1f} MB")
        print(f"  Secciones encontradas: {list(first_by_sec.keys())}")
        mem_peak = max(mem_peak, mem_after_metadata)

        # Fase 3: Procesar primera sección
        seccion = settings.FOCUS_SECTIONS[0]
        print(f"\n→ Fase 3: Procesando sección '{seccion}'…")
        res, wm, driver_cols = pipeline._run_section(
            selected, seccion, first_by_sec[seccion], None
        )
        gc.collect()
        mem_after_section = get_memory_mb()
        print(f"  Memoria después de sección: {mem_after_section:.1f} MB")
        print(f"  Filas generadas: {res.height if res.height else 0}")
        mem_peak = max(mem_peak, mem_after_section)

        # Resumen
        print("\n" + "=" * 70)
        print("✓ TEST COMPLETADO SIN ERRORES")
        print("=" * 70)
        print("\n📈 Estadísticas de Memoria:")
        print(f"  Inicial:        {mem_start:.1f} MB")
        print(f"  Pico máximo:    {mem_peak:.1f} MB")
        print(
            f"  Aumento total:  {mem_peak - mem_start:.1f} MB ({100 * (mem_peak - mem_start) / mem_start:.1f}%)"
        )
        print("\n💡 Recomendación:")
        if mem_peak < 500:
            print("  ✅ Uso de memoria EXCELENTE (<500 MB)")
        elif mem_peak < 1000:
            print("  ✅ Uso de memoria BUENO (<1 GB)")
        elif mem_peak < 2000:
            print("  ⚠️  Uso de memoria MODERADO (1-2 GB)")
        else:
            print(
                "  ❌ Uso de memoria ALTO (>2 GB) - Revisar --limit-series o --max-memory-mb"
            )

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
