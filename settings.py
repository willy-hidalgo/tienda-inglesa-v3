"""
Configuración central del proyecto de forecasting jerárquico RLS.
Secciones 1 y 23 · niveles: sección → SKU → local (tienda).
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

APP_VERSION: str = "13.2.6"

# ── Modo de demostración de selección de datos ──────────────────────────────
DEMO_MODE = os.environ.get("TI_DEMO_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}

# No existe corrección post-hoc de sesgo por período: in-sample, OOS y
# forecast-only usan exactamente la misma familia/modelo en cada origen.

# Ejecución multi-cadencia memory-safe. Para cadencias menores a 28d se limita
# el paralelismo y se persisten resultados por sección para mantener la RAM acotada.
MULTIBLOCK_MEMORY_SAFE: bool = True
MULTIBLOCK_MEMORY_SAFE_THRESHOLD_DAYS: int = 28
MULTIBLOCK_MAX_JOBS: dict[int, int] = {1: 1, 7: 2, 14: 4, 28: 8}

# ─────────────────────────────────────────────────────────────────────────────
# Rutas (ancladas al propio archivo → independientes del cwd)
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = ROOT / "data" / "input"
OUT_DIR = ROOT / "data" / "output"

MASTER_PATH = OUT_DIR / "master.parquet"
SALES_PATH = OUT_DIR / "sales.parquet"
SELECTED_PATH = OUT_DIR / "selected.parquet"
FORECAST_PATH = OUT_DIR / "forecast.parquet"

MASTER_XLSX_FILENAME = "Mercadologico_Tienda_secc_1_y_23.xlsx"
MASTER_DAT_FILENAME = "MercadologicoTienda.dat"
SALES_FILES_GLOB = "Valida_Profi*"

# ─────────────────────────────────────────────────────────────────────────────
# Secciones y filtros
# ─────────────────────────────────────────────────────────────────────────────
FOCUS_SECTIONS = ["1", "23"]

SALES_DATE_FORMAT = "%d-%m-%Y %H:%M:%S"
SOURCE_ENCODING = "cp1252"

# ─────────────────────────────────────────────────────────────────────────────
# Forecasting: RLS expanding-28 + SKU+tienda SES+RLS
# ─────────────────────────────────────────────────────────────────────────────
# Fechas globales de compatibilidad; las ventanas operativas se definen por
# sección en SECCIONES y se resuelven con section_horizons().
METRIC_HORIZON_DAYS = 28

# El artefacto wmape.parquet del pipeline prioriza OOS para evitar agregaciones
# históricas costosas al cierre de cada sección.
PIPELINE_WMAPE_OOS_ONLY: bool = True

# Requisito del cliente:
#   actuals 1..28   -> modelo para forecast 29..56
#   actuals 1..56   -> modelo para forecast 57..84
#   actuals 1..84   -> modelo para forecast 85..112
# La implementación usa un único recorrido RLS y snapshots de coeficientes en
# cada corte de 28 días; evita refits completos y conserva la semántica expansiva.
# Cadencia de actualización del modelo: 1/7/14/28 días.
# OOS y forecast-only permanecen fijos en METRIC_HORIZON_DAYS=28 para que
# los escenarios sean directamente comparables.
UPDATE_BLOCK_OPTIONS: tuple[int, ...] = (1, 7, 14, 28)
RLS_BLOCK_DAYS: int = 28
RLS_INITIAL_SEED_DAYS: int = 28
UPDATE_BLOCKS_DIR = OUT_DIR / "update_blocks"

def update_block_out_dir(days: int) -> Path:
    days = int(days)
    if days not in UPDATE_BLOCK_OPTIONS:
        raise ValueError(f"Bloque de actualización inválido: {days}; use {UPDATE_BLOCK_OPTIONS}")
    return UPDATE_BLOCKS_DIR / f"block_{days:02d}d"

def update_block_forecast_path(days: int) -> Path:
    return update_block_out_dir(days) / "forecast.parquet"

# Performance contract: RLS numerical kernels must execute in Numba
# nopython mode. With ~100 drivers the covariance update is O(p^2); executing
# that double loop in Python is not production-viable. cache=True avoids paying
# compilation cost on every process invocation.
RLS_NUMBA_KERNELS_REQUIRED: bool = True


# Métricas por defecto de sección/tienda: forecasts acumulados emitidos por los
# bloques expanding-28. "current" restaura el cálculo anterior.
METRICS_MODE: str = "rolling_28"

# SKU+tienda: contrato productivo único y explicable (v13).
#   1) Nivel = SES de magnitudes positivas desestacionalizadas.
#   2) Estado inicial = mediana positiva de los primeros 28 días calendario.
#   3) El mejor alpha SES se selecciona causalmente con bloques previos.
#   4) RLS tienda/sección aporta el efecto no-intercepto de sus coeficientes
#      en escala log1p; el SES aporta el nivel base de la hoja.
#   5) Tienda vs sección se elige por wMAPE RLS acumulado de bloques previos.
#   6) La MISMA composición genera in-sample, OOS y forecast-only.
LEAF_INITIAL_LEVEL_DAYS: int = 28
LEAF_SES_ALPHA: float = 0.10
LEAF_SES_ALPHA_CANDIDATES: tuple[float, ...] = (
    0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80
)
LEAF_PARENT_DEFAULT: str = "store"
LEAF_DRIVER_EFFECT_LIMIT: float = 20.0  # guard numérico, no calibración del modelo

# Optimización estadística v13.x dentro de la MISMA familia SES+RLS.
# v13.2.6 RESTAURA la configuración productiva estadística de v13.1.1 como
# baseline estable después de detectar regresiones visuales/end-to-end en v13.2.x.
# Las extensiones posteriores permanecen solo como diagnóstico hasta superar
# un gate de regresión completo en 1d/7d/14d/28d. El OOS actual no hace tuning.
STAT_OPTIMIZATION_ENABLED: bool = True
STAT_OPTIMIZATION_PROMOTE_AUTOMATICALLY: bool = False
# Phase 2/3/4 mantiene un espacio diagnóstico alrededor del productivo.
# En v13.2.6 alpha=0.30 y lambda=0.990/0.9975 vuelven a ser SOLO diagnóstico.
# Ninguno participa en producción hasta demostrar no-regresión end-to-end.
STAT_OPT_SES_EXTRA_ALPHAS: tuple[float, ...] = (0.0025, 0.0075, 0.30, 0.50)
STAT_OPT_RLS_EXTRA_LAMBDAS: tuple[float, ...] = (0.990, 0.9975, 1.0)
# Refit exacto sobre el mismo RLS. No crea drivers nuevos.
STAT_OPT_REFIT_DRIVER_GROUPS: tuple[str, ...] = (
    "weekday", "month", "holiday_other", "price"
)
# Máximo de filas de detalle residual persistidas. Los resúmenes se calculan
# sobre todos los puntos; este límite solo evita un artefacto de detalle excesivo.
STAT_OPT_RESIDUAL_HISTORY_DAYS: int = 168
STAT_OPT_MAX_RESIDUAL_EXTREMES: int = 50_000
# Phase 4: calibración diagnóstica de la intensidad del efecto RLS en hojas.
# gamma=1.0 reproduce exactamente producción; el resto solo se evalúa sobre
# forecasts causales ya emitidos y nunca se promueve automáticamente.
STAT_OPT_DRIVER_STRENGTH_CANDIDATES: tuple[float, ...] = (
    0.0, 0.25, 0.50, 0.75, 0.90, 1.0, 1.10, 1.25, 1.50
)
STAT_OPT_DRIVER_STRENGTH_MIN_FOLDS: int = 4

# Phase 5: selección diagnóstica del parent RLS en contexto SKU+Tienda.
# Sigue siendo la MISMA composición SES + RLS; la alternativa solo cambia
# qué parent ya existente (Sección o Tienda) aporta el efecto al mismo nivel SES.
# La historia cerrada crea candidatos y el OOS actual únicamente valida/veta.
STAT_OPT_PARENT_MIN_FOLDS: int = 4
STAT_OPT_PARENT_MIN_WMAPE_IMPROVEMENT: float = 0.010
STAT_OPT_PARENT_MIN_POSITIVE_FOLD_SHARE: float = 0.60
STAT_OPT_PARENT_MAX_HISTORY_ABS_BIAS_DETERIORATION: float = 0.010
STAT_OPT_PARENT_HOLDOUT_VALIDATE_MAX_ABS_BIAS_DETERIORATION: float = 0.005
STAT_OPT_PARENT_HOLDOUT_VETO_ABS_BIAS_DETERIORATION: float = 0.010

# Ranking SKU: mínimo de días con actual distinto de cero.
RANKING_SKU_MIN_NONZERO_POINTS: int = 15
OOS_METRIC_MODE: str = "active"  # active | all
OOS_ACTIVE_MIN_NONZERO_DAYS: int = 7
DASHBOARD_METRIC_IDENTITY_TOLERANCE: float = 1e-9

FECHAS_TRAIN = (dt.date(2024, 5, 1), dt.date(2025, 10, 26))

# --------------------------------------------------------------------------- #
# Secciones y locales (tablas del cliente)
# --------------------------------------------------------------------------- #
# Las fechas test/forecast dentro de cada sección son referencias de respaldo.
# El horizonte operativo se deriva de los datos disponibles y mantiene OOS=28d.
SECCIONES = {
    "1": {
        "test_start": dt.date(2026, 3, 29),
        "test_end": dt.date(2026, 4, 26),
        "forecast_start": dt.date(2026, 5, 4),
        "forecast_end": dt.date(2026, 5, 26),
        "locales": [
            "00122",
            "00154",
            "00211",
            "00063",
            "00001",
            "00003",
            "00411",
            "00025",
        ],
        "local_names": {
            "00122": "ABAST. COSTA VERDE",
            "00154": "HIPERPREC",
            "00211": "ABAST.SUPER DONATO",
            "00063": "CARRUSEL",
            "00001": "CENTRAL",
            "00003": "UNION",
            "00411": "ATLANTICO",
            "00025": "SOLANAS",
        },
    },
    "23": {
        "test_start": dt.date(2025, 11, 30),
        "test_end": dt.date(2025, 12, 7),
        "forecast_start": dt.date(2026, 1, 5),
        "forecast_end": dt.date(2026, 2, 1),
        "locales": [
            "00155",
            "00200",
            "00046",
            "00095",
            "00006",
            "00005",
            "00022",
            "00012",
        ],
        "local_names": {
            "00155": "DELSOL",
            "00200": "ELTANO",
            "00046": "EL MORRO",
            "00095": "EXPRESS 1",
            "00006": "SHOPPING",
            "00005": "POSADAS",
            "00022": "LA BARRA",
            "00012": "ATLANTIDA",
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Niveles de agregación
# ─────────────────────────────────────────────────────────────────────────────
# IDs canónicos:
#   "1"                          → sección
#   "1||T:00122"                 → tienda
#   "1||T:00122||S:SKU123"       → SKU+tienda
AGGREGATION_LEVELS = {
    "SECCION": "seccion",
    "STORE_ID": "store",
    "SKU_ID": "sku",
}


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros del modelo RLS
# ─────────────────────────────────────────────────────────────────────────────
RMSE_ERROR = 0.2
FORGETTING_FACTOR = 0.995
RLS_FORGETTING_FACTOR_CANDIDATES: tuple[float, ...] = (0.970, 0.985, 0.995)
RLS_AUTOREGRESSIVE_DRIVERS: bool = True
# AR no se fuerza. Cada bloque elige causalmente entre RLS base y RLS+AR
# usando exclusivamente el wMAPE acumulado de bloques anteriores.
RLS_DYNAMICS_CANDIDATES: tuple[str, ...] = ("base", "ar")
RLS_DEFAULT_DYNAMICS: str = "base"
# Valor($) vuelve al diseño productivo estable v13.1.1 sin asp/edp/discount.
# Las pruebas con price continúan disponibles exclusivamente en diagnóstico.
RLS_VALUE_PRICE_NODE_IDS: tuple[str, ...] = ()
# v13.2.6 desactiva las exclusiones productivas promovidas en v13.2.2.
# Permanecen como evidencia experimental, no como configuración productiva.
RLS_DRIVER_GROUP_EXCLUSIONS: dict[str, dict[str, tuple[str, ...]]] = {}
# Los drivers AR del bloque objetivo se construyen recursivamente: nunca usan
# actuals del propio bloque que todavía no eran conocidos al emitir el forecast.
MIN_Y_TO_UPDATE = 1.0  # actualiza RLS solo cuando y supera este umbral


# ─────────────────────────────────────────────────────────────────────────────
# Columnas canónicas
# ─────────────────────────────────────────────────────────────────────────────
DATE_COLUMN = "SALES_DAY"
QUANTITY_COLUMN = "SLS_QTY"
PRICE_COLUMN = "SLS_VAL"

# ─────────────────────────────────────────────────────────────────────────────
# Festividades (Uruguay / retail)
# ─────────────────────────────────────────────────────────────────────────────
HOLIDAYS = {
    "xmas": {
        "month": 12,
        "day": 24,
        "window_before_days": 3,
        "window_after_days": 1,
        "active": 1,
    },
    "new_year": {
        "month": 12,
        "day": 31,
        "window_before_days": 3,
        "window_after_days": 1,
        "active": 1,
    },
    "mothers_day": {
        "rule": "2nd_sunday_may",
        "relative_ocurrence": 2,
        "absolute_weekday": 7,
        "absolute_month": 5,
        "window_before_days": 15,
        "window_after_days": 0,
        "active": 1,
    },
    "dia_trabajo": {
        "month": 4,
        "day": 30,
        "window_before_days": 2,
        "window_after_days": 2,
        "active": 1,
    },
    "promo_feb": {
        "month": 2,
        "day": 17,
        "window_before_days": 2,
        "window_after_days": 2,
        "active": 1,
    },
    "promo_mar": {
        "month": 3,
        "day": 12,
        "window_before_days": 1,
        "window_after_days": 5,
        "active": 1,
    },
    "promo_abr": {
        "month": 4,
        "day": 22,
        "window_before_days": 2,
        "window_after_days": 1,
        "active": 1,
    },
    "promo_abr_2": {
        "month": 4,
        "day": 30,
        "window_before_days": 1,
        "window_after_days": 1,
        "active": 1,
    },
    "promo_may": {
        "rule": "1nd_sunday_may",
        "relative_ocurrence": 1,
        "absolute_weekday": 7,
        "absolute_month": 5,
        "window_before_days": 2,
        "window_after_days": 2,
        "active": 1,
    },
    "promo_may_2": {
        "rule": "4nd_sunday_may",
        "relative_ocurrence": 4,
        "absolute_weekday": 7,
        "absolute_month": 5,
        "window_before_days": 2,
        "window_after_days": 2,
        "active": 1,
    },
    "promo_sep": {
        "month": 9,
        "day": 3,
        "window_before_days": 1,
        "window_after_days": 15,
        "active": 1,
    },
    "promo_oct": {
        "month": 10,
        "day": 1,
        "window_before_days": 1,
        "window_after_days": 1,
        "active": 1,
    },
    "promo_nov": {
        "month": 11,
        "day": 3,
        "window_before_days": 3,
        "window_after_days": 0,
        "active": 1,
    },
    "promo_dic": {
        "month": 12,
        "day": 7,
        "window_before_days": 1,
        "window_after_days": 1,
        "active": 1,
    },
    "fathers_day": {
        "rule": "2nd_sunday_july",
        "relative_ocurrence": 2,
        "absolute_weekday": 7,
        "absolute_month": 7,
        "window_before_days": 13,
        "window_after_days": 0,
        "active": 1,
    },
    "black_friday": {
        "rule": "4th_friday_november",
        "relative_ocurrence": 4,
        "absolute_weekday": 5,
        "absolute_month": 11,
        "window_before_days": 6,
        "window_after_days": 0,
        "active": 1,
    },
}

CURRENT_ZONE = "America/Montevideo"
ID_EJECUCION = "ASSESSMENT TIENDA INGLESA – Secciones 1 y 23"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de fecha
# ─────────────────────────────────────────────────────────────────────────────
def nearest_sunday_on_or_before(d: dt.date) -> dt.date:
    """Domingo más próximo ≤ d (weekday: lunes=0 … domingo=6)."""
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def nearest_sunday_on_or_after(d: dt.date) -> dt.date:
    """Domingo más próximo ≥ d."""
    return d + dt.timedelta(days=(6 - d.weekday()) % 7)


def nearest_monday_on_or_after(d: dt.date) -> dt.date:
    """Lunes más próximo ≥ d."""
    return d + dt.timedelta(days=(7 - d.weekday()) % 7)


def compute_train_window(
    first_data: dt.date, train_end: dt.date, block_days: int = 28
) -> tuple[dt.date, dt.date]:
    """Inclusive train window aligned to the first available Monday."""
    if block_days <= 0:
        raise ValueError("block_days must be > 0")
    train_start = nearest_monday_on_or_after(first_data)
    if train_end < train_start:
        return train_start, train_end
    n_days = (train_end - train_start).days + 1
    n_complete = n_days // block_days
    if n_complete < 1:
        return train_start, train_end
    aligned_end = train_start + dt.timedelta(days=n_complete * block_days - 1)
    return train_start, aligned_end


def _aligned_block_start_on_or_after(
    first_monday: dt.date, preferred: dt.date, block_days: int
) -> dt.date:
    """First block boundary >= preferred on the grid rooted at first_monday."""
    preferred_monday = nearest_monday_on_or_after(preferred)
    delta = max(0, (preferred_monday - first_monday).days)
    k = (delta + block_days - 1) // block_days
    return first_monday + dt.timedelta(days=k * block_days)


def section_horizons(
    seccion: str,
    first_data: dt.date | None = None,
    last_actual: dt.date | None = None,
) -> dict:
    """Resolve the production train/OOS/forecast-only windows.

    Contrato operativo:
    - OOS son siempre los últimos 28 días calendario con actuals;
    - forecast-only empieza al día siguiente y dura 28 días;
    - la historia previa se alinea a la cadencia seleccionada 1/7/14/28d;
    - in-sample, OOS y forecast-only usan la misma familia de modelo.
    """
    cfg = SECCIONES[seccion]
    update_days = int(RLS_BLOCK_DAYS)
    metric_days = int(METRIC_HORIZON_DAYS)

    if first_data is None:
        first_data = FECHAS_TRAIN[0]
    if last_actual is None:
        # Compatibility fallback when no data frame is available. In the real
        # pipeline last_actual is always read from the section data.
        last_actual = cfg["test_end"]

    if last_actual < first_data:
        raise ValueError(
            f"Sección {seccion}: last_actual={last_actual} < first_data={first_data}."
        )

    # El horizonte de evaluación permanece fijo en 28 días para poder comparar
    # cadencias de actualización distintas sobre exactamente las mismas fechas.
    oos_end = last_actual
    oos_start = oos_end - dt.timedelta(days=metric_days - 1)
    if oos_start < first_data:
        raise ValueError(
            f"Sección {seccion}: no hay {metric_days} días calendario para OOS "
            f"({first_data} → {last_actual})."
        )

    # El histórico previo al OOS se alinea a la cadencia seleccionada. Durante
    # los 28 días OOS se emiten forecasts secuenciales en bloques de update_days
    # y, al cerrarse cada bloque, sus actuals pasan al siguiente ajuste.
    delta_days = (oos_start - first_data).days
    offset = delta_days % update_days
    train_start = first_data + dt.timedelta(days=offset)
    train_end = oos_start - dt.timedelta(days=1)
    train_days = (train_end - train_start).days + 1
    if train_days < update_days or train_days % update_days != 0:
        raise ValueError(
            f"Sección {seccion}: train no queda alineado a bloques de "
            f"{update_days} días ({train_start} → {train_end}, {train_days})."
        )

    forecast_start = oos_end + dt.timedelta(days=1)
    forecast_end = forecast_start + dt.timedelta(days=metric_days - 1)

    return {
        # Ancla informativa de la cadencia de actualización.
        # that still reads this key. It is the aligned model start and
        # is not required to be Monday.
        "first_monday": train_start,
        "model_start": train_start,
        "last_actual": last_actual,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": oos_start,
        "test_end": oos_end,
        "actual_extension_start": None,
        "actual_extension_end": None,
        "observed_tail_start": None,
        "observed_tail_end": None,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "raw_test_start": cfg.get("test_start"),
        "update_block_days": update_days,
        "metric_horizon_days": metric_days,
        "raw_test_end": cfg.get("test_end"),
        "raw_forecast_start": cfg.get("forecast_start"),
        "raw_forecast_end": cfg.get("forecast_end"),
        "locales": list(cfg["locales"]),
        "local_names": dict(cfg["local_names"]),
    }


def make_unique_id(
    seccion: str, store: str | None = None, sku: str | None = None
) -> str:
    """
    Construye el unique_id según el nuevo esquema de filtros independientes
    (tienda y sku ya no son jerárquicos entre sí, pueden combinarse en
    cualquier orden):
      - "1"                    → sección
      - "1||T:00122"           → sección + tienda (todos los SKU)
      - "1||S:SKU123"          → sección + sku (todas las tiendas)
      - "1||T:00122||S:SKU123" → sección + tienda + sku
    El orden interno del id siempre es T antes de S (canónico), sin importar
    el orden en que el usuario haya elegido los filtros en la UI.
    """
    parts = [seccion]
    if store is not None:
        parts.append(f"T:{store}")
    if sku is not None:
        parts.append(f"S:{sku}")
    return "||".join(parts)


def split_unique_id(unique_id: str) -> dict[str, str | None]:
    """Descompone unique_id en seccion / store / sku (esquema T:/S: — ver `make_unique_id`)."""
    parts = unique_id.split("||")
    out: dict[str, str | None] = {"seccion": parts[0], "store": None, "sku": None}
    for p in parts[1:]:
        if p.startswith("T:"):
            out["store"] = p[2:]
        elif p.startswith("S:"):
            out["sku"] = p[2:]
    return out


def display_label(
    unique_id: str,
    sku_desc: str | None = None,
    store_name: str | None = None,
) -> str:
    """
    Etiqueta visible sin código de sección.
    - sección:          "Sección 1"
    - tienda:            "00122 — CENTRAL"
    - sku (todas tiendas): "SKU123 — DESCRIPCION"
    - tienda + sku:      "SKU123 — DESCRIPCION @ 00122 — CENTRAL"
    """
    p = split_unique_id(unique_id)
    store, sku = p["store"], p["sku"]
    if store is None and sku is None:
        return f"Sección {p['seccion']}"

    store_part = f"{store} — {store_name}" if store_name else store
    sku_part = f"{sku} — {sku_desc}" if sku_desc else sku

    if store is not None and sku is None:
        return store_part
    if store is None and sku is not None:
        return sku_part
    return f"{sku_part} @ {store_part}"


def ranking_code(unique_id: str) -> str:
    """Código visible en ranking (sin sección): prioridad tienda > sku."""
    p = split_unique_id(unique_id)
    if p["store"] is not None:
        return p["store"]
    if p["sku"] is not None:
        return p["sku"]
    return p["seccion"]


def ranking_description(
    unique_id: str,
    sku_desc: str | None = None,
    store_name: str | None = None,
) -> str:
    """Descripción para ranking: nombre de tienda o DESCRIPCION de SKU."""
    p = split_unique_id(unique_id)
    if p["store"] is not None and p["sku"] is None:
        return store_name or ""
    if p["sku"] is not None and p["store"] is None:
        return sku_desc or ""
    if p["store"] is not None and p["sku"] is not None:
        return sku_desc or store_name or ""
    return f"Sección {p['seccion']}"


# ─────────────────────────────────────────────────────────────────────────────
# Métricas (WMAPE / BIAS)
# ─────────────────────────────────────────────────────────────────────────────
# Regla del cliente: la métrica oficial se calcula SOLO en días con venta y!=0.
# Los falsos positivos de días sin venta se reportan aparte como zero-demand.


def compute_wmape(y, yhat) -> float:
    """WMAPE oficial = Σ|y-ŷ|/Σ|y| sobre pares finitos con y != 0."""
    import numpy as np

    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(yhat) & (y != 0)
    if not mask.any():
        return 0.0
    y_m, yh_m = y[mask], yhat[mask]
    denom = float(np.abs(y_m).sum())
    if denom == 0:
        return 0.0
    return float(np.abs(y_m - yh_m).sum()) / denom


def compute_bias(y, yhat) -> float:
    """BIAS oficial = Σ(ŷ-y)/Σ|y| sobre pares finitos con y != 0."""
    import numpy as np

    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(yhat) & (y != 0)
    if not mask.any():
        return 0.0
    y_m, yh_m = y[mask], yhat[mask]
    denom = float(np.abs(y_m).sum())
    if denom == 0:
        return 0.0
    return float((yh_m - y_m).sum()) / denom
