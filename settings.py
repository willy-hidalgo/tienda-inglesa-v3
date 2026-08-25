"""
Configuración central del proyecto de forecasting jerárquico RLS.
Secciones 1 y 23 · niveles: sección → SKU → local (tienda).
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

APP_VERSION: str = "9.1"

# ── Flag de modelo a nivel hoja ──────────────────────────────────────────────
DEMO_MODE = os.environ.get("TI_DEMO_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}

# Corrección de sesgo OOS/forecast: factor = Σy/Σŷ en in_sample (por unique_id).
# Se aplica solo a out_sample y forecast_only (in_sample queda crudo).
BIAS_CORRECTION: bool = True  # solo nodos agregados; SKU+tienda se excluye estructuralmente
BIAS_CORRECTION_MIN_POINTS: int = 7
BIAS_CORRECTION_CLIP: tuple[float, float] = (0.5, 2.0)

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
WMAPE_PATH = OUT_DIR / "wmape.parquet"

MASTER_XLSX_FILENAME = "Mercadologico_Tienda_secc_1_y_23.xlsx"
MASTER_DAT_FILENAME = "MercadologicoTienda.dat"
SALES_FILES_GLOB = "Valida_Profi*"

# ─────────────────────────────────────────────────────────────────────────────
# Secciones y filtros
# ─────────────────────────────────────────────────────────────────────────────
FOCUS_SECTIONS = ["1", "23"]
EXCLUDED_CATEGORIES_FOR_RANGE: list[str] = []

SALES_DATE_FORMAT = "%d-%m-%Y %H:%M:%S"
SOURCE_ENCODING = "cp1252"

# ─────────────────────────────────────────────────────────────────────────────
# Forecasting v5: RLS expanding-28 + hojas rápidas con efectos de drivers
# ─────────────────────────────────────────────────────────────────────────────
# Fechas globales de compatibilidad; las ventanas operativas se definen por
# sección en SECCIONES y se resuelven con section_horizons().
TEST_START = dt.date(2025, 11, 1)
TEST_END = dt.date(2026, 1, 1)
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
RLS_FIT_MODE: str = "expanding_28"  # "current" restaura el fit único anterior
RLS_BLOCK_DAYS: int = 28

# Métricas por defecto de sección/tienda: forecasts acumulados emitidos por los
# bloques expanding-28. "current" restaura el cálculo anterior.
METRICS_MODE: str = "rolling_28"

# SKU+tienda: ruta productiva escalable.
# Los primeros 28 días son warm-up: nivel inicial = promedio de actuals
# realmente disponibles (no se diluye sum/28 por fechas ausentes). Desde el
# día 29 SES se actualiza sobre el residuo log-space únicamente al cierre de
# cada bloque con actuals conocidos.
FAST_LEAF_MODE: bool = True
LEAF_INITIAL_LEVEL_DAYS: int = 28
LEAF_INITIAL_MIN_POINTS: int = 7  # legacy/ignored v8.9; warm-up = sum(actuals)/28 calendar days
LEAF_SES_ALPHA: float = 0.10  # fallback conservador cuando aún no hay score causal
LEAF_SES_ALPHA_CANDIDATES: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80)
LEAF_PARENT_SELECTION: str = "prior_cumulative_wmape"
LEAF_ALPHA_SELECTION: str = "pure_ses_prior_cumulative_wmape"
LEAF_SES_NEAR_BEST_REL_TOLERANCE: float = 0.05
LEAF_SES_NEAR_BEST_ABS_TOLERANCE: float = 1e-6
LEAF_SES_STABILITY_ENABLED: bool = True
LEAF_SES_STABILITY_WINDOW_DAYS: int = 112
LEAF_SES_LEVEL_MAX_RECENT_RATIO: float = 1.50
LEAF_SES_STABILITY_EPS: float = 1e-9
LEAF_SES_SCORE_DECAY: float = 0.85
LEAF_REGIME_DENSE_COVERAGE: float = 0.85
LEAF_REGIME_SHOCK_RATIO: float = 1.50
LEAF_REGIME_DECLINE_RATIO: float = 0.60
LEAF_REGIME_RECENT14_WEIGHT: float = 0.65
LEAF_REGIME_SCORE_TOLERANCE: float = 0.20
LEAF_REGIME_GROWTH_RATIO: float = 1.15
LEAF_REGIME_SPARSE_SHOCK_RATIO: float = 1.35
LEAF_REGIME_SPARSE_STABILITY_RATIO: float = 1.25
# Motor computacional; no modifica el espacio estadístico de candidatos.
LEAF_CANDIDATE_ENGINE: str = "vectorized_block"
LEAF_REQUIRE_PARENT_DRIVERS: bool = True
WMAPE_INCLUDE_ZERO_ACTUAL_DAYS: bool = True
# SES modela directamente el nivel observable de cada hoja en escala
# original. Los drivers RLS solo modifican la forma temporal del bloque.
LEAF_SES_SCALE: str = "original"
FAST_LEAF_DRIVER_EFFECTS: bool = True
# El driver RLS modifica la FORMA diaria, nunca el nivel medio del SKU.
# Factor multiplicativo relativo, normalizado a media 1 por bloque.
FAST_LEAF_DRIVER_FACTOR_CLIP: tuple[float, float] = (0.50, 2.00)
# Después del warm-up, ausencia de fila diaria significa venta cero
# para el estado SES y para el score histórico. Train y OOS usan así la misma
# semántica de calendario diario.
LEAF_ZERO_FILL_MISSING_CALENDAR_DAYS: bool = True
LEAF_DRIVER_MEAN_TOLERANCE: float = 1e-6
LEAF_DRIVER_STRENGTH_CANDIDATES: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
LEAF_DRIVER_DEFAULT_STRENGTH: float = 0.50
LEAF_DRIVER_SCORE_DECAY: float = 0.70
LEAF_DRIVER_NEAR_BEST_REL_TOLERANCE: float = 0.02
LEAF_DRIVER_DIRECTION_CONFLICT_RATIO: float = 0.85
LEAF_DRIVER_RECENT_TREND_FLOOR: float = 0.95
LEAF_DRIVER_RECENT_TREND_CEILING: float = 1.05
LEAF_DRIVER_DIRECTION_GUARD_MAX_STRENGTH: float = 0.25
LEAF_FORECAST_LEVEL_MEAN_TOLERANCE: float = 0.02
LEAF_LEVEL_REFERENCE_WARN_RATIO: float = 4.0
# Diagnóstico obligatorio de transiciones OOS/forecast-only.
LEAF_DIAGNOSTIC_TOP_N: int = 10
LEAF_DIAGNOSTIC_LEVEL_RATIO: float = 3.0
LEAF_DIAGNOSTIC_FORECAST_RATIO: float = 3.0

# Métricas oficiales: siempre bottom-up desde SKU+tienda y desde el día 29.
METRICS_START_DAY: int = 29

# Ranking SKU: mínimo de días con actual distinto de cero.
RANKING_SKU_MIN_NONZERO_POINTS: int = 15

FECHAS_TRAIN = (dt.date(2024, 5, 1), dt.date(2025, 10, 26))
FECHAS_TEST = (TEST_START, TEST_END)

# --------------------------------------------------------------------------- #
# Secciones y locales (tablas del cliente)
# --------------------------------------------------------------------------- #
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

FORECAST_LEVELS = ["seccion", "store", "sku"]
NOMBRES_NIVELES = ["seccion", "store", "sku"]

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros del modelo RLS
# ─────────────────────────────────────────────────────────────────────────────
RMSE_ERROR = 0.2
CORRECTION_FACTOR = False
FORGETTING_FACTOR = 0.995
RLS_FORGETTING_FACTOR_CANDIDATES: tuple[float, ...] = (0.970, 0.985, 0.995)
RLS_AUTOREGRESSIVE_DRIVERS: bool = True
# v8.1: AR no se fuerza. Cada bloque elige causalmente entre RLS base y RLS+AR
# usando exclusivamente el wMAPE acumulado de bloques anteriores. Esto protege
# OOS contra deriva recursiva de lag/rolling sin eliminar los picos cuando AR gana.
RLS_DYNAMICS_CANDIDATES: tuple[str, ...] = ("base", "ar")
RLS_DEFAULT_DYNAMICS: str = "base"
RLS_AR_LAGS: tuple[int, ...] = (7, 28)
RLS_AR_ROLLING_WINDOWS: tuple[int, ...] = (7, 28)
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
    """Resolve aligned train/OOS/actual-extension/forecast-only windows.

    Rules:
    - day 1 is the first Monday available for the section;
    - every modeling block is 28 days, Monday through Sunday;
    - OOS is exactly one complete 28-day block;
    - OOS start is the first aligned block boundary on/after the configured
      preferred test_start; if there are not 28 actual days available there,
      step backwards by complete blocks until a valid OOS is found;
    - forecast-only is exactly the 28 days immediately after OOS;
    - any actuals that happen to exist in forecast-only are not consumed before
      generating that forecast, preventing leakage;
    - forecast-only therefore starts Monday and ends Sunday with no visual gap.
    """
    cfg = SECCIONES[seccion]
    block_days = int(RLS_BLOCK_DAYS)

    if first_data is None:
        first_data = FECHAS_TRAIN[0]
    first_monday = nearest_monday_on_or_after(first_data)

    if last_actual is None:
        # Compatibility fallback for callers that do not have data bounds.
        last_actual = cfg.get("test_end") or TEST_END

    preferred = cfg.get("test_start") or first_monday
    oos_start = _aligned_block_start_on_or_after(
        first_monday, preferred, block_days
    )
    oos_end = oos_start + dt.timedelta(days=block_days - 1)

    # OOS must be fully covered by actuals. Move backward by full blocks if the
    # configured/preferred boundary is too recent for the available data.
    while oos_end > last_actual and oos_start - dt.timedelta(days=block_days) >= first_monday:
        oos_start -= dt.timedelta(days=block_days)
        oos_end -= dt.timedelta(days=block_days)

    if oos_end > last_actual:
        raise ValueError(
            f"Sección {seccion}: no hay 28 días completos de actuals para OOS "
            f"desde el primer lunes disponible {first_monday}."
        )

    train_start = first_monday
    train_end = oos_start - dt.timedelta(days=1)
    train_days = (train_end - train_start).days + 1
    if train_days < block_days or train_days % block_days != 0:
        raise ValueError(
            f"Sección {seccion}: train no queda alineado a bloques de {block_days} días "
            f"({train_start} → {train_end}, {train_days} días)."
        )

    # Forecast-only is ALWAYS the immediately following 28-day block.
    # Actuals that may already exist inside this period are deliberately not
    # used to update RLS/SES before creating the forecast (no leakage).
    forecast_start = oos_end + dt.timedelta(days=1)
    forecast_end = forecast_start + dt.timedelta(days=block_days - 1)

    # Kept in the public horizon contract for compatibility, but disabled for
    # the current production forecast: post-OOS actuals are not consumed.
    extension_start = None
    actual_extension_end = None
    n_extension_blocks = 0

    return {
        "first_monday": first_monday,
        "last_actual": last_actual,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": oos_start,
        "test_end": oos_end,
        "actual_extension_start": None,
        "actual_extension_end": actual_extension_end,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "raw_test_start": cfg.get("test_start"),
        "raw_forecast_start": cfg.get("forecast_start"),
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
# WMAPE TOTAL REAL = Σ_i |y_i − ŷ_i| / Σ_j |y_j|
# Los días con y=0 permanecen: aportan |ŷ| al numerador y 0 al denominador.
# Solo se excluyen pares no finitos/nulos.
#
# BIAS = Σ (ŷ − y) / Σ y sobre los mismos pares finitos.


def compute_wmape(y, yhat) -> float:
    """WMAPE en [0,+∞). Incluye días y==0 en el error absoluto."""
    import numpy as np

    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not mask.any():
        return 0.0
    y_m, yh_m = y[mask], yhat[mask]
    denom = float(np.abs(y_m).sum())
    if denom == 0:
        return 0.0
    return float(np.abs(y_m - yh_m).sum()) / denom


def compute_bias(y, yhat) -> float:
    """BIAS = Σ(ŷ−y)/Σy sobre pares finitos, incluyendo y==0."""
    import numpy as np

    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not mask.any():
        return 0.0
    y_m, yh_m = y[mask], yhat[mask]
    denom = float(y_m.sum())
    if denom == 0:
        return 0.0
    return float((yh_m - y_m).sum()) / denom
