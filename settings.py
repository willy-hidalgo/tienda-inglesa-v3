"""
Configuración central del proyecto de forecasting jerárquico RLS.
Secciones 1 y 23 · niveles: sección → SKU → local (tienda).
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

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

# Compatibilidad global (el pipeline usa SECCIONES por sección)
TEST_START = dt.date(2025, 11, 1)
TEST_END = dt.date(2026, 1, 1)
METRIC_HORIZON_DAYS = 28
ROLLING_HORIZON_DAYS = 28  # yhat28 / valuehat28 walk-forward
COMPUTE_ROLLING_28 = False  # si False, forecasts.py no calcula yhat28/valuehat28
# (el dashboard detecta la ausencia de estas
# columnas en el parquet para ocultar el control
# de selección de series y la sección de métricas
# rolling28; ver README § Rolling 28d). Desde el
# cambio de arquitectura (RLS solo a nivel
# sección), el rolling28 SOLO se calcula para los
# nodos de sección — ver README § Modelo jerárquico.

# ─────────────────────────────────────────────────────────────────────────────
# Modelo jerárquico: RLS solo a nivel sección; tienda/sku/tienda+sku se
# derivan sin RLS (efecto de drivers + suavización exponencial sobre el
# residuo). Ver README § Modelo jerárquico para el detalle del método.
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANTE: Los coeficientes RLS son DIFERENTES para cantidad vs precio,
# pero la SES también debe ser diferenciada porque las escalas y volatilidades
# son MUY distintas. Ver ISSUE_SES_ALPHA_DIFFERENTIATION.md para detalles.

SES_ALPHA = 0.1  # DEPRECATED: usar SES_ALPHA_QUANTITY y SES_ALPHA_VALUE en su lugar
SES_ALPHA_QUANTITY = 0.15  # α para suavización de residuos en CANTIDAD
# Escala pequeña (1-1000 unidades) → más reactivo
SES_ALPHA_VALUE = 0.05  # α para suavización de residuos en VALOR/PRECIO
# Escala grande ($100-$10k) → menos reactivo
# Criterio de diferenciación: Cantidad tiene menos inercia que precio,
# así que responde más rápido a cambios (drivers de promoción, etc).
# Precio es más "pegajoso" (stickier) en dinámicas corto-plazo.

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
# Niveles de agregación (solo 3)
# ─────────────────────────────────────────────────────────────────────────────
# unique_id:
#   "1"                     → sección
#   "1||00122"              → tienda (local)
#   "1||00122||SKU123"      → SKU dentro de la tienda
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
MIN_Y_TO_update = 1.0

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
    "promo_mar": {
        "month": 3,
        "day": 12,
        "window_before_days": 1,
        "window_after_days": 5,
        "active": 1,
    },
    "promo_sep": {
        "month": 9,
        "day": 3,
        "window_before_days": 1,
        "window_after_days": 15,
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
    """
    train_end = test_start de la sección.
    train_start = max(S0, T*), donde:
      S0 = domingo ≥ first_data
      T* = train_end - n*block_days  (n máximo tal que T* ≥ S0)
    """
    s0 = nearest_sunday_on_or_after(first_data)
    if train_end < s0:
        return s0, train_end
    n = (train_end - s0).days // block_days
    t_star = train_end - dt.timedelta(days=n * block_days)
    train_start = max(s0, t_star)
    return train_start, train_end


def section_horizons(seccion: str, first_data: dt.date | None = None) -> dict:
    """
    Ventanas de una sección:
      train:  [train_start, train_end]  train_end = test_start original
      OOS:    lunes ≥ test_start → test_end
      forecast-only (sin actuals): test_end + 1 → forecast_end (settings.SECCIONES)
    """
    cfg = SECCIONES[seccion]
    test_start = cfg["test_start"]
    test_end = cfg["test_end"]
    train_end = test_start
    if first_data is None:
        first_data = FECHAS_TRAIN[0]
    train_start, train_end = compute_train_window(first_data, train_end)
    oos_start = nearest_monday_on_or_after(test_start)
    # Solo-forecast: día siguiente a test_end hasta forecast_end de la sección
    forecast_start = test_end + dt.timedelta(days=1)
    forecast_end = cfg.get("forecast_end") or (
        test_end + dt.timedelta(days=METRIC_HORIZON_DAYS)
    )
    forecast_end = max(forecast_end, forecast_start)
    return {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": oos_start,
        "test_end": test_end,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "raw_test_start": test_start,
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
# WMAPE TOTAL REAL = Σ_i |y_i − ŷ_i| / Σ_j y_j
# (equiv. a Σ_i (|err_i|/y_i) · (y_i / Σ y) )
# Se excluyen observaciones con y = 0 (o nulas).
#
# BIAS = Σ (ŷ − y) / Σ y   (mismo filtro y ≠ 0)


def compute_wmape(y, yhat) -> float:
    """WMAPE en [0, +∞) como fracción (no porcentaje). Excluye y==0."""
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
    """BIAS = Σ(ŷ−y)/Σy. Excluye y==0."""
    import numpy as np

    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(yhat) & (y != 0)
    if not mask.any():
        return 0.0
    y_m, yh_m = y[mask], yhat[mask]
    denom = float(y_m.sum())
    if denom == 0:
        return 0.0
    return float((yh_m - y_m).sum()) / denom
