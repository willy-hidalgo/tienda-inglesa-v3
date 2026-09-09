"""
Configuración central del proyecto de forecasting jerárquico RLS.
Secciones 1 y 23 · niveles: sección → SKU → local (tienda).
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

APP_VERSION: str = "12.9.12"

# ── Flag de modelo a nivel hoja ──────────────────────────────────────────────
DEMO_MODE = os.environ.get("TI_DEMO_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}

# Corrección de sesgo OOS/forecast: factor = Σy/Σŷ en in_sample (por unique_id).
# Se aplica solo a out_sample y forecast_only (in_sample queda crudo).
BIAS_CORRECTION: bool = True  # solo nodos agregados; SKU+tienda se excluye estructuralmente
BIAS_CORRECTION_MIN_POINTS: int = 7
BIAS_CORRECTION_CLIP: tuple[float, float] = (0.5, 2.0)
WRITE_SECTION_CHECKPOINTS: bool = False  # checkpoints completos son opt-in; costosos en I/O

# v12.8.5 memory-safe multi-cadence demo.  Para cadencias menores a 28d
# los estados leaf intermedios crecen casi inversamente con el tamaño de bloque.
# Se derraman a parquet temporal y se limita el paralelismo para mantener la RAM
# acotada. El escenario 28d conserva el camino rápido histórico.
MULTIBLOCK_MEMORY_SAFE: bool = True
MULTIBLOCK_MEMORY_SAFE_THRESHOLD_DAYS: int = 28
MULTIBLOCK_MAX_JOBS: dict[int, int] = {1: 1, 7: 2, 14: 4, 28: 8}
MULTIBLOCK_SPILL_COMPRESSION: str = "zstd"

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
# Forecasting: RLS expanding-28 + SKU+tienda SES+RLS
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
# Cadencia de actualización del modelo. Para la demo puede ser 1/7/14/28 días.
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

# v11.2 performance contract: RLS numerical kernels must execute in Numba
# nopython mode. With ~100 drivers the covariance update is O(p^2); executing
# that double loop in Python is not production-viable. cache=True avoids paying
# compilation cost on every process invocation.
RLS_NUMBA_KERNELS_REQUIRED: bool = True


# Métricas por defecto de sección/tienda: forecasts acumulados emitidos por los
# bloques expanding-28. "current" restaura el cálculo anterior.
METRICS_MODE: str = "rolling_28"

# SKU+tienda: arquitectura v11.5 (OOS final + fallbacks robustos causales).
# Regla simple y auditable:
#   1) SES puro determina el NIVEL base leaf.
#   2) RLS tienda o sección aporta FORMA y puede aportar UPLIFT DE NIVEL.
#   3) shape_only y level_shape compiten causalmente con bloques ya cerrados.
#   4) No existe modelo leaf "none" ni selección mirando el bloque objetivo.
FAST_LEAF_MODE: bool = True
LEAF_INITIAL_LEVEL_DAYS: int = 28
LEAF_SES_ALPHA: float = 0.10
LEAF_SES_ALPHA_CANDIDATES: tuple[float, ...] = (
    0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80
)
LEAF_ALPHA_SCORE_WEIGHTS: tuple[float, float, float, float] = (
    0.60, 0.30, 0.10, 0.00
)
LEAF_ALPHA_BIAS_WEIGHT: float = 0.20
LEAF_ALPHA_SCORE_MIN_DEN: float = 1e-9

# Detección conservadora de régimen. El nivel estructural es la mediana de
# hasta 5 bloques COMPLETOS anteriores a B0. Trend requiere tres transiciones
# consecutivas B3→B2→B1→B0; uno/dos bloques extremos se tratan como transitorio.
LEAF_REGIME_HISTORY_BLOCKS: int = 5
LEAF_REGIME_MIN_HISTORY_BLOCKS: int = 4
LEAF_REGIME_TREND_UP_RATIO: float = 1.05
LEAF_REGIME_TREND_DOWN_RATIO: float = 0.95
LEAF_REGIME_TREND_MAX_STEP_RATIO: float = 2.00
LEAF_REGIME_TRANSIENT_RATIO: float = 1.75
LEAF_REGIME_EPS: float = 1e-9
LEAF_RESET_TRANSIENT: bool = True
LEAF_TRANSIENT_DOWN_ALPHA: float = 0.20

# RLS obligatorio: sección o tienda, nunca "none", siempre con efecto completo.
LEAF_PARENT_CANDIDATES: tuple[str, str] = ("store", "section")
LEAF_PARENT_DEFAULT: str = "store"
LEAF_PARENT_DRIVER_STRENGTH: float = 1.00

# v11.5: para cada padre compiten causalmente dos interpretaciones del RLS:
# - shape_only: forma diaria con media 1 (comportamiento v11.4);
# - level_shape: misma forma + cambio de nivel del bloque que anticipa el RLS
#   frente al nivel real del bloque inmediatamente anterior.
# La selección usa exclusivamente errores de bloques ya cerrados.
LEAF_PARENT_DRIVER_MODES: tuple[str, str] = ("shape_only", "level_shape")
# v11.6: v11.5 showed that parent level_shape reduced BIAS but worsened wMAPE
# materially (and produced store-level overshoot). Keep the implementation for
# audit/A-B work, but production selection uses parent shape only.
LEAF_PARENT_LEVEL_SHAPE_SELECTABLE: bool = False
LEAF_PARENT_DEFAULT_DRIVER_MODE: str = "shape_only"

# SKU-specific annual seasonality: 13 x 28 days = 364 days, aligned by weekday.
# The factor is calculated from SKU aggregate sales across stores and selected
# causally against the existing forecast on the preceding validation block.
LEAF_SKU_SEASONAL_ENABLED: bool = True
LEAF_SKU_SEASONAL_LAG_BLOCKS: int = 13
LEAF_SKU_SEASONAL_MIN_POSITIVE_OBS: int = 14
LEAF_SKU_SEASONAL_FULL_WEIGHT_OBS: int = 56
LEAF_SKU_SEASONAL_FACTOR_CLIP: tuple[float, float] = (0.50, 2.50)

# La forma diaria se acota con una única escala alrededor de 1. El multiplicador
# de nivel de level_shape usa el mismo límite causal de salto de régimen definido
# por LEAF_REGIME_TREND_MAX_STEP_RATIO.
FAST_LEAF_DRIVER_FACTOR_CLIP: tuple[float, float] = (0.20, 5.00)
LEAF_DRIVER_MEAN_TOLERANCE: float = 1e-6
LEAF_REQUIRE_PARENT_DRIVERS: bool = True

LEAF_ZERO_FILL_MISSING_CALENDAR_DAYS: bool = True
LEAF_FORECAST_LEVEL_MEAN_TOLERANCE: float = 0.02
LEAF_LEVEL_REFERENCE_WARN_RATIO: float = 4.0
LEAF_DIAGNOSTIC_TOP_N: int = 10
LEAF_DIAGNOSTIC_LEVEL_RATIO: float = 3.0
LEAF_DIAGNOSTIC_FORECAST_RATIO: float = 3.0

# v11.2: OOS siempre es el último bloque observado de 28 días. No puede haber
# actuals posteriores a OOS; forecast-only empieza exactamente al día siguiente.
OOS_USE_LAST_28_ACTUAL_DAYS: bool = True

# Fallbacks causales para hojas de alto error. El trigger usa únicamente error
# histórico anterior al bloque objetivo. Nunca se selecciona un modelo mirando
# los actuals del propio OOS.
LEAF_FALLBACK_ENABLED: bool = True
LEAF_FALLBACK_ENGINE: str = "vectorized_sparse"
LEAF_FALLBACK_TRIGGER_WMAPE: float = 0.60
LEAF_FALLBACK_MIN_IMPROVEMENT: float = 0.02
LEAF_FALLBACK_BIAS_WEIGHT: float = 0.20
LEAF_FALLBACK_MIN_HISTORY_DAYS: int = 84
LEAF_FALLBACK_MIN_POSITIVE_HISTORY: int = 14
# Fallbacks are selected only from validation blocks with enough observed
# sales days to be statistically defensible and consistent with Active OOS.
# v11.5 preserves fallback level conditional on actual>0 because the official
# client wMAPE/BIAS uses exactly that same support. Zero-demand impact remains
# reported separately and is not hidden from acceptance.
LEAF_FALLBACK_MIN_VALIDATION_SALES: int = 7
LEAF_FALLBACK_PEAK_MAD_MULTIPLIER: float = 4.0
LEAF_FALLBACK_PEAK_MEDIAN_MULTIPLIER: float = 3.0

# Diagnóstico detallado se deja fuera del hot path. Las invariantes de producción
# siguen activas; el análisis exhaustivo queda en validate_v11/diagnose_leaf.
LEAF_RUNTIME_DETAILED_DIAGNOSTICS: bool = False
LEAF_SORT_OUTPUT: bool = False

# --------------------------------------------------------------------------- #
# v12 challenger: SKU total diario + occurrence/store-share allocation
# --------------------------------------------------------------------------- #
# v11.6.1 remains the incumbent. v12 is selected per leaf/target only from
# strictly CLOSED 28-day blocks. No actual from the target block is used.
V12_LEAF_CHALLENGER_ENABLED: bool = True
V12_HISTORY_DAYS: int = 84
V12_MIN_HISTORY_DAYS: int = 56
V12_MIN_VALIDATION_SALES: int = 7
V12_MIN_IMPROVEMENT: float = 0.02
# Historical closed-block scoring pool. v12.9 materializes 5 blocks so Value
# can form 3 walk-forward pseudo-OOS folds. Quantity explicitly slices back to
# the first 4 blocks, preserving the v12.8.6 policy.
V12_BIAS_WEIGHT: float = 0.00  # retained only for backwards config compatibility
V12_SELECTION_BLOCKS: int = 5
V12_SELECTION_MIN_BLOCKS: int = 2
V12_SELECTION_MIN_WINS: int = 2
V12_REQUIRE_RECENT_WIN: bool = True
V12_RECENT_MIN_IMPROVEMENT: float = 0.02
# v12.8: occurrence/share remains shared because quantity and value describe
# the same retail event, but the FINAL magnitude family is target-specific.
# The v12.7 OOS oracle showed that forcing Qty/Valor to switch together can
# improve one target while degrading the other.
V12_REQUIRE_JOINT_TARGET_WIN: bool = False
V12_MAX_BLOCK_DEGRADATION: float = 0.08
V12_MAX_ABS_BIAS_WORSEN: float = 0.10
V12_SECTION_MIN_IMPROVEMENT: float = 0.005
V12_SECTION_MIN_WINS: int = 2
V12_SECTION_REQUIRE_RECENT_WIN: bool = True

# v12.7: selector utility = wMAPE + lambda*|BIAS|, with an explicit
# stability penalty across closed discovery blocks. wMAPE remains the primary
# objective and the section portfolio gate still requires non-negative wMAPE
# improvement; these knobs only make leaf selection less brittle to one lucky
# block while preserving causal validation.
V12_SELECTOR_BIAS_WEIGHT: float = 0.25
V12_SELECTOR_STABILITY_WEIGHT: float = 0.10
V12_SELECTOR_MIN_UTILITY_IMPROVEMENT: float = 0.0025
V12_SELECTOR_MIN_WMAPE_IMPROVEMENT: float = 0.02

# v12.8 causal meta-selector.  One pooled regularized logistic model per
# section/target learns whether v12 beats v11 in the NEXT closed 28-day block.
# Training labels come from the latest closed block and features come only from
# older blocks; the target forecast is scored from its own closed history.
V12_META_SELECTOR_ENABLED: bool = True
V12_META_SELECTOR_THRESHOLD: float = 0.55
V12_META_SELECTOR_THRESHOLDS: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65)
V12_META_SELECTOR_L2: float = 1.00
V12_META_SELECTOR_MAX_ITER: int = 30
V12_META_SELECTOR_MIN_TRAIN_ROWS: int = 500
V12_META_SELECTOR_MIN_FEATURE_BLOCKS: int = 2
V12_META_SELECTOR_RECENCY_WEIGHTS: tuple[float, float, float] = (0.55, 0.30, 0.15)
V12_META_SELECTOR_MAX_RECENT_DEGRADATION: float = 0.08
# v12.8.2: the leaf selector no longer depends on the legacy section gate.
# A temporally older meta-model scores the latest CLOSED validation block;
# that block then selects both the probability threshold and the causal
# portfolio mode (v11_all / v12_all / meta_leaf).  The production meta-model
# is shifted one block forward and never sees target actuals.
V12_META_SELECTOR_ADAPTIVE_POLICY: bool = True
V12_META_SELECTOR_BIAS_GUARD_TOLERANCE: float = 0.03
V12_META_PORTFOLIO_BIAS_GUARD_TOLERANCE: float = 0.03
V12_META_PORTFOLIO_MIN_UTILITY_GAIN: float = 0.0025

# v12.8.3 Value Portfolio Safety. Quantity keeps the v12.8.2 policy unchanged.
# Value applies a second, strictly causal safety layer before an all-portfolio
# switch is authorized.  All checks are calibrated/evaluated on CLOSED blocks.
V12_VALUE_PORTFOLIO_SAFETY_ENABLED: bool = True
V12_VALUE_ALL_MIN_RECENT_CONFIRMATIONS: int = 2
V12_VALUE_ALL_BIAS_COVERAGE_GRID: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65)
V12_VALUE_META_LEAF_MIN_UTILITY_GAIN: float = 0.0025
V12_VALUE_SAFETY_MIN_CLOSED_LEAVES: int = 100

# v12.9.2: walk-forward portfolio selector for Value ($). Quantity remains
# frozen on the v12.8.6 four-block policy.  Value uses three strictly closed
# pseudo-OOS folds and bottom-up volume-weighted portfolio statistics before a
# full-family switch is promoted.
V129_VALUE_WALK_FORWARD_ENABLED: bool = True
V129_VALUE_WF_FOLDS: int = 3
V129_VALUE_WF_MIN_FOLDS: int = 3
V129_VALUE_WF_MIN_WIN_RATE: float = 2.0 / 3.0
V129_VALUE_WF_MIN_MEDIAN_WMAPE_GAIN: float = 0.0025
V129_VALUE_WF_MAX_WORST_WMAPE_DEGRADATION: float = 0.04
V129_VALUE_WF_MAX_ABS_BIAS_WORSEN: float = 0.02
V129_VALUE_WF_MIN_PROMOTION_UTILITY_GAIN: float = 0.0030
V129_VALUE_WF_META_MIN_FOLDS: int = 2
V129_VALUE_WF_META_MARGIN_OVER_ALL: float = 0.0025
V129_QTY_FROZEN_SELECTION_BLOCKS: int = 4

# v12.9.2: selector leaf de Valor basado en expected gain causal.
# Se entrena con pseudo-OOS cerrados y sample_weight proporcional al denominador
# de wMAPE.  Un margen conservador + estabilidad histórica decide cuándo v12
# merece desplazar al incumbente v11.
V1291_VALUE_EXPECTED_GAIN_ENABLED: bool = True
V1291_VALUE_EXPECTED_GAIN_LABEL_BLOCKS: tuple[int, ...] = (1, 2, 3)
V1291_VALUE_EXPECTED_GAIN_FEATURE_BLOCKS: int = 3
V1291_VALUE_EXPECTED_GAIN_L2: float = 2.0
V1291_VALUE_EXPECTED_GAIN_MIN_TRAIN_ROWS: int = 1000
V1291_VALUE_EXPECTED_GAIN_MIN_GAIN: float = 0.0050
V1291_VALUE_EXPECTED_GAIN_MIN_CONFIDENCE: float = 0.10
V1291_VALUE_EXPECTED_GAIN_MAX_GAIN_RANGE: float = 1.00
V1291_VALUE_EXPECTED_GAIN_MAX_RECENT_DEGRADATION: float = 0.08
V1291_VALUE_EXPECTED_GAIN_PORTFOLIO_MIN_EXPECTED_GAIN: float = 0.0025
V1291_VALUE_EXPECTED_GAIN_PORTFOLIO_MAX_SHARE: float = 0.80

# v12.9.2: calibrated expected-impact selector. Raw ridge magnitude is mapped
# through strictly-temporal OOF buckets; large-impact leaves face asymmetric
# downside guards and the final portfolio has a capped volume budget.
V1292_VALUE_EXPECTED_IMPACT_ENABLED: bool = True
V1292_VALUE_MIN_CALIBRATION_ROWS: int = 1000
V1292_VALUE_MIN_CALIBRATED_GAIN: float = 0.0040
V1292_VALUE_MIN_CONFIDENCE: float = 0.15
V1292_VALUE_MAX_BUCKET_LOSS_RATE: float = 0.55
V1292_VALUE_MAX_LEAF_LOSS_RATE: float = 0.67
V1292_VALUE_MIN_P10_GAIN: float = -0.08
V1292_VALUE_TOP10_MIN_GAIN: float = 0.0080
V1292_VALUE_TOP10_MIN_CONFIDENCE: float = 0.25
V1292_VALUE_TOP10_MAX_LEAF_LOSS_RATE: float = 0.50
V1292_VALUE_TOP10_MIN_P10_GAIN: float = -0.04
V1292_VALUE_TOP02_MIN_GAIN: float = 0.0150
V1292_VALUE_TOP02_MIN_CONFIDENCE: float = 0.35
V1292_VALUE_TOP02_MAX_LEAF_LOSS_RATE: float = 0.34
V1292_VALUE_TOP02_MIN_P10_GAIN: float = -0.02
V1292_VALUE_PORTFOLIO_MIN_EXPECTED_GAIN: float = 0.0025
V1292_VALUE_PORTFOLIO_MAX_VOLUME_SHARE: float = 0.45

# v12.9.3: hierarchical, interpretable segment walk-forward selector for Value ($).
# Productive only for the canonical 28-day update cadence. The v12.9.1/2 ridge
# expected-impact layer remains diagnostic and never authorizes production.
V1293_VALUE_SEGMENT_SELECTOR_ENABLED: bool = True
V1293_VALUE_SEGMENT_FOLDS: int = 3
V1293_VALUE_SEGMENT_MAX_VOLUME_SHARE: float = 0.45
V1293_VALUE_SECTION_STRONG_GAIN: float = 0.030
V1293_VALUE_L4_MIN_LEAVES: int = 20
V1293_VALUE_L3_MIN_LEAVES: int = 40
V1293_VALUE_L2_MIN_LEAVES: int = 80
V1293_VALUE_L1_MIN_LEAVES: int = 500

# v12.9.4: freeze Sec.1 and add strict volume-risk control only for Value Sec.23.
V1294_VALUE_SEC23_MAX_VOLUME_SHARE: float = 0.25
V1294_VALUE_SEC23_MAX_SEGMENT_VOLUME_SHARE: float = 0.08
V1294_VALUE_TOP01_PERCENTILE: float = 0.99
V1294_VALUE_TOP05_PERCENTILE: float = 0.95
V1294_VALUE_TOP01_MIN_WEIGHTED_GAIN: float = 0.040
V1294_VALUE_TOP01_MIN_WORST_GAIN: float = 0.020
V1294_VALUE_TOP05_MIN_WEIGHTED_GAIN: float = 0.025
V1294_VALUE_TOP05_MIN_WORST_GAIN: float = 0.010
V1294_VALUE_TOP01_MAX_BIAS_WORSEN: float = 0.000
V1294_VALUE_TOP05_MAX_BIAS_WORSEN: float = 0.010
V1294_VALUE_SEGMENT_DOWNSIDE_WEIGHT: float = 5.0

# v12.9.5: long-horizon segment stress test. Diagnostic only: it must never
# authorize a production v12 switch. We materialize up to 12 historical closed
# 28-day blocks while the productive v12.9.4 selector continues to use its
# original five-block pool / three-fold segment policy.
V1295_STRESS_TEST_ENABLED: bool = True
V1295_STRESS_TEST_BLOCKS: int = 12
V1295_STRESS_TEST_MIN_FOLDS: int = 8
V1295_STRESS_MIN_WIN_RATE: float = 0.75
V1295_STRESS_MIN_MEDIAN_GAIN: float = 0.0
V1295_STRESS_MIN_P25_GAIN: float = 0.0
V1295_STRESS_MIN_P10_GAIN: float = -0.03
V1295_STRESS_MAX_BIAS_P90_WORSEN: float = 0.02
V1295_STRESS_MAX_CONSECUTIVE_LOSSES: int = 2

# v12.9.6: Sec23 Value long-horizon stress-gated production selector.
# Sec1 remains frozen on the v12.9.3/5 policy; Sec23 starts from v11 and only
# promotes sufficiently specific L4 (or evidence-insufficient L3 backoff) segments.
V1296_SEC23_STRESS_GATE_ENABLED: bool = True
V1296_SEC23_MIN_FOLDS: int = 8
V1296_SEC23_MIN_WIN_RATE: float = 0.75
V1296_SEC23_MIN_MEDIAN_GAIN: float = 0.0
V1296_SEC23_MIN_P25_GAIN: float = 0.0
V1296_SEC23_MIN_P10_GAIN: float = -0.01
V1296_SEC23_MAX_BIAS_P90_WORSEN: float = 0.015
V1296_SEC23_MAX_CONSECUTIVE_LOSSES: int = 2
V1296_SEC23_ALLOW_L3_BACKOFF: bool = True
V1296_SEC23_MAX_VOLUME_SHARE: float = 0.15
V1296_SEC23_TOP05_PERCENTILE: float = 0.95
V1296_SEC23_TOP01_PERCENTILE: float = 0.99
V1296_SEC23_HIGH_IMPACT_MIN_P10_GAIN: float = 0.0

# v12.9.7: multi-cutoff temporal robustness replay. Diagnostic only.
# Replays the promoted Sec23 Value policy on historical pseudo-OOS cutoffs,
# training each cutoff strictly on older 28-day blocks. Production remains
# frozen on the v12.9.6 L4 long-horizon stress gate.
V1297_TEMPORAL_REPLAY_ENABLED: bool = True
V1297_TEMPORAL_REPLAY_MAX_CUTOFFS: int = 4
V1297_TEMPORAL_REPLAY_MIN_TRAIN_FOLDS: int = 8
V1297_TEMPORAL_REPLAY_MIN_WIN_RATE: float = 0.75
V1297_TEMPORAL_REPLAY_MIN_MEDIAN_GAIN: float = 0.0
V1297_TEMPORAL_REPLAY_MIN_WORST_GAIN: float = -0.005
V1297_TEMPORAL_REPLAY_MAX_BIAS_WORSEN: float = 0.01

# v12.9.8: causal sparse-demand bias rescue challenger.  Diagnostic only in
# this version: it estimates a conservative multiplicative uplift for leaves
# whose latest CLOSED 28d block had only 7-13 positive-sales days.  The factor
# is learned exclusively from older closed blocks and is scored on current OOS
# only in the acceptance report.  Production forecasts remain v12.9.7.
V1298_SPARSE_BIAS_DIAGNOSTIC_ENABLED: bool = True
V1298_SPARSE_BIAS_MIN_FOLDS: int = 8
V1298_SPARSE_BIAS_BUCKETS: tuple[str, ...] = ("07-09", "10-13")
V1298_SPARSE_BIAS_MIN_MEDIAN_RATIO: float = 1.05
V1298_SPARSE_BIAS_MIN_P25_RATIO: float = 1.00
V1298_SPARSE_BIAS_MIN_WORST_RATIO: float = 0.90
V1298_SPARSE_BIAS_SHRINK: float = 0.50
V1298_SPARSE_BIAS_FACTOR_CLIP: tuple[float, float] = (1.00, 1.15)

# v12.9.9 — Value-only segmented sparse-bias rescue (diagnostic, non-productive).
# The v12.9.7 productive selector remains frozen.  Factors are selected from
# exact positive-day wMAPE on CLOSED blocks and must pass a long-horizon gate.
V1299_VALUE_SPARSE_DIAGNOSTIC_ENABLED: bool = True
V1299_VALUE_SPARSE_BUCKETS: tuple[str, ...] = ("07-09", "10-13")
V1299_VALUE_SPARSE_FACTOR_GRID: tuple[float, ...] = (
    1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15
)
V1299_VALUE_SPARSE_MIN_FOLDS: int = 8
V1299_VALUE_SPARSE_MIN_LEAVES_PER_FOLD: int = 20
V1299_VALUE_SPARSE_MIN_WIN_RATE: float = 0.75
V1299_VALUE_SPARSE_MIN_MEDIAN_GAIN: float = 0.0
V1299_VALUE_SPARSE_MIN_WORST_GAIN: float = -0.005
V1299_VALUE_SPARSE_MAX_BIAS_WORSEN: float = 0.01
V1299_VALUE_SPARSE_REPLAY_CUTOFFS: int = 4

# v12.9.10 — Productive Value sparse-rescue promotion gate.
# The factor/segments still come from v12.9.9 closed-block learning, but a
# section is allowed to write back to valuehat only when the official ACTIVE
# cohort passes an independent causal multi-cutoff replay.  Current OOS actuals
# are never used for promotion.
V12910_VALUE_SPARSE_PROMOTION_ENABLED: bool = True
V12910_VALUE_SPARSE_ACTIVE_MIN_NONZERO_DAYS: int = 7
V12910_VALUE_SPARSE_MIN_REPLAY_CUTOFFS: int = 4
V12910_VALUE_SPARSE_MIN_REPLAY_WIN_RATE: float = 1.00
V12910_VALUE_SPARSE_MIN_REPLAY_MEDIAN_GAIN: float = 0.0025
V12910_VALUE_SPARSE_MIN_REPLAY_WORST_GAIN: float = 0.0
V12910_VALUE_SPARSE_MIN_REPLAY_WEIGHTED_GAIN: float = 0.0025
V12910_VALUE_SPARSE_MAX_REPLAY_BIAS_WORSEN: float = 0.0
V12910_VALUE_SPARSE_MAX_SELECTED_VOLUME_SHARE: float = 0.30


# v12.9.11 — Sec23 Value residual selector-gap challenger (diagnostic only).
# Searches for leaves that the productive v12.9.10 baseline still leaves on v11
# but whose own long closed-block history supports v12.  Selection is strictly
# causal and volume-limited; current OOS is audit-only and never authorizes a
# switch.  v12.9.10 remains the productive forecast policy in this version.
V12911_SEC23_VALUE_RESIDUAL_ENABLED: bool = True
V12911_SEC23_VALUE_MIN_FOLDS: int = 8
V12911_SEC23_VALUE_MIN_WIN_RATE: float = 0.75
V12911_SEC23_VALUE_MIN_MEDIAN_GAIN: float = 0.0
V12911_SEC23_VALUE_MIN_P25_GAIN: float = -0.005
V12911_SEC23_VALUE_MIN_WEIGHTED_GAIN: float = 0.005
V12911_SEC23_VALUE_MIN_RECENT_GAIN: float = 0.0
V12911_SEC23_VALUE_MIN_WORST_GAIN: float = -0.10
V12911_SEC23_VALUE_MAX_BIAS_WORSEN: float = 0.05
V12911_SEC23_VALUE_MAX_VOLUME_SHARE: float = 0.10
V12911_SEC23_VALUE_TOP05_PERCENTILE: float = 0.95
V12911_SEC23_VALUE_TOP01_PERCENTILE: float = 0.99
V12911_SEC23_VALUE_REPLAY_CUTOFFS: int = 4
V12911_SEC23_VALUE_REPLAY_MIN_WIN_RATE: float = 0.75
V12911_SEC23_VALUE_REPLAY_MIN_MEDIAN_GAIN: float = 0.0
V12911_SEC23_VALUE_REPLAY_MIN_WORST_GAIN: float = -0.0025
V12911_SEC23_VALUE_REPLAY_MAX_BIAS_WORSEN: float = 0.01


# v12.9.12 — ultra-stable Sec23 Value residual diagnostic (NO promotion).
# Reuses the v12.9.11 residual candidate metadata but applies a much stricter
# evidence gate before a leaf is considered statistically actionable.  This
# version remains diagnostic-only: productive forecasts stay on v12.9.10 policy.
V12912_ULTRA_STABLE_ENABLED: bool = True
V12912_ULTRA_MIN_WIN_RATE: float = 1.00
V12912_ULTRA_MIN_MEDIAN_GAIN: float = 0.0010
V12912_ULTRA_MIN_P25_GAIN: float = 0.0005
V12912_ULTRA_MIN_WORST_GAIN: float = 0.0
V12912_ULTRA_MIN_WEIGHTED_GAIN: float = 0.0010
V12912_ULTRA_MIN_RECENT_GAIN: float = 0.0
V12912_ULTRA_MAX_BIAS_WORSEN: float = 0.0
V12912_ULTRA_MAX_VOLUME_SHARE: float = 0.03


# SKU-total model: recent 84-day daily level × weekday profile, blended with
# a scaled 364-day path when a valid prior-year reference exists.
V12_SKU_RECENT28_WEIGHT: float = 0.60
V12_SKU_WEEKDAY_FACTOR_CLIP: tuple[float, float] = (0.40, 2.50)
V12_SKU_ANNUAL_LAG_DAYS: int = 364
V12_SKU_ANNUAL_WEIGHT: float = 0.50
V12_SKU_ANNUAL_SCALE_CLIP: tuple[float, float] = (0.50, 2.00)
# v12.4: causal SKU-total ensemble. The previous fixed blend is no longer the
# only SKU magnitude model. Each 28-day target selects among the incumbent SKU
# sum, recent weekday level, recent same-weekday average, lag-28 seasonal naive,
# and the scaled annual blend using only closed blocks before the target origin.
V12_SKU_SELECTION_BLOCKS: int = 3
V12_SKU_SELECTION_MIN_BLOCKS: int = 2
V12_SKU_SELECTION_MIN_POSITIVE_DAYS: int = 7
V12_SKU_RECENT_SCORE_WEIGHT: float = 0.25

# v12.7 calibration formula is retained in v12.8 only as a diagnostic
# challenger.  The factor is causal (closed blocks only), robust, shrunk and
# clipped, but production reverts to the uncalibrated v12.6 SKU-total baseline
# unless V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER is explicitly enabled.
V12_SKU_LEVEL_CALIBRATION_ENABLED: bool = True
# v12.8: calculate the calibrated path as a diagnostic challenger, but do NOT
# replace the v12.6 uncalibrated SKU-total baseline unless explicitly enabled.
V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER: bool = False
V12_SKU_LEVEL_CALIBRATION_RECENT_WEIGHT: float = 0.50
V12_SKU_LEVEL_CALIBRATION_PRIOR_BLOCKS: float = 1.50
V12_SKU_LEVEL_CALIBRATION_CLIP: tuple[float, float] = (0.60, 2.00)
V12_SKU_LEVEL_CALIBRATION_MIN_BLOCKS: int = 2

# Allocation model. v12.5 keeps the v12.3 shared true-hurdle occurrence gate,
# but replaces the noisy weekday store-share with the robust rolling winner:
# 70% of the overall SKU×store share from the last 28 days + 30% from the last
# 84 days. The blend is normalized only inside the productive support.
V12_SHARE_RECENT_DAYS: int = 28
V12_SHARE_STABLE_DAYS: int = 84
V12_SHARE_RECENT_WEIGHT: float = 0.70
V12_ALLOCATION_FULL_WEIGHT_POSITIVE_DAYS: int = 14
V12_OCCURRENCE_GATE: float = 0.10
# Deprecated compatibility knob: kept so existing local configs do not fail.
# It is no longer used by the v12.3 allocation formula.
V12_OCCURRENCE_WEIGHT_FLOOR: float = 0.05

# v12.6: pooled LightGBM at SKU aggregate, SHAPE ONLY.  Hyperparameters and
# gamma are frozen from the 12-block PRE-OOS rolling backtest.  The model is
# fitted separately by section/target on older closed 28-day blocks and its
# output is renormalized to preserve the existing SKU 28-day total exactly.
V12_SKU_SHAPE_LGBM_ENABLED: bool = True
V12_SKU_SHAPE_LGBM_HISTORY_BLOCKS: int = 16
V12_SKU_SHAPE_LGBM_GAMMA: float = 1.00
V12_SKU_SHAPE_LGBM_SMOOTH_FRAC: float = 0.02
V12_SKU_SHAPE_LGBM_Z_CLIP: float = 1.50
V12_SKU_SHAPE_LGBM_ROUNDS: int = 160
V12_SKU_SHAPE_LGBM_SEED: int = 20260831
V12_SKU_SHAPE_LGBM_LEARNING_RATE: float = 0.035
V12_SKU_SHAPE_LGBM_NUM_LEAVES: int = 15
V12_SKU_SHAPE_LGBM_MAX_DEPTH: int = 4
V12_SKU_SHAPE_LGBM_MIN_DATA_IN_LEAF: int = 500
V12_SKU_SHAPE_LGBM_BAGGING_FRACTION: float = 0.85
V12_SKU_SHAPE_LGBM_FEATURE_FRACTION: float = 0.85
V12_SKU_SHAPE_LGBM_L1: float = 0.20
V12_SKU_SHAPE_LGBM_L2: float = 8.0
V12_SKU_SHAPE_LGBM_MAX_BIN: int = 127
V12_SKU_SHAPE_LGBM_TOTAL_TOLERANCE: float = 1e-9

# Métricas oficiales: siempre bottom-up desde SKU+tienda y desde el día 29.
METRICS_START_DAY: int = 29

# Ranking SKU: mínimo de días con actual distinto de cero.
RANKING_SKU_MIN_NONZERO_POINTS: int = 15
OOS_METRIC_MODE: str = "active"  # active | all
OOS_ACTIVE_MIN_NONZERO_DAYS: int = 7
OOS_ZERO_DEMAND_POLICY: str = "separate"
SHOW_WMAPE_ALL: bool = True
SHOW_ZERO_DEMAND_IMPACT: bool = True
DASHBOARD_METRIC_IDENTITY_TOLERANCE: float = 1e-9

FECHAS_TRAIN = (dt.date(2024, 5, 1), dt.date(2025, 10, 26))
FECHAS_TEST = (TEST_START, TEST_END)

# --------------------------------------------------------------------------- #
# Secciones y locales (tablas del cliente)
# --------------------------------------------------------------------------- #
# Las fechas test/forecast dentro de cada sección son referencias legacy para
# compatibilidad. v11.2 deriva OOS/forecast desde last_actual y NO las usa para
# desplazar el OOS operativo.
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
# AR no se fuerza. Cada bloque elige causalmente entre RLS base y RLS+AR
# usando exclusivamente el wMAPE acumulado de bloques anteriores.
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
    """Resolve the production train/OOS/forecast-only windows.

    v11.2 contract:
    - OOS is ALWAYS the latest 28 calendar days with actuals;
    - therefore no actual can exist after OOS;
    - forecast-only starts at ``last_actual + 1`` and lasts 28 days;
    - train is aligned backwards from OOS so every expanding block is exactly
      28 days. Up to 27 earliest calendar days may be discarded to preserve
      the 28x28 contract without moving OOS away from the latest actuals.

    The old configured ``test_start/test_end`` values remain metadata only and
    are not allowed to create an observed tail after OOS.
    """
    cfg = SECCIONES[seccion]
    update_days = int(RLS_BLOCK_DAYS)
    metric_days = int(METRIC_HORIZON_DAYS)

    if first_data is None:
        first_data = FECHAS_TRAIN[0]
    if last_actual is None:
        # Compatibility fallback when no data frame is available. In the real
        # pipeline last_actual is always read from the section data.
        last_actual = cfg.get("test_end") or TEST_END

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
        # ``first_monday`` is kept only for backward compatibility with code
        # that still reads this key. In v11.2 it is the aligned model start and
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
