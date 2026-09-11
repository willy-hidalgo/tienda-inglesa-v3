"""Fast pre-forecast contract gate for v13 production invariants."""
from __future__ import annotations
from pathlib import Path
import sys
import settings

ROOT = Path(__file__).resolve().parents[2]

def check() -> list[str]:
    errors=[]
    if tuple(settings.FOCUS_SECTIONS)!=('1','23'): errors.append('secciones foco deben ser exactamente 1 y 23')
    if tuple(settings.UPDATE_BLOCK_OPTIONS)!=(1,7,14,28): errors.append('UPDATE_BLOCK_OPTIONS cambiado')
    if settings.RLS_BLOCK_DAYS!=28 or settings.METRIC_HORIZON_DAYS!=28: errors.append('contrato 28d cambiado')
    if tuple(settings.LEAF_SES_ALPHA_CANDIDATES)!=(0.005,0.01,0.02,0.05,0.10,0.20,0.40,0.60,0.70,0.80): errors.append('alphas productivos cambiados')
    if tuple(settings.RLS_FORGETTING_FACTOR_CANDIDATES)!=(0.970,0.985,0.995): errors.append('lambdas productivos cambiados')
    if settings.RLS_DRIVER_GROUP_EXCLUSIONS!={}: errors.append('exclusiones drivers productivas reactivadas')
    if settings.STAT_OPTIMIZATION_PROMOTE_AUTOMATICALLY: errors.append('promoción automática activada')
    if not settings.LEAF_YOY_SEASONAL_ENABLED: errors.append('factor YoY leaf productivo desactivado')
    if settings.LEAF_YOY_SEASONAL_MIN_POSITIVE_DAYS != 7: errors.append('mínimo soporte YoY cambiado')
    if settings.LEAF_YOY_SEASONAL_FULL_RELIABILITY_DAYS != 14: errors.append('confiabilidad YoY cambiada')
    if (settings.LEAF_YOY_SEASONAL_FACTOR_MIN, settings.LEAF_YOY_SEASONAL_FACTOR_MAX) != (0.50, 1.50): errors.append('guard YoY leaf cambiado')
    runner=(ROOT/'app/forecasting/runner.py').read_text(encoding='utf-8')
    leaf=(ROOT/'app/forecasting/leaf_ses_rls.py').read_text(encoding='utf-8')
    cats=(ROOT/'app/categories_selector.py').read_text(encoding='utf-8')
    dash=(ROOT/'app/dashboard.py').read_text(encoding='utf-8')
    settings_text=(ROOT/'settings.py').read_text(encoding='utf-8')
    if 'TI_ENABLE_EXCEPTION_ROUTING' in settings_text: errors.append('exception routing productivo debe estar eliminado en v13.3.3')
    exception_lab=(ROOT/'app/forecasting/exception_lab.py').read_text(encoding='utf-8') if (ROOT/'app/forecasting/exception_lab.py').exists() else ''
    if 'fixed_oos_start' in runner or 'fixed_oos_boundary' in runner: errors.append('OOS global fixed-origin viola recurrencia por bloque')
    if 'history_mask = source_period[s:e] == "in_sample"' not in runner: errors.append('RLS OOS podría participar en selección')
    if 'if pc == 0:  # only closed in-sample history selects alpha' not in leaf: errors.append('SES OOS podría participar en selección')
    if 'oos_forecast_state_y' in leaf: errors.append('SES OOS congelado globalmente; debe recurrir por bloque cerrado')
    if '_attach_leaf_yoy_monthly_seasonality' not in leaf or 'leaf_total_factor_y' not in leaf: errors.append('transición YoY leaf causal ausente')
    if 'select_best_skus' in cats or 'best_skus' in cats: errors.append('muestreo/whitelist SKU reintroducido')
    if '.head(' in cats or '.sample(' in cats: errors.append('top/sample SKU reintroducido')
    if 'mask = np.isfinite(y) & np.isfinite(yhat) & (y != 0)' not in settings_text: errors.append('soporte oficial wMAPE/BIAS y!=0 cambiado')
    if 'def _dashboard_dataframe' not in dash or dash.count('st.dataframe(')!=1: errors.append('formatter global dashboard roto')
    if exception_lab:
        for productive in (runner, leaf, (ROOT/'app/forecasting/pipeline.py').read_text(encoding='utf-8'), (ROOT/'app/forecasts.py').read_text(encoding='utf-8')):
            if 'exception_lab' in productive: errors.append('laboratorio de excepciones conectado al forecast productivo')
        if 'DEFAULT_MAX_SERIES = 250' not in exception_lab or 'HARD_MAX_SERIES = 2000' not in exception_lab: errors.append('cap de velocidad del laboratorio de excepciones cambiado')
        if 'no automatic routing/promotion' not in exception_lab.lower(): errors.append('contrato diagnostic-only del laboratorio no explícito')
    removed = [
        'app/forecasting/exception_routing.py',
        'app/forecasting/exception_routing_gate.py',
        'app/forecasting/freeze_exception_routing.py',
        'app/forecasting/exception_routing_candidate_registry.csv',
    ]
    for rel in removed:
        if (ROOT / rel).exists():
            errors.append(f'módulo/registry de exception routing productivo debe estar eliminado: {rel}')
    if 'apply_exception_routing' in runner:
        errors.append('runner no debe invocar exception routing en v13.3.3')
    return errors

def main() -> int:
    errors=check()
    if errors:
        print(f'CONTRACT GATE v{settings.APP_VERSION}: ERROR')
        for e in errors: print(f'- {e}')
        return 1
    print(f'CONTRACT GATE v{settings.APP_VERSION}: OK')
    print('- arquitectura productiva SES leaf + RLS parent + YoY leaf causal')
    print('- 28d y cadencias 1/7/14/28 preservadas')
    print('- OOS no hace tuning; bloque cerrado sí puede actualizar estado siguiente')
    print('- todos los SKU+Tienda preservados; sin select_best_skus')
    print('- formatter global dashboard preservado')
    print('- exception routing productivo eliminado; ACTIVE routes=0')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
