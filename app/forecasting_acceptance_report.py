"""Executive/statistical acceptance report for the coherent v13 production model.

The report is intentionally compact.  It audits the single explainable model
family and the contractual metrics; it does not contain experimental selectors,
challengers or tuning layers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import settings
from app.dashboard_artifacts import artifacts_dir
from app.forecasting.validate_v13 import validate as validate_model


def _fmt_pct(x: object) -> str:
    try:
        return f"{100.0 * float(x):8.2f}%"
    except (TypeError, ValueError):
        return "     N/A"


def _metrics_path(forecast: Path, in_sample: bool = False) -> Path:
    name = "metrics_in_sample.parquet" if in_sample else "metrics.parquet"
    return artifacts_dir(forecast) / name


def _print_section_metrics(path: Path, title: str) -> None:
    print(f"\n=== {title} ===")
    if not path.exists():
        print(f"No existe {path}; ejecute `uv run python -m app.dashboard_artifacts --forecast ...`")
        return
    m = pl.read_parquet(path)
    if m.height == 0:
        print("Sin métricas")
        return
    sec = m.filter(~pl.col("unique_id").str.contains(r"\|\|", literal=False)).sort(["unidad", "unique_id"])
    for row in sec.iter_rows(named=True):
        print(
            f"{str(row.get('unidad')):<10} sec={str(row.get('unique_id')):>3} | "
            f"wMAPE={_fmt_pct(row.get('wmape'))} | BIAS={_fmt_pct(row.get('bias'))} | "
            f"acid wMAPE={_fmt_pct(row.get('wmape_all_points'))} | acid BIAS={_fmt_pct(row.get('bias_all_points'))} | "
            f"active={int(row.get('n_leaf_active') or 0):5d} sparse={int(row.get('n_leaf_sparse') or 0):5d} zero={int(row.get('n_leaf_zero') or 0):5d}"
        )


def _leaf_summary(forecast: Path) -> None:
    print("\n=== 3) COHERENCIA DE MODELO SKU+TIENDA ===")
    lf = pl.scan_parquet(forecast)
    cols = set(lf.collect_schema().names())
    required = {
        "unique_id", "period_type", "parent_model_y", "parent_model_value",
        "ses_alpha_y", "ses_alpha_value", "initial_level_y", "initial_level_value",
        "warmup_positive_days", "modelo_seleccionado", "rls_metric_eligible",
    }
    if not required.issubset(cols):
        print(f"Faltan columnas: {sorted(required - cols)}")
        return
    leaf = lf.filter(pl.col("unique_id").str.contains(r"\|\|T:[^|]+\|\|S:[^|]+"))
    base = leaf.filter(pl.col("period_type") == "out_sample")
    stats = base.select(
        pl.col("unique_id").n_unique().alias("leaves"),
        pl.col("modelo_seleccionado").n_unique().alias("model_labels"),
        pl.col("ses_alpha_y").n_unique().alias("alpha_y_values"),
        pl.col("ses_alpha_value").n_unique().alias("alpha_value_values"),
    ).collect()
    print(
        f"hojas={int(stats['leaves'][0])} | familia única=SES+RLS | "
        f"alphas Qty={int(stats['alpha_y_values'][0])} Valor={int(stats['alpha_value_values'][0])}"
    )

    for target, col in (("Unidades", "parent_model_y"), ("Valor ($)", "parent_model_value")):
        dist = (
            base.group_by(col).agg(pl.col("unique_id").n_unique().alias("leaves"))
            .sort("leaves", descending=True).collect()
        )
        txt = ", ".join(f"{r[col]}={int(r['leaves'])}" for r in dist.iter_rows(named=True))
        print(f"parent RLS {target}: {txt}")

    alpha = base.select(
        pl.col("ses_alpha_y").median().alias("alpha_y_med"),
        pl.col("ses_alpha_value").median().alias("alpha_v_med"),
        pl.col("warmup_positive_days").median().alias("warmup_pos_med"),
    ).collect()
    print(
        f"alpha mediano Qty={float(alpha['alpha_y_med'][0] or 0):.3f} | "
        f"Valor={float(alpha['alpha_v_med'][0] or 0):.3f} | "
        f"días positivos warm-up mediana={float(alpha['warmup_pos_med'][0] or 0):.1f}"
    )


def _period_contract(forecast: Path) -> None:
    print("\n=== 4) MISMA FAMILIA EN IN-SAMPLE / OOS / FORECAST-ONLY ===")
    lf = pl.scan_parquet(forecast).filter(
        pl.col("unique_id").str.contains(r"\|\|T:[^|]+\|\|S:[^|]+")
    )
    x = (
        lf.group_by("period_type")
        .agg(
            pl.len().alias("rows"),
            pl.col("unique_id").n_unique().alias("leaves"),
            pl.col("modelo_seleccionado").n_unique().alias("model_labels"),
            pl.col("ds").n_unique().alias("calendar_days"),
        )
        .sort("period_type")
        .collect()
    )
    for r in x.iter_rows(named=True):
        print(
            f"{r['period_type']:<14} rows={int(r['rows']):>9,} leaves={int(r['leaves']):>6,} "
            f"calendar-days={int(r['calendar_days']):>4} family=SES+RLS"
        )
    print("Regla: cambia el origen/información disponible; no cambia la composición del modelo.")


def _schema_hygiene(forecast: Path) -> None:
    print("\n=== 5) HIGIENE / TRAZABILIDAD ===")
    cols = pl.scan_parquet(forecast).collect_schema().names()
    legacy_tokens = ("v11_", "v12_", "v129", "lgbm", "occ_share", "sparse_rescue", "meta_selector")
    legacy = [c for c in cols if any(t in c.lower() for t in legacy_tokens)]
    print(f"APP_VERSION={settings.APP_VERSION} | columns={len(cols)} | legacy-model-columns={len(legacy)}")
    print("Métrica oficial: y!=0. Métricas all-points: solo diagnóstico, no ranking/selección.")
    print("Optimización estadística experimental: PAUSADA.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aceptación estadística v13")
    parser.add_argument("--forecast", default=str(settings.FORECAST_PATH))
    args = parser.parse_args(argv)
    forecast = Path(args.forecast)
    if not forecast.exists():
        print(f"No existe forecast: {forecast}")
        return 1

    print(f"ACEPTACIÓN ESTADÍSTICA | APP_VERSION={settings.APP_VERSION}")
    print(f"Forecast: {forecast}")
    print(f"Bloque de actualización: {int(settings.RLS_BLOCK_DAYS)} días | OOS: {int(settings.METRIC_HORIZON_DAYS)} días | forecast-only: {int(settings.METRIC_HORIZON_DAYS)} días")
    print("Modelo productivo: RLS sección/tienda + SES SKU+tienda (mediana positiva inicial) + parent RLS elegido causalmente.")

    errors = validate_model(forecast)
    print("\n=== 1) CONTRATO ESTRUCTURAL ===")
    if errors:
        print("ERROR")
        for e in errors:
            print(f" - {e}")
    else:
        print("OK")

    _print_section_metrics(_metrics_path(forecast, False), "2A) MÉTRICAS OOS BOTTOM-UP (oficial + prueba ácida)")
    _print_section_metrics(_metrics_path(forecast, True), "2B) MÉTRICAS IN-SAMPLE BOTTOM-UP (oficial + prueba ácida)")
    _leaf_summary(forecast)
    _period_contract(forecast)
    _schema_hygiene(forecast)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
