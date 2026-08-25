#!/usr/bin/env python3
"""
Diagnóstico Fase A v2 — separa ruido de bajo volumen del problema real
========================================================================

Contexto (resultado de Fase A v1, corrido por Willy sobre forecast.parquet):

    n_hojas_totales_con_ventas_oos: 24,082
    wmape_global_bottom_up_oos:     73.4%
    sobreestiman >30%:  3,370 hojas (14.0%) -> explican 21.2% del error absoluto
    subestiman   >30%: 17,341 hojas (72.0%) -> explican 51.3% del error absoluto

Y el top-30 por `bias_leaf` (sesgo relativo) resultó dominado por hojas con
`sum_y` de apenas 1-3 unidades vendidas en los 28 días de OOS (p.ej. vender 1
unidad y pronosticar 336). Eso es matemáticamente correcto pero es RUIDO
estadístico de bajo volumen, no el problema que Willy señaló originalmente
(SKU 587833 / 261298 @ tienda 00001, que sí tienen volumen real y SÍ siguen
sobreestimando: 261298 aparece en el top con sum_y=12, sum_yhat=448).

Este script v2:

  1. Separa las hojas en dos cohortes por volumen real de ventas OOS:
       - "volumen_bajo": sum_y < --min-volume (ruido, difícil de arreglar,
         cuantificado pero no priorizado)
       - "volumen_significativo": sum_y >= --min-volume (el problema real
         y accionable)
  2. Dentro de volumen_significativo, rankea por impacto absoluto Y por sesgo
     relativo, y para esas hojas corre el diagnóstico completo (régimen
     denso/sparse, contrato driver=1, continuidad OOS->forecast_only, alpha
     seleccionado) EN UNA SOLA TABLA exportable (v1 solo exportaba el ranking
     básico; había que volver a correr para ver régimen/driver/continuidad).
  3. Busca explícitamente los 2 casos de referencia de VALIDATION_v9.1.md
     (587833 y 261298 @ tienda 00001) para confirmar si v9.1 los corrigió.
  4. Rollup por tienda/sección: cuánto del wMAPE bottom-up de cada tienda
     viene de la cohorte volumen_significativo que sobreestima.

Uso
---
    python diagnostico_fase_a_v2.py
    python diagnostico_fase_a_v2.py --forecast "D:\\...\\forecast.parquet" --min-volume 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

DEFAULT_FORECAST_PATH = r"D:\Documents\GitHub\tienda-inglesa-v3\data\output\forecast.parquet"
DEFAULT_DENSE_COVERAGE = 0.85
REFERENCE_UIDS = ["1||T:00001||S:587833", "1||T:00001||S:261298"]

_SETTINGS = None
for _cand in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent):
    if (_cand / "settings.py").exists():
        sys.path.insert(0, str(_cand))
        try:
            import settings as _SETTINGS  # type: ignore
        except Exception:
            _SETTINGS = None
        break


def _setting(name: str, default):
    return default if _SETTINGS is None else getattr(_SETTINGS, name, default)


LEAF_COLS = [
    "unique_id", "ds", "period_type", "y", "value", "yhat", "valuehat",
    "yhat_raw", "valuehat_raw",
    "ses_level_y", "ses_alpha_y", "pure_ses_wmape_y",
    "ses_recent28_coverage_y", "ses_regime_anchor_y",
    "ses_sparse_robust_y", "ses_sparse_shock_y",
    "ses_robust_block_median_y", "ses_stability_reference_y",
    "driver_factor_y", "driver_strength_y", "driver_strength_selected_y",
    "parent_model_y", "modelo_seleccionado", "seccion", "store_name",
]


def _leaf_filter() -> pl.Expr:
    uid = pl.col("unique_id").cast(pl.Utf8)
    return uid.str.contains(r"\|\|T:[^|]+") & uid.str.contains(r"\|\|S:[^|]+")


def load(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el parquet en: {path}")
    df = pl.read_parquet(path)
    cols = [c for c in LEAF_COLS if c in df.columns]
    return df.select(cols)


def per_leaf_oos(df: pl.DataFrame) -> pl.DataFrame:
    base = df.filter(
        (pl.col("period_type") == "out_sample")
        & pl.col("y").is_not_null() & pl.col("yhat").is_not_null()
        & pl.col("y").is_finite() & pl.col("yhat").is_finite()
    )
    meta_cols = [c for c in ("seccion", "store_name") if c in df.columns]
    agg = [
        pl.len().alias("n_dias"),
        pl.col("y").sum().alias("sum_y"),
        pl.col("yhat").sum().alias("sum_yhat"),
        (pl.col("y") - pl.col("yhat")).abs().sum().alias("sum_abs_err"),
        (pl.col("yhat") - pl.col("y")).sum().alias("sum_signed_err"),
    ] + [pl.col(c).first().alias(c) for c in meta_cols]
    per_leaf = (
        base.group_by("unique_id").agg(agg)
        .filter(pl.col("sum_y") > 0)
        .with_columns(
            (pl.col("sum_abs_err") / pl.col("sum_y")).alias("wmape_leaf"),
            (pl.col("sum_signed_err") / pl.col("sum_y")).alias("bias_leaf"),
            (pl.col("sum_yhat") / pl.col("sum_y")).alias("overestimation_ratio"),
        )
    )
    return per_leaf


def cohort_summary(per_leaf: pl.DataFrame, min_volume: float) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    low = per_leaf.filter(pl.col("sum_y") < min_volume)
    sig = per_leaf.filter(pl.col("sum_y") >= min_volume)
    total_abs_err = per_leaf["sum_abs_err"].sum()

    def _stats(sub: pl.DataFrame, label: str) -> dict:
        n = sub.height
        ae = sub["sum_abs_err"].sum()
        return {
            "cohorte": label,
            "n_hojas": n,
            "pct_hojas": (n / per_leaf.height * 100) if per_leaf.height else None,
            "sum_abs_err": float(ae),
            "pct_error_abs_total": float(ae / total_abs_err * 100) if total_abs_err else None,
            "bias_leaf_mediana": float(sub["bias_leaf"].median()) if n else None,
            "wmape_leaf_mediana": float(sub["wmape_leaf"].median()) if n else None,
            "n_sobreestiman_gt_30pct": int((sub["bias_leaf"] > 0.30).sum()) if n else 0,
        }

    summary = {
        "min_volume_umbral_unidades_oos": min_volume,
        "volumen_bajo": _stats(low, "volumen_bajo (ruido)"),
        "volumen_significativo": _stats(sig, "volumen_significativo (accionable)"),
    }
    return low, sig, summary


def enrich_diagnostics(df: pl.DataFrame, uids: list[str], dense_coverage: float) -> pl.DataFrame:
    """Para una lista de unique_id, arma UNA tabla con régimen, contrato driver,
    continuidad OOS->forecast_only y alpha — todo en un solo export."""
    sub = df.filter(pl.col("unique_id").is_in(uids))

    has_cov = "ses_recent28_coverage_y" in df.columns
    has_drv = "driver_factor_y" in df.columns
    has_lvl = "ses_level_y" in df.columns
    has_alpha = "ses_alpha_y" in df.columns
    has_pure = "pure_ses_wmape_y" in df.columns

    oos = sub.filter(pl.col("period_type") == "out_sample")
    fo = sub.filter(pl.col("period_type") == "forecast_only")

    agg_exprs = [pl.col("yhat").mean().alias("mean_yhat_oos")]
    if has_cov:
        agg_exprs.append(pl.col("ses_recent28_coverage_y").mean().alias("coverage_oos"))
    if has_drv:
        agg_exprs += [
            pl.col("driver_factor_y").mean().alias("mean_driver_factor_oos"),
        ]
    if has_lvl:
        agg_exprs.append(pl.col("ses_level_y").mean().alias("mean_ses_level_oos"))
    if has_alpha:
        agg_exprs.append(pl.col("ses_alpha_y").first().alias("ses_alpha_y"))
    if has_pure:
        agg_exprs.append(pl.col("pure_ses_wmape_y").first().alias("pure_ses_wmape_y"))
    if "modelo_seleccionado" in df.columns:
        agg_exprs.append(pl.col("modelo_seleccionado").first().alias("modelo_seleccionado"))

    oos_stats = oos.group_by("unique_id").agg(agg_exprs)

    fo_agg = [pl.col("yhat").mean().alias("mean_yhat_forecast_only")]
    if has_drv:
        fo_agg.append(pl.col("driver_factor_y").mean().alias("mean_driver_factor_forecast_only"))
    if has_lvl:
        fo_agg.append(pl.col("ses_level_y").mean().alias("mean_ses_level_forecast_only"))
    fo_stats = (
        fo.group_by("unique_id").agg(fo_agg)
        if fo.height else
        pl.DataFrame(schema={"unique_id": pl.Utf8, **{c.output_name: pl.Float64 for c in fo_agg if hasattr(c, "output_name")}})
    )

    out = oos_stats.join(fo_stats, on="unique_id", how="left")

    if has_cov:
        out = out.with_columns(
            pl.when(pl.col("coverage_oos") >= dense_coverage)
            .then(pl.lit("dense")).otherwise(pl.lit("sparse")).alias("regimen")
        )
    if has_drv:
        out = out.with_columns(
            (pl.col("mean_driver_factor_oos") - 1.0).abs().alias("desvio_contrato_driver_oos")
        )
    if "mean_yhat_forecast_only" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("mean_yhat_oos") > 0)
            .then(pl.col("mean_yhat_forecast_only") / pl.col("mean_yhat_oos"))
            .otherwise(None)
            .alias("ratio_forecast_only_vs_oos")
        )
    return out


def check_reference_leaves(df: pl.DataFrame, per_leaf: pl.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("REVISIÓN CASOS DE REFERENCIA (VALIDATION_v9.1.md: 587833 y 261298 @ tienda 00001)")
    print("=" * 78)
    for uid in REFERENCE_UIDS:
        row = per_leaf.filter(pl.col("unique_id") == uid)
        if row.height == 0:
            print(f"  {uid}: NO tiene ventas OOS>0 registradas (o no está en el parquet).")
            continue
        r = row.to_dicts()[0]
        print(
            f"  {uid}: sum_y={r['sum_y']:.1f}  sum_yhat={r['sum_yhat']:.1f}  "
            f"wmape_leaf={r['wmape_leaf']:.2f}  bias_leaf={r['bias_leaf']:.2f}  "
            f"overestimation_ratio={r['overestimation_ratio']:.2f}x"
        )
    print(
        "\n  Si bias_leaf sigue siendo positivo y grande, el guard sparse de v9.1 "
        "no resolvió (o resolvió solo parcialmente) el caso que originó la versión."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", default=DEFAULT_FORECAST_PATH)
    ap.add_argument("--min-volume", type=float, default=15.0,
                     help="Umbral de sum_y (unidades vendidas en OOS) para separar ruido de bajo volumen del problema accionable.")
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args()

    path = Path(args.forecast)
    print(f"Leyendo: {path}")
    df = load(path).filter(_leaf_filter())
    print(f"Filas hoja SKU+tienda cargadas: {df.height:,}")

    per_leaf = per_leaf_oos(df)
    dense_coverage = float(_setting("LEAF_REGIME_DENSE_COVERAGE", DEFAULT_DENSE_COVERAGE))

    # ── Casos de referencia primero (siempre, sin importar volumen) ────────
    check_reference_leaves(df, per_leaf)

    # ── Split por volumen ────────────────────────────────────────────────────
    low, sig, cohort_json = cohort_summary(per_leaf, args.min_volume)
    print("\n" + "=" * 78)
    print(f"SPLIT POR VOLUMEN (umbral sum_y >= {args.min_volume} unidades en OOS)")
    print("=" * 78)
    print(json.dumps(cohort_json, ensure_ascii=False, indent=2))

    if sig.height == 0:
        print("\n(No hay hojas en la cohorte de volumen significativo con este umbral; prueba --min-volume más bajo.)")
        return 0

    # ── Top ofensoras dentro de volumen_significativo ───────────────────────
    top_impact = sig.sort("sum_abs_err", descending=True).head(args.top_n)
    top_bias = sig.filter(pl.col("bias_leaf") > 0).sort("bias_leaf", descending=True).head(args.top_n)
    offender_uids = list(set(top_impact["unique_id"].to_list()) | set(top_bias["unique_id"].to_list()) | set(REFERENCE_UIDS))

    print(f"\n--- TOP {args.top_n} volumen_significativo POR IMPACTO ABSOLUTO ---")
    with pl.Config(tbl_cols=-1, tbl_rows=args.top_n, fmt_str_lengths=50):
        print(top_impact.select(
            "unique_id", "sum_y", "sum_yhat", "sum_abs_err", "wmape_leaf", "bias_leaf"
        ))

    # ── Diagnóstico enriquecido (régimen + driver + continuidad + alpha) ───
    enriched = enrich_diagnostics(df, offender_uids, dense_coverage)
    full = per_leaf.filter(pl.col("unique_id").is_in(offender_uids)).join(
        enriched, on="unique_id", how="left"
    ).sort("bias_leaf", descending=True)

    print("\n" + "=" * 78)
    print(f"DIAGNÓSTICO COMPLETO — {full.height} hojas ofensoras (impacto + sesgo + referencias)")
    print("  columnas: régimen | desvío contrato driver=1 | continuidad forecast_only/oos | alpha")
    print("=" * 78)
    with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=50):
        print(full)

    if "regimen" in full.columns:
        print("\n--- Distribución de régimen entre ofensoras de volumen significativo ---")
        print(full.group_by("regimen").agg(
            pl.len().alias("n"),
            pl.col("bias_leaf").mean().alias("bias_leaf_medio"),
        ))

    if "desvio_contrato_driver_oos" in full.columns:
        viol = full.filter(pl.col("desvio_contrato_driver_oos") > 0.02)
        print(f"\n--- Violaciones de contrato media(driver_factor)=1 en OOS: {viol.height}/{full.height} ---")

    # ── Rollup por tienda/sección ────────────────────────────────────────────
    if "seccion" in per_leaf.columns:
        rollup = (
            per_leaf.group_by("seccion")
            .agg(
                pl.len().alias("n_hojas"),
                pl.col("sum_abs_err").sum().alias("sum_abs_err_total"),
                pl.col("sum_y").sum().alias("sum_y_total"),
            )
            .with_columns((pl.col("sum_abs_err_total") / pl.col("sum_y_total")).alias("wmape_seccion"))
            .sort("wmape_seccion", descending=True)
        )
        print("\n" + "=" * 78)
        print("wMAPE BOTTOM-UP POR SECCIÓN (todas las hojas, para contexto)")
        print("=" * 78)
        print(rollup)

    # ── Export ────────────────────────────────────────────────────────────────
    out_dir = path.parent
    try:
        full.write_csv(out_dir / "diagnostico_fase_a_v2_ofensoras_enriquecido.csv")
        with open(out_dir / "diagnostico_fase_a_v2_cohortes.json", "w", encoding="utf-8") as f:
            json.dump(cohort_json, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nExportado: {out_dir / 'diagnostico_fase_a_v2_ofensoras_enriquecido.csv'}")
        print(f"Exportado: {out_dir / 'diagnostico_fase_a_v2_cohortes.json'}")
    except Exception as e:
        print(f"\n(no se pudo exportar: {e})")

    print("\nListo. Comparte el CSV enriquecido + el JSON de cohortes para pasar a Fase B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
