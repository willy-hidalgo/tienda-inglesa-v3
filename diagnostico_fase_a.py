#!/usr/bin/env python3
"""
Diagnóstico Fase A — v9.1 — Sobreestimación OOS en hojas SKU+tienda
=====================================================================

Objetivo
--------
Willy reporta que existen combinaciones SKU+tienda que sobreestiman en el
periodo OOS (elevando el nivel sin razón válida), lo que a su vez eleva el
wMAPE bottom-up de tienda/sección, y sospecha que el forecast-only hereda el
mismo sesgo. VALIDATION_v9.1.md documenta que la versión v9.1 ya intentó
corregir exactamente este síntoma (SKU 587833 y 261298 @ tienda 00001) vía un
guard de régimen sparse/shock-heavy, pero el problema persiste.

Este script NO modifica nada. Solo lee `forecast.parquet` y produce evidencia
para decidir cuál de estas 4 hipótesis está pasando:

  H1. Las hojas afectadas son DENSAS (coverage >= LEAF_REGIME_DENSE_COVERAGE),
      régimen que v9.1 no tocó -> el guard sparse nunca podía arreglarlas.
  H2. Las hojas afectadas son SPARSE pero el guard no dispara o no alcanza
      (ratio/tolerancia mal calibrados para estos casos concretos).
  H3. El nivel SES (`ses_level_y`) está correcto, pero el DRIVER RLS
      (`driver_factor_y`) infla el forecast (violación del contrato
      "media(driver_factor)=1 por bloque").
  H4. El forecast_only simplemente hereda el estado (nivel + driver) del
      último bloque OOS sin ningún mecanismo propio de corrección, por lo que
      el sesgo se propaga sin cambios.

Uso
---
    python diagnostico_fase_a.py
    python diagnostico_fase_a.py --forecast "D:\\ruta\\forecast.parquet"
    python diagnostico_fase_a.py --top-n 40 --bias-threshold 0.30
    python diagnostico_fase_a.py --uid "1||T:00001||S:587833"   # foco en 1 hoja

Salida
------
Imprime en consola las 6 secciones de diagnóstico (resumen + tablas) y además
escribe, junto al parquet de entrada:
    diagnostico_fase_a_top_offenders.csv   -> tabla completa de hojas top-N
    diagnostico_fase_a_summary.json        -> resumen numérico agregable

Comparte ambos archivos (o el output de consola) para pasar a Fase B.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

# ── bootstrap: intenta usar settings.py del proyecto si está disponible ─────
DEFAULT_FORECAST_PATH = r"D:\Documents\GitHub\tienda-inglesa-v3\data\output\forecast.parquet"
DEFAULT_DENSE_COVERAGE = 0.85
DEFAULT_SPARSE_SHOCK_RATIO = 1.35
DEFAULT_SPARSE_STABILITY_RATIO = 1.25

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
    if _SETTINGS is None:
        return default
    return getattr(_SETTINGS, name, default)


# ── columnas esperadas (ver app/forecasting/diagnose_leaf.py FIELDS) ────────
LEAF_COLS = [
    "unique_id", "ds", "period_type", "y", "value", "yhat", "valuehat",
    "yhat_raw", "valuehat_raw",
    "ses_level_y", "ses_alpha_y", "ses_alpha_value",
    "recent28_mean_y", "ses_recent28_y", "ses_recent14_y",
    "ses_recent28_coverage_y", "ses_recent28_coverage_value",
    "ses_regime_anchor_y", "ses_sparse_robust_y", "ses_sparse_shock_y",
    "ses_robust_block_median_y", "ses_stability_reference_y",
    "ses_stability_guard_y", "pure_ses_wmape_y",
    "driver_factor_y", "driver_strength_y", "driver_strength_selected_y",
    "driver_direction_guard_y", "parent_model_y",
    "modelo_seleccionado",
]


def _leaf_filter() -> pl.Expr:
    uid = pl.col("unique_id").cast(pl.Utf8)
    return uid.str.contains(r"\|\|T:[^|]+") & uid.str.contains(r"\|\|S:[^|]+")


def load(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el parquet en: {path}\n"
            "Usa --forecast <ruta> para indicar la ubicación correcta."
        )
    df = pl.read_parquet(path)
    missing_required = [c for c in ("unique_id", "ds", "period_type", "y", "yhat") if c not in df.columns]
    if missing_required:
        raise ValueError(f"Faltan columnas obligatorias en el parquet: {missing_required}")
    cols = [c for c in LEAF_COLS if c in df.columns]
    return df.select(cols)


def section1_ranking(leaves_oos: pl.DataFrame, top_n: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ranking de hojas por sesgo relativo (bias_leaf) y por impacto absoluto."""
    base = leaves_oos.filter(
        pl.col("y").is_not_null() & pl.col("yhat").is_not_null()
        & pl.col("y").is_finite() & pl.col("yhat").is_finite()
    )
    per_leaf = (
        base.group_by("unique_id")
        .agg(
            pl.len().alias("n_dias"),
            pl.col("y").sum().alias("sum_y"),
            pl.col("yhat").sum().alias("sum_yhat"),
            pl.col("y").mean().alias("mean_y"),
            pl.col("yhat").mean().alias("mean_yhat"),
            (pl.col("y") - pl.col("yhat")).abs().sum().alias("sum_abs_err"),
            (pl.col("yhat") - pl.col("y")).sum().alias("sum_signed_err"),
        )
        .filter(pl.col("sum_y") > 0)
        .with_columns(
            (pl.col("sum_abs_err") / pl.col("sum_y")).alias("wmape_leaf"),
            (pl.col("sum_signed_err") / pl.col("sum_y")).alias("bias_leaf"),
            (pl.col("sum_yhat") / pl.col("sum_y")).alias("overestimation_ratio"),
        )
        .sort("bias_leaf", descending=True)
    )
    top_bias = per_leaf.head(top_n)
    top_impact = per_leaf.sort("sum_abs_err", descending=True).head(top_n)
    return per_leaf, top_bias.join(
        top_impact.select("unique_id"), on="unique_id", how="outer_coalesce"
    ) if False else top_bias  # (impact table returned separately más abajo)


def section2_impact(per_leaf: pl.DataFrame, bias_threshold: float) -> dict:
    total_abs_err = per_leaf["sum_abs_err"].sum()
    total_y = per_leaf["sum_y"].sum()
    overest = per_leaf.filter(pl.col("bias_leaf") > bias_threshold)
    underest = per_leaf.filter(pl.col("bias_leaf") < -bias_threshold)
    n_total = per_leaf.height
    global_wmape = float(total_abs_err / total_y) if total_y else None
    return {
        "n_hojas_totales_con_ventas_oos": n_total,
        "wmape_global_bottom_up_oos": global_wmape,
        "n_hojas_sobreestiman_gt_%d_pct" % int(bias_threshold * 100): overest.height,
        "pct_hojas_sobreestiman": (overest.height / n_total * 100) if n_total else None,
        "pct_error_abs_explicado_por_sobreestimadoras": (
            float(overest["sum_abs_err"].sum() / total_abs_err * 100) if total_abs_err else None
        ),
        "n_hojas_subestiman_gt_%d_pct" % int(bias_threshold * 100): underest.height,
        "pct_error_abs_explicado_por_subestimadoras": (
            float(underest["sum_abs_err"].sum() / total_abs_err * 100) if total_abs_err else None
        ),
    }


def section3_regime(df: pl.DataFrame, per_leaf: pl.DataFrame, dense_coverage: float) -> pl.DataFrame:
    if "ses_recent28_coverage_y" not in df.columns:
        return pl.DataFrame()
    cov = (
        df.filter(pl.col("period_type") == "out_sample")
        .group_by("unique_id")
        .agg(pl.col("ses_recent28_coverage_y").mean().alias("coverage_oos_mean"))
        .with_columns(
            pl.when(pl.col("coverage_oos_mean") >= dense_coverage)
            .then(pl.lit("dense"))
            .otherwise(pl.lit("sparse"))
            .alias("regimen")
        )
    )
    joined = per_leaf.join(cov, on="unique_id", how="left")
    return (
        joined.group_by("regimen")
        .agg(
            pl.len().alias("n_hojas"),
            pl.col("bias_leaf").mean().alias("bias_leaf_medio"),
            pl.col("wmape_leaf").mean().alias("wmape_leaf_medio"),
            (pl.col("bias_leaf") > 0.30).sum().alias("n_sobreestiman_fuerte"),
        )
        .sort("regimen")
    )


def section4_level_vs_driver(df: pl.DataFrame, top_bias_uids: list[str]) -> pl.DataFrame:
    if "driver_factor_y" not in df.columns:
        return pl.DataFrame()
    sub = df.filter(
        pl.col("unique_id").is_in(top_bias_uids)
        & pl.col("period_type").is_in(["out_sample", "forecast_only"])
    )
    per_block = (
        sub.group_by(["unique_id", "period_type"])
        .agg(
            pl.col("driver_factor_y").mean().alias("mean_driver_factor"),
            pl.col("driver_factor_y").min().alias("min_driver_factor"),
            pl.col("driver_factor_y").max().alias("max_driver_factor"),
            (pl.col("ses_level_y").mean() if "ses_level_y" in df.columns else pl.lit(None)).alias("mean_ses_level"),
            pl.col("yhat").mean().alias("mean_yhat"),
        )
        .with_columns((pl.col("mean_driver_factor") - 1.0).abs().alias("desvio_contrato_media_1"))
        .sort(["unique_id", "period_type"])
    )
    return per_block


def section5_continuity(df: pl.DataFrame, top_bias_uids: list[str]) -> pl.DataFrame:
    sub = df.filter(
        pl.col("unique_id").is_in(top_bias_uids)
        & pl.col("period_type").is_in(["out_sample", "forecast_only"])
    )
    agg_cols = [pl.col("yhat").mean().alias("mean_yhat")]
    if "ses_level_y" in df.columns:
        agg_cols.append(pl.col("ses_level_y").mean().alias("mean_ses_level"))
    if "driver_factor_y" in df.columns:
        agg_cols.append(pl.col("driver_factor_y").mean().alias("mean_driver_factor"))
    per_period = (
        sub.group_by(["unique_id", "period_type"]).agg(agg_cols)
    )
    pivot = per_period.pivot(
        values=[c for c in ("mean_yhat", "mean_ses_level", "mean_driver_factor") if c in per_period.columns],
        index="unique_id",
        on="period_type",
    )
    yhat_oos_col = "mean_yhat_out_sample" if "mean_yhat_out_sample" in pivot.columns else None
    yhat_fo_col = "mean_yhat_forecast_only" if "mean_yhat_forecast_only" in pivot.columns else None
    if yhat_oos_col and yhat_fo_col:
        pivot = pivot.with_columns(
            (pl.col(yhat_fo_col) / pl.col(yhat_oos_col)).alias("ratio_forecast_only_vs_oos")
        )
    return pivot


def section6_alpha_check(df: pl.DataFrame, per_leaf: pl.DataFrame, top_n: int) -> pl.DataFrame:
    cols = [c for c in ("ses_alpha_y", "pure_ses_wmape_y", "modelo_seleccionado") if c in df.columns]
    if not cols:
        return pl.DataFrame()
    meta = (
        df.filter(pl.col("period_type") == "out_sample")
        .group_by("unique_id")
        .agg([pl.col(c).first().alias(c) for c in cols])
    )
    return (
        per_leaf.sort("bias_leaf", descending=True)
        .head(top_n)
        .join(meta, on="unique_id", how="left")
        .select(
            ["unique_id", "bias_leaf", "wmape_leaf", "overestimation_ratio", "sum_y", "sum_yhat"]
            + cols
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", default=DEFAULT_FORECAST_PATH)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--bias-threshold", type=float, default=0.30,
                     help="Umbral de sesgo relativo (yhat-y)/y para considerar 'sobreestimación fuerte'.")
    ap.add_argument("--uid", default=None, help="Si se indica, imprime detalle diario de esa hoja y termina.")
    args = ap.parse_args()

    path = Path(args.forecast)
    print(f"Leyendo: {path}")
    df = load(path)
    df = df.filter(_leaf_filter())
    print(f"Filas hoja SKU+tienda cargadas: {df.height:,}")

    if args.uid:
        leaf = df.filter(pl.col("unique_id") == args.uid).sort("ds")
        if leaf.height == 0:
            print(f"No se encontró la hoja {args.uid}")
            return 1
        with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=40):
            for pt in ("in_sample", "out_sample", "forecast_only"):
                g = leaf.filter(pl.col("period_type") == pt)
                if g.height:
                    print(f"\n=== {args.uid} | {pt} | {g['ds'].min()} → {g['ds'].max()} ({g.height} filas) ===")
                    print(g)
        return 0

    dense_coverage = float(_setting("LEAF_REGIME_DENSE_COVERAGE", DEFAULT_DENSE_COVERAGE))
    leaves_oos = df.filter(pl.col("period_type") == "out_sample")
    if leaves_oos.height == 0:
        print("No hay filas out_sample en el parquet. ¿Ruta correcta?")
        return 1

    # ── Sección 1: ranking de sesgo ─────────────────────────────────────────
    per_leaf, top_bias = section1_ranking(leaves_oos, args.top_n)
    print("\n" + "=" * 78)
    print(f"SECCIÓN 1 — TOP {args.top_n} HOJAS POR SESGO OOS (bias_leaf = Σ(ŷ-y)/Σy)")
    print("=" * 78)
    with pl.Config(tbl_cols=-1, tbl_rows=args.top_n, fmt_str_lengths=50):
        print(top_bias.select(
            "unique_id", "n_dias", "sum_y", "sum_yhat", "wmape_leaf",
            "bias_leaf", "overestimation_ratio"
        ))
    top_impact = per_leaf.sort("sum_abs_err", descending=True).head(args.top_n)
    print(f"\n--- TOP {args.top_n} HOJAS POR IMPACTO ABSOLUTO (mayor Σ|y-ŷ|) ---")
    with pl.Config(tbl_cols=-1, tbl_rows=args.top_n, fmt_str_lengths=50):
        print(top_impact.select(
            "unique_id", "n_dias", "sum_y", "sum_yhat", "sum_abs_err",
            "wmape_leaf", "bias_leaf"
        ))

    # ── Sección 2: magnitud del problema ────────────────────────────────────
    impact_summary = section2_impact(per_leaf, args.bias_threshold)
    print("\n" + "=" * 78)
    print("SECCIÓN 2 — MAGNITUD GLOBAL DEL PROBLEMA")
    print("=" * 78)
    for k, v in impact_summary.items():
        print(f"  {k}: {v}")

    # ── Sección 3: régimen dense vs sparse ──────────────────────────────────
    regime_tbl = section3_regime(df, per_leaf, dense_coverage)
    print("\n" + "=" * 78)
    print(f"SECCIÓN 3 — RÉGIMEN DENSO (coverage>={dense_coverage}) vs SPARSE")
    print("  -> Responde H1/H2: ¿las hojas que sobreestiman son densas (guard v9.1")
    print("     nunca las tocó) o sparse (guard debería aplicar pero no alcanza)?")
    print("=" * 78)
    if regime_tbl.height:
        print(regime_tbl)
    else:
        print("  (columna ses_recent28_coverage_y no disponible en este parquet)")

    # ── Sección 4: nivel SES vs driver RLS ──────────────────────────────────
    top_bias_uids = top_bias["unique_id"].to_list()
    level_driver_tbl = section4_level_vs_driver(df, top_bias_uids)
    print("\n" + "=" * 78)
    print("SECCIÓN 4 — NIVEL SES vs DRIVER RLS (contrato: media(driver_factor)=1/bloque)")
    print("  -> Responde H3: ¿el nivel SES ya viene alto, o el driver RLS infla")
    print("     el forecast violando el contrato de media 1 por bloque?")
    print("=" * 78)
    if level_driver_tbl.height:
        with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=50):
            print(level_driver_tbl)
        viol = level_driver_tbl.filter(pl.col("desvio_contrato_media_1") > 0.02)
        print(f"\n  Bloques con |media(driver_factor)-1| > 0.02 (violación de contrato): {viol.height} / {level_driver_tbl.height}")
    else:
        print("  (columna driver_factor_y no disponible en este parquet)")

    # ── Sección 5: continuidad OOS -> forecast_only ─────────────────────────
    continuity_tbl = section5_continuity(df, top_bias_uids)
    print("\n" + "=" * 78)
    print("SECCIÓN 5 — CONTINUIDAD OOS -> FORECAST_ONLY (¿se hereda el sesgo?)")
    print("  -> Responde H4: ratio_forecast_only_vs_oos cercano a 1 sugiere que")
    print("     forecast_only simplemente extiende el mismo nivel ya sesgado.")
    print("=" * 78)
    if continuity_tbl.height:
        with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=50):
            print(continuity_tbl)
    else:
        print("  (no hay filas forecast_only para las hojas top-bias, o faltan columnas)")

    # ── Sección 6: selección de alpha ───────────────────────────────────────
    alpha_tbl = section6_alpha_check(df, per_leaf, args.top_n)
    print("\n" + "=" * 78)
    print("SECCIÓN 6 — ALPHA SES SELECCIONADO EN HOJAS OFENSORAS")
    print("  -> ¿El alpha elegido causalmente (score in-sample) generaliza mal")
    print("     a OOS, o se concentra en valores altos (reactivo a picos)?")
    print("=" * 78)
    if alpha_tbl.height:
        with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=50):
            print(alpha_tbl)
    else:
        print("  (columnas ses_alpha_y / pure_ses_wmape_y no disponibles)")

    # ── Export ───────────────────────────────────────────────────────────────
    out_dir = path.parent
    csv_path = out_dir / "diagnostico_fase_a_top_offenders.csv"
    json_path = out_dir / "diagnostico_fase_a_summary.json"
    try:
        top_bias.write_csv(csv_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(impact_summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nExportado: {csv_path}")
        print(f"Exportado: {json_path}")
    except Exception as e:
        print(f"\n(no se pudo exportar a disco: {e})")

    print("\nListo. Comparte la salida de consola (o los 2 archivos exportados) para Fase B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
