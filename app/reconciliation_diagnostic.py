from __future__ import annotations

import math
from pathlib import Path

import polars as pl

import settings

EPS = 1e-12


def _pct(x: float | None) -> str:
    if x is None or not math.isfinite(float(x)):
        return "   n/a"
    return f"{100.0 * float(x):7.2f}%"


def _metric(df: pl.DataFrame, actual: str, pred: str, by: list[str]) -> pl.DataFrame:
    return (
        df.filter(pl.col(actual).fill_null(0.0) > 0.0)
        .group_by(by)
        .agg(
            (pl.col(pred).fill_null(0.0) - pl.col(actual)).abs().sum().alias("ae"),
            (pl.col(pred).fill_null(0.0) - pl.col(actual)).sum().alias("se"),
            pl.col(actual).abs().sum().alias("den"),
        )
        .with_columns(
            pl.when(pl.col("den") > EPS).then(pl.col("ae") / pl.col("den")).otherwise(None).alias("wmape"),
            pl.when(pl.col("den") > EPS).then(pl.col("se") / pl.col("den")).otherwise(None).alias("bias"),
        )
    )


def _zero_penalty(df: pl.DataFrame, actual: str, pred: str, by: list[str]) -> pl.DataFrame:
    return (
        df.filter(pl.col(actual).fill_null(0.0) <= 0.0)
        .group_by(by)
        .agg(pl.col(pred).fill_null(0.0).clip(lower_bound=0.0).sum().alias("zero_forecast"))
    )


def _print_section_table(title: str, table: pl.DataFrame) -> None:
    print(f"\n{title}")
    for r in table.sort(["target", "seccion", "method"]).iter_rows(named=True):
        print(
            f"{r['target']:<10} sec={str(r['seccion']):>3} | {r['method']:<18} "
            f"wMAPE={_pct(r['wmape'])} BIAS={_pct(r['bias'])} "
            f"zero={float(r.get('zero_forecast') or 0.0):,.2f}"
        )


def main() -> None:
    path = Path(settings.FORECAST_PATH)
    if not path.exists():
        raise SystemExit(f"No existe forecast.parquet: {path}")

    cols = [
        "unique_id", "ds", "period_type", "y", "value", "yhat", "valuehat"
    ]
    schema = pl.scan_parquet(path).collect_schema().names()
    missing = [c for c in cols if c not in schema]
    if missing:
        raise SystemExit(f"forecast.parquet no contiene columnas requeridas: {missing}")

    df = (
        pl.scan_parquet(path)
        .select(cols)
        .filter(pl.col("period_type") == "out_sample")
        .with_columns(
            pl.col("ds").cast(pl.Date),
            pl.col("unique_id").str.split("||").list.first().alias("seccion"),
            pl.col("unique_id").str.count_matches(r"\|\|", literal=False).alias("_depth"),
        )
        .collect()
    )

    leaf = (
        df.filter(pl.col("_depth") == 2)
        .with_columns(
            pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("store_uid"),
            pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("store"),
        )
    )
    store = (
        df.filter(pl.col("_depth") == 1)
        .rename({"unique_id": "store_uid"})
        .with_columns(pl.col("store_uid").str.extract(r"\|\|T:([^|]+)", 1).alias("store"))
    )

    if leaf.height == 0 or store.height == 0:
        raise SystemExit("No se encontraron filas OOS leaf/store en forecast.parquet")

    print("DIAGNÓSTICO DE RECONCILIACIÓN JERÁRQUICA OOS")
    print(f"Forecast: {path}")
    print(f"Leaf OOS rows : {leaf.height:,}")
    print(f"Store OOS rows: {store.height:,}")
    print("Los factores hipotéticos usan SOLO forecasts leaf+store; no usan actual OOS.")

    section_rows: list[pl.DataFrame] = []
    store_rows: list[pl.DataFrame] = []
    factor_rows: list[pl.DataFrame] = []

    for target, actual, pred in (
        ("Unidades", "y", "yhat"),
        ("Valor ($)", "value", "valuehat"),
    ):
        # Current leaf forecast.
        current = leaf.select("unique_id", "store_uid", "store", "seccion", "ds", actual, pred)

        leaf_store_day = (
            current.group_by(["store_uid", "store", "seccion", "ds"])
            .agg(
                pl.col(pred).fill_null(0.0).sum().alias("leaf_sum_pred"),
                pl.col(actual).fill_null(0.0).sum().alias("leaf_sum_actual"),
            )
        )
        parent = store.select(
            "store_uid", "store", "seccion", "ds",
            pl.col(pred).fill_null(0.0).alias("parent_pred"),
            pl.col(actual).fill_null(0.0).alias("parent_actual"),
        )
        coherence = leaf_store_day.join(parent, on=["store_uid", "store", "seccion", "ds"], how="inner")

        # Exact daily store reconciliation. Factor is purely forecast/forecast.
        daily_factor = coherence.select(
            "store_uid", "ds",
            pl.when(pl.col("leaf_sum_pred") > EPS)
            .then(pl.col("parent_pred") / pl.col("leaf_sum_pred"))
            .otherwise(1.0)
            .clip(lower_bound=0.0, upper_bound=10.0)
            .alias("factor_daily"),
        )

        # Exact 28-day block reconciliation. One factor per store/horizon.
        block_factor = (
            coherence.group_by("store_uid")
            .agg(
                pl.col("parent_pred").sum().alias("parent_sum"),
                pl.col("leaf_sum_pred").sum().alias("leaf_sum"),
            )
            .with_columns(
                pl.when(pl.col("leaf_sum") > EPS)
                .then(pl.col("parent_sum") / pl.col("leaf_sum"))
                .otherwise(1.0)
                .clip(lower_bound=0.0, upper_bound=10.0)
                .alias("factor_block")
            )
            .select("store_uid", "factor_block")
        )

        eval_leaf = (
            current.join(daily_factor, on=["store_uid", "ds"], how="left")
            .join(block_factor, on="store_uid", how="left")
            .with_columns(
                (pl.col(pred).fill_null(0.0) * pl.col("factor_daily").fill_null(1.0)).alias("pred_daily_recon"),
                (pl.col(pred).fill_null(0.0) * pl.col("factor_block").fill_null(1.0)).alias("pred_block_recon"),
                pl.col(pred).fill_null(0.0).alias("pred_current"),
            )
        )

        for method, pcol in (
            ("current_leaf", "pred_current"),
            ("store_block_recon", "pred_block_recon"),
            ("store_daily_recon", "pred_daily_recon"),
        ):
            m = _metric(eval_leaf, actual, pcol, ["seccion"]).with_columns(
                pl.lit(target).alias("target"), pl.lit(method).alias("method")
            )
            z = _zero_penalty(eval_leaf, actual, pcol, ["seccion"])
            section_rows.append(m.join(z, on="seccion", how="left"))

        # Direct store parent quality vs current leaf-bottom-up store quality.
        parent_metric = _metric(parent, "parent_actual", "parent_pred", ["seccion", "store"]).with_columns(
            pl.lit(target).alias("target"), pl.lit("direct_store_parent").alias("method")
        )
        bu_metric = _metric(coherence, "leaf_sum_actual", "leaf_sum_pred", ["seccion", "store"]).with_columns(
            pl.lit(target).alias("target"), pl.lit("leaf_bottom_up_store").alias("method")
        )
        store_rows.extend([parent_metric, bu_metric])

        # Factor diagnostics.
        f = (
            coherence.group_by(["seccion", "store", "store_uid"])
            .agg(
                pl.col("parent_pred").sum().alias("parent_fcst"),
                pl.col("leaf_sum_pred").sum().alias("leaf_fcst"),
                pl.col("parent_actual").sum().alias("actual"),
            )
            .join(block_factor, on="store_uid", how="left")
            .with_columns(pl.lit(target).alias("target"))
        )
        factor_rows.append(f)

    section_table = pl.concat(section_rows, how="vertical_relaxed")
    _print_section_table("=== A) LEAF OOS: ACTUAL vs RECONCILIACIÓN HIPOTÉTICA ===", section_table)

    stores = pl.concat(store_rows, how="vertical_relaxed")
    print("\n=== B) CALIDAD DIRECTA DEL PADRE STORE vs BOTTOM-UP LEAF ===")
    for target in ("Unidades", "Valor ($)"):
        print(f"\n{target}:")
        s = stores.filter(pl.col("target") == target)
        for r in s.sort(["seccion", "store", "method"]).iter_rows(named=True):
            print(
                f"  sec={r['seccion']:>3} store={r['store']} {r['method']:<21} "
                f"wMAPE={_pct(r['wmape'])} BIAS={_pct(r['bias'])}"
            )

    factors = pl.concat(factor_rows, how="vertical_relaxed")
    print("\n=== C) FACTOR DE COHERENCIA 28D: parent store / suma leaf ===")
    for target in ("Unidades", "Valor ($)"):
        t = factors.filter(pl.col("target") == target)
        q = t.select(
            pl.col("factor_block").min().alias("min"),
            pl.col("factor_block").quantile(0.25).alias("p25"),
            pl.col("factor_block").median().alias("p50"),
            pl.col("factor_block").quantile(0.75).alias("p75"),
            pl.col("factor_block").max().alias("max"),
        ).row(0, named=True)
        print(
            f"{target:<10} min={q['min']:.3f} p25={q['p25']:.3f} p50={q['p50']:.3f} "
            f"p75={q['p75']:.3f} max={q['max']:.3f}"
        )
        print("  Por tienda:")
        for r in t.sort(["seccion", "store"]).iter_rows(named=True):
            parent_vs_actual = (r["parent_fcst"] / r["actual"]) if r["actual"] and r["actual"] > EPS else float("nan")
            leaf_vs_actual = (r["leaf_fcst"] / r["actual"]) if r["actual"] and r["actual"] > EPS else float("nan")
            print(
                f"    sec={r['seccion']:>3} store={r['store']} factor={r['factor_block']:.3f} "
                f"parent/actual={parent_vs_actual:.3f} leaf/actual={leaf_vs_actual:.3f}"
            )

    print("\nINTERPRETACIÓN:")
    print("- Si direct_store_parent es mucho mejor que leaf_bottom_up_store y store_block_recon reduce fuerte wMAPE/BIAS,")
    print("  el siguiente paso correcto es reconciliar SKU+tienda al forecast RLS de tienda.")
    print("- Si el padre store también tiene wMAPE alto, la reconciliación no resolverá el problema y hay que cambiar el modelo leaf.")
    print("- Este diagnóstico no modifica forecast.parquet ni dashboard.")


if __name__ == "__main__":
    main()
