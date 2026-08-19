"""Compare exact client bottom-up OOS WMAPE before/after leaf optimization."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import settings


def _metrics(df: pl.DataFrame, pred_col: str) -> pl.DataFrame:
    leaves = df.filter(
        (pl.col("period_type") == "out_sample")
        & (pl.col("unique_id").str.count_matches(r"\|\|") == 2)
        & pl.col("y").is_not_null()
        & pl.col(pred_col).is_not_null()
        & pl.col("y").is_finite()
        & pl.col(pred_col).is_finite()
        & (pl.col("y") != 0)
    ).with_columns(
        pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("_store"),
        pl.col("unique_id").str.split("||").list.first().alias("_section"),
        (pl.col("y") - pl.col(pred_col)).abs().alias("_err"),
        pl.col("y").abs().alias("_abs_y"),
    )
    rows=[]
    for level, key in [("sku_tienda","unique_id"),("tienda","_store"),("seccion","_section")]:
        agg=leaves.group_by(key).agg(pl.col("_err").sum().alias("err"),pl.col("_abs_y").sum().alias("den"))
        rows.append(pl.DataFrame({"nivel":[level],"wmape":[float(agg["err"].sum()/agg["den"].sum()) if agg.height and float(agg["den"].sum()) else None]}))
    return pl.concat(rows)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--forecast",type=Path,default=settings.FORECAST_PATH)
    args=parser.parse_args()
    df=pl.read_parquet(args.forecast)
    if "yhat_pre_leaf_optimizer" not in df.columns:
        raise SystemExit("forecast.parquet no contiene yhat_pre_leaf_optimizer; ejecute el pipeline actualizado")
    before=_metrics(df.filter(pl.col("yhat_pre_leaf_optimizer").is_not_null()),"yhat_pre_leaf_optimizer").rename({"wmape":"wmape_pre"})
    after=_metrics(df,"yhat").rename({"wmape":"wmape_post"})
    out=before.join(after,on="nivel",how="inner").with_columns(
        ((pl.col("wmape_pre")-pl.col("wmape_post"))/pl.col("wmape_pre")*100).alias("improvement_pct")
    )
    print(out)

if __name__ == "__main__":
    main()
