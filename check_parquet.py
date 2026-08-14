import polars as pl
from pathlib import Path
df = pl.read_parquet("data/output/forecast.parquet")
u = df["unique_id"]
print("rows:", df.height)
print("n unique_id:", u.n_unique())
print("sample ids:", u.unique().sort().to_list()[:6])
m = df.group_by("period_type").agg(
    pl.col("yhat").is_finite().alias("yhat_finite"),
    pl.col("yhat").is_nan().alias("yhat_nan"),
    pl.col("y").is_null().alias("y_null"),
)
print(m.sort("period_type"))
