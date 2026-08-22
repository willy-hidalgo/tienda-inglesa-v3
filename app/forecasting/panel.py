"""Dense daily panel construction."""
from __future__ import annotations
import datetime as dt
import logging
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

def densify_section_panel(
    df: pl.DataFrame,
    date_start: dt.date,
    date_end: dt.date,
    *,
    fill_y: float = 0.0,
    fill_value: float = 0.0,
    extra_uids: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Panel denso: (unique_id × cada día en [date_start, date_end]).
    Sin venta → y=0, value=0. Métricas excluyen ceros aparte.

    Grid vía np.repeat/tile (más barato que join cross de Polars).
    """
    if date_end < date_start:
        return df
    if df.height == 0 and (extra_uids is None or extra_uids.height == 0):
        return df

    dates = pl.date_range(date_start, date_end, interval="1d", eager=True)
    n_days = len(dates)

    if df.height:
        df = df.with_columns(pl.col("ds").cast(pl.Date))
        uids_s = df.get_column("unique_id").unique()
    else:
        uids_s = pl.Series("unique_id", [], dtype=pl.Utf8)
    if extra_uids is not None and extra_uids.height:
        extra_s = extra_uids.get_column("unique_id")
        uids_s = pl.concat([uids_s, extra_s]).unique()

    n_uid = uids_s.len()
    if n_uid == 0:
        return df

    # Grid denso sin cross-join: O(n) construcción
    uid_arr = uids_s.to_numpy()
    ds_arr = dates.to_numpy()
    grid = pl.DataFrame(
        {
            "unique_id": np.repeat(uid_arr, n_days),
            "ds": np.tile(ds_arr, n_uid),
        }
    ).with_columns(pl.col("ds").cast(pl.Date))

    meta_cols = [
        c
        for c in ("sku_desc", "store_name", "seccion", "conteo_sku")
        if df.height and c in df.columns
    ]
    if meta_cols:
        meta = (
            df.select(["unique_id"] + meta_cols)
            .group_by("unique_id")
            .agg([pl.col(c).drop_nulls().first().alias(c) for c in meta_cols])
        )
        grid = grid.join(meta, on="unique_id", how="left")

    if df.height:
        data_cols = [
            c for c in df.columns if c not in ("unique_id", "ds") and c not in meta_cols
        ]
        join_df = df.select(["unique_id", "ds"] + data_cols)
        out = grid.join(join_df, on=["unique_id", "ds"], how="left")
    else:
        out = grid

    fills = []
    if "y" in out.columns:
        fills.append(pl.col("y").fill_null(fill_y))
    else:
        fills.append(pl.lit(fill_y).alias("y"))
    if "value" in out.columns:
        fills.append(pl.col("value").fill_null(fill_value))
    else:
        fills.append(pl.lit(fill_value).alias("value"))
    if "conteo_sku" in out.columns:
        fills.append(pl.col("conteo_sku").fill_null(0))
    if fills:
        out = out.with_columns(fills)

    logger.info(
        "densify: %d series × %d días = %d filas",
        n_uid,
        n_days,
        out.height,
    )
    # Orden natural del grid: uid bloqueado × días → ya casi ordenado
    return out.sort(["unique_id", "ds"])
