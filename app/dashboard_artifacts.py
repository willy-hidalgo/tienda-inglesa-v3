"""
Artefactos precalculados del dashboard.
=======================================
Objetivo: el Streamlit solo lee y pinta. Cero WMAPE / agregaciones pesadas
en el hot path.

Salida (junto al forecast.parquet, o en settings.DASHBOARD_DIR si existe):

  dashboard/
    index.json              — secciones, tiendas, skus, horizontes, flags
                              (SIN label_map: va en labels.parquet)
    labels.parquet          — unique_id → label, description
    metrics.parquet         — una fila por (unique_id, unidad)
    series/seccion=<s>/...  — panel slim particionado por sección
                              (predicate pushdown + menos I/O)

Uso:
  python -m app.dashboard_artifacts
  python -m app.dashboard_artifacts --forecast path/to/forecast.parquet
  from app.dashboard_artifacts import build_artifacts, load_index, load_metrics, load_series
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings

try:
    from app import backend
except ImportError:  # pragma: no cover
    import backend  # type: ignore

logger = logging.getLogger(__name__)

SERIES_COLS_PREFERRED = [
    "unique_id",
    "ds",
    "y",
    "yhat",
    "yhat28",
    "value",
    "valuehat",
    "valuehat28",
    "period_type",
    "sku_desc",
    "store_name",
    "seccion",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "forecast_start",
    "forecast_end",
    "driver_effect",
    "driver_effect_value",
]


def artifacts_dir(forecast_path: Path | None = None) -> Path:
    """Directorio de artefactos. Prefer settings.DASHBOARD_DIR; si no, junto al parquet."""
    custom = getattr(settings, "DASHBOARD_DIR", None)
    if custom:
        return Path(custom)
    if forecast_path is not None:
        return Path(forecast_path).resolve().parent / "dashboard"
    return Path(settings.FORECAST_PATH).resolve().parent / "dashboard"


def _index_path(adir: Path) -> Path:
    return adir / "index.json"


def _labels_path(adir: Path) -> Path:
    return adir / "labels.parquet"


def _metrics_path(adir: Path) -> Path:
    return adir / "metrics.parquet"


def _series_dir(adir: Path) -> Path:
    return adir / "series"


def _series_legacy_path(adir: Path) -> Path:
    """Compat: un solo series.parquet (versiones anteriores)."""
    return adir / "series.parquet"


def artifacts_exist(forecast_path: Path | None = None) -> bool:
    adir = artifacts_dir(forecast_path)
    has_series = _series_dir(adir).is_dir() or _series_legacy_path(adir).exists()
    return (
        _index_path(adir).exists()
        and _metrics_path(adir).exists()
        and has_series
    )


def _parse_uid_parts(uids: list[str]) -> pl.DataFrame:
    """Vectoriza seccion / store / sku / node_kind desde unique_id."""
    sec, store, sku, kind = [], [], [], []
    for uid in uids:
        p = settings.split_unique_id(uid)
        s, t, k = p["seccion"], p.get("store"), p.get("sku")
        sec.append(s)
        store.append(t)
        sku.append(k)
        if t is not None and k is not None:
            kind.append("tienda_sku")
        elif k is not None:
            kind.append("sku")
        elif t is not None:
            kind.append("tienda")
        else:
            kind.append("seccion")
    return pl.DataFrame(
        {
            "unique_id": uids,
            "seccion": sec,
            "store": store,
            "sku": sku,
            "node_kind": kind,
        }
    )


def _wmape_table_for_unit(
    unit_df: pl.DataFrame,
    all_ids: list[str],
    unidad: str,
) -> pl.DataFrame:
    """
    Métricas por unique_id para una unidad (Valor o Unidades).

    Un solo group_by sobre el panel completo (no un wmape_por_id por sección).
    n_spine se asigna después por sección vía join.
    """
    empty = pl.DataFrame(
        schema={
            "unique_id": pl.Utf8,
            "wmape": pl.Float64,
            "sum_y": pl.Float64,
            "n_points": pl.UInt32,
            "n_with_sales": pl.UInt32,
            "unidad": pl.Utf8,
            "seccion": pl.Utf8,
            "n_spine": pl.UInt32,
        }
    )
    if unit_df.height == 0:
        return empty

    # Secciones: parseo una sola vez
    sec_from_id = (
        pl.DataFrame({"unique_id": all_ids})
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion")
        )
        .unique()
    )
    secciones = (
        sec_from_id.filter(~pl.col("unique_id").str.contains(r"\|\|", literal=False))
        ["seccion"]
        .unique()
        .to_list()
    )
    if not secciones:
        secciones = sec_from_id["seccion"].unique().to_list()

    # n_spine por sección (settings + opcional override por datos)
    spine_rows = []
    for seccion in secciones:
        hz = settings.section_horizons(seccion)
        n_spine = (hz["forecast_end"] - hz["train_start"]).days + 1
        spine_rows.append({"seccion": seccion, "n_spine": int(n_spine)})
    spine_df = pl.DataFrame(spine_rows).with_columns(
        pl.col("n_spine").cast(pl.UInt32)
    )

    # Un solo WMAPE sobre todo el panel (sin lista de ids → group_by directo)
    # n_points se rellena después con n_spine de la sección.
    tabla = backend.wmape_all_ids(unit_df, n_fechas_spine=0)
    if tabla.height == 0:
        return empty

    tabla = (
        tabla.join(sec_from_id, on="unique_id", how="left")
        .join(spine_df, on="seccion", how="left")
        .with_columns(
            pl.col("n_spine").fill_null(0).alias("n_points"),
            pl.lit(unidad).alias("unidad"),
        )
        .select(
            [
                "unique_id",
                "wmape",
                "sum_y",
                "n_points",
                "n_with_sales",
                "unidad",
                "seccion",
                "n_spine",
            ]
        )
    )
    return tabla


def _build_pure_sku_series_vectorized(res_df: pl.DataFrame) -> pl.DataFrame:
    """
    Materializa series de SKU puro (sin tienda) en una sola pasada.

    Hojas = unique_id con T: y S:. Agrupa por (seccion, sku, ds).
    """
    if res_df.height == 0 or "unique_id" not in res_df.columns:
        return pl.DataFrame()

    # Parse vectorizado de hojas tienda+sku
    leaves = res_df.filter(
        pl.col("unique_id").str.contains(r"\|\|T:", literal=False)
        & pl.col("unique_id").str.contains(r"\|\|S:", literal=False)
    )
    if leaves.height == 0:
        return pl.DataFrame()

    leaves = leaves.with_columns(
        pl.col("unique_id").str.split("||").list.get(0).alias("_sec"),
        pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("_sku"),
    )

    sum_cols = [
        c
        for c in ("y", "yhat", "yhat28", "value", "valuehat", "valuehat28")
        if c in leaves.columns
    ]
    first_cols = [
        c
        for c in (
            "period_type",
            "sku_desc",
            "seccion",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "forecast_start",
            "forecast_end",
        )
        if c in leaves.columns
    ]
    aggs: list[pl.Expr] = [pl.col(c).sum() for c in sum_cols]
    aggs.extend(pl.col(c).first() for c in first_cols)

    pure = (
        leaves.group_by(["_sec", "_sku", "ds"])
        .agg(aggs)
        .with_columns(
            pl.concat_str(
                [pl.col("_sec"), pl.lit("||S:"), pl.col("_sku")]
            ).alias("unique_id"),
            # SKU puro no tiene store_name
            pl.lit(None).cast(pl.Utf8).alias("store_name")
            if "store_name" in res_df.columns
            else pl.lit(None).alias("store_name"),
        )
        .drop(["_sec", "_sku"])
        .sort("unique_id", "ds")
    )
    return pure


def _build_index(
    res_df: pl.DataFrame,
    all_ids: list[str],
    metrics: pl.DataFrame,
    forecast_path: Path,
    label_map: dict[str, str],
    desc_map: dict[str, str],
) -> dict[str, Any]:
    """Index liviano: sin label_map/desc_map (van a labels.parquet)."""
    secciones = backend.secciones_disponibles(all_ids)
    stores_by_sec: dict[str, list[str]] = {}
    skus_by_sec: dict[str, list[str]] = {}
    stores_for_sku: dict[str, dict[str, list[str]]] = {}
    skus_for_store: dict[str, dict[str, list[str]]] = {}
    n_spine_by_sec: dict[str, int] = {}
    horizons_by_sec: dict[str, dict[str, str | None]] = {}

    for seccion in secciones:
        stores_by_sec[seccion] = backend.all_stores_in_section(all_ids, seccion)
        skus_by_sec[seccion] = backend.all_skus_in_section(all_ids, seccion)
        stores_for_sku[seccion] = {
            sku: backend.stores_for_sku(all_ids, seccion, sku)
            for sku in skus_by_sec[seccion]
        }
        skus_for_store[seccion] = {
            store: backend.skus_for_store(all_ids, seccion, store)
            for store in stores_by_sec[seccion]
        }
        hz = settings.section_horizons(seccion)
        horizons_by_sec[seccion] = {
            k: (v.isoformat() if v is not None else None)
            for k, v in hz.items()
            if hasattr(v, "isoformat") or v is None
        }
        sub = metrics.filter(pl.col("seccion") == seccion)
        if sub.height and "n_spine" in sub.columns:
            n_spine_by_sec[seccion] = int(sub["n_spine"][0])
        else:
            n_spine_by_sec[seccion] = (
                hz["forecast_end"] - hz["train_start"]
            ).days + 1

    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    has_rolling28 = (
        "yhat28" in cols and res_df.select("yhat28").drop_nulls().height > 0
    ) or (
        "valuehat28" in cols
        and res_df.select("valuehat28").drop_nulls().height > 0
    )

    # labels para SKU puro virtuales
    for seccion, skus in skus_by_sec.items():
        for sku in skus:
            pure_id = settings.make_unique_id(seccion, sku=sku)
            if pure_id not in label_map:
                hit = next(
                    (
                        desc_map[k]
                        for k in desc_map
                        if f"||S:{sku}" in k and desc_map[k]
                    ),
                    "",
                )
                label_map[pure_id] = settings.display_label(pure_id, hit or None, None)
                desc_map[pure_id] = hit or settings.ranking_description(
                    pure_id, hit or None, None
                )

    return {
        "version": 2,
        "forecast_path": str(forecast_path.resolve()),
        "forecast_mtime": forecast_path.stat().st_mtime if forecast_path.exists() else 0.0,
        "secciones": secciones,
        "stores_by_sec": stores_by_sec,
        "skus_by_sec": skus_by_sec,
        "stores_for_sku": stores_for_sku,
        "skus_for_store": skus_for_store,
        "n_spine_by_sec": n_spine_by_sec,
        "horizons_by_sec": horizons_by_sec,
        # labels viven en labels.parquet (v2); se mantienen vacíos aquí por compat
        "label_map": {},
        "desc_map": {},
        "has_value": has_value,
        "has_rolling28": has_rolling28,
        "n_unique_ids": len(all_ids),
        "n_rows_source": res_df.height,
        "series_partitioned": True,
    }


def _write_series_partitioned(series: pl.DataFrame, adir: Path) -> None:
    """Escribe series/seccion=<s>/data.parquet ordenado por unique_id, ds."""
    sdir = _series_dir(adir)
    if sdir.exists():
        # limpiar particiones previas
        for p in sdir.rglob("*.parquet"):
            p.unlink()
        for p in sorted(sdir.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
    sdir.mkdir(parents=True, exist_ok=True)

    if series.height == 0:
        return

    # Garantizar columna seccion para particionar
    if "seccion" not in series.columns:
        series = series.with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion")
        )
    else:
        # rellenar nulls desde unique_id
        series = series.with_columns(
            pl.when(pl.col("seccion").is_null() | (pl.col("seccion") == ""))
            .then(pl.col("unique_id").str.split("||").list.get(0))
            .otherwise(pl.col("seccion"))
            .alias("seccion")
        )

    for seccion, part in series.partition_by("seccion", as_dict=True).items():
        sec_val = seccion[0] if isinstance(seccion, tuple) else seccion
        sec_str = str(sec_val)
        out = sdir / f"seccion={sec_str}"
        out.mkdir(parents=True, exist_ok=True)
        (
            part.sort("unique_id", "ds")
            .write_parquet(
                out / "data.parquet",
                compression="zstd",
                compression_level=3,
                statistics=True,
            )
        )


def _labels_df(label_map: dict[str, str], desc_map: dict[str, str]) -> pl.DataFrame:
    uids = sorted(set(label_map) | set(desc_map))
    return pl.DataFrame(
        {
            "unique_id": uids,
            "label": [label_map.get(u, "") for u in uids],
            "description": [desc_map.get(u, "") for u in uids],
        }
    )


def build_artifacts(
    forecast_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
) -> Path:
    """
    Lee forecast.parquet y materializa index + labels + metrics + series particionadas.
    Devuelve el directorio de artefactos.
    """
    fpath = Path(forecast_path or settings.FORECAST_PATH)
    if not fpath.exists():
        raise FileNotFoundError(f"No existe forecast: {fpath}")

    adir = Path(out_dir) if out_dir else artifacts_dir(fpath)
    adir.mkdir(parents=True, exist_ok=True)
    logger.info("Construyendo artefactos dashboard en %s …", adir)

    res_df = backend.load_forecast_parquet(fpath)
    all_ids = res_df["unique_id"].unique().to_list()
    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols

    label_map, desc_map = backend.build_label_maps(res_df)

    # ── metrics (unidades + valor si hay) ──────────────────────────────────
    metric_parts: list[pl.DataFrame] = []
    unit_df_u = backend.prepare_unit_df(res_df, "Unidades", has_value)
    metric_parts.append(_wmape_table_for_unit(unit_df_u, all_ids, "Unidades"))
    if has_value:
        unit_df_v = backend.prepare_unit_df(res_df, "Valor ($)", has_value)
        metric_parts.append(_wmape_table_for_unit(unit_df_v, all_ids, "Valor ($)"))
    metrics = pl.concat([p for p in metric_parts if p.height], how="diagonal_relaxed")

    if metrics.height:
        # unique_id se repite por unidad (Unidades / Valor $). parts_df debe
        # tener 1 fila por uid; si no, el join multiplica filas → ranking duplicado.
        parts_df = _parse_uid_parts(
            metrics["unique_id"].unique().to_list()
        ).unique(subset=["unique_id"])
        metrics = metrics.drop(
            [c for c in ("store", "sku", "node_kind", "seccion") if c in metrics.columns]
        )
        metrics = metrics.join(parts_df, on="unique_id", how="left")

    # ── series slim + SKU puro vectorizado ────────────────────────────────
    keep = [c for c in SERIES_COLS_PREFERRED if c in res_df.columns]
    series = res_df.select(keep)

    pure = _build_pure_sku_series_vectorized(res_df)
    pure_ids: list[str] = []
    if pure.height:
        for c in keep:
            if c not in pure.columns:
                pure = pure.with_columns(pl.lit(None).alias(c))
        pure = pure.select(keep)
        series = pl.concat([series, pure], how="diagonal_relaxed")

        pure_ids = pure["unique_id"].unique().to_list()
        unit_specs: list[tuple[str, pl.DataFrame]] = [("Unidades", pure)]
        if has_value and "value" in pure.columns and "valuehat" in pure.columns:
            exprs = [pl.col("value").alias("y"), pl.col("valuehat").alias("yhat")]
            if "valuehat28" in pure.columns:
                exprs.append(pl.col("valuehat28").alias("yhat28"))
            unit_specs.append(("Valor ($)", pure.with_columns(exprs)))
        pure_metric_parts: list[pl.DataFrame] = []
        for unidad, pure_unit in unit_specs:
            # Un solo group_by sobre todas las series SKU-puro
            tabla = _wmape_table_for_unit(pure_unit, pure_ids, unidad)
            if tabla.height:
                pure_metric_parts.append(tabla)
        if pure_metric_parts:
            pure_metrics = pl.concat(pure_metric_parts, how="diagonal_relaxed")
            pure_parts = _parse_uid_parts(
                pure_metrics["unique_id"].unique().to_list()
            ).unique(subset=["unique_id"])
            pure_metrics = pure_metrics.drop(
                [
                    c
                    for c in ("store", "sku", "node_kind", "seccion")
                    if c in pure_metrics.columns
                ]
            )
            pure_metrics = pure_metrics.join(pure_parts, on="unique_id", how="left")
            metrics = pl.concat([metrics, pure_metrics], how="diagonal_relaxed")

        # labels SKU puro
        for uid in pure_ids:
            if uid not in label_map:
                p = settings.split_unique_id(uid)
                sku = p.get("sku")
                hit = next(
                    (
                        desc_map[k]
                        for k in desc_map
                        if sku and f"||S:{sku}" in k and desc_map[k]
                    ),
                    "",
                )
                label_map[uid] = settings.display_label(uid, hit or None, None)
                desc_map[uid] = hit or settings.ranking_description(
                    uid, hit or None, None
                )

    # Seguridad: una fila por (unique_id, unidad)
    if metrics.height and "unidad" in metrics.columns:
        metrics = metrics.unique(subset=["unique_id", "unidad"], keep="first")
    elif metrics.height:
        metrics = metrics.unique(subset=["unique_id"], keep="first")

    # ── index + labels ────────────────────────────────────────────────────
    index = _build_index(res_df, all_ids, metrics, fpath, label_map, desc_map)
    labels = _labels_df(label_map, desc_map)

    # ── write ─────────────────────────────────────────────────────────────
    _index_path(adir).write_text(
        json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    labels.write_parquet(
        _labels_path(adir), compression="zstd", compression_level=3, statistics=True
    )
    metrics.write_parquet(
        _metrics_path(adir), compression="zstd", compression_level=3, statistics=True
    )
    _write_series_partitioned(series, adir)
    # eliminar legacy monolítico si existía
    legacy = _series_legacy_path(adir)
    if legacy.exists():
        legacy.unlink()

    logger.info(
        "✓ Artefactos: index (%d secciones), metrics (%d filas), series particionadas, labels (%d)",
        len(index["secciones"]),
        metrics.height,
        labels.height,
    )
    return adir


# ── loaders (usados por el dashboard) ─────────────────────────────────────────


def load_index(forecast_path: Path | None = None) -> dict[str, Any]:
    path = _index_path(artifacts_dir(forecast_path))
    if not path.exists():
        raise FileNotFoundError(f"Falta index de dashboard: {path}")
    index = json.loads(path.read_text(encoding="utf-8"))
    # v2: labels en parquet; v1: embebidos en index
    if not index.get("label_map"):
        labels_path = _labels_path(artifacts_dir(forecast_path))
        if labels_path.exists():
            lab = pl.read_parquet(labels_path)
            index["label_map"] = dict(
                zip(lab["unique_id"].to_list(), lab["label"].to_list())
            )
            index["desc_map"] = dict(
                zip(lab["unique_id"].to_list(), lab["description"].to_list())
            )
    return index


def load_metrics(forecast_path: Path | None = None) -> pl.DataFrame:
    path = _metrics_path(artifacts_dir(forecast_path))
    if not path.exists():
        raise FileNotFoundError(f"Falta metrics de dashboard: {path}")
    return pl.read_parquet(path)


def _series_scan_path(adir: Path, unique_id: str | None = None) -> Path | list[Path]:
    """
    Resuelve path(s) de series.
    Prefer partición por sección; fallback a series.parquet legacy.
    """
    sdir = _series_dir(adir)
    if sdir.is_dir():
        if unique_id is not None:
            seccion = unique_id.split("||", 1)[0]
            part = sdir / f"seccion={seccion}" / "data.parquet"
            if part.exists():
                return part
            # buscar cualquier partición (por si seccion no matchea el nombre)
            parts = list(sdir.glob("seccion=*/data.parquet"))
            if parts:
                return parts
        else:
            return list(sdir.glob("seccion=*/data.parquet"))
    legacy = _series_legacy_path(adir)
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Falta series de dashboard en {adir}")


def load_series(
    unique_id: str,
    forecast_path: Path | None = None,
    *,
    unidad: str = "Unidades",
    has_value: bool = False,
) -> pl.DataFrame:
    """
    Lee SOLO la serie del unique_id pedido (partición de sección + filter).
    Aplica prepare_unit_df según unidad.
    """
    adir = artifacts_dir(forecast_path)
    path = _series_scan_path(adir, unique_id)
    if isinstance(path, list):
        df = (
            pl.scan_parquet(path)
            .filter(pl.col("unique_id") == unique_id)
            .collect()
            .sort("ds")
        )
    else:
        df = (
            pl.scan_parquet(path)
            .filter(pl.col("unique_id") == unique_id)
            .collect()
            .sort("ds")
        )
    if df.height == 0:
        return df
    return backend.prepare_unit_df(df, unidad, has_value)


def load_series_many(
    unique_ids: list[str],
    forecast_path: Path | None = None,
) -> pl.DataFrame:
    if not unique_ids:
        return pl.DataFrame()
    adir = artifacts_dir(forecast_path)
    # agrupar por sección para leer solo particiones necesarias
    by_sec: dict[str, list[str]] = {}
    for uid in unique_ids:
        by_sec.setdefault(uid.split("||", 1)[0], []).append(uid)

    chunks: list[pl.DataFrame] = []
    sdir = _series_dir(adir)
    if sdir.is_dir():
        for seccion, uids in by_sec.items():
            part = sdir / f"seccion={seccion}" / "data.parquet"
            if not part.exists():
                continue
            chunks.append(
                pl.scan_parquet(part)
                .filter(pl.col("unique_id").is_in(uids))
                .collect()
            )
    else:
        legacy = _series_legacy_path(adir)
        if legacy.exists():
            chunks.append(
                pl.scan_parquet(legacy)
                .filter(pl.col("unique_id").is_in(unique_ids))
                .collect()
            )
    if not chunks:
        return pl.DataFrame()
    return pl.concat(chunks, how="diagonal_relaxed").sort("unique_id", "ds")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    parser = argparse.ArgumentParser(description="Construir artefactos del dashboard")
    parser.add_argument(
        "--forecast",
        type=str,
        default=None,
        help="Path a forecast.parquet (default: settings.FORECAST_PATH)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directorio de salida (default: <forecast_dir>/dashboard)",
    )
    args = parser.parse_args(argv)
    adir = build_artifacts(args.forecast, out_dir=args.out_dir)
    print(f"Artefactos listos en: {adir}")


if __name__ == "__main__":
    main()
