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
    metrics.parquet         — una fila por (unique_id, unidad); wMAPE = out-of-sample
    series/seccion=<s>/...  — panel slim particionado por sección
                              (predicate pushdown + menos I/O)

Uso:
  python -m app.dashboard_artifacts
  python -m app.dashboard_artifacts --forecast path/to/forecast.parquet
  from app.dashboard_artifacts import build_artifacts, load_index, load_metrics, load_series
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
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

ARTIFACT_VERSION = 6

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
    "ses_level_y",
    "ses_level_value",
    "driver_effect",
    "driver_effect_value",
    "driver_factor_y",
    "driver_factor_value",
    "driver_strength_selected_y",
    "driver_strength_selected_value",
    "driver_strength_y",
    "driver_strength_value",
    "driver_direction_guard_y",
    "driver_direction_guard_value",
    "ses_recent28_y",
    "ses_recent28_value",
    "ses_recent14_y",
    "ses_recent14_value",
    "ses_recent28_coverage_y",
    "ses_recent28_coverage_value",
    "ses_regime_anchor_y",
    "ses_regime_anchor_value",
    "ses_stability_reference_y",
    "ses_stability_reference_value",
    "ses_stability_guard_y",
    "ses_stability_guard_value",
    "rls_metric_eligible",
    "rls_block",
    "rls_train_days",
]

# Columnas mínimas para WMAPE (reduce picos de memoria al preparar unidades)
_WMAPE_COLS = ("unique_id", "ds", "y", "yhat", "period_type", "value", "valuehat", "valuehat28")


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def _progress(msg: str, t0: float | None = None) -> None:
    """Log de avance con timestamp relativo y RSS si está disponible."""
    rss = _rss_mb()
    mem = f" | RSS={rss:.0f} MB" if rss >= 0 else ""
    if t0 is None:
        logger.info("%s%s", msg, mem)
    else:
        logger.info("%s (%.1fs)%s", msg, time.perf_counter() - t0, mem)


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


def source_fingerprint(
    forecast_path: Path,
    *,
    n_rows: int | None = None,
) -> dict[str, int | str]:
    """Fingerprint barato y estricto del forecast que originó los artefactos."""
    path = Path(forecast_path)
    stat = path.stat()
    fp: dict[str, int | str] = {
        "mtime_ns": int(stat.st_mtime_ns),
        "size_bytes": int(stat.st_size),
        "app_version": str(getattr(settings, "APP_VERSION", "")),
    }
    if n_rows is not None:
        fp["n_rows"] = int(n_rows)
    return fp


def artifacts_match_source(
    forecast_path: Path,
    index: dict[str, Any] | None = None,
) -> bool:
    """True solo si index y forecast pertenecen exactamente a la misma corrida."""
    path = Path(forecast_path)
    if not path.exists():
        return False
    if index is None:
        ipath = _index_path(artifacts_dir(path))
        if not ipath.exists():
            return False
        try:
            index = json.loads(ipath.read_text(encoding="utf-8"))
        except Exception:
            return False
    expected = index.get("forecast_fingerprint") or {}
    if not expected:
        return False
    current = source_fingerprint(path)
    stat_match = (
        int(expected.get("mtime_ns") or -1) == int(current["mtime_ns"])
        and int(expected.get("size_bytes") or -1) == int(current["size_bytes"])
        and str(expected.get("app_version") or "") == str(current["app_version"])
    )
    if not stat_match:
        return False

    # Row count is part of the identity too. Parquet metadata makes this check
    # cheap and it prevents a ranking/KPI from being served from a different
    # forecast even in the unlikely case of matching file size/timestamps.
    expected_rows = int(expected.get("n_rows") or -1)
    if expected_rows < 0:
        return False
    try:
        current_rows = int(
            pl.scan_parquet(path)
            .select(pl.len().alias("_n"))
            .collect()
            .item()
        )
    except Exception:
        return False
    return current_rows == expected_rows


def artifacts_exist(forecast_path: Path | None = None) -> bool:
    adir = artifacts_dir(forecast_path)
    has_series = _series_dir(adir).is_dir() or _series_legacy_path(adir).exists()
    ipath = _index_path(adir)
    if not (ipath.exists() and _metrics_path(adir).exists() and has_series):
        return False
    try:
        payload = json.loads(ipath.read_text(encoding="utf-8"))
        return (
            int(payload.get("version") or 0) == ARTIFACT_VERSION
            and artifacts_match_source(
                Path(forecast_path or settings.FORECAST_PATH),
                payload,
            )
        )
    except Exception:
        return False


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


def _slim_for_wmape(df: pl.DataFrame) -> pl.DataFrame:
    """Solo columnas necesarias para WMAPE — evita duplicar el panel completo."""
    keep = [c for c in _WMAPE_COLS if c in df.columns]
    return df.select(keep)


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
        ranking_days = (hz["test_end"] - hz["test_start"]).days + 1
        spine_rows.append(
            {
                "seccion": seccion,
                "n_spine": int(n_spine),
                "ranking_days": int(ranking_days),
            }
        )
    spine_df = pl.DataFrame(spine_rows).with_columns(
        pl.col("n_spine").cast(pl.UInt32),
        pl.col("ranking_days").cast(pl.UInt32),
    )

    # WMAPE de rankings = solo out-of-sample (única fuente de verdad en tablas).
    # n_points se rellena después con n_spine de la sección.
    tabla = backend.wmape_all_ids(
        unit_df, n_fechas_spine=0, period_types=["out_sample"]
    )
    if tabla.height == 0:
        return empty

    tabla = (
        tabla.with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion")
        )
        .join(spine_df, on="seccion", how="left")
        .with_columns(
            pl.col("ranking_days").fill_null(0).alias("n_points"),
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


def _parts_from_ids(all_ids: list[str]) -> pl.DataFrame:
    """Parseo vectorizado de unique_ids → seccion/store/sku/node_kind (una sola vez)."""
    return (
        pl.DataFrame({"unique_id": all_ids})
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.when(pl.col("unique_id").str.contains(r"\|\|T:", literal=False))
            .then(pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1))
            .otherwise(None)
            .alias("store"),
            pl.when(pl.col("unique_id").str.contains(r"\|\|S:", literal=False))
            .then(pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1))
            .otherwise(None)
            .alias("sku"),
        )
        .with_columns(
            pl.when(pl.col("store").is_not_null() & pl.col("sku").is_not_null())
            .then(pl.lit("tienda_sku"))
            .when(pl.col("sku").is_not_null())
            .then(pl.lit("sku"))
            .when(pl.col("store").is_not_null())
            .then(pl.lit("tienda"))
            .otherwise(pl.lit("seccion"))
            .alias("node_kind")
        )
    )


def _build_index(
    res_df: pl.DataFrame,
    all_ids: list[str],
    metrics: pl.DataFrame,
    forecast_path: Path,
    label_map: dict[str, str],
    desc_map: dict[str, str],
) -> dict[str, Any]:
    """Index liviano: sin label_map/desc_map (van a labels.parquet). Vectorizado."""
    parts = _parts_from_ids(all_ids)

    # Secciones = nodos sin store ni sku
    secciones = (
        parts.filter(pl.col("node_kind") == "seccion")["seccion"]
        .unique()
        .sort()
        .to_list()
    )
    if not secciones:
        secciones = parts["seccion"].unique().sort().to_list()

    stores_by_sec: dict[str, list[str]] = {}
    skus_by_sec: dict[str, list[str]] = {}
    stores_for_sku: dict[str, dict[str, list[str]]] = {}
    skus_for_store: dict[str, dict[str, list[str]]] = {}
    n_spine_by_sec: dict[str, int] = {}
    horizons_by_sec: dict[str, dict[str, str | None]] = {}

    # Tiendas y SKUs por sección (nodos tienda / cualquier id con sku)
    for seccion in secciones:
        sec_parts = parts.filter(pl.col("seccion") == seccion)
        stores_by_sec[seccion] = (
            sec_parts.filter(pl.col("node_kind") == "tienda")["store"]
            .drop_nulls()
            .unique()
            .sort()
            .to_list()
        )
        skus_by_sec[seccion] = (
            sec_parts.filter(pl.col("sku").is_not_null())["sku"]
            .unique()
            .sort()
            .to_list()
        )

        # Hojas: store×sku presentes
        leaves = sec_parts.filter(pl.col("node_kind") == "tienda_sku")
        if leaves.height:
            # stores_for_sku[seccion][sku] = [stores…]
            sfs: dict[str, list[str]] = {}
            for row in (
                leaves.group_by("sku")
                .agg(pl.col("store").unique().sort().alias("stores"))
                .iter_rows(named=True)
            ):
                sfs[str(row["sku"])] = [str(x) for x in row["stores"] if x is not None]
            stores_for_sku[seccion] = sfs

            # skus_for_store[seccion][store] = [skus…]
            sft: dict[str, list[str]] = {}
            for row in (
                leaves.group_by("store")
                .agg(pl.col("sku").unique().sort().alias("skus"))
                .iter_rows(named=True)
            ):
                sft[str(row["store"])] = [str(x) for x in row["skus"] if x is not None]
            skus_for_store[seccion] = sft
        else:
            stores_for_sku[seccion] = {}
            skus_for_store[seccion] = {}

        hz = settings.section_horizons(seccion)
        horizons_by_sec[seccion] = {
            k: (v.isoformat() if v is not None else None)
            for k, v in hz.items()
            if hasattr(v, "isoformat") or v is None
        }
        sub = metrics.filter(pl.col("seccion") == seccion) if metrics.height else metrics
        n_val = None
        if sub.height and "n_spine" in sub.columns:
            raw = sub["n_spine"][0]
            if raw is not None:
                try:
                    n_val = int(raw)
                except (TypeError, ValueError):
                    n_val = None
        if n_val is None or n_val <= 0:
            n_val = (hz["forecast_end"] - hz["train_start"]).days + 1
        n_spine_by_sec[seccion] = int(n_val)

    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    has_rolling28 = False
    if "yhat28" in cols:
        # sample cheap: null count vs height on a projection
        has_rolling28 = res_df.select(pl.col("yhat28").null_count()).item() < res_df.height
    if not has_rolling28 and "valuehat28" in cols:
        has_rolling28 = (
            res_df.select(pl.col("valuehat28").null_count()).item() < res_df.height
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
        "version": ARTIFACT_VERSION,
        "forecast_path": str(forecast_path.resolve()),
        "forecast_mtime": forecast_path.stat().st_mtime if forecast_path.exists() else 0.0,
        "forecast_fingerprint": source_fingerprint(
            forecast_path,
            n_rows=res_df.height,
        ),
        "app_version": str(getattr(settings, "APP_VERSION", "")),
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


def _clear_series_dir(sdir: Path) -> None:
    if not sdir.exists():
        return
    for p in sdir.rglob("*.parquet"):
        p.unlink()
    for p in sorted(sdir.rglob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass


def _write_one_section_series(part: pl.DataFrame, sdir: Path, seccion: str) -> None:
    """Escribe una partición seccion=<s>/data.parquet ordenada."""
    out = sdir / f"seccion={seccion}"
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


def _write_series_partitioned(series: pl.DataFrame, adir: Path) -> None:
    """Escribe series/seccion=<s>/data.parquet ordenado por unique_id, ds."""
    sdir = _series_dir(adir)
    _clear_series_dir(sdir)
    sdir.mkdir(parents=True, exist_ok=True)

    if series.height == 0:
        return

    # Garantizar columna seccion para particionar
    if "seccion" not in series.columns:
        series = series.with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion")
        )
    else:
        series = series.with_columns(
            pl.when(pl.col("seccion").is_null() | (pl.col("seccion") == ""))
            .then(pl.col("unique_id").str.split("||").list.get(0))
            .otherwise(pl.col("seccion"))
            .alias("seccion")
        )

    n_sec = series.select(pl.col("seccion").n_unique()).item()
    _progress(f"  escribiendo series particionadas ({n_sec} secciones)…")
    for i, (seccion, part) in enumerate(
        series.partition_by("seccion", as_dict=True).items(), start=1
    ):
        sec_val = seccion[0] if isinstance(seccion, tuple) else seccion
        sec_str = str(sec_val)
        _write_one_section_series(part, sdir, sec_str)
        if i == 1 or i % 5 == 0 or i == n_sec:
            _progress(f"  serie sección {i}/{n_sec}: {sec_str} ({part.height:,} filas)")


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

    Diseñado para datasets grandes:
    - logs de avance por etapa + RSS
    - paneles WMAPE slim (solo columnas necesarias)
    - libera intermedios con gc
    - index vectorizado (sin O(ids × skus) en Python)
    - series escritas por sección
    """
    t_all = time.perf_counter()
    fpath = Path(forecast_path or settings.FORECAST_PATH)
    if not fpath.exists():
        raise FileNotFoundError(f"No existe forecast: {fpath}")

    adir = Path(out_dir) if out_dir else artifacts_dir(fpath)
    adir.mkdir(parents=True, exist_ok=True)
    _progress(f"Construyendo artefactos dashboard en {adir}")
    _progress(f"Fuente: {fpath} ({fpath.stat().st_size / (1024**2):.1f} MB)")

    # ── 1. Carga ──────────────────────────────────────────────────────────
    t = time.perf_counter()
    _progress("1/6 Cargando forecast.parquet…")
    res_df = backend.load_forecast_parquet(fpath)
    all_ids = res_df["unique_id"].unique().to_list()
    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    _progress(
        f"1/6 Cargado: {res_df.height:,} filas, {len(all_ids):,} unique_ids, "
        f"{len(cols)} cols",
        t,
    )

    # ── 2. Labels ─────────────────────────────────────────────────────────
    t = time.perf_counter()
    _progress("2/6 Construyendo label_map / desc_map…")
    label_map, desc_map = backend.build_label_maps(res_df)
    _progress(f"2/6 Labels: {len(label_map):,} ids", t)

    # ── 3. Metrics (slim + liberar unit dfs) ───────────────────────────────
    t = time.perf_counter()
    _progress("3/6 Calculando metrics (WMAPE OOS bottom-up)…")
    metric_parts: list[pl.DataFrame] = []

    slim = _slim_for_wmape(res_df)
    _progress(f"  panel slim WMAPE: {slim.height:,} filas, {len(slim.columns)} cols")

    unit_df_u = backend.prepare_unit_df(slim, "Unidades", has_value)
    _progress("  WMAPE Unidades…")
    metric_parts.append(_wmape_table_for_unit(unit_df_u, all_ids, "Unidades"))
    del unit_df_u
    gc.collect()

    if has_value:
        unit_df_v = backend.prepare_unit_df(slim, "Valor ($)", has_value)
        _progress("  WMAPE Valor ($)…")
        metric_parts.append(_wmape_table_for_unit(unit_df_v, all_ids, "Valor ($)"))
        del unit_df_v
        gc.collect()

    del slim
    gc.collect()

    metrics = pl.concat([p for p in metric_parts if p.height], how="diagonal_relaxed")
    del metric_parts
    gc.collect()

    if metrics.height:
        # unique_id se repite por unidad. parts_df 1 fila por uid.
        parts_df = _parse_uid_parts(
            metrics["unique_id"].unique().to_list()
        ).unique(subset=["unique_id"])
        metrics = metrics.drop(
            [c for c in ("store", "sku", "node_kind", "seccion") if c in metrics.columns]
        )
        metrics = metrics.join(parts_df, on="unique_id", how="left")
        del parts_df

    if metrics.height and "unidad" in metrics.columns:
        metrics = metrics.unique(subset=["unique_id", "unidad"], keep="first")
    elif metrics.height:
        metrics = metrics.unique(subset=["unique_id"], keep="first")

    _progress(f"3/6 Metrics: {metrics.height:,} filas", t)

    # ── 4. Series slim + SKU puro ─────────────────────────────────────────
    t = time.perf_counter()
    _progress("4/6 Materializando series slim + SKU puro…")
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
        _progress(f"  SKU puro: {len(pure_ids):,} ids, {pure.height:,} filas")

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
        del pure
        gc.collect()
    else:
        del pure

    # Ya no necesitamos el panel completo
    n_rows_source = res_df.height
    # Mantener res_df solo para flags has_rolling28 en index (proyección barata)
    # → liberamos después de index
    _progress(f"4/6 Series en memoria: {series.height:,} filas", t)

    # ── 5. Index + labels ─────────────────────────────────────────────────
    t = time.perf_counter()
    _progress("5/6 Construyendo index + labels…")
    index = _build_index(res_df, all_ids, metrics, fpath, label_map, desc_map)
    # Corregir n_rows_source por si res_df se usó antes de pure
    index["n_rows_source"] = n_rows_source
    labels = _labels_df(label_map, desc_map)
    del label_map, desc_map, all_ids
    del res_df
    gc.collect()
    _progress(
        f"5/6 Index: {len(index['secciones'])} secciones, labels={labels.height:,}",
        t,
    )

    # ── 6. Write ──────────────────────────────────────────────────────────
    t = time.perf_counter()
    _progress("6/6 Escribiendo artefactos a disco…")
    _index_path(adir).write_text(
        json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    labels.write_parquet(
        _labels_path(adir), compression="zstd", compression_level=3, statistics=True
    )
    metrics.write_parquet(
        _metrics_path(adir), compression="zstd", compression_level=3, statistics=True
    )
    del labels, metrics
    gc.collect()

    _write_series_partitioned(series, adir)
    del series
    gc.collect()

    legacy = _series_legacy_path(adir)
    if legacy.exists():
        legacy.unlink()

    _progress(
        f"✓ Artefactos listos: {len(index['secciones'])} secciones, "
        f"series particionadas en {adir}",
        t_all,
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


def load_leaves_for_scope(
    seccion: str,
    store: str | None = None,
    sku: str | None = None,
    forecast_path: Path | None = None,
    *,
    unidad: str = "Unidades",
    has_value: bool = False,
) -> pl.DataFrame:
    """
    Hojas sku+tienda del alcance desde artefactos (métricas bottom-up).
    """
    adir = artifacts_dir(forecast_path)
    sdir = _series_dir(adir)
    part = sdir / f"seccion={seccion}" / "data.parquet"
    if part.exists():
        lf = pl.scan_parquet(part)
    else:
        legacy = _series_legacy_path(adir)
        if not legacy.exists():
            return pl.DataFrame()
        lf = pl.scan_parquet(legacy).filter(
            (pl.col("unique_id") == seccion)
            | pl.col("unique_id").str.starts_with(f"{seccion}||")
        )

    lf = lf.filter(pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)

    if store is not None and sku is not None:
        uid = f"{seccion}||T:{store}||S:{sku}"
        lf = lf.filter(pl.col("unique_id") == uid)
    elif store is not None:
        prefix = f"{seccion}||T:{store}||"
        lf = lf.filter(pl.col("unique_id").str.starts_with(prefix))
    elif sku is not None:
        needle = f"||S:{sku}"
        lf = lf.filter(pl.col("unique_id").str.ends_with(needle))

    df = lf.collect()
    if df.height == 0:
        return df
    return backend.prepare_unit_df(df, unidad, has_value)


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
