#!/usr/bin/env python3
"""
Validación numérica del wMAPE bottom-up.

Método (única fuente de verdad):
  1. Hojas = unique_id con forma sec||T:tienda||S:sku
  2. Excluir period_type == forecast_only y filas con y == 0 / nulo
  3. abs_err = |y − ŷ| en cada fila hoja
  4. wMAPE(nivel) = Σ abs_err / Σ |y|  sobre el alcance del nivel:
       - sku+tienda: solo esa hoja
       - tienda:     todas las hojas de esa tienda
       - sección:    todas las hojas de la sección

Uso (desde la raíz del proyecto, donde está settings.py):

    python artifacts/validate_wmape_bottom_up.py
    python artifacts/validate_wmape_bottom_up.py --unidad Unidades
    python artifacts/validate_wmape_bottom_up.py --seccion 1 --store 00063 --sku 127360
    python artifacts/validate_wmape_bottom_up.py --in-sample-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# ── bootstrap path ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
for candidate in [ROOT, ROOT.parent, Path.cwd(), Path.cwd().parent]:
    if (candidate / "settings.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        ROOT = candidate
        break
else:
    # app/ junto al script (monorepo artifacts/)
    if (ROOT / "app" / "backend.py").exists() and str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import settings  # noqa: E402
from app import backend  # noqa: E402


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.4f}%"


def _scored_leaves_manual(unit_df: pl.DataFrame) -> pl.DataFrame:
    """Réplica explícita de backend._scored_leaves (para assert cruzado)."""
    leaves = unit_df.filter(
        pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2
    )
    if "period_type" in leaves.columns:
        leaves = leaves.filter(pl.col("period_type") != "forecast_only")
    leaves = leaves.filter(
        pl.col("y").is_not_null()
        & pl.col("yhat").is_not_null()
        & (pl.col("y") != 0)
    )
    if leaves.height == 0:
        return leaves
    return leaves.with_columns(
        (pl.col("y") - pl.col("yhat")).abs().alias("abs_err"),
        pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("store_uid"),
        pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
        pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("sku"),
    )


def wmape_manual(leaves: pl.DataFrame, mask: pl.Expr) -> dict[str, float | int]:
    sub = leaves.filter(mask)
    if sub.height == 0:
        return {
            "n_rows": 0,
            "n_leaves": 0,
            "sum_y": 0.0,
            "sum_abs_err": 0.0,
            "wmape": 0.0,
        }
    sum_y = float(sub["y"].sum())
    sum_err = float(sub["abs_err"].sum())
    return {
        "n_rows": int(sub.height),
        "n_leaves": int(sub["unique_id"].n_unique()),
        "sum_y": sum_y,
        "sum_abs_err": sum_err,
        "wmape": (sum_err / abs(sum_y)) if sum_y else 0.0,
    }


def serie_agregada_wmape(unit_df: pl.DataFrame, uid: str) -> float | None:
    """Método viejo (incorrecto para tienda/sección): error sobre la serie del nodo."""
    sub = unit_df.filter(pl.col("unique_id") == uid)
    if "period_type" in sub.columns:
        sub = sub.filter(pl.col("period_type") != "forecast_only")
    sub = sub.filter(pl.col("y").is_not_null() & (pl.col("y") != 0))
    if sub.height == 0:
        return None
    sum_y = float(sub["y"].sum())
    if sum_y == 0:
        return 0.0
    return float((sub["y"] - sub["yhat"]).abs().sum()) / abs(sum_y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seccion", default="1")
    ap.add_argument("--store", default="00063")
    ap.add_argument("--sku", default="127360")
    ap.add_argument(
        "--unidad",
        default="Valor ($)",
        choices=["Valor ($)", "Unidades"],
    )
    ap.add_argument(
        "--in-sample-only",
        action="store_true",
        help="Restringir a period_type == in_sample",
    )
    ap.add_argument(
        "--forecast-path",
        default=None,
        help="Override de settings.FORECAST_PATH",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Tolerancia relativa manual vs backend.wmape_bottom_up",
    )
    args = ap.parse_args()

    fpath = Path(args.forecast_path or settings.FORECAST_PATH)
    print("=" * 72)
    print("VALIDACIÓN wMAPE bottom-up")
    print("=" * 72)
    print(f"forecast     : {fpath}")
    print(f"exists       : {fpath.exists()}")
    print(f"unidad       : {args.unidad}")
    print(f"in_sample    : {args.in_sample_only}")
    print(f"seccion/store/sku : {args.seccion} / {args.store} / {args.sku}")

    if not fpath.exists():
        print("ERROR: no existe FORECAST_PATH", file=sys.stderr)
        return 2

    res_df = backend.load_forecast_parquet(fpath)
    has_value = "value" in res_df.columns and "valuehat" in res_df.columns
    unit_df = backend.prepare_unit_df(res_df, args.unidad, has_value)

    if args.in_sample_only and "period_type" in unit_df.columns:
        unit_df = unit_df.filter(pl.col("period_type") == "in_sample")

    uid_sec = str(args.seccion)
    uid_store = settings.make_unique_id(args.seccion, store=args.store)
    uid_leaf = settings.make_unique_id(
        args.seccion, store=args.store, sku=args.sku
    )

    leaves = _scored_leaves_manual(unit_df)
    print(f"\nHojas scorables: {leaves.height} filas | {leaves['unique_id'].n_unique() if leaves.height else 0} series")

    cases = [
        ("sku+tienda", uid_leaf, pl.col("unique_id") == uid_leaf),
        ("tienda", uid_store, pl.col("store_uid") == uid_store),
        ("seccion", uid_sec, pl.col("seccion") == uid_sec),
    ]

    # Backend vectorizado
    bu = backend.wmape_bottom_up(unit_df)
    bu_map = {
        r["unique_id"]: r
        for r in bu.to_dicts()
    }

    ok_all = True
    print("\n" + "-" * 72)
    print(f"{'nivel':<12} {'unique_id':<32} {'manual':>12} {'backend':>12} {'Δ':>12} {'OK':>4}")
    print("-" * 72)

    rows_out = []
    for nivel, uid, mask in cases:
        m = wmape_manual(leaves, mask)
        b = bu_map.get(uid)
        b_wmape = float(b["wmape"]) if b else None
        if b_wmape is None:
            delta = None
            match = False
        else:
            delta = m["wmape"] - b_wmape
            # match numérico
            denom = max(abs(m["wmape"]), abs(b_wmape), 1e-15)
            match = abs(delta) / denom <= args.rtol or abs(delta) < 1e-12
        ok_all = ok_all and match

        old = serie_agregada_wmape(unit_df, uid)

        print(
            f"{nivel:<12} {uid:<32} {_fmt_pct(m['wmape']):>12} "
            f"{_fmt_pct(b_wmape) if b_wmape is not None else '—':>12} "
            f"{(f'{delta * 100:+.6f}pp' if delta is not None else '—'):>12} "
            f"{'✓' if match else '✗':>4}"
        )
        print(
            f"  {'':12} n_rows={m['n_rows']}  n_leaves={m['n_leaves']}  "
            f"Σ|y|={m['sum_y']:,.4f}  Σabs_err={m['sum_abs_err']:,.4f}"
        )
        if old is not None and nivel != "sku+tienda":
            print(
                f"  {'':12} (ref. método viejo serie agregada) {_fmt_pct(old)}  ← no usar"
            )
        elif old is not None and nivel == "sku+tienda":
            print(
                f"  {'':12} (serie hoja = bottom-up) {_fmt_pct(old)}  "
                f"{'✓' if abs(old - m['wmape']) < 1e-12 else '✗ diverge'}"
            )

        rows_out.append(
            {
                "nivel": nivel,
                "unique_id": uid,
                "wmape_manual": m["wmape"],
                "wmape_backend": b_wmape,
                "sum_y": m["sum_y"],
                "sum_abs_err": m["sum_abs_err"],
                "n_rows": m["n_rows"],
                "n_leaves": m["n_leaves"],
                "match": match,
                "wmape_serie_agregada": old,
            }
        )

    print("-" * 72)

    # Checks estructurales
    print("\nChecks estructurales:")
    checks = []

    # 1) hoja ⊆ tienda ⊆ sección en sum_y / sum_abs_err
    by = {r["nivel"]: r for r in rows_out}
    if by["sku+tienda"]["n_rows"] and by["tienda"]["n_rows"]:
        c = by["sku+tienda"]["sum_y"] <= by["tienda"]["sum_y"] + 1e-6
        checks.append(("Σ|y| hoja ≤ Σ|y| tienda", c))
        c2 = by["sku+tienda"]["sum_abs_err"] <= by["tienda"]["sum_abs_err"] + 1e-6
        checks.append(("Σabs_err hoja ≤ Σabs_err tienda", c2))
    if by["tienda"]["n_rows"] and by["seccion"]["n_rows"]:
        c = by["tienda"]["sum_y"] <= by["seccion"]["sum_y"] + 1e-6
        checks.append(("Σ|y| tienda ≤ Σ|y| sección", c))
        c2 = by["tienda"]["sum_abs_err"] <= by["seccion"]["sum_abs_err"] + 1e-6
        checks.append(("Σabs_err tienda ≤ Σabs_err sección", c2))

    # 2) wmape_por_id coincide con bottom_up para estos ids
    tabla_ids = backend.wmape_por_id(
        [uid_leaf, uid_store, uid_sec], unit_df, fill_missing=False
    )
    for uid in (uid_leaf, uid_store, uid_sec):
        row_bu = bu_map.get(uid)
        row_id = tabla_ids.filter(pl.col("unique_id") == uid)
        if row_bu is None or row_id.height == 0:
            checks.append((f"wmape_por_id contiene {uid}", False))
            continue
        w1 = float(row_bu["wmape"])
        w2 = float(row_id["wmape"][0])
        checks.append(
            (f"wmape_por_id == bottom_up ({uid})", abs(w1 - w2) < 1e-12)
        )

    for name, passed in checks:
        print(f"  [{'✓' if passed else '✗'}] {name}")
        ok_all = ok_all and passed

    print("\n" + "=" * 72)
    if ok_all:
        print("RESULTADO: OK — manual y backend coinciden; checks estructurales OK")
        return 0
    print("RESULTADO: FAIL — revisar divergencias arriba", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
