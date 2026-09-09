"""v12.5 read-only rolling backtest: pooled LightGBM SHAPE residual at SKU-total.

Why shape-only
--------------
The pooled Ridge diagnostic showed the only supervised component that generalized
was ``ridge_shape`` in section 1; ``ridge_level`` and ``ridge_full`` were strongly
unstable.  This diagnostic therefore tests whether non-linearity/interactions can
improve DAILY SHAPE while preserving the current v12.5 28-day SKU total exactly.

Causality
---------
* One pooled model per section/target/pseudo-OOS; never SKU×store.
* Training uses only older closed 28-day blocks.
* Production OOS is used once as a final holdout after PRE-OOS selection.
* Current v12.5 occurrence and 28d×84d store-share remain fixed.
* No OOS/future actual or price is used as a feature.

Model target
------------
For each SKU/day in a closed block, fit a robust residual of normalized daily
shape rather than level::

    z_shape = log(actual_share_smooth) - log(base_share_smooth)

The LightGBM prediction is shrunk by gamma and converted back to a daily shape.
For every SKU and 28-day horizon, the candidate is then renormalized to the
EXACT current v12.5 SKU-total.  Thus any improvement is attributable only to
shape.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from statistics import median

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings
from app.forecasting.leaf_v12 import _block_candidate
from app.v12_sku_total_calendar_driver_backtest import (
    EPS,
    _baseline_block,
    _calendar_table,
    _eval_leaf,
    _eval_sku,
    _history_from_selected,
    _incumbent_rows,
    _oos_bounds,
    _pct,
    _pp,
)
from app.v12_sku_total_pooled_ridge_backtest import LAGS, _attach_causal_features

GAMMAS = (0.25, 0.50, 0.75, 1.00)
SHAPE_SMOOTH_FRAC = 0.02
SHAPE_Z_CLIP = 1.5


def _require_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as exc:  # pragma: no cover - user environment dependent
        raise RuntimeError(
            "Este diagnóstico requiere LightGBM. Ejecútalo sin modificar el proyecto con:\n"
            "  uv run --with lightgbm python -m app.v12_sku_total_pooled_lgbm_shape_backtest --blocks 12 --history-max 16 --n-jobs 8"
        ) from exc
    return lgb


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x.astype(np.float64, copy=False), 1e-12, None))


def _block_total(x: np.ndarray, sku: list[str]) -> np.ndarray:
    # Input block is normally sorted by SKU/day, but use Polars to avoid relying on order.
    tmp = pl.DataFrame({"sku": sku, "x": x}).with_columns(pl.col("x").sum().over("sku").alias("tot"))
    return tmp.get_column("tot").to_numpy().astype(np.float64, copy=False)


def _shape_share(x: np.ndarray, total: np.ndarray) -> np.ndarray:
    # 2% pseudo-total spread uniformly over the 28-day horizon; scale-free for U and V.
    pc = np.maximum(1e-9, SHAPE_SMOOTH_FRAC * np.maximum(total, 0.0) / 28.0)
    den = np.maximum(total + 28.0 * pc, 1e-9)
    return (np.maximum(x, 0.0) + pc) / den


def _cat_values(blocks: list[pl.DataFrame]) -> list[str]:
    vals: set[str] = set()
    for b in blocks:
        if "_cat" not in b.columns:
            continue
        vals.update(str(x) for x in b.get_column("_cat").drop_nulls().unique().to_list())
    return sorted(vals)


def _feature_names(event_cols: list[str], cats: list[str]) -> list[str]:
    names = [
        "log_base", "log_base_total", "log_base_share",
        "log_lag28", "log_lag56", "log_lag84", "log_lag364", "log_lag365",
        "log_lag28_share", "log_lag56_share", "log_lag84_share", "log_lag364_share", "log_lag365_share",
        "recent_state", "trend_state", "price_state",
        "h_sin", "h_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos",
    ]
    names += [f"event:{c.replace('_ev_', '')}" for c in event_cols]
    names += [f"cat:{c}" for c in cats]
    return names


def _shape_matrix(
    df: pl.DataFrame,
    suffix: str,
    event_cols: list[str],
    cats: list[str],
    training: bool,
):
    base = df.get_column(f"base_{suffix}").to_numpy().astype(np.float64, copy=False)
    sku = [str(x) for x in df.get_column("_v12_sku").to_list()]
    n = len(base)
    names = _feature_names(event_cols, cats)
    if n == 0:
        return np.empty((0, len(names))), np.empty(0), np.empty(0), names

    base_total = _block_total(base, sku)
    base_share = _shape_share(base, base_total)
    cols: list[np.ndarray] = [
        np.log1p(np.maximum(base, 0.0)),
        np.log1p(np.maximum(base_total, 0.0)),
        _safe_log(base_share),
    ]

    lag_values: dict[int, np.ndarray] = {}
    lag_totals: dict[int, np.ndarray] = {}
    for lag in LAGS:
        v = df.get_column(f"lag{lag}_{suffix}").to_numpy().astype(np.float64, copy=False)
        lag_values[lag] = v
        lag_totals[lag] = _block_total(v, sku)
        cols.append(np.log1p(np.maximum(v, 0.0)))
    for lag in LAGS:
        cols.append(_safe_log(_shape_share(lag_values[lag], lag_totals[lag])))

    cols.extend([
        df.get_column(f"recent_state_{suffix}").to_numpy().astype(np.float64, copy=False),
        df.get_column(f"trend_state_{suffix}").to_numpy().astype(np.float64, copy=False),
        df.get_column("_price_state").to_numpy().astype(np.float64, copy=False),
    ])

    h = df.get_column("_h").to_numpy().astype(np.float64, copy=False)
    dow = df.get_column("_dow").to_numpy().astype(np.float64, copy=False)
    mon = df.get_column("_mon").to_numpy().astype(np.float64, copy=False)
    cols.extend([
        np.sin(2*np.pi*h/28.0), np.cos(2*np.pi*h/28.0),
        np.sin(2*np.pi*(dow-1.0)/7.0), np.cos(2*np.pi*(dow-1.0)/7.0),
        np.sin(2*np.pi*(mon-1.0)/12.0), np.cos(2*np.pi*(mon-1.0)/12.0),
    ])
    for ev in event_cols:
        if ev in df.columns:
            cols.append(df.get_column(ev).to_numpy().astype(np.float64, copy=False))
        else:
            cols.append(np.zeros(n, dtype=np.float64))

    cat_arr = np.array([str(x) if x is not None else "" for x in df.get_column("_cat").to_list()], dtype=object)
    for c in cats:
        cols.append((cat_arr == c).astype(np.float64))

    X = np.column_stack(cols).astype(np.float32, copy=False)
    X[~np.isfinite(X)] = 0.0
    if not training:
        return X, np.empty(0), np.empty(0), names

    actual = df.get_column(f"actual_{suffix}").to_numpy().astype(np.float64, copy=False)
    actual_total = _block_total(actual, sku)
    actual_share = _shape_share(actual, actual_total)
    z = np.clip(_safe_log(actual_share) - _safe_log(base_share), -SHAPE_Z_CLIP, SHAPE_Z_CLIP)

    # Align with official positive-day metric while preserving volume emphasis.
    mask = np.isfinite(actual) & (actual > 0) & np.isfinite(z) & (base_total > EPS) & (actual_total > EPS)
    w = np.clip(np.sqrt(np.maximum(actual[mask], 0.0)), 0.1, 1000.0).astype(np.float32)
    return X[mask], z[mask].astype(np.float32), w, names


def _fit_model(blocks: list[pl.DataFrame], suffix: str, event_cols: list[str], n_jobs: int, seed: int):
    lgb = _require_lightgbm()
    cats = _cat_values(blocks)
    xs=[]; ys=[]; ws=[]; names=None
    for b in blocks:
        X,y,w,nm = _shape_matrix(b, suffix, event_cols, cats, True)
        if X.shape[0]:
            xs.append(X); ys.append(y); ws.append(w); names=nm
    if not xs:
        return None, cats, names or _feature_names(event_cols,cats), 0
    X=np.vstack(xs); y=np.concatenate(ys); w=np.concatenate(ws)
    model=lgb.LGBMRegressor(
        objective="huber",
        n_estimators=160,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=500,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.20,
        reg_lambda=8.0,
        max_bin=127,
        random_state=int(seed),
        n_jobs=int(n_jobs),
        verbosity=-1,
    )
    model.fit(X,y,sample_weight=w)
    return model,cats,names or [],len(y)


def _predict_candidates(df: pl.DataFrame, suffix: str, event_cols: list[str], fit):
    model,cats,names,_=fit
    out=df.select("_v12_sku","ds",f"actual_{suffix}",f"base_{suffix}")
    if model is None or df.height==0:
        for g in GAMMAS:
            tag=str(g).replace(".","p")
            out=out.with_columns(pl.col(f"base_{suffix}").alias(f"fc_lgbm_shape_g{tag}_{suffix}"))
        return out,names
    X,_,_,_=_shape_matrix(df,suffix,event_cols,cats,False)
    z=np.asarray(model.predict(X),dtype=np.float64)
    z=np.clip(z,-SHAPE_Z_CLIP,SHAPE_Z_CLIP)
    base=df.get_column(f"base_{suffix}").to_numpy().astype(np.float64,copy=False)
    sku=[str(x) for x in df.get_column("_v12_sku").to_list()]
    base_total=_block_total(base,sku)
    pc=np.maximum(1e-9,SHAPE_SMOOTH_FRAC*np.maximum(base_total,0.0)/28.0)
    base_adj=np.maximum(base,0.0)+pc
    for g in GAMMAS:
        raw=base_adj*np.exp(float(g)*z)
        raw_sum=_block_total(raw,sku)
        fc=np.where(raw_sum>EPS,raw*base_total/np.maximum(raw_sum,EPS),base)
        fc=np.maximum(fc,0.0)
        tag=str(g).replace(".","p")
        out=out.with_columns(pl.Series(f"fc_lgbm_shape_g{tag}_{suffix}",fc))
    return out,names


def _candidate_names():
    return tuple(f"lgbm_shape_g{str(g).replace('.', 'p')}" for g in GAMMAS)


CANDIDATES=_candidate_names()


def _summary(rows:list[dict],sec:str,suffix:str,metric:str):
    base={r["eval_rank"]:r for r in rows if r["sec"]==sec and r["suffix"]==suffix and r["candidate"]=="current_selector"}
    out=[]
    for cand in CANDIDATES:
        rs=[r for r in rows if r["sec"]==sec and r["suffix"]==suffix and r["candidate"]==cand]
        if not rs: continue
        ae=sum(float(r[f"{metric}_ae"]) for r in rs); den=sum(float(r[f"{metric}_den"]) for r in rs)
        bae=sum(float(base[r["eval_rank"]][f"{metric}_ae"]) for r in rs); bden=sum(float(base[r["eval_rank"]][f"{metric}_den"]) for r in rs)
        pooled=ae/max(EPS,den); bpooled=bae/max(EPS,bden)
        gains=[base[r["eval_rank"]][f"{metric}_wmape"]-r[f"{metric}_wmape"] for r in rs]
        recent=[g for r,g in zip(rs,gains) if r["eval_rank"]<=4]
        out.append({"candidate":cand,"pooled":pooled,"gain":bpooled-pooled,"median":median(gains),"recent4":sum(recent)/len(recent) if recent else float("nan"),"win":sum(g>0 for g in gains)/len(gains),"worst":min(gains)})
    return sorted(out,key=lambda r:(-r["gain"],r["pooled"]))


def _print_summary(title:str,rows:list[dict]):
    print(f"\n{title}")
    print("  candidate                 pooled    gain    median  recent4  win%   worst")
    for r in rows:
        print(f"  {r['candidate']:24s} {_pct(r['pooled'])} {_pp(r['gain'])} {_pp(r['median'])} {_pp(r['recent4'])} {100*r['win']:5.0f}% {_pp(r['worst'])}")


def _robust_pick(summary:list[dict]):
    eligible=[r for r in summary if r["gain"]>0.0025 and r["recent4"]>0 and r["win"]>=0.58 and r["worst"]>=-0.02]
    return max(eligible,key=lambda r:(r["gain"],r["recent4"],r["win"])) if eligible else None


def _importance(model,names:list[str],topn:int=12):
    if model is None: return []
    gains=np.asarray(model.booster_.feature_importance(importance_type="gain"),dtype=np.float64)
    if gains.size==0 or gains.sum()<=0: return []
    gains=gains/gains.sum()
    idx=np.argsort(gains)[::-1][:topn]
    return [(names[i],float(gains[i])) for i in idx if i<len(names)]


def _run_section(selected_path:Path,forecast_path:Path,sec:str,oos_origin:dt.date,blocks:int,history_max:int,active_min:int,n_jobs:int,out_dir:Path):
    prepared,sku_daily,sku_first,uid_first,ids,meta,cat_name=_history_from_selected(selected_path,sec)
    incumbent=_incumbent_rows(forecast_path,sec)
    block_days=int(getattr(settings,"RLS_BLOCK_DAYS",28))
    max_block=blocks+history_max
    all_lo=oos_origin-dt.timedelta(days=block_days*max_block+370)
    all_hi=oos_origin+dt.timedelta(days=block_days-1)
    _,event_cols=_calendar_table(all_lo,all_hi)

    print(f"\n=== SECCIÓN {sec} | pooled LightGBM SHAPE rolling PRE-OOS={blocks} | history={history_max} | categoria={cat_name} ===")
    print(f"event drivers={len(event_cols)} | gammas={GAMMAS} | n_jobs={n_jobs}")
    print(f"precomputando {max_block} bloques current SKU-total + causal lags ...")
    daily_by_id:dict[int,pl.DataFrame]={}
    for bid in range(1,max_block+1):
        start=oos_origin-dt.timedelta(days=block_days*bid); end=start+dt.timedelta(days=block_days-1)
        b=_baseline_block(prepared,sku_daily,sku_first,ids,meta,incumbent,start,end,bid)
        daily_by_id[bid]=_attach_causal_features(b,sku_daily,start,end)
        print(f"  block {bid:02d}/{max_block}: {start}→{end} SKU={daily_by_id[bid].select('_v12_sku').n_unique()}")

    rows=[]
    for eval_rank in range(blocks,0,-1):
        start=oos_origin-dt.timedelta(days=block_days*eval_rank); end=start+dt.timedelta(days=block_days-1)
        target=daily_by_id[eval_rank]
        hist_ids=list(range(eval_rank+1,min(max_block,eval_rank+history_max)+1))
        hist_blocks=[daily_by_id[h] for h in hist_ids]
        share_rows=_block_candidate(prepared,sku_daily,sku_first,uid_first,ids,incumbent,start,end).select("unique_id","_v12_sku","ds","v12_store_share_y","v12_store_share_value")
        for suffix in ("y","v"):
            fit=_fit_model(hist_blocks,suffix,event_cols,n_jobs,seed=20260831+eval_rank+(0 if suffix=="y" else 1000))
            pred,_=_predict_candidates(target,suffix,event_cols,fit)
            bsku=_eval_sku(target,f"base_{suffix}",suffix)
            bleaf=_eval_leaf(share_rows,target,f"base_{suffix}",prepared,start,end,suffix,active_min)
            rows.append({"sec":sec,"eval_rank":eval_rank,"start":start,"end":end,"suffix":suffix,"candidate":"current_selector","sku_wmape":bsku[0],"sku_bias":bsku[1],"sku_ae":bsku[2],"sku_den":bsku[3],"leaf_wmape":bleaf[0],"leaf_bias":bleaf[1],"leaf_ae":bleaf[2],"leaf_den":bleaf[3],"active":bleaf[4],"train_rows":fit[3]})
            for cand in CANDIDATES:
                fc=f"fc_{cand}_{suffix}"
                csku=_eval_sku(pred,fc,suffix); cleaf=_eval_leaf(share_rows,pred,fc,prepared,start,end,suffix,active_min)
                rows.append({"sec":sec,"eval_rank":eval_rank,"start":start,"end":end,"suffix":suffix,"candidate":cand,"sku_wmape":csku[0],"sku_bias":csku[1],"sku_ae":csku[2],"sku_den":csku[3],"leaf_wmape":cleaf[0],"leaf_bias":cleaf[1],"leaf_ae":cleaf[2],"leaf_den":cleaf[3],"active":cleaf[4],"train_rows":fit[3]})
        by=next(r for r in rows if r["sec"]==sec and r["eval_rank"]==eval_rank and r["suffix"]=="y" and r["candidate"]=="current_selector")
        bv=next(r for r in rows if r["sec"]==sec and r["eval_rank"]==eval_rank and r["suffix"]=="v" and r["candidate"]=="current_selector")
        print(f"  pseudo-OOS {start}→{end} | current leaf U/V={_pct(by['leaf_wmape'])}/{_pct(bv['leaf_wmape'])}")

    pl.DataFrame(rows).write_csv(out_dir/f"pooled_lgbm_shape_rolling_sec_{sec}.csv")
    picks={}
    print(f"\n--- RESUMEN PRE-OOS sec={sec} ---")
    for suffix,label in (("y","Unidades"),("v","Valor ($)")):
        sku_sum=_summary(rows,sec,suffix,"sku"); leaf_sum=_summary(rows,sec,suffix,"leaf")
        _print_summary(f"SKU-TOTAL | sec={sec} | {label}",sku_sum)
        _print_summary(f"LEAF share v12.5 fijo | sec={sec} | {label}",leaf_sum)
        pick=_robust_pick(leaf_sum); picks[suffix]=pick
        if pick:
            print(f"  ==> PRE-OOS winner causal: {pick['candidate']} | gain={_pp(pick['gain'])} recent4={_pp(pick['recent4'])} win={100*pick['win']:.0f}% worst={_pp(pick['worst'])}")
        else:
            print("  ==> sin pooled-LightGBM shape challenger que cumpla guardas robustas")

    oos_end=oos_origin+dt.timedelta(days=block_days-1)
    oos=_attach_causal_features(_baseline_block(prepared,sku_daily,sku_first,ids,meta,incumbent,oos_origin,oos_end,0),sku_daily,oos_origin,oos_end)
    hist_ids=list(range(1,min(history_max,max_block)+1)); hist_blocks=[daily_by_id[h] for h in hist_ids]
    oos_share=_block_candidate(prepared,sku_daily,sku_first,uid_first,ids,incumbent,oos_origin,oos_end).select("unique_id","_v12_sku","ds","v12_store_share_y","v12_store_share_value")
    hold=[]
    print(f"\n=== HOLDOUT OOS PRODUCCIÓN sec={sec} | {oos_origin}→{oos_end} ===")
    for suffix,label in (("y","Unidades"),("v","Valor ($)")):
        fit=_fit_model(hist_blocks,suffix,event_cols,n_jobs,seed=20260831+(0 if suffix=="y" else 1000))
        pred,names=_predict_candidates(oos,suffix,event_cols,fit)
        bsku=_eval_sku(oos,f"base_{suffix}",suffix); bleaf=_eval_leaf(oos_share,oos,f"base_{suffix}",prepared,oos_origin,oos_end,suffix,active_min)
        imp=_importance(fit[0],names,10)
        print(f"  {label:10s}: train rows={fit[3]:,} | top gain: "+", ".join(f"{n}={100*v:.1f}%" for n,v in imp[:6]))
        pick=picks[suffix]
        if pick is None:
            print(f"  {label:10s}: current SKU={_pct(bsku[0])} LEAF={_pct(bleaf[0])} | sin challenger causal")
            hold.append({"sec":sec,"suffix":suffix,"candidate":"current_selector","sku_wmape":bsku[0],"sku_bias":bsku[1],"leaf_wmape":bleaf[0],"leaf_bias":bleaf[1],"gain_leaf":0.0})
            continue
        cand=pick["candidate"]; fc=f"fc_{cand}_{suffix}"
        csku=_eval_sku(pred,fc,suffix); cleaf=_eval_leaf(oos_share,pred,fc,prepared,oos_origin,oos_end,suffix,active_min); gain=bleaf[0]-cleaf[0]
        print(f"  {label:10s}: PRE-OOS pick={cand:20s} | current LEAF={_pct(bleaf[0])} → candidate={_pct(cleaf[0])} gain={_pp(gain)} | BIAS {_pct(bleaf[1])}→{_pct(cleaf[1])}")
        hold.append({"sec":sec,"suffix":suffix,"candidate":cand,"sku_wmape":csku[0],"sku_bias":csku[1],"leaf_wmape":cleaf[0],"leaf_bias":cleaf[1],"gain_leaf":gain,"baseline_sku_wmape":bsku[0],"baseline_leaf_wmape":bleaf[0],"baseline_leaf_bias":bleaf[1]})
    pl.DataFrame(hold).write_csv(out_dir/f"pooled_lgbm_shape_holdout_sec_{sec}.csv")
    return rows,hold


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--blocks",type=int,default=12)
    ap.add_argument("--history-max",type=int,default=16)
    ap.add_argument("--active-min",type=int,default=int(getattr(settings,"OOS_ACTIVE_MIN_NONZERO_DAYS",7)))
    ap.add_argument("--n-jobs",type=int,default=8)
    args=ap.parse_args()
    if args.blocks<8: raise ValueError("--blocks debe ser >=8")
    if args.history_max<13: raise ValueError("--history-max debe ser >=13")
    _require_lightgbm()

    selected_path=Path(settings.SELECTED_PATH)
    forecast_path=Path(getattr(settings,"FORECAST_PATH",Path(getattr(settings,"OUT_DIR","data/output"))/"forecast.parquet"))
    if not selected_path.exists(): raise FileNotFoundError(selected_path)
    if not forecast_path.exists(): raise FileNotFoundError(forecast_path)
    out_dir=Path(getattr(settings,"OUT_DIR","data/output"))/"diagnostics"/"v12_sku_total_pooled_lgbm_shape"
    out_dir.mkdir(parents=True,exist_ok=True)

    print("DIAGNÓSTICO v12.5 — POOLED LIGHTGBM SHAPE SKU-TOTAL")
    print(f"Selected: {selected_path}")
    print(f"Forecast: {forecast_path}")
    print(f"Salida  : {out_dir}")
    print("Read-only. Un LightGBM pooled por sección/target; nunca SKU×store. OOS solo holdout final.")
    print(f"blocks={args.blocks} | history_max={args.history_max} | gammas={GAMMAS} | n_jobs={args.n_jobs}")
    print("Target = residual de SHAPE normalizada; total SKU 28d, occurrence y store-share v12.5 quedan fijos exactamente.")

    bounds=_oos_bounds(forecast_path); all_rows=[]; holdouts=[]
    for sec in sorted(bounds):
        origin,_=bounds[sec]
        r,h=_run_section(selected_path,forecast_path,sec,origin,int(args.blocks),int(args.history_max),int(args.active_min),int(args.n_jobs),out_dir)
        all_rows.extend(r); holdouts.extend(h)
    if all_rows: pl.DataFrame(all_rows).write_csv(out_dir/"pooled_lgbm_shape_rolling_all.csv")
    if holdouts: pl.DataFrame(holdouts).write_csv(out_dir/"pooled_lgbm_shape_holdout_all.csv")

    print("\n=== DECISIÓN HOLDOUT ===")
    good=[r for r in holdouts if r.get("candidate")!="current_selector" and float(r.get("gain_leaf",0.0))>0]
    for r in holdouts:
        print(f"  sec={r['sec']} {'U' if r['suffix']=='y' else 'V'} | {r['candidate']} | holdout leaf={_pct(float(r['leaf_wmape']))} gain={_pp(float(r.get('gain_leaf',0.0)))}")
    print(f"- Mejoran {len(good)}/{len(holdouts)} paneles OOS con picks hechos solo PRE-OOS.")
    print("- 4/4 con ganancia material => candidato real para v12.6 shape-only.")
    print("- Si sec1 mejora pero sec23 no, mantener ridge/GBDT como challenger por sección; no promover globalmente.")
    print("- Si tampoco mejora, el siguiente cuello no es no-linealidad de shape con estas variables: hacen falta señales externas/comerciales adicionales (promoción real, stock/listing, precio futuro planificado) o reformular el target.")
    print("- OOS nunca participa en entrenamiento, gamma ni selección PRE-OOS.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
