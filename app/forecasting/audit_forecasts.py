"""Post-run production audit for temporal alignment and OOS metrics."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import settings


def wmape_bias(df: pl.DataFrame, actual: str, forecast: str) -> tuple[float, float, int]:
    q=df.filter(pl.col(actual).is_finite() & pl.col(forecast).is_finite() & (pl.col(actual)!=0))
    if not q.height: return float('nan'), float('nan'), 0
    den=float(q.select(pl.col(actual).abs().sum()).item())
    ae=float(q.select((pl.col(actual)-pl.col(forecast)).abs().sum()).item())
    err=float(q.select((pl.col(forecast)-pl.col(actual)).sum()).item())
    return (ae/den if den else float('nan'), err/den if den else float('nan'), q.height)


def audit(path: Path | None=None) -> pl.DataFrame:
    path=path or Path(settings.FORECAST_PATH)
    df=pl.read_parquet(path).with_columns(pl.col('ds').cast(pl.Date))
    rows=[]
    for sec in sorted(df.get_column('seccion').drop_nulls().cast(pl.Utf8).unique().to_list()):
        q=df.filter(pl.col('seccion').cast(pl.Utf8)==sec)
        for unit,a,f in [('units','y','yhat'),('value','value','valuehat')]:
            o=q.filter((pl.col('period_type')=='out_sample') & pl.col('unique_id').str.contains(r'\\|\\|T:') & pl.col('unique_id').str.contains(r'\\|\\|S:'))
            w,b,n=wmape_bias(o,a,f)
            dates=o.select('ds').unique().sort('ds')
            contiguous=(dates.height==28 and (dates['ds'][-1]-dates['ds'][0]).days==27) if dates.height else False
            rows.append({'section':sec,'unit':unit,'level':'bottom_up_leaf','wmape':w,'bias':b,'n':n,'oos_28_contiguous':contiguous})
        # Direct RLS nodes, useful to diagnose parent shape independently.
        for level,expr in [('section',pl.col('unique_id')==sec),('store',pl.col('unique_id').str.count_matches(r'\\|\\|')==1)]:
            qq=q.filter((pl.col('period_type')=='out_sample') & expr)
            for unit,a,f in [('units','y','yhat'),('value','value','valuehat')]:
                w,b,n=wmape_bias(qq,a,f)
                rows.append({'section':sec,'unit':unit,'level':level,'wmape':w,'bias':b,'n':n,'oos_28_contiguous':None})
    return pl.DataFrame(rows)

if __name__=='__main__':
    out=audit()
    print(out.sort(['section','unit','level']))
