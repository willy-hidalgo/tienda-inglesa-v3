from pathlib import Path
import ast, re

root=Path('/mnt/data/p2d')
app=root/'app'; fp=app/'forecasts.py'
s=fp.read_text()
t=ast.parse(s)

def find(name, parent=None):
    nodes=(parent.body if parent is not None else t.body)
    for n in nodes:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==name:
            return n
    raise KeyError(name)

def seg(n): return ast.get_source_segment(s,n)

runner=find('RLSForecastRunner')
wm=find('_compute_wmape',runner)
dens=find('densify_section_panel')
wm_text=seg(wm)
# Convert staticmethod body into module function, preserving exact implementation.
wm_text=wm_text.replace('    def _compute_wmape(', 'def compute_wmape(', 1)
# Remove 4-space indentation from entire function body after signature.
wm_lines=wm_text.splitlines()
wm_lines=[wm_lines[0]]+[ln[4:] if ln.startswith('    ') else ln for ln in wm_lines[1:]]
wm_text='\n'.join(wm_lines)+'\n'

metrics='''"""Forecasting evaluation metrics.\n\nPure metric functions extracted from the forecasting runner.\n"""\n\nfrom __future__ import annotations\n\nimport polars as pl\n\n'''+wm_text
(app/'forecasting'/'metrics.py').write_text(metrics)

panel='''"""Panel construction utilities for forecasting."""\n\nfrom __future__ import annotations\n\nimport datetime as dt\nimport logging\n\nimport numpy as np\nimport polars as pl\n\nlogger = logging.getLogger(__name__)\n\n'''+seg(dens)+'\n'
(app/'forecasting'/'panel.py').write_text(panel)

# Remove densify function from forecasts and add import.
lines=s.splitlines(True)
for node in sorted([dens], key=lambda n:n.lineno, reverse=True):
    del lines[node.lineno-1:node.end_lineno]
new=''.join(lines)
# Remove compute_wmape method from runner.
# Re-parse after densify removal and locate method by name.
t2=ast.parse(new)
r2=next(n for n in t2.body if isinstance(n,ast.ClassDef) and n.name=='RLSForecastRunner')
wm2=next(n for n in r2.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_wmape')
lines=new.splitlines(True)
del lines[wm2.lineno-1:wm2.end_lineno]
new=''.join(lines)
# Insert imports.
anchor='from app.forecasting.features import CalendarFeatureBuilder\n'
imports='from app.forecasting.metrics import compute_wmape\nfrom app.forecasting.panel import densify_section_panel\n'
if imports not in new:
    new=new.replace(anchor,anchor+imports)
# Replace static call body with compatibility wrapper.
needle='''    @staticmethod\n    def _compute_wmape(res_df: pl.DataFrame) -> pl.DataFrame:\n'''
# Since method removed, insert after __init__ block at class start, before next method.
marker='    def _correction_factor('
wrapper='''    @staticmethod\n    def _compute_wmape(res_df: pl.DataFrame) -> pl.DataFrame:\n        """Backward-compatible wrapper around the pure WMAPE metric."""\n        return compute_wmape(res_df)\n\n'''
if wrapper not in new:
    new=new.replace(marker,wrapper+marker,1)
fp.write_text(new)

# Update exports.
init=app/'forecasting'/'__init__.py'
it=init.read_text()
if 'compute_wmape' not in it:
    it=it.replace('from .features import CalendarFeatureBuilder\n','from .features import CalendarFeatureBuilder\nfrom .metrics import compute_wmape\nfrom .panel import densify_section_panel\n')
    it=it.replace('    "CalendarFeatureBuilder",\n','    "CalendarFeatureBuilder",\n    "compute_wmape",\n    "densify_section_panel",\n')
init.write_text(it)

# Migrate densify test to new module; retain one compatibility test via wrapper in runner indirectly not needed.
tp=root/'tests'/'test_densify.py'
txt=tp.read_text()
txt=txt.replace('from forecasts import densify_section_panel','from app.forecasting.panel import densify_section_panel')
tp.write_text(txt)
# Migrate WMAPE direct runner test to metric function, while wrapper remains tested implicitly by existing compatibility if needed.
tw=root/'tests'/'test_wmape.py'
txt=tw.read_text()
txt=txt.replace('from forecasts import RLSForecastRunner','from forecasts import RLSForecastRunner\nfrom app.forecasting.metrics import compute_wmape')
txt=txt.replace('w = RLSForecastRunner._compute_wmape(df)','w = compute_wmape(df)')
tw.write_text(txt)

print('forecasts lines:',len(fp.read_text().splitlines()))
print('new modules:',[(p.name,len(p.read_text().splitlines())) for p in (app/'forecasting').glob('*.py')])
