from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_nonactive_table_has_oos_forecast_and_percent_formats():
    dash = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    data = (ROOT / "app" / "dashboard_data.py").read_text(encoding="utf-8")
    assert 'fc_label = "Pronóstico in-sample" if is_in else "Pronóstico OOS"' in data
    assert 'total_col = "forecast_in_sample_total" if is_in else "forecast_oos_total"' in data
    assert '"Pronóstico OOS", "Cohort"' in dash
    assert 'format="%.2f%%"' in dash
    assert 'format="%.1f%%"' in dash
    assert '"wMAPE (%)": [float(w) * 100.0 if w is not None else None' in data
    assert '"BIAS (%)": [float(b) * 100.0 if b is not None else None' in data
    assert '"% ≠0": [float(p) for p in pct]' in data


def test_excel_percent_is_literal_not_scaled_again():
    dash = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert "0.00\"%\"" in dash


def test_nonactive_oos_forecast_is_full_horizon_artifact():
    art = (ROOT / "app" / "dashboard_artifacts.py").read_text(encoding="utf-8")
    assert 'total_col = "forecast_oos_total" if period_type == "out_sample" else "forecast_in_sample_total"' in art
    assert 'unit_df.filter(pl.col("period_type") == period_type)' in art
    assert '.agg(pl.col("yhat").sum().alias(total_col))' in art
    assert '"forecast_oos_total": pl.Float64' in art
