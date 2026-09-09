import datetime as dt

import pytest


def test_temporal_aggregation_never_mixes_in_sample_and_oos_bucket():
    pl = pytest.importorskip("polars")
    from app import backend

    # Corte dentro de la misma semana: antes del fix ambos periodos se sumaban
    # en un único bucket y period_type=max() ocultaba la mezcla.
    df = pl.DataFrame(
        {
            "ds": [dt.date(2025, 12, 3), dt.date(2025, 12, 4), dt.date(2025, 12, 5), dt.date(2025, 12, 6)],
            "y": [10.0, 20.0, 30.0, 40.0],
            "yhat": [9.0, 19.0, 29.0, 39.0],
            "period_type": ["in_sample", "in_sample", "out_sample", "out_sample"],
            "unique_id": ["23"] * 4,
        }
    )
    out = backend.aggregate_temporal(df, "Semanal")
    assert out.height == 2
    assert set(out["period_type"].to_list()) == {"in_sample", "out_sample"}
    sums = {r["period_type"]: r["y"] for r in out.iter_rows(named=True)}
    assert sums["in_sample"] == 30.0
    assert sums["out_sample"] == 70.0


def test_chart_enforces_section_specific_oos_boundaries_even_with_bad_labels():
    pl = pytest.importorskip("polars")
    from app import backend

    cutoff = dt.date(2025, 12, 4)
    test_end = dt.date(2026, 1, 1)
    df = pl.DataFrame(
        {
            "ds": [dt.date(2025, 12, 4), dt.date(2025, 12, 5), dt.date(2025, 12, 6), dt.date(2026, 1, 2)],
            "y": [1.0, 2.0, 3.0, None],
            "yhat": [1.0, 2.0, 3.0, 4.0],
            # Etiqueta incorrecta deliberada dentro del OOS.
            "period_type": ["in_sample", "in_sample", "out_sample", "forecast_only"],
        }
    )
    ch = backend.build_chart_series(
        df,
        test_end,
        dt.date(2026, 1, 2),
        dt.date(2026, 1, 29),
        cutoff,
        True,
    )
    assert ch["in_ds"] == [dt.date(2025, 12, 4)]
    assert ch["oos_ds"] == [dt.date(2025, 12, 6)]
    assert ch["fcst_ds"] == [dt.date(2026, 1, 2)]


def test_global_leaf_ranking_ignores_dashboard_filters_by_contract():
    pl = pytest.importorskip("polars")
    from app.dashboard_data import global_leaf_ranking_from_metrics

    metrics = pl.DataFrame(
        {
            "unique_id": ["1||T:00001||S:10", "23||T:00022||S:20", "1", "23||T:00022"],
            "unidad": ["Unidades", "Valor ($)", "Unidades", "Valor ($)"],
            "seccion": ["1", "23", "1", "23"],
            "store": ["00001", "00022", None, "00022"],
            "sku": ["10", "20", None, None],
            "wmape": [0.20, 0.10, 0.05, 0.06],
            "bias": [-0.1, -0.2, 0.0, 0.0],
            "n_points": [28, 28, 28, 28],
            "n_with_sales": [20, 10, 28, 28],
            "metric_cohort": ["active", "active", "active", "active"],
            "sum_abs_y": [100.0, 200.0, 300.0, 400.0],
            "sum_abs_error": [20.0, 20.0, 15.0, 24.0],
        }
    )
    out = global_leaf_ranking_from_metrics(metrics)
    assert out.height == 2
    assert set(out["Sección"].to_list()) == {"1", "23"}
    assert set(out["Unidad"].to_list()) == {"Unidades", "Valor ($)"}
    assert all("||T:" in x and "||S:" in x for x in out["unique_id"].to_list())


def test_dashboard_renders_actual_as_area_and_has_excel_global_ranking():
    text = open("app/dashboard.py", encoding="utf-8").read()
    assert 'name="Actual · in-sample", mode="lines", fill="tozeroy"' in text
    assert 'name="Actual · OOS", mode="lines", fill="tozeroy"' in text
    assert "Ranking global SKU+Tienda" in text
    assert "Exportar ranking global SKU+Tienda a Excel" in text
    assert "global_leaf_ranking_unit" in text
    backend = open("app/backend.py", encoding="utf-8").read()
    assert 'group_keys.append("period_type")' in backend
    assert 'pl.col("ds").min().alias("ds")' in backend
