from pathlib import Path


def test_discrepancy_rule_contract_is_symmetric_and_material():
    from app.discrepancy import discrepancy_points

    ds = [1, 2, 3]
    out = discrepancy_points(ds, [1000.0, 20.0, 100.0], [50.0, 5000.0, 105.0])
    flagged = {row["ds"] for row in out}
    assert 1 in flagged
    assert 2 in flagged
    assert 3 not in flagged


def test_dashboard_uses_actual_vs_forecast_discrepancies_and_secondary_edp_axis():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    assert "Marcar discrepancias Actual vs Forecast" in text
    assert "discrepancy_points" in text
    assert 'yaxis="y2"' in text
    assert 'title="EDP ($/unidad)"' in text
    assert "EDP SKU+Tienda" in text


def test_leaf_audit_contains_requested_calculation_columns():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    required = [
        "Actual desestacionalizado",
        "Nivel SES antes de drivers",
        "Efecto conjunto drivers (log)",
        "Factor conjunto drivers",
        "Forecast reconstruido raw",
        "Error absoluto",
        "wMAPE numerador",
        "BIAS numerador",
        "EDP observado SKU+Tienda",
        "Discrepancia outlier",
    ]
    for item in required:
        assert item in text
