"""Tests de configuración, fechas por sección y etiquetas."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import settings


def test_focus_sections():
    assert settings.FOCUS_SECTIONS == ["1", "23"]


def test_secciones_keys():
    assert set(settings.SECCIONES.keys()) == {"1", "23"}
    assert "locales" in settings.SECCIONES["1"]
    assert "local_names" in settings.SECCIONES["23"]


def test_aggregation_levels_order():
    keys = list(settings.AGGREGATION_LEVELS.keys())
    assert keys == ["SECCION", "STORE_ID", "SKU_ID"]


def test_nearest_sunday_on_or_before():
    assert settings.nearest_sunday_on_or_before(dt.date(2025, 11, 1)) == dt.date(
        2025, 10, 26
    )
    assert settings.nearest_sunday_on_or_before(dt.date(2025, 10, 26)) == dt.date(
        2025, 10, 26
    )


def test_nearest_sunday_on_or_after():
    assert settings.nearest_sunday_on_or_after(dt.date(2025, 11, 1)) == dt.date(
        2025, 11, 2
    )


def test_nearest_monday_on_or_after():
    # 2025-11-30 es domingo → lunes = 2025-12-01
    assert settings.nearest_monday_on_or_after(dt.date(2025, 11, 30)) == dt.date(
        2025, 12, 1
    )
    # ya es lunes
    assert settings.nearest_monday_on_or_after(dt.date(2025, 12, 1)) == dt.date(
        2025, 12, 1
    )
    # sábado
    assert settings.nearest_monday_on_or_after(dt.date(2025, 11, 29)) == dt.date(
        2025, 12, 1
    )


def test_compute_train_window_aligned_28():
    first = dt.date(2024, 5, 3)  # viernes
    requested_end = dt.date(2025, 11, 30)
    start, end = settings.compute_train_window(first, requested_end, block_days=28)
    assert start.weekday() == 0
    assert end.weekday() == 6
    assert end <= requested_end
    assert ((end - start).days + 1) % 28 == 0



def test_section_horizons_23():
    last = dt.date(2026, 2, 15)
    hz = settings.section_horizons(
        "23",
        first_data=dt.date(2024, 5, 1),
        last_actual=last,
    )
    assert hz["test_end"] == last
    assert hz["test_start"] == last - dt.timedelta(days=27)
    assert (hz["test_end"] - hz["test_start"]).days + 1 == 28
    assert ((hz["train_end"] - hz["train_start"]).days + 1) % 28 == 0
    assert hz["forecast_start"] == last + dt.timedelta(days=1)
    assert (hz["forecast_end"] - hz["forecast_start"]).days + 1 == 28



def test_section_horizons_1():
    last = dt.date(2026, 5, 6)
    hz = settings.section_horizons(
        "1",
        first_data=dt.date(2024, 6, 1),
        last_actual=last,
    )
    assert hz["test_end"] == last
    assert hz["test_start"] == last - dt.timedelta(days=27)
    assert (hz["test_end"] - hz["test_start"]).days + 1 == 28
    assert ((hz["train_end"] - hz["train_start"]).days + 1) % 28 == 0
    assert hz["forecast_start"] == last + dt.timedelta(days=1)



def test_make_unique_id_and_split_roundtrip():
    assert settings.make_unique_id("1") == "1"
    assert settings.make_unique_id("1", store="00122") == "1||T:00122"
    assert settings.make_unique_id("1", sku="SKU99") == "1||S:SKU99"
    assert (
        settings.make_unique_id("1", store="00122", sku="SKU99")
        == "1||T:00122||S:SKU99"
    )
    # orden de construcción no importa; el id siempre es T antes de S
    assert settings.make_unique_id("1", sku="SKU99", store="00122") == "1||T:00122||S:SKU99"

    for uid, expected in [
        ("1", {"seccion": "1", "store": None, "sku": None}),
        ("1||T:00122", {"seccion": "1", "store": "00122", "sku": None}),
        ("1||S:SKU99", {"seccion": "1", "store": None, "sku": "SKU99"}),
        (
            "1||T:00122||S:SKU99",
            {"seccion": "1", "store": "00122", "sku": "SKU99"},
        ),
    ]:
        assert settings.split_unique_id(uid) == expected


def test_display_label_all_combinations():
    assert "Sección" in settings.display_label("1")

    lbl_store = settings.display_label("1||T:00122", store_name="CENTRAL")
    assert "00122" in lbl_store
    assert "CENTRAL" in lbl_store
    assert not lbl_store.startswith("1||")

    lbl_sku_only = settings.display_label("1||S:SKU99", sku_desc="LECHE")
    assert "SKU99" in lbl_sku_only
    assert "LECHE" in lbl_sku_only
    assert "@" not in lbl_sku_only  # sin tienda no hay "@ tienda"

    lbl_combo = settings.display_label(
        "1||T:00122||S:SKU99", sku_desc="LECHE", store_name="CENTRAL"
    )
    assert "SKU99" in lbl_combo
    assert "LECHE" in lbl_combo
    assert "00122" in lbl_combo
    assert "CENTRAL" in lbl_combo


def test_ranking_code_and_description():
    assert settings.ranking_code("1") == "1"
    assert settings.ranking_code("1||T:00122") == "00122"
    assert settings.ranking_code("1||S:SKU9") == "SKU9"
    assert settings.ranking_code("1||T:00122||S:SKU9") == "00122"  # prioridad tienda
    assert settings.ranking_description("1||T:00122", store_name="CENTRAL") == "CENTRAL"
    assert (
        settings.ranking_description("1||S:SKU9", sku_desc="LECHE") == "LECHE"
    )


def test_levels_order():
    assert list(settings.AGGREGATION_LEVELS.keys()) == [
        "SECCION",
        "STORE_ID",
        "SKU_ID",
    ]


def test_client_required_rls_defaults():
    assert settings.RLS_FIT_MODE == "expanding_28"
    assert settings.RLS_BLOCK_DAYS == 28
    assert settings.METRICS_MODE == "rolling_28"
    assert settings.FAST_LEAF_DRIVER_FACTOR_CLIP == (0.20, 5.00)


def test_v11_leaf_and_metric_contract():
    assert settings.APP_VERSION == "12.9.12"
    assert settings.RLS_FIT_MODE == "expanding_28"
    assert settings.RLS_BLOCK_DAYS == 28
    assert settings.METRICS_MODE == "rolling_28"
    assert settings.METRICS_START_DAY == 29
    assert settings.LEAF_INITIAL_LEVEL_DAYS == 28
    assert 0 < settings.LEAF_SES_ALPHA <= 1
    assert settings.LEAF_SES_ALPHA in settings.LEAF_SES_ALPHA_CANDIDATES
    assert settings.LEAF_PARENT_CANDIDATES == ("store", "section")
    assert settings.LEAF_PARENT_DRIVER_STRENGTH == 1.0
    assert settings.LEAF_REGIME_HISTORY_BLOCKS == 5
    assert settings.LEAF_RESET_TRANSIENT is True
    assert settings.RANKING_SKU_MIN_NONZERO_POINTS == 15


def test_train_windows_are_exact_multiples_of_28():
    for sec in settings.FOCUS_SECTIONS:
        hz = settings.section_horizons(sec)
        n_days = (hz["train_end"] - hz["train_start"]).days + 1
        assert n_days % settings.RLS_BLOCK_DAYS == 0
        assert n_days >= settings.RLS_BLOCK_DAYS


def test_section_horizon_oos_is_latest_28_actual_days():
    first = dt.date(2024, 5, 1)
    last = dt.date(2026, 5, 10)
    for sec in settings.FOCUS_SECTIONS:
        hz = settings.section_horizons(sec, first, last)
        assert hz["test_end"] == last
        assert hz["test_start"] == last - dt.timedelta(days=27)
        assert (hz["test_end"] - hz["test_start"]).days + 1 == 28
        assert hz["train_end"] == hz["test_start"] - dt.timedelta(days=1)
        assert (hz["train_end"] - hz["train_start"]).days + 1 >= 28
        assert ((hz["train_end"] - hz["train_start"]).days + 1) % 28 == 0


def test_forecast_only_never_overlaps_known_actuals():
    first = dt.date(2024, 5, 1)
    last = dt.date(2026, 4, 30)
    hz = settings.section_horizons("1", first, last)
    assert hz["test_start"] == dt.date(2026, 4, 3)
    assert hz["test_end"] == last
    assert hz["forecast_start"] == last + dt.timedelta(days=1)
    assert (hz["forecast_end"] - hz["forecast_start"]).days + 1 == 28
    assert hz["actual_extension_start"] is None
    assert hz["observed_tail_start"] is None


def test_no_actual_extension_or_observed_tail_after_oos():
    hz = settings.section_horizons(
        "1", dt.date(2024, 5, 1), dt.date(2026, 6, 28)
    )
    assert hz["test_end"] == dt.date(2026, 6, 28)
    assert hz["actual_extension_start"] is None
    assert hz["actual_extension_end"] is None
    assert hz["observed_tail_start"] is None
    assert hz["observed_tail_end"] is None
    assert hz["forecast_start"] == dt.date(2026, 6, 29)

