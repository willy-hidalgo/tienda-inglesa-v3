#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import settings
from app.categories_selector import DemandAnalysisPipeline
from app.forecasts import DataAggregator, CalendarFeatureBuilder, HolidayCalendar
import polars as pl

def main():
    print("=== Understanding Data Structure ===")

    # Load the selected data
    pipeline = DemandAnalysisPipeline()
    selected_df = pipeline.run()
    print(f"Selected data shape: {selected_df.shape}")
    print(f"Selected data columns: {selected_df.columns}")

    # Look at data for section 1
    sec1_df = selected_df.filter(pl.col("SECCION") == "1")
    print(f"Section 1 data shape: {sec1_df.shape}")

    # Aggregate to see what we get at different levels
    aggregator = DataAggregator(
        settings.DATE_COLUMN,
        settings.QUANTITY_COLUMN,
        settings.PRICE_COLUMN,
        settings.AGGREGATION_LEVELS,
    )

    # Aggregate section 1 data
    agg_df = aggregator.aggregate(sec1_df)
    print(f"\nAggregated data shape: {agg_df.shape}")
    print(f"Aggregated data columns: {agg_df.columns}")

    # Show some sample data
    print("\nSample aggregated data (first 5 rows):")
    print(agg_df.head())

    # Check what unique_ids we have at each level
    print("\nUnique IDs by level:")
    for depth in range(3):
        level_count = pl.col("unique_id").str.count_matches(r"\|\|", literal=False)
        depth_df = agg_df.filter(level_count == depth)
        if depth_df.height > 0:
            unique_ids = depth_df["unique_id"].unique().to_list()
            print(f"  Depth {depth} ({'section' if depth==0 else 'store' if depth==1 else 'sku'}): {len(unique_ids)} unique IDs")
            if depth == 0:
                print(f"    Section IDs: {unique_ids[:5]}")  # Show first 5
            elif depth == 1:
                print(f"    Store IDs: {unique_ids[:5]}")  # Show first 5
            else:
                print(f"    SKU IDs: {unique_ids[:5]}")  # Show first 5
        else:
            print(f"  Depth {depth}: No data")

    # Now let's see what happens when we add features
    print("\n=== Adding Features ===")

    # Get first data date for section 1
    first_data = sec1_df[settings.DATE_COLUMN].min()
    if hasattr(first_data, 'date'):
        first_data = first_data.date()
    print(f"First data date for section 1: {first_data}")

    # Get section horizons
    hz = settings.section_horizons("1", first_data)
    print(f"Section 1 horizons: {hz}")

    # Filter to training period
    train_df = sec1_df.filter(
        (pl.col(settings.DATE_COLUMN) >= hz["train_start"]) &
        (pl.col(settings.DATE_COLUMN) <= hz["train_end"])
    )
    print(f"Training data shape: {train_df.shape}")

    # Aggregate training data
    train_agg = aggregator.aggregate(train_df)
    print(f"Training aggregated shape: {train_agg.shape}")

    # Add features
    holiday_calendar = HolidayCalendar(settings.HOLIDAYS, 2026)  # Using 2026 as current year
    feature_builder = CalendarFeatureBuilder(holiday_calendar)

    # Get driver columns (what would be used for modeling)
    train_with_features = feature_builder.extract_drivers(train_agg)
    print(f"After adding features shape: {train_with_features.shape}")
    print(f"After adding features columns: {train_with_features.columns}")

    # Determine driver columns (same logic as in forecasts.py)
    driver_cols = [
        c
        for c in train_with_features.columns
        if c
        not in (
            "ds",
            "y",
            "value",
            "valuehat",
            "unique_id",
            "conteo_sku",
            "sku_desc",
            "store_name",
            "seccion",
            "intercept",
        )
    ]
    print(f"\nDriver columns ({len(driver_cols)}): {driver_cols}")

    # Show some feature data
    print("\nSample feature data:")
    feature_sample = train_with_features.select(["ds", "y"] + driver_cols[:5])  # Show first 5 drivers
    print(feature_sample.head())

if __name__ == "__main__":
    main()