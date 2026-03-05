"""
Result Export and Merging Module

Merges Pareto ranking results (from an Excel table) back into the original
geospatial grid (GeoPackage).
"""

import geopandas as gpd
import pandas as pd


def merge_grid_with_ranking(
    grid_path: str,
    excel_path: str,
    output_path: str,
    grid_id_col: str = "GridID",
    rank_col: str = "pareto_rank",
) -> None:
    """
    Merge Pareto ranking results into the original grid GeoPackage.

    A left join is used so that all grid features are retained.  Grid cells
    without a matching rank (e.g., those excluded from ranking) will have
    a null value in the ``rank_col`` column.

    Parameters:
        grid_path: Path to the input grid GeoPackage.
        excel_path: Path to the Excel file containing ranking results.
        output_path: Path for the output merged GeoPackage.
        grid_id_col: Name of the grid identifier column (must exist in both
            the grid and the Excel file).
        rank_col: Name of the Pareto rank column in the Excel file.
    """
    # --- Read grid ---
    print(f"Reading grid: {grid_path}")
    gdf = gpd.read_file(grid_path)

    # --- Read Excel ranking ---
    print(f"Reading ranking table: {excel_path}")
    excel_df = pd.read_excel(excel_path)

    # Validate required columns
    for col in [grid_id_col, rank_col]:
        if col not in excel_df.columns:
            raise ValueError(f"Column '{col}' not found in Excel file")

    # --- Merge (left join) ---
    print("Merging ranking into grid...")
    merged = gdf.merge(
        excel_df[[grid_id_col, rank_col]],
        on=grid_id_col,
        how="left",
    )

    matched = merged[rank_col].notna().sum()
    total = len(merged)
    missing = total - matched
    print(f"  Total grid features: {total}")
    print(f"  Matched with ranking: {matched}")
    print(f"  Unmatched (null rank): {missing}")

    # --- Save ---
    print(f"Saving merged grid to: {output_path}")
    merged.to_file(output_path, driver="GPKG")
    print("Export complete.")
