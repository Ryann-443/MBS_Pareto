"""
Suitability Identification Module

This module overlays grid polygons with a binary suitability raster to assign
an unsuitability flag to each grid cell.

"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask


def add_unsuitability_to_grid(
    grid_path: str,
    suitability_raster_path: str,
    output_path: str,
) -> tuple:
    """
    Add an ``unsuit`` attribute to each grid feature by overlaying with a
    binary suitability raster.

    Parameters:
        grid_path: Path to the input grid GeoPackage.
        suitability_raster_path: Path to the suitability raster (GeoTIFF).
        output_path: Path for the output GeoPackage with the new attribute.

    Returns:
        Tuple of (suitable_count, unsuitable_count, error_count).
    """
    # --- Validate input files ---
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    if not os.path.exists(suitability_raster_path):
        raise FileNotFoundError(f"Raster file not found: {suitability_raster_path}")

    # --- Read grid ---
    print(f"Reading grid file: {grid_path}")
    gdf = gpd.read_file(grid_path)
    original_crs = gdf.crs
    print(f"Grid CRS: {original_crs}")

    if gdf.crs is None:
        print("Warning: Grid has no CRS; defaulting to EPSG:4326")
        gdf.set_crs("EPSG:4326", inplace=True)
        original_crs = gdf.crs

    # --- Read suitability raster ---
    print(f"Reading suitability raster: {suitability_raster_path}")
    with rasterio.open(suitability_raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        nodata = src.nodata
        print(f"  Raster CRS: {raster_crs}")
        print(f"  Raster size: {src.width} x {src.height}")
        print(f"  Raster bounds: {raster_bounds}")
        print(f"  NoData value: {nodata}")

        # --- Reproject grid if CRS differs ---
        working_gdf = gdf.copy()
        if working_gdf.crs != raster_crs:
            print(f"Reprojecting grid from {working_gdf.crs} to {raster_crs}")
            working_gdf = working_gdf.to_crs(raster_crs)

        # --- Check spatial overlap ---
        gdf_bounds = working_gdf.total_bounds
        overlap = not (
            gdf_bounds[2] < raster_bounds[0]
            or gdf_bounds[0] > raster_bounds[2]
            or gdf_bounds[3] < raster_bounds[1]
            or gdf_bounds[1] > raster_bounds[3]
        )
        if not overlap:
            raise ValueError("No spatial overlap between grid and raster!")
        print("Spatial overlap detected. Processing features...")

        # --- Process each feature ---
        processed_count = 0
        suitable_count = 0
        unsuitable_count = 0
        error_count = 0
        unsuit_values = []

        for i, row in working_gdf.iterrows():
            try:
                geometry = row.geometry
                geom_bounds = geometry.bounds

                # Feature entirely outside raster extent -> mark unsuitable
                if (
                    geom_bounds[2] < raster_bounds[0]
                    or geom_bounds[0] > raster_bounds[2]
                    or geom_bounds[3] < raster_bounds[1]
                    or geom_bounds[1] > raster_bounds[3]
                ):
                    unsuit_values.append(1)
                    unsuitable_count += 1
                    continue

                # Clip raster to feature geometry
                try:
                    masked_data, _ = mask(
                        src, [geometry], crop=True, filled=False, all_touched=True
                    )

                    # Extract valid pixels (excluding NoData)
                    if nodata is not None:
                        valid_pixels = masked_data[0][
                            (~masked_data.mask[0]) & (masked_data[0] != nodata)
                        ]
                    else:
                        valid_pixels = masked_data[0][~masked_data.mask[0]]

                    if len(valid_pixels) > 0:
                        # Check for suitable pixels (value == 0)
                        if np.any(valid_pixels == 0):
                            unsuit_values.append(0)  # At least one suitable pixel
                            suitable_count += 1
                        else:
                            unsuit_values.append(1)  # All pixels unsuitable
                            unsuitable_count += 1
                        processed_count += 1
                    else:
                        # No valid pixels -> mark unsuitable
                        unsuit_values.append(1)
                        unsuitable_count += 1

                except Exception as mask_error:
                    unsuit_values.append(1)
                    unsuitable_count += 1
                    error_count += 1
                    if error_count <= 10:
                        print(f"  Warning: Feature {i} mask failed: {mask_error}")

            except Exception as feature_error:
                unsuit_values.append(1)
                unsuitable_count += 1
                error_count += 1
                if error_count <= 10:
                    print(f"  Warning: Feature {i} error: {feature_error}")

            if (i + 1) % 5000 == 0:
                print(f"  Processed: {i + 1}/{len(working_gdf)} features")

        # --- Attach results and save ---
        gdf["unsuit"] = unsuit_values
        print(f"Saving results to: {output_path}")
        gdf.to_file(output_path, driver="gpkg")

        print(f"\nProcessing complete!")
        print(f"  Total features: {len(gdf)}")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Suitable (unsuit=0): {suitable_count}")
        print(f"  Unsuitable (unsuit=1): {unsuitable_count}")
        print(f"  Errors: {error_count}")

        return suitable_count, unsuitable_count, error_count


def validate_result(input_path: str, output_path: str) -> None:
    """
    Validate spatial consistency between input and output grids.

    Checks CRS, bounding box, feature count, and prints unsuitability
    statistics for the output file.

    Parameters:
        input_path: Path to the original grid file.
        output_path: Path to the output grid file.
    """
    print("\n=== Spatial consistency check ===")
    input_gdf = gpd.read_file(input_path)
    output_gdf = gpd.read_file(output_path)

    print(f"  Input CRS:  {input_gdf.crs}")
    print(f"  Output CRS: {output_gdf.crs}")

    # Compare bounding boxes
    bounds_diff = np.abs(
        np.array(input_gdf.total_bounds) - np.array(output_gdf.total_bounds)
    )
    if np.max(bounds_diff) < 1e-6:
        print("  OK: Bounding boxes match")
    else:
        print(f"  WARNING: Bounding box mismatch (max diff = {np.max(bounds_diff)})")

    # Compare feature counts
    if len(input_gdf) == len(output_gdf):
        print("  OK: Feature counts match")
    else:
        print(f"  WARNING: Feature count mismatch: "
              f"{len(input_gdf)} vs {len(output_gdf)}")

    # Unsuitability statistics
    if "unsuit" in output_gdf.columns:
        counts = output_gdf["unsuit"].value_counts().sort_index()
        total = len(output_gdf)
        for val, cnt in counts.items():
            pct = cnt / total * 100
            label = "suitable" if val == 0 else "unsuitable"
            print(f"  unsuit={val} ({label}): {cnt} ({pct:.2f}%)")
