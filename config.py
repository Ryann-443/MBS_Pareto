"""
This file centralizes all file paths and key parameters used throughout
the pipeline. Modify the paths below to match your local data layout.
"""

import os

# =============================================================================
# Input file paths
# =============================================================================

# Grid file with greenness prediction, accessibility, and PVOUT attributes
GRID_INPUT_PATH = r"path/to/your/grid.gpkg"

# Suitability raster file (binary: 0 = suitable, non-0 = unsuitable)
SUITABILITY_RASTER_PATH = r"path/to/your/PVsuitability.tif"

# =============================================================================
# Output file paths
# =============================================================================

# Output directory
OUTPUT_DIR = r"path/to/output"

# Grid with unsuitability attribute added
GRID_WITH_UNSUIT_PATH = os.path.join(
    OUTPUT_DIR, "Grid_with_unsuit.gpkg"
)

# Excel file with full Pareto ranking results
RANKING_EXCEL_PATH = os.path.join(
    os.path.dirname(OUTPUT_DIR), "Ranking_results.xlsx"
)

# Final output grid with Pareto ranks merged
FINAL_GRID_PATH = os.path.join(OUTPUT_DIR, "Grid_rank.gpkg")

# =============================================================================
# Model parameters
# =============================================================================

# Number of quantile bins for objective discretization.
DISCRETIZATION_BINS = 10

# Column name mappings: original name -> standardized name
COLUMN_MAPPING = {
    "Predicted_Probability": "greenness",    # Ecological benefit
    "pvout_mean": "generation",              # Power generation potential
    "access": "accessibility",               # Infrastructure accessibility
    "unsuit": "unsuitability",               # Unsuitability flag (0/1)
}

# Objectives to maximize (higher is better)
MAXIMIZE_OBJECTIVES = ["greenness", "generation"]

# Objectives to minimize (lower is better; inverted during normalization)
MINIMIZE_OBJECTIVES = ["accessibility"]
