
## Overview

This repository contains the custom code for the Spatial Pareto non-dominated sorting described in the associated publication. To identify priorities for PV deployment on suitable marginal land, we applied a spatial Pareto non-dominated sorting algorithm at the grid scale. This approach could avoid the compensation effects and weight sensitivity inherent in multi-objective decision-making. Given that the suitability assessment has considered multiple bottom-line constraints, we defined three key optimization objectives for prioritizing PV deployment based on natural endowment and potential impacts: vegetation restoration probability, PV power efficiency, and infrastructure accessibility. Higher restoration probability, greater PV efficiency, and better accessibility correspond to higher deployment priority. Grid cells were ranked by their ability to simultaneously optimize the three objectives, with top-ranked cells designated as priority sites.

## Project Structure

```
pareto_pv_code/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── config.py                          # Configuration and file paths
├── main.py                            # Main entry point (runs the full pipeline)
└── src/
    ├── __init__.py
    ├── suitability_identification.py  # Step 1: Suitability identification
    ├── pareto_ranking.py              # Step 2: Multi-objective Pareto ranking
    └── export_results.py              # Step 3: Result export and merging
```


## Usage

### Step 1: Configure file paths

Edit `config.py` to set your input/output file paths:

```python
GRID_INPUT_PATH = "path/to/your/grid.gpkg"
SUITABILITY_RASTER_PATH = "path/to/your/PVsuitability.tif"
OUTPUT_DIR = "path/to/output"
```

### Step 2: Run the full pipeline

```bash
python main.py
```


## Methodology

### 1. Suitability Identification

The module overlays grid features with a suitability raster to assign binary unsuitability flags:

- `unsuit = 0`: At least one suitable pixel (value = 0) exists within the grid cell.
- `unsuit = 1`: All pixels within the grid cell are unsuitable (value != 0).

### 2. Multi-Objective Pareto Ranking

Three objectives are optimized simultaneously using non-dominated sorting:

| Objective              | Direction | Source Column            |
|------------------------|-----------|--------------------------|
| Ecological benefit     | Maximize  | `Predicted_Probability`  |
| Power generation       | Maximize  | `pvout_mean`             |
| Accessibility          | Minimize  | `access`                 |


**Normalization:**
- Maximize objectives use standard min-max scaling to [0, 1].
- Minimize objectives use inverted min-max scaling: `(max - value) / (max - min)`, so that higher normalized values always indicate better performance.

**Exclusion criteria:**
- Grids with `greenness == 0`, `generation == 0`, or `unsuitability == 1` are excluded from ranking.

### 3. Result Export

Merges Pareto ranking results back to the original geospatial grid for visualization and analysis in GIS software.


## License

MIT License

## Citation

If you use this code, please cite the associated publication and this code repository.
